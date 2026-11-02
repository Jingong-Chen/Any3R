# Copyright (c) 2026 The Any3R Authors.
#
# This source code is licensed under the MIT license found in the LICENSE file in the root directory
# of this source tree.

"""Central camera models and their ray fields.

Any3R takes the camera of each view as a ray field: the unit direction, in the camera frame, of the
centre of every pixel. This module computes ray fields from calibrations.

Conventions
-----------
* Camera frame (OpenCV): +x right, +y down, +z along the optical axis.
* Pixel coordinates (COLMAP): the top-left corner of the image is (0, 0), so the centre of pixel
  (row i, column j) is (u, v) = (j + 0.5, i + 0.5). OpenCV puts pixel centres at integers; use
  :func:`from_opencv`, which adds the half pixel to the principal point.
* Pixels outside the lens (e.g. the corners of a circular fisheye) are marked invalid and get the
  ray (0, 0, 1); this is also how Any3R saw them in training. Their image pixels should be black.

Every model implements

    ray_map(height, width) -> rays (H, W, 3) float32 unit vectors, valid (H, W) bool
    project(rays (..., 3)) -> u (...), v (...), valid (...)

and the two are inverse to each other on the valid domain (:meth:`CameraModel.roundtrip_error`).

Models: pinhole, Brown-Conrady (OpenCV radial-tangential, optionally rational), equidistant,
Kannala-Brandt (OpenCV fisheye), MEI (unified camera model), double sphere, extended unified
(EUCM), equisolid, stereographic, FOV (Devernay-Faugeras), cylindrical panorama, equirectangular
panorama and an equirectangular window. :func:`from_colmap` and :func:`read_colmap_cameras` read
COLMAP calibrations; :func:`from_fov` builds a centred camera of a given model and field of view.
"""
import math
import os
import struct

import numpy as np

TWO_PI = 2.0 * math.pi

__all__ = [
    "CameraModel", "Pinhole", "BrownConrady", "Equidistant", "KB4", "MEI", "DoubleSphere", "EUCM",
    "Equisolid", "Stereographic", "FOV", "Cylindrical", "ERP", "ERPCrop",
    "OpenCV", "OpenCVFisheye", "KannalaBrandt", "UnifiedCamera", "Equirectangular",
    "CAMERA_KINDS", "camera_from_dict", "from_fov", "from_colmap", "from_opencv", "read_colmap_cameras",
]


def _unit(v, eps=1e-12):
    n = np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)
    return v / n


def _fov_to_theta(fov_deg, default=None):
    return math.radians(fov_deg / 2.0) if fov_deg else default


class CameraModel:
    """Base class. Subclasses implement pixel_to_ray (unnormalised directions and a validity mask)
    and project. All arithmetic is float64; ray maps are returned as float32."""

    kind = "base"

    def __init__(self, fx, fy, cx, cy, width=None, height=None):
        self.fx = float(fx); self.fy = float(fy)
        self.cx = float(cx); self.cy = float(cy)
        self.width = width; self.height = height

    # -- public API ------------------------------------------------------------------------
    def ray_map(self, height=None, width=None):
        """Unit ray of every pixel centre, (H, W, 3) float32, and the lens mask (H, W) bool.
        The size defaults to the one the camera was constructed with."""
        H = self.height if height is None else height
        W = self.width if width is None else width
        if H is None or W is None:
            raise ValueError(f"{type(self).__name__}: give the image size (here or at construction)")
        return self.unproject(int(H), int(W))

    def unproject(self, H, W):
        u, v = self._grid(H, W)
        dirs, valid = self.pixel_to_ray(u, v)
        d = _unit(dirs)
        # The half-angle limit is enforced here for every model: for MEI and double sphere the
        # validity of pixel_to_ray only says the projection equation has a solution, which for
        # some parameters admits rays well beyond the lens' field of view.
        tm = getattr(self, "theta_max", None)
        if tm is not None and tm < math.pi:
            ang = np.arccos(np.clip(d[..., 2], -1.0, 1.0))
            valid = valid & (ang <= tm + 1e-9)
        # invalid or non-finite pixels (e.g. outside a fisheye circle, where iterative inversions
        # diverge) get the ray (0, 0, 1)
        finite = np.isfinite(d).all(-1)
        valid = valid & finite
        d = np.where(valid[..., None], d, np.array([0., 0., 1.], d.dtype))
        return d.astype(np.float32), valid

    def pixel_to_ray(self, u, v):
        raise NotImplementedError

    def project(self, dirs):
        raise NotImplementedError

    def resized(self, width, height):
        """The same lens imaged at another resolution (pixel coordinates scale with the image)."""
        if self.width is None or self.height is None:
            raise ValueError("resized() needs the camera's own width and height")
        sx, sy = width / self.width, height / self.height
        cam = self._copy()
        cam.fx, cam.cx, cam.fy, cam.cy = self.fx * sx, self.cx * sx, self.fy * sy, self.cy * sy
        cam.width, cam.height = int(width), int(height)
        return cam

    def roundtrip_error(self, height=None, width=None):
        """Self-check: unproject every pixel centre, project it back, and return the largest error in
        pixels over valid pixels together with their count. Should be below 1e-4."""
        H = self.height if height is None else height
        W = self.width if width is None else width
        uu, vv = self._grid(H, W)
        dirs, valid = self.unproject(H, W)
        pu, pv, pvalid = self.project(dirs.astype(np.float64))
        m = valid & pvalid & self._in_image(pu, pv, H, W)
        if m.sum() == 0:
            return float("nan"), 0
        err = np.sqrt((pu - uu) ** 2 + (pv - vv) ** 2)[m]
        return float(err.max()), int(m.sum())

    def to_dict(self):
        d = {"model": self.kind, "width": self.width, "height": self.height}
        d.update({k: v for k, v in self._params().items()})
        return d

    def __repr__(self):
        p = ", ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in self._params().items())
        return f"{type(self).__name__}({p}, width={self.width}, height={self.height})"

    # -- helpers ---------------------------------------------------------------------------
    def _params(self):
        p = {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}
        if type(self) in (Pinhole, BrownConrady) and getattr(self, "theta_max", None) is not None:
            p["fov_deg"] = 2.0 * math.degrees(self.theta_max)
        return p

    def _copy(self):
        cam = object.__new__(type(self))
        cam.__dict__.update(self.__dict__)
        return cam

    @staticmethod
    def _grid(H, W):
        uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        return uu.astype(np.float64), vv.astype(np.float64)

    @staticmethod
    def _in_image(u, v, H, W):
        return (u >= -0.5) & (u <= W - 0.5) & (v >= -0.5) & (v <= H - 0.5)


# ------------------------------------------------------------------------------------- pinhole
class Pinhole(CameraModel):
    """Rectilinear perspective camera (COLMAP PINHOLE / SIMPLE_PINHOLE)."""

    kind = "pinhole"

    def __init__(self, fx, fy, cx, cy, fov_deg=None, width=None, height=None):
        super().__init__(fx, fy, cx, cy, width, height)
        if fov_deg:
            self.theta_max = _fov_to_theta(fov_deg)

    def pixel_to_ray(self, u, v):
        a = (u - self.cx) / self.fx
        b = (v - self.cy) / self.fy
        d = np.stack([a, b, np.ones_like(a)], -1)
        return d, np.ones(u.shape, bool)

    def project(self, dirs):
        z = dirs[..., 2]
        sz = np.where(np.abs(z) < 1e-9, 1e-9, z)
        u = self.fx * dirs[..., 0] / sz + self.cx
        v = self.fy * dirs[..., 1] / sz + self.cy
        return u, v, z > 1e-9


# ------------------------------------------------------------------------------ Brown-Conrady
class BrownConrady(CameraModel):
    """Radial-tangential distortion (COLMAP OPENCV / FULL_OPENCV, OpenCV's default model).

    x_d = x * rad + 2 p1 x y + p2 (r^2 + 2 x^2), y_d = y * rad + p1 (r^2 + 2 y^2) + 2 p2 x y with
    rad = 1 + k1 r^2 + k2 r^4 + k3 r^6, or the rational (1 + k1 r^2 + k2 r^4 + k3 r^6) /
    (1 + k4 r^2 + k5 r^4 + k6 r^6) when any of k4..k6 is non-zero.
    """

    kind = "brown_conrady"

    def __init__(self, fx, fy, cx, cy, k1=0., k2=0., p1=0., p2=0., k3=0., k4=0., k5=0., k6=0.,
                 fov_deg=None, width=None, height=None):
        super().__init__(fx, fy, cx, cy, width, height)
        self.k1, self.k2, self.p1, self.p2, self.k3 = map(float, (k1, k2, p1, p2, k3))
        self.k4, self.k5, self.k6 = map(float, (k4, k5, k6))
        if fov_deg:
            self.theta_max = _fov_to_theta(fov_deg)

    def _params(self):
        p = super()._params()
        p.update(k1=self.k1, k2=self.k2, p1=self.p1, p2=self.p2, k3=self.k3)
        if self.k4 or self.k5 or self.k6:
            p.update(k4=self.k4, k5=self.k5, k6=self.k6)
        return p

    def _radial(self, r2):
        num = 1 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 ** 3
        if self.k4 or self.k5 or self.k6:
            return num / (1 + self.k4 * r2 + self.k5 * r2 * r2 + self.k6 * r2 ** 3)
        return num

    def _distort(self, a, b):
        r2 = a * a + b * b
        rad = self._radial(r2)
        xd = a * rad + 2 * self.p1 * a * b + self.p2 * (r2 + 2 * a * a)
        yd = b * rad + self.p1 * (r2 + 2 * b * b) + 2 * self.p2 * a * b
        return xd, yd

    def pixel_to_ray(self, u, v):
        """Fixed-point undistortion. The iteration diverges for strong barrel distortion and past
        the lens' fold-over radius, so convergence is checked: pixels whose re-distorted position
        misses the input are returned invalid."""
        xd = (u - self.cx) / self.fx
        yd = (v - self.cy) / self.fy
        a, b = xd.copy(), yd.copy()
        for _ in range(50):
            r2 = a * a + b * b
            rad = self._radial(r2)
            # rad -> 0 or flips sign past the fold-over radius: freeze those pixels
            safe = np.abs(rad) > 1e-6
            rad = np.where(safe, rad, 1.0)
            dx = 2 * self.p1 * a * b + self.p2 * (r2 + 2 * a * a)
            dy = self.p1 * (r2 + 2 * b * b) + 2 * self.p2 * a * b
            an, bn = (xd - dx) / rad, (yd - dy) / rad
            a = np.where(safe, an, a)
            b = np.where(safe, bn, b)
            if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
                a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
                b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
        xr, yr = self._distort(a, b)
        resid = np.hypot(xr - xd, yr - yd)
        valid = np.isfinite(resid) & (resid < 1e-8)
        if not valid.all():
            a, b, valid = self._newton_undistort(xd, yd, a, b, valid)
        a = np.where(valid, a, 0.0)
        b = np.where(valid, b, 0.0)
        d = np.stack([a, b, np.ones_like(a)], -1)
        return d, valid

    def _newton_undistort(self, xd, yd, a, b, valid):
        """Second pass for the pixels where the fixed point failed: Newton's method on the 2-D
        distortion map, started from the radial-only solution along the pixel's own direction.
        Pixels it does not invert to 1e-8 stay invalid; converged pixels are left untouched."""
        bad = ~valid
        x0, y0 = xd[bad], yd[bad]
        rd = np.hypot(x0, y0)
        # radial-only start: solve r * rad(r) = rd by bisection on the monotone branch
        lo, hi = np.zeros_like(rd), np.full_like(rd, 1e-3)
        for _ in range(60):                              # grow hi until r * rad(r) passes rd
            grow = (hi * self._radial(hi * hi) < rd) & (hi < 1e3)
            if not grow.any():
                break
            hi = np.where(grow, hi * 2.0, hi)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            below = mid * self._radial(mid * mid) < rd
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        r0 = 0.5 * (lo + hi)
        s = np.where(rd > 1e-12, r0 / np.maximum(rd, 1e-12), 1.0)
        xa, ya = x0 * s, y0 * s
        h = 1e-7
        for _ in range(30):
            fx_, fy_ = self._distort(xa, ya)
            ex, ey = fx_ - x0, fy_ - y0
            # Jacobian by central differences (the map is smooth on the monotone branch)
            ax1, ay1 = self._distort(xa + h, ya); ax0, ay0 = self._distort(xa - h, ya)
            bx1, by1 = self._distort(xa, ya + h); bx0, by0 = self._distort(xa, ya - h)
            j11, j21 = (ax1 - ax0) / (2 * h), (ay1 - ay0) / (2 * h)
            j12, j22 = (bx1 - bx0) / (2 * h), (by1 - by0) / (2 * h)
            det = j11 * j22 - j12 * j21
            det = np.where(np.abs(det) < 1e-12, 1e-12, det)
            xa = xa - (j22 * ex - j12 * ey) / det
            ya = ya - (-j21 * ex + j11 * ey) / det
        fx_, fy_ = self._distort(xa, ya)
        ok = np.isfinite(xa) & np.isfinite(ya) & (np.hypot(fx_ - x0, fy_ - y0) < 1e-8)
        # stay on the branch the lens actually images: the distortion must still be increasing
        # radially there (beyond the fold-over radius a second, spurious solution exists)
        r = np.hypot(xa, ya)
        eps = 1e-6 * np.maximum(r, 1e-6)
        ok &= (r + eps) * self._radial((r + eps) ** 2) > (r - eps) * self._radial((r - eps) ** 2)
        a, b, valid = a.copy(), b.copy(), valid.copy()
        a[bad] = np.where(ok, xa, a[bad]); b[bad] = np.where(ok, ya, b[bad]); valid[bad] = ok
        return a, b, valid

    def project(self, dirs):
        z = dirs[..., 2]
        sz = np.where(np.abs(z) < 1e-9, 1e-9, z)
        a, b = dirs[..., 0] / sz, dirs[..., 1] / sz
        xd, yd = self._distort(a, b)
        return self.fx * xd + self.cx, self.fy * yd + self.cy, z > 1e-9


# ---------------------------------------------------------------------------------- fisheyes
class _RadialFisheye(CameraModel):
    """Fisheye with a radial mapping r(theta) and an optional field-of-view limit."""

    def __init__(self, fx, fy, cx, cy, fov_deg=None, width=None, height=None, _default_theta=math.pi):
        super().__init__(fx, fy, cx, cy, width, height)
        self.theta_max = _fov_to_theta(fov_deg, _default_theta)

    def _params(self):
        p = super()._params()
        p["fov_deg"] = 2.0 * math.degrees(self.theta_max)
        return p

    @staticmethod
    def _dir(theta, psi):
        st = np.sin(theta)
        return np.stack([st * np.cos(psi), st * np.sin(psi), np.cos(theta)], -1)


class Equidistant(_RadialFisheye):
    """Equidistant fisheye, r = f * theta."""

    kind = "equidistant"

    def pixel_to_ray(self, u, v):
        xd = (u - self.cx) / self.fx
        yd = (v - self.cy) / self.fy
        theta = np.sqrt(xd * xd + yd * yd)
        psi = np.arctan2(yd, xd)
        return self._dir(theta, psi), theta <= self.theta_max + 1e-9

    def project(self, dirs):
        d = _unit(dirs)
        theta = np.arccos(np.clip(d[..., 2], -1, 1))
        rt = np.maximum(np.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2), 1e-12)
        u = self.fx * theta * (d[..., 0] / rt) + self.cx
        v = self.fy * theta * (d[..., 1] / rt) + self.cy
        return u, v, theta <= self.theta_max + 1e-9


class KB4(CameraModel):
    """Kannala-Brandt fisheye (COLMAP / OpenCV OPENCV_FISHEYE):
    r = f * theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)."""

    kind = "kb4"

    def __init__(self, fx, fy, cx, cy, k1=0., k2=0., k3=0., k4=0., fov_deg=None, theta_max=None,
                 width=None, height=None):
        super().__init__(fx, fy, cx, cy, width, height)
        self.k = (float(k1), float(k2), float(k3), float(k4))
        if theta_max is None:
            theta_max = _fov_to_theta(fov_deg, math.pi)
        self.theta_max = float(theta_max)

    def _params(self):
        p = super()._params()
        p.update(zip(("k1", "k2", "k3", "k4"), self.k))
        p["fov_deg"] = 2.0 * math.degrees(self.theta_max)
        return p

    def _theta_d(self, t):
        k1, k2, k3, k4 = self.k
        t2 = t * t
        return t * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))

    def pixel_to_ray(self, u, v):
        xd = (u - self.cx) / self.fx
        yd = (v - self.cy) / self.fy
        td = np.sqrt(xd * xd + yd * yd)
        psi = np.arctan2(yd, xd)
        theta = np.clip(td, 0, math.pi)            # Newton on theta_d(theta) = td, clipped every
        for _ in range(25):                        # iteration so out-of-lens pixels cannot overflow
            k1, k2, k3, k4 = self.k
            t2 = theta * theta
            f = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - td
            fp = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
            theta = np.clip(theta - f / np.where(np.abs(fp) < 1e-9, 1e-9, fp), 0, math.pi)
        # with strong coefficients theta_d is not monotone and past its fold-over radius there is
        # no solution: accept only pixels that actually invert
        resid = np.abs(self._theta_d(theta) - td)
        ok = np.isfinite(resid) & (resid < 1e-9) & (theta <= self.theta_max + 1e-6)
        theta = np.where(ok, theta, 0.0)
        return _RadialFisheye._dir(theta, psi), ok

    def project(self, dirs):
        d = _unit(dirs)
        theta = np.arccos(np.clip(d[..., 2], -1, 1))
        td = self._theta_d(theta)
        rt = np.maximum(np.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2), 1e-12)
        u = self.fx * td * (d[..., 0] / rt) + self.cx
        v = self.fy * td * (d[..., 1] / rt) + self.cy
        return u, v, theta <= self.theta_max + 1e-6


class Equisolid(_RadialFisheye):
    """Equisolid-angle fisheye, r = 2 f sin(theta / 2)."""

    kind = "equisolid"

    def pixel_to_ray(self, u, v):
        xd = (u - self.cx) / self.fx
        yd = (v - self.cy) / self.fy
        r = np.sqrt(xd * xd + yd * yd)
        theta = 2.0 * np.arcsin(np.clip(r / 2.0, -1.0, 1.0))      # no solution for r > 2
        psi = np.arctan2(yd, xd)
        return self._dir(theta, psi), (r <= 2.0 + 1e-9) & (theta <= self.theta_max + 1e-9)

    def project(self, dirs):
        d = _unit(dirs)
        theta = np.arccos(np.clip(d[..., 2], -1, 1))
        rt = np.maximum(np.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2), 1e-12)
        r = 2.0 * np.sin(theta / 2.0)
        u = self.fx * r * (d[..., 0] / rt) + self.cx
        v = self.fy * r * (d[..., 1] / rt) + self.cy
        return u, v, theta <= self.theta_max + 1e-9


class Stereographic(_RadialFisheye):
    """Stereographic fisheye, r = 2 f tan(theta / 2). Without fov_deg the limit is 179 degrees off axis."""

    kind = "stereographic"

    def __init__(self, fx, fy, cx, cy, fov_deg=None, width=None, height=None):
        super().__init__(fx, fy, cx, cy, fov_deg, width, height, _default_theta=math.radians(179.0))

    def pixel_to_ray(self, u, v):
        xd = (u - self.cx) / self.fx
        yd = (v - self.cy) / self.fy
        r = np.sqrt(xd * xd + yd * yd)
        theta = 2.0 * np.arctan(r / 2.0)
        psi = np.arctan2(yd, xd)
        return self._dir(theta, psi), theta <= self.theta_max + 1e-9

    def project(self, dirs):
        d = _unit(dirs)
        theta = np.arccos(np.clip(d[..., 2], -1, 1))
        rt = np.maximum(np.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2), 1e-12)
        r = 2.0 * np.tan(np.clip(theta, 0, math.radians(179.5)) / 2.0)
        u = self.fx * r * (d[..., 0] / rt) + self.cx
        v = self.fy * r * (d[..., 1] / rt) + self.cy
        return u, v, theta <= self.theta_max + 1e-9


class FOV(_RadialFisheye):
    """FOV model of Devernay and Faugeras (COLMAP FOV): r_d = arctan(2 r_u tan(w / 2)) / w, with
    r_u the pinhole radius tan(theta)."""

    kind = "fov"

    def __init__(self, fx, fy, cx, cy, w=0.9, fov_deg=None, width=None, height=None):
        super().__init__(fx, fy, cx, cy, fov_deg, width, height)
        self.w = float(w)

    def _params(self):
        p = super()._params()
        p["w"] = self.w
        return p

    def pixel_to_ray(self, u, v):
        xd = (u - self.cx) / self.fx
        yd = (v - self.cy) / self.fy
        rd = np.sqrt(xd * xd + yd * yd)
        tw = math.tan(self.w / 2.0)
        ru = np.where(rd < 1e-9, rd, np.tan(rd * self.w) / (2.0 * tw))
        theta = np.arctan(ru)
        psi = np.arctan2(yd, xd)
        # beyond rd * w = pi / 2 the tangent changes sign: no solution there
        return self._dir(theta, psi), (rd * self.w < math.pi / 2 - 1e-6) & (theta <= self.theta_max + 1e-9)

    def project(self, dirs):
        d = _unit(dirs)
        z = np.clip(d[..., 2], 1e-9, 1)
        ru = np.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2) / z
        tw = math.tan(self.w / 2.0)
        rd = np.where(ru < 1e-9, ru, np.arctan(2.0 * ru * tw) / self.w)
        rt = np.maximum(np.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2), 1e-12)
        u = self.fx * rd * (d[..., 0] / rt) + self.cx
        v = self.fy * rd * (d[..., 1] / rt) + self.cy
        theta = np.arccos(np.clip(d[..., 2], -1, 1))
        return u, v, (d[..., 2] > 1e-6) & (theta <= self.theta_max + 1e-9)


# ------------------------------------------------------------------------- spherical models
class MEI(CameraModel):
    """Unified camera model of Mei and Rives (omnidirectional camera), parameter xi."""

    kind = "mei"

    def __init__(self, fx, fy, cx, cy, xi=0.0, fov_deg=None, width=None, height=None):
        super().__init__(fx, fy, cx, cy, width, height)
        self.xi = float(xi)
        if fov_deg:
            self.theta_max = _fov_to_theta(fov_deg)

    def _params(self):
        p = super()._params()
        p["xi"] = self.xi
        if getattr(self, "theta_max", None) is not None:
            p["fov_deg"] = 2.0 * math.degrees(self.theta_max)
        return p

    def pixel_to_ray(self, u, v):
        mx = (u - self.cx) / self.fx
        my = (v - self.cy) / self.fy
        r2 = mx * mx + my * my
        disc = 1 + (1 - self.xi * self.xi) * r2
        valid = disc >= 0
        s = (self.xi + np.sqrt(np.maximum(disc, 0))) / (r2 + 1)
        d = np.stack([s * mx, s * my, s - self.xi], -1)
        return d, valid

    def project(self, dirs):
        d = _unit(dirs)
        den = d[..., 2] + self.xi
        sden = np.where(np.abs(den) < 1e-9, 1e-9, den)
        u = self.fx * d[..., 0] / sden + self.cx
        v = self.fy * d[..., 1] / sden + self.cy
        return u, v, den > 1e-9


class DoubleSphere(CameraModel):
    """Double sphere model of Usenko et al., parameters xi and alpha."""

    kind = "double_sphere"

    def __init__(self, fx, fy, cx, cy, xi=0.0, alpha=0.5, fov_deg=None, width=None, height=None):
        super().__init__(fx, fy, cx, cy, width, height)
        self.xi = float(xi); self.alpha = float(alpha)
        if fov_deg:
            self.theta_max = _fov_to_theta(fov_deg)

    def _params(self):
        p = super()._params()
        p.update(xi=self.xi, alpha=self.alpha)
        if getattr(self, "theta_max", None) is not None:
            p["fov_deg"] = 2.0 * math.degrees(self.theta_max)
        return p

    def pixel_to_ray(self, u, v):
        mx = (u - self.cx) / self.fx
        my = (v - self.cy) / self.fy
        r2 = mx * mx + my * my
        a = self.alpha
        disc1 = 1 - (2 * a - 1) * r2
        valid = disc1 >= 0
        mz = (1 - a * a * r2) / (a * np.sqrt(np.maximum(disc1, 0)) + 1 - a)
        num = mz * self.xi + np.sqrt(np.maximum(mz * mz + (1 - self.xi * self.xi) * r2, 0))
        coef = num / (mz * mz + r2 + 1e-12)
        d = np.stack([coef * mx, coef * my, coef * mz - self.xi], -1)
        return d, valid

    def project(self, dirs):
        d = _unit(dirs)
        x, y, z = d[..., 0], d[..., 1], d[..., 2]
        d1 = np.sqrt(x * x + y * y + z * z)
        d2 = np.sqrt(x * x + y * y + (self.xi * d1 + z) ** 2)
        den = self.alpha * d2 + (1 - self.alpha) * (self.xi * d1 + z)
        sden = np.where(np.abs(den) < 1e-9, 1e-9, den)
        u = self.fx * x / sden + self.cx
        v = self.fy * y / sden + self.cy
        w1 = self.alpha / (1 - self.alpha) if self.alpha <= 0.5 else (1 - self.alpha) / self.alpha
        w2 = (w1 + self.xi) / math.sqrt(2 * w1 * self.xi + self.xi * self.xi + 1)
        return u, v, z > -w2 * d1


class EUCM(_RadialFisheye):
    """Extended unified camera model of Khomutenko et al., parameters alpha and beta."""

    kind = "eucm"

    def __init__(self, fx, fy, cx, cy, alpha=0.55, beta=1.0, fov_deg=None, width=None, height=None):
        super().__init__(fx, fy, cx, cy, fov_deg, width, height)
        self.alpha = float(alpha); self.beta = float(beta)

    def _params(self):
        p = super()._params()
        p.update(alpha=self.alpha, beta=self.beta)
        return p

    def pixel_to_ray(self, u, v):
        mx = (u - self.cx) / self.fx
        my = (v - self.cy) / self.fy
        r2 = mx * mx + my * my
        a, b = self.alpha, self.beta
        disc = 1.0 - (2.0 * a - 1.0) * b * r2
        ok = disc >= 0
        mz = (1.0 - b * a * a * r2) / (a * np.sqrt(np.maximum(disc, 0)) + (1.0 - a))
        n = np.sqrt(np.maximum(mx * mx + my * my + mz * mz, 1e-18))
        d = np.stack([mx / n, my / n, mz / n], -1)
        theta = np.arccos(np.clip(d[..., 2], -1, 1))
        return d, ok & (theta <= self.theta_max + 1e-9)

    def project(self, dirs):
        d = _unit(dirs)
        x, y, z = d[..., 0], d[..., 1], d[..., 2]
        rho = np.sqrt(self.beta * (x * x + y * y) + z * z)
        den = self.alpha * rho + (1.0 - self.alpha) * z
        sden = np.where(np.abs(den) < 1e-9, 1e-9, den)
        u = self.fx * x / sden + self.cx
        v = self.fy * y / sden + self.cy
        theta = np.arccos(np.clip(z, -1, 1))
        return u, v, (den > 1e-9) & (theta <= self.theta_max + 1e-9)


# ------------------------------------------------------------------------------- panoramas
class _Panorama(CameraModel):
    """Charts defined by the image size alone: ray_map(H, W) rebinds the camera to that size."""

    def resized(self, width, height):
        cam = self._copy()
        cam._bind(int(height), int(width))
        return cam

    def unproject(self, H, W):
        self._bind(H, W)
        u, v = self._grid(H, W)
        d, valid = self.pixel_to_ray(u, v)
        return d.astype(np.float32), valid

    def _bind(self, H, W):
        self.height, self.width = H, W

    def _wh(self):
        if self.width is None or self.height is None:
            raise ValueError(f"{type(self).__name__} needs its image size")
        return self.height, self.width

    def _params(self):
        return {}


class ERP(_Panorama):
    """Equirectangular 360 x 180 degree panorama: azimuth linear in u, elevation linear in v."""

    kind = "erp"

    def __init__(self, width=None, height=None):
        CameraModel.__init__(self, 0.0, 0.0, 0.0, 0.0, width, height)
        self.theta_max = math.pi
        if width is not None and height is not None:
            self._bind(height, width)

    def _bind(self, H, W):
        self.height, self.width = H, W
        self.fx, self.fy, self.cx, self.cy = W / TWO_PI, H / math.pi, W / 2, H / 2

    def pixel_to_ray(self, u, v):
        H, W = self._wh()
        az = (u / W - 0.5) * TWO_PI
        el = -(v / H - 0.5) * math.pi
        d = np.stack([np.cos(el) * np.sin(az), -np.sin(el), np.cos(el) * np.cos(az)], -1)
        return d, np.ones(np.shape(u), bool)

    def project(self, dirs):
        H, W = self._wh()
        x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
        r = np.maximum(np.linalg.norm(dirs, axis=-1), 1e-9)
        u = (np.arctan2(x, z) / TWO_PI + 0.5) * W
        v = (0.5 - np.arcsin(np.clip(-y / r, -1, 1)) / math.pi) * H
        return u, v, np.ones(np.shape(x), bool)


class ERPCrop(_Panorama):
    """Equirectangular window centred on the optical axis: azimuth in [-az_max, az_max] linear in
    u, elevation in [-el_max, el_max] linear in v (angles in radians). (pi, pi/2) is ERP."""

    kind = "erp_crop"

    def __init__(self, az_max, el_max, width=None, height=None):
        if not (0 < az_max <= math.pi and 0 < el_max <= math.pi / 2):
            raise ValueError(f"bad window az_max={az_max} el_max={el_max}")
        CameraModel.__init__(self, 1.0, 1.0, 0.0, 0.0, width, height)
        self.az_max, self.el_max = float(az_max), float(el_max)
        self._theta_max = None

    def _params(self):
        return {"h_fov_deg": 2.0 * math.degrees(self.az_max), "v_fov_deg": 2.0 * math.degrees(self.el_max)}

    @property
    def theta_max(self):
        """Half-angle of the smallest cone around the optical axis that contains the window."""
        if self._theta_max is not None:
            return self._theta_max
        c = (math.cos(self.el_max) * math.cos(self.az_max)) if self.az_max <= math.pi / 2 else math.cos(self.az_max)
        return math.acos(max(-1.0, min(1.0, c)))

    @theta_max.setter
    def theta_max(self, value):
        self._theta_max = None if value is None else float(value)

    def pixel_to_ray(self, u, v):
        H, W = self._wh()
        az = (u / W - 0.5) * 2.0 * self.az_max
        el = -(v / H - 0.5) * 2.0 * self.el_max
        d = np.stack([np.cos(el) * np.sin(az), -np.sin(el), np.cos(el) * np.cos(az)], -1)
        return d, np.ones(np.shape(u), bool)

    def unproject(self, H, W):
        self._bind(H, W)
        u, v = self._grid(H, W)
        d, _ = self.pixel_to_ray(u, v)
        return d.astype(np.float32), np.ones((H, W), bool)

    def project(self, dirs):
        H, W = self._wh()
        x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
        r = np.maximum(np.linalg.norm(dirs, axis=-1), 1e-9)
        az = np.arctan2(x, z)
        el = np.arcsin(np.clip(-y / r, -1, 1))
        u = (az / (2.0 * self.az_max) + 0.5) * W
        v = (0.5 - el / (2.0 * self.el_max)) * H
        return u, v, (np.abs(az) <= self.az_max + 1e-9) & (np.abs(el) <= self.el_max + 1e-9)


class Cylindrical(_Panorama):
    """360-degree cylindrical panorama: azimuth linear in u, tan(elevation) linear in v, vertical
    field of view v_fov_deg (a strip panorama stitched from a rotating camera)."""

    kind = "cylindrical"

    def __init__(self, width=None, height=None, v_fov_deg=90.0):
        self.v_half = math.radians(v_fov_deg / 2.0)
        CameraModel.__init__(self, 1.0, 1.0, 0.0, 0.0, width, height)
        self.theta_max = math.pi
        if width is not None and height is not None:
            self._bind(height, width)

    def _params(self):
        return {"v_fov_deg": 2.0 * math.degrees(self.v_half)}

    def _bind(self, H, W):
        self.height, self.width = H, W
        self.fx, self.cx, self.cy = W / TWO_PI, W / 2, H / 2
        self.fy = (H / 2.0) / math.tan(self.v_half)

    def pixel_to_ray(self, u, v):
        H, W = self._wh()
        az = (u / W - 0.5) * TWO_PI
        el = -np.arctan((v - self.cy) / self.fy)
        ce = np.cos(el)
        d = np.stack([ce * np.sin(az), -np.sin(el), ce * np.cos(az)], -1)
        return d, np.abs(el) <= self.v_half + 1e-9

    def project(self, dirs):
        H, W = self._wh()
        d = _unit(dirs)
        x, y, z = d[..., 0], d[..., 1], d[..., 2]
        az = np.arctan2(x, z)
        el = -np.arcsin(np.clip(y, -1, 1))
        u = (az / TWO_PI + 0.5) * W
        v = self.cy + self.fy * np.tan(np.clip(-el, -self.v_half, self.v_half))
        return u, v, np.abs(el) <= self.v_half + 1e-9


# aliases under the names used by COLMAP / OpenCV
OpenCV = BrownConrady
OpenCVFisheye = KannalaBrandt = KB4
UnifiedCamera = MEI
Equirectangular = ERP

_KINDS = {c.kind: c for c in (Pinhole, BrownConrady, Equidistant, KB4, MEI, DoubleSphere, EUCM, Equisolid,
                              Stereographic, FOV, Cylindrical, ERP, ERPCrop)}
CAMERA_KINDS = tuple(_KINDS)


# ------------------------------------------------------------------------------- factories
def from_fov(kind, width, height, fov_deg, principal_point=None, circular=None):
    """A camera of the given model whose field of view across the image height is fov_deg, with
    the reference shape parameters of the paper's benchmark (e.g. Kannala-Brandt k1..k4 = 0.04,
    -0.01, 0.002, -3e-4; MEI xi = 1.6; double sphere xi = 0.25, alpha = 0.55; EUCM alpha = 0.55,
    beta = 1). Use it when only the lens type and its field of view are known.

    For 'brown_conrady' fov_deg is that of the undistorted pinhole with the same focal length (the
    barrel distortion widens the actual view). For 'erp' fov_deg is ignored; for 'erp_crop' it is the horizontal extent of a window 90 degrees
    tall; for 'cylindrical' the panorama is 360 degrees wide and 90 degrees tall.
    principal_point defaults to the image centre (width / 2, height / 2).

    circular limits the view to the cone of half-angle fov_deg / 2, i.e. to a circle touching the
    top and bottom of the image for radially symmetric lenses, and blacks out the rest. It defaults
    to True for fisheye models (a circular fisheye) and False for 'pinhole' and 'brown_conrady'.
    The paper's benchmark renders every model with circular=True."""
    W, H = int(width), int(height)
    cx, cy = principal_point if principal_point is not None else (W / 2.0, H / 2.0)
    if kind not in ("erp", "cylindrical"):
        limit = 180.0 if kind in ("pinhole", "brown_conrady", "fov") else 360.0
        if not 0.0 < fov_deg < limit:
            raise ValueError(f"{kind}: field of view must lie in (0, {limit:g}) degrees, got {fov_deg}")
    tm = math.radians(fov_deg / 2)
    if kind == "erp":
        return ERP(width=W, height=H)
    if kind == "erp_crop":
        cam = ERPCrop(math.radians(fov_deg / 2), math.radians(90.0 / 2), width=W, height=H)
        cam.theta_max = float(min(math.pi, math.hypot(math.radians(fov_deg / 2), math.radians(45.0))))
        return cam
    if kind == "cylindrical":
        return Cylindrical(width=W, height=H, v_fov_deg=90.0)
    half = H / 2.0
    if kind == "pinhole":
        f = half / math.tan(tm)
        cam = Pinhole(f, f, cx, cy)
    elif kind == "brown_conrady":
        f = half / math.tan(tm)
        cam = BrownConrady(f, f, cx, cy, k1=-0.18, k2=0.02, p1=8e-4, p2=-6e-4)
    elif kind == "equidistant":
        f = half / tm
        cam = Equidistant(f, f, cx, cy, fov_deg=fov_deg)
    elif kind == "kb4":
        k1, k2, k3, k4 = 0.04, -0.01, 0.002, -3e-4
        f = half / (tm * (1 + k1 * tm ** 2 + k2 * tm ** 4 + k3 * tm ** 6 + k4 * tm ** 8))
        cam = KB4(f, f, cx, cy, k1=k1, k2=k2, k3=k3, k4=k4, theta_max=tm)
    elif kind == "mei":
        xi = 1.6
        f = half * (math.cos(tm) + xi) / math.sin(tm)
        cam = MEI(f, f, cx, cy, xi=xi)
    elif kind == "double_sphere":
        xi, alpha = 0.25, 0.55
        st, ct = math.sin(tm), math.cos(tm)
        d2 = math.sqrt((xi + ct) ** 2 + st ** 2)
        f = half * (alpha * d2 + (1 - alpha) * (xi + ct)) / st
        cam = DoubleSphere(f, f, cx, cy, xi=xi, alpha=alpha)
    elif kind == "equisolid":
        f = half / (2.0 * math.sin(tm / 2.0))
        cam = Equisolid(f, f, cx, cy, fov_deg=fov_deg)
    elif kind == "stereographic":
        f = half / (2.0 * math.tan(tm / 2.0))
        cam = Stereographic(f, f, cx, cy, fov_deg=fov_deg)
    elif kind == "fov":
        w = 0.9
        f = half / (math.atan(2.0 * math.tan(tm) * math.tan(w / 2.0)) / w)
        cam = FOV(f, f, cx, cy, w=w, fov_deg=fov_deg)
    elif kind == "eucm":
        al, be = 0.55, 1.0
        st, ct = math.sin(tm), math.cos(tm)
        f = half * (al * math.sqrt(be * st * st + ct * ct) + (1 - al) * ct) / st
        cam = EUCM(f, f, cx, cy, alpha=al, beta=be, fov_deg=fov_deg)
    else:
        raise ValueError(f"unknown camera model {kind!r}; choose from {', '.join(CAMERA_KINDS)}")
    if circular is None:
        circular = kind not in ("pinhole", "brown_conrady")
    if circular:
        if getattr(cam, "theta_max", None) is None:
            cam.theta_max = tm
    elif kind in ("pinhole", "brown_conrady", "mei", "double_sphere"):
        cam.__dict__.pop("theta_max", None)
    else:
        cam.theta_max = {"stereographic": math.radians(179.0)}.get(kind, math.pi)
    cam.width, cam.height = W, H
    return cam


# COLMAP camera models: name -> (model id, parameter names)
_COLMAP = {
    "SIMPLE_PINHOLE": (0, ("f", "cx", "cy")),
    "PINHOLE": (1, ("fx", "fy", "cx", "cy")),
    "SIMPLE_RADIAL": (2, ("f", "cx", "cy", "k")),
    "RADIAL": (3, ("f", "cx", "cy", "k1", "k2")),
    "OPENCV": (4, ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2")),
    "OPENCV_FISHEYE": (5, ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4")),
    "FULL_OPENCV": (6, ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")),
    "FOV": (7, ("fx", "fy", "cx", "cy", "omega")),
    "SIMPLE_RADIAL_FISHEYE": (8, ("f", "cx", "cy", "k")),
    "RADIAL_FISHEYE": (9, ("f", "cx", "cy", "k1", "k2")),
    "THIN_PRISM_FISHEYE": (10, ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1")),
    "RAD_TAN_THIN_PRISM_FISHEYE": (11, ("fx", "fy", "cx", "cy", "k0", "k1", "k2", "k3", "k4", "k5",
                                        "p0", "p1", "s0", "s1", "s2", "s3")),
}
_COLMAP_BY_ID = {mid: name for name, (mid, _) in _COLMAP.items()}


def from_colmap(model, width, height, params, fov_deg=None):
    """Camera from a COLMAP model name and parameter list (COLMAP's pixel convention is the one
    used here, so no conversion is needed). fov_deg optionally limits the field of view of
    fisheye models, which is useful when the calibration is only valid inside the image circle."""
    model = str(model).upper()
    if model not in _COLMAP:
        raise ValueError(f"unknown COLMAP camera model {model!r}")
    names = _COLMAP[model][1]
    if len(params) != len(names):
        raise ValueError(f"{model} takes {len(names)} parameters {names}, got {len(params)}")
    p = dict(zip(names, map(float, params)))
    kw = dict(width=int(width), height=int(height))
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        p["fx"] = p["fy"] = p.pop("f")
    if model in ("SIMPLE_PINHOLE", "PINHOLE"):
        return Pinhole(p["fx"], p["fy"], p["cx"], p["cy"], **kw)
    if model == "SIMPLE_RADIAL":
        return BrownConrady(p["fx"], p["fy"], p["cx"], p["cy"], k1=p["k"], **kw)
    if model == "RADIAL":
        return BrownConrady(p["fx"], p["fy"], p["cx"], p["cy"], k1=p["k1"], k2=p["k2"], **kw)
    if model == "OPENCV":
        return BrownConrady(p["fx"], p["fy"], p["cx"], p["cy"], k1=p["k1"], k2=p["k2"], p1=p["p1"], p2=p["p2"], **kw)
    if model == "FULL_OPENCV":
        return BrownConrady(p["fx"], p["fy"], p["cx"], p["cy"], k1=p["k1"], k2=p["k2"], p1=p["p1"], p2=p["p2"],
                            k3=p["k3"], k4=p["k4"], k5=p["k5"], k6=p["k6"], **kw)
    if model == "OPENCV_FISHEYE":
        return KB4(p["fx"], p["fy"], p["cx"], p["cy"], k1=p["k1"], k2=p["k2"], k3=p["k3"], k4=p["k4"],
                   fov_deg=fov_deg, **kw)
    if model == "SIMPLE_RADIAL_FISHEYE":
        return KB4(p["fx"], p["fy"], p["cx"], p["cy"], k1=p["k"], fov_deg=fov_deg, **kw)
    if model == "RADIAL_FISHEYE":
        return KB4(p["fx"], p["fy"], p["cx"], p["cy"], k1=p["k1"], k2=p["k2"], fov_deg=fov_deg, **kw)
    if model == "FOV":
        return FOV(p["fx"], p["fy"], p["cx"], p["cy"], w=p["omega"], fov_deg=fov_deg, **kw)
    raise NotImplementedError(f"COLMAP model {model} is not supported yet; convert it to OPENCV_FISHEYE")


def from_opencv(K, dist, width, height, fisheye=False, fov_deg=None):
    """Camera from an OpenCV calibration: K (3x3) and distortion coefficients, as returned by
    cv2.calibrateCamera ((k1, k2, p1, p2[, k3[, k4, k5, k6]])) or cv2.fisheye.calibrate
    ((k1, k2, k3, k4), fisheye=True). OpenCV puts pixel centres at integer coordinates, so half a
    pixel is added to the principal point."""
    K = np.asarray(K, np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2] + 0.5, K[1, 2] + 0.5
    d = [float(x) for x in np.asarray(dist if dist is not None else [], np.float64).ravel()]
    kw = dict(width=int(width), height=int(height))
    if fisheye:
        d = (d + [0.0] * 4)[:4]
        return KB4(fx, fy, cx, cy, *d, fov_deg=fov_deg, **kw)
    if len(d) > 8 or any(d[8:]):
        raise NotImplementedError("thin-prism / tilted OpenCV coefficients are not supported")
    d = (d + [0.0] * 8)[:8]
    if not any(d):
        return Pinhole(fx, fy, cx, cy, **kw)
    k1, k2, p1, p2, k3, k4, k5, k6 = d
    return BrownConrady(fx, fy, cx, cy, k1=k1, k2=k2, p1=p1, p2=p2, k3=k3, k4=k4, k5=k5, k6=k6, **kw)


def camera_from_dict(spec):
    """Camera from a JSON-style dict. Accepted forms:

        {"model": "kb4", "width": 1024, "height": 1024, "fx": ..., "fy": ..., "cx": ..., "cy": ..., "k1": ...}
        {"model": "OPENCV_FISHEYE", "width": 1024, "height": 1024, "params": [fx, fy, cx, cy, k1, k2, k3, k4]}
        {"model": "equidistant", "width": 1024, "height": 1024, "fov_deg": 190}      (see from_fov)
        {"model": "erp", "width": 2048, "height": 1024}

    i.e. one of this module's model names with keyword parameters, a COLMAP model with its
    parameter list, or a model name with only a field of view."""
    spec = dict(spec)
    model = spec.pop("model")
    W, H = spec.pop("width", None), spec.pop("height", None)
    if "params" in spec:
        return from_colmap(model, W, H, spec.pop("params"), fov_deg=spec.pop("fov_deg", None))
    if model not in _KINDS:
        raise ValueError(f"unknown camera model {model!r}; choose from {', '.join(CAMERA_KINDS)} or a COLMAP model")
    cls = _KINDS[model]
    if model in ("erp", "cylindrical"):
        return cls(width=W, height=H, **spec)
    if model == "erp_crop":
        if "h_fov_deg" in spec:
            return ERPCrop(math.radians(spec["h_fov_deg"] / 2), math.radians(spec.get("v_fov_deg", 90.0) / 2),
                           width=W, height=H)
        return from_fov(model, W, H, spec["fov_deg"])
    if "fx" not in spec and "f" not in spec:
        pp = (spec["cx"], spec["cy"]) if "cx" in spec else None
        return from_fov(model, W, H, spec["fov_deg"], principal_point=pp)
    if "f" in spec:
        spec["fx"] = spec["fy"] = spec.pop("f")
    return cls(width=W, height=H, **spec)


def read_colmap_cameras(path):
    """Read a COLMAP sparse model's cameras.txt or cameras.bin -> {camera_id: CameraModel}.
    `path` may be the file or the model directory."""
    if os.path.isdir(path):
        for name in ("cameras.bin", "cameras.txt"):
            if os.path.exists(os.path.join(path, name)):
                path = os.path.join(path, name)
                break
        else:
            raise FileNotFoundError(f"no cameras.bin or cameras.txt in {path}")
    cams = {}
    if path.endswith(".bin"):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n):
                cam_id, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
                name = _COLMAP_BY_ID[model_id]
                k = len(_COLMAP[name][1])
                params = struct.unpack("<" + "d" * k, f.read(8 * k))
                cams[cam_id] = (name, w, h, params)
    else:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                t = line.split()
                cams[int(t[0])] = (t[1], int(t[2]), int(t[3]), [float(x) for x in t[4:]])
    return {cid: from_colmap(name, w, h, params) for cid, (name, w, h, params) in cams.items()}
