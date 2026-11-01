// Point-cloud viewer for the Any3R project page (three.js).
// Round, opaque, perspective-scaled points; camera glyphs coloured by field of view; optional
// ceiling clip; viewers can be linked so that they share one camera.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const VERT = `
uniform float uSize;
uniform float uScale;
uniform float uReveal;
attribute vec3 color;
attribute float vid;
varying vec3 vColor;
varying float vY;
varying float vG;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  vG = clamp(uReveal - vid, 0.0, 1.0);
  gl_PointSize = clamp(uSize * uScale / -mv.z, 1.0, 48.0) * vG;
  vColor = color;
  vY = position.y;
}`;
const FRAG = `
uniform float uClipY;
varying vec3 vColor;
varying float vY;
varying float vG;
void main() {
  if (vY > uClipY || vG <= 0.0) discard;
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(c, c);
  if (r2 > 1.0) discard;
  gl_FragColor = vec4(vColor * (1.0 - 0.22 * r2), 1.0);
}`;

export async function loadPoints(url, lo, hi, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  let buf;
  if (onProgress && res.body && total) {
    const reader = res.body.getReader();
    const chunks = []; let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value); got += value.length; onProgress(got / total);
    }
    buf = new Uint8Array(got); let o = 0;
    for (const c of chunks) { buf.set(c, o); o += c.length; }
    buf = buf.buffer;
  } else {
    buf = await res.arrayBuffer();
  }
  const n = new Uint32Array(buf, 0, 1)[0];
  const q = new Uint16Array(buf, 4, n * 3);
  const c = new Uint8Array(buf, 4 + n * 6, n * 3);
  const vid = buf.byteLength >= 4 + n * 10 ? Float32Array.from(new Uint8Array(buf, 4 + n * 9, n)) : null;
  const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      pos[3 * i + k] = lo[k] + (q[3 * i + k] / 65535) * (hi[k] - lo[k]);
      col[3 * i + k] = c[3 * i + k] / 255;
    }
  }
  return { n, pos, col, vid };
}

function cameraGlyph(cam, size) {
  // geometry in the OpenCV camera frame (x right, y down, z forward), then placed by the pose
  const v = [];
  const seg = (a, b) => v.push(...a, ...b);
  const circle = (r, z, axis = "z", n = 40) => {
    for (let i = 0; i < n; i++) {
      const a0 = (i / n) * 2 * Math.PI, a1 = ((i + 1) / n) * 2 * Math.PI;
      const p = (a) => axis === "z" ? [r * Math.cos(a), r * Math.sin(a), z]
        : axis === "y" ? [r * Math.cos(a), z, r * Math.sin(a)] : [z, r * Math.cos(a), r * Math.sin(a)];
      seg(p(a0), p(a1));
    }
  };
  const s = size;
  if (cam.fov >= 300) {                                   // panorama: a gyroscope
    circle(0.6 * s, 0, "z"); circle(0.6 * s, 0, "y"); circle(0.6 * s, 0, "x");
    seg([0, 0, 0], [0, 0, 1.1 * s]);
  } else if (cam.fov >= 170) {                            // fisheye: a dome reaching past 90 degrees
    const th = Math.min((cam.fov / 2) * Math.PI / 180, Math.PI * 0.95);
    const r = 0.7 * s;
    for (let m = 0; m < 4; m++) {                         // meridians from the axis to the rim
      const phi = (m / 4) * 2 * Math.PI, n = 14;
      for (let i = 0; i < n; i++) {
        const t0 = (i / n) * th, t1 = ((i + 1) / n) * th;
        const p = (t) => [r * Math.sin(t) * Math.cos(phi), r * Math.sin(t) * Math.sin(phi), r * Math.cos(t)];
        seg(p(t0), p(t1));
      }
    }
    circle(r * Math.sin(th), r * Math.cos(th), "z");      // the rim of the field of view
    circle(r, 0, "z", 32);
  } else {                                                // rectilinear: a frustum
    const t = Math.tan(Math.min(cam.fov / 2, 60) * Math.PI / 180), z = s;
    const c = [[-t * z, -t * z, z], [t * z, -t * z, z], [t * z, t * z, z], [-t * z, t * z, z]];
    for (let i = 0; i < 4; i++) { seg([0, 0, 0], c[i]); seg(c[i], c[(i + 1) % 4]); }
    seg([-0.3 * t * z, -t * z, z], [0, -1.35 * t * z, z]); seg([0, -1.35 * t * z, z], [0.3 * t * z, -t * z, z]);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(v, 3));
  const lines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: cam.color }));
  const dot = new THREE.Mesh(new THREE.SphereGeometry(0.09 * s, 12, 8), new THREE.MeshBasicMaterial({ color: cam.color }));
  const grp = new THREE.Group();
  grp.add(lines, dot);
  const R = cam.rotation, p = cam.position;
  const m = new THREE.Matrix4().set(R[0][0], R[0][1], R[0][2], p[0], R[1][0], R[1][1], R[1][2], p[1],
                                    R[2][0], R[2][1], R[2][2], p[2], 0, 0, 0, 1);
  grp.applyMatrix4(m);
  return grp;
}

export class PointViewer {
  constructor(container, { background = 0xffffff, pointSize = 1.0, baseSize = 0.006 } = {}) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(background, 1);
    container.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.005, 200);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.screenSpacePanning = true;
    this.controls.autoRotateSpeed = 0.8;
    this.material = new THREE.ShaderMaterial({
      uniforms: { uSize: { value: baseSize * pointSize }, uScale: { value: 500 }, uClipY: { value: 1e9 },
                  uReveal: { value: 1e9 } },
      vertexShader: VERT, fragmentShader: FRAG,
    });
    this.baseSize = baseSize;
    this.pointSize = pointSize;
    this.points = null;
    this.cams = new THREE.Group();
    this.scene.add(this.cams);
    this.linked = [];
    this._syncing = false;
    this.controls.addEventListener("change", () => this._sync());
    this._resize = () => this.resize();
    new ResizeObserver(this._resize).observe(container);
    this.resize();
    const loop = () => { this.controls.update(); this.renderer.render(this.scene, this.camera); requestAnimationFrame(loop); };
    loop();
    this.controls.addEventListener("start", () => this.container.dispatchEvent(new Event("interact")));
  }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = "100%";
    this.renderer.domElement.style.height = "100%";
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.material.uniforms.uScale.value = (h * this.renderer.getPixelRatio()) / (2 * Math.tan((this.camera.fov * Math.PI) / 360));
  }

  setPoints({ n, pos, col, vid }) {
    if (this.points) { this.scene.remove(this.points); this.points.geometry.dispose(); }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    g.setAttribute("vid", new THREE.BufferAttribute(vid || new Float32Array(n), 1));
    this.material.uniforms.uReveal.value = 1e9;
    this.points = new THREE.Points(g, this.material);
    this.points.frustumCulled = false;
    this.scene.add(this.points);
    this._ys = null;
    this._pos = pos;
  }

  setCameras(cams, size = 0.06) {
    this.cams.clear();
    for (const c of cams || []) {
      const g = cameraGlyph(c, size);
      g.userData.rank = c.rank ?? 0;
      g.userData.base = g.scale.clone();
      this.cams.add(g);
    }
  }

  // views with rank < r are shown; the view of rank floor(r) is growing in
  setReveal(r) {
    this.material.uniforms.uReveal.value = r;
    for (const g of this.cams.children) {
      const k = Math.max(0, Math.min(1, r - g.userData.rank));
      g.visible = k > 0;
      g.scale.copy(g.userData.base).multiplyScalar(0.3 + 0.7 * k);
    }
  }

  // play the reveal over `seconds`; resolves when done (or when superseded)
  playReveal(nViews, seconds) {
    const token = (this._revealToken = (this._revealToken || 0) + 1);
    const t0 = performance.now();
    return new Promise((resolve) => {
      const step = () => {
        if (token !== this._revealToken) return resolve(false);
        const t = Math.min(1, (performance.now() - t0) / (seconds * 1000));
        this.setReveal(t * nViews + (t >= 1 ? 1e9 : 0));
        if (t < 1) requestAnimationFrame(step); else resolve(true);
      };
      step();
    });
  }

  stopReveal() { this._revealToken = (this._revealToken || 0) + 1; this.setReveal(1e9); }

  // faint latitude / longitude lines of the unit sphere
  sphereGuide(on = true) {
    if (!this._guide) {
      const v = [];
      const n = 64;
      for (let lat = -60; lat <= 60; lat += 30) {
        const e = (lat * Math.PI) / 180;
        for (let i = 0; i < n; i++) {
          const a0 = (i / n) * 2 * Math.PI, a1 = ((i + 1) / n) * 2 * Math.PI;
          v.push(Math.cos(e) * Math.sin(a0), Math.sin(e), Math.cos(e) * Math.cos(a0),
                 Math.cos(e) * Math.sin(a1), Math.sin(e), Math.cos(e) * Math.cos(a1));
        }
      }
      for (let lon = 0; lon < 360; lon += 30) {
        const a = (lon * Math.PI) / 180;
        for (let i = 0; i < n; i++) {
          const e0 = -Math.PI / 2 + (i / n) * Math.PI, e1 = -Math.PI / 2 + ((i + 1) / n) * Math.PI;
          v.push(Math.cos(e0) * Math.sin(a), Math.sin(e0), Math.cos(e0) * Math.cos(a),
                 Math.cos(e1) * Math.sin(a), Math.sin(e1), Math.cos(e1) * Math.cos(a));
        }
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(v.map((x) => x * 0.995), 3));
      this._guide = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0xc9d0dc, transparent: true, opacity: 0.6 }));
      this.scene.add(this._guide);
    }
    this._guide.visible = on;
  }

  showCameras(on) { this.cams.visible = on; }
  setPointSize(k) { this.pointSize = k; this.material.uniforms.uSize.value = this.baseSize * k; }
  setAutoRotate(on) { this.controls.autoRotate = on; }

  // clip everything above the given quantile of the cloud's own heights (1 = no clip)
  setClipQuantile(qt) {
    if (qt >= 1 || !this._pos) { this.material.uniforms.uClipY.value = 1e9; return; }
    if (!this._ys) {
      const n = this._pos.length / 3, step = Math.max(1, Math.floor(n / 50000)), ys = [];
      for (let i = 0; i < n; i += step) ys.push(this._pos[3 * i + 1]);
      this._ys = ys.sort((a, b) => a - b);
    }
    this.material.uniforms.uClipY.value = this._ys[Math.floor(qt * (this._ys.length - 1))];
  }

  setView(position, target) {
    this.camera.position.set(...position);
    this.controls.target.set(...target);
    this.camera.up.set(0, 1, 0);
    this.controls.update();
    this.home = { position: [...position], target: [...target] };
  }

  reset() { if (this.home) this.setView(this.home.position, this.home.target); }

  link(other) { this.linked.push(other); other.linked.push(this); }

  _sync() {
    if (this._syncing) return;
    for (const o of this.linked) {
      o._syncing = true;
      o.camera.position.copy(this.camera.position);
      o.camera.quaternion.copy(this.camera.quaternion);
      o.controls.target.copy(this.controls.target);
      o.controls.update();
      o._syncing = false;
    }
  }
}
