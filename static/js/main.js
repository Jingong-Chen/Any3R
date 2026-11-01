import { PointViewer, loadPoints } from "./viewer.js";

const KIND = {
  pinhole: "Pinhole", brown_conrady: "Brown–Conrady", equidistant: "Equidistant", kb4: "Kannala–Brandt",
  mei: "MEI", double_sphere: "Double sphere", eucm: "EUCM", equisolid: "Equisolid", stereographic: "Stereographic",
  fov: "FOV", cylindrical: "Cylindrical", erp: "Equirectangular", erp_crop: "ERP window",
};
const $ = (s, el = document) => el.querySelector(s);
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const fmtFov = (f) => `${Math.round(f)}°`;

// ------------------------------------------------------------------------------ gallery
const VIEW = {   // azimuth (deg, relative to the first camera's heading), elevation (deg), distance
  street_mixed: { az: 155, el: 32, dist: 2.1 },
  fiord_meetingroom: { az: 180, el: 58, dist: 2.3 },
  fiord_parakennus: { az: 200, el: 68, dist: 2.6 },
  panocity_kb4: { az: 150, el: 38, dist: 2.7 },
  panocity_pinhole: { az: 150, el: 38, dist: 2.7 },
  panocity_erp: { az: 150, el: 35, dist: 2.0 },
};

function viewFor(meta, preset) {
  const c0 = meta.cameras[0];
  const f = [c0.rotation[0][2], c0.rotation[1][2], c0.rotation[2][2]];
  let h = Math.atan2(f[0], f[2]);
  const az = h + (preset.az * Math.PI) / 180, elv = (preset.el * Math.PI) / 180, d = preset.dist;
  return [[d * Math.cos(elv) * Math.sin(az), d * Math.sin(elv), d * Math.cos(elv) * Math.cos(az)], [0, 0, 0]];
}

async function initGallery() {
  const root = $("#gallery");
  if (!root) return;
  const index = await (await fetch("static/pcd/gallery.json")).json();
  const viewer = new PointViewer($(".viewer-canvas", root), { baseSize: 0.0055 });
  const status = $(".viewer-status", root), info = $(".viewer-info", root), strip = $(".input-strip", root);
  const picker = $(".scene-picker", root);
  let autoTimer = null;
  const state = { cams: true, rotate: true, size: 1.0, cut: true };
  let current = null;
  viewer.setAutoRotate(true);
  root.querySelector(".viewer-canvas").addEventListener("interact", () => {
    if (!state.rotate) return;
    viewer.setAutoRotate(false); clearTimeout(autoTimer);
    autoTimer = setTimeout(() => state.rotate && viewer.setAutoRotate(true), 6000);
  });

  async function show(i) {
    const g = index[i];
    picker.querySelectorAll(".scene-card").forEach((c, k) => c.classList.toggle("active", k === i));
    status.hidden = false; status.textContent = "Loading…";
    const meta = await (await fetch(`static/pcd/${g.name}.json`)).json();
    const pts = await loadPoints(`static/pcd/${g.name}.bin`, meta.lo, meta.hi,
      (p) => { status.textContent = `Loading ${Math.round(p * 100)}%`; });
    viewer.setPoints(pts);
    viewer.setCameras(meta.cameras, 0.055);
    viewer.showCameras(state.cams);
    current = meta;
    viewer.setClipQuantile(state.cut ? meta.clip ?? 1 : 1);
    $(".btn-gcut", root).hidden = !(meta.clip < 1);
    viewer.setView(...viewFor(meta, VIEW[meta.name] || { az: 180, el: 35, dist: 2.2 }));
    status.hidden = true;
    reveal();
    const kinds = [...new Set(meta.cameras.map((c) => c.kind))];
    info.innerHTML = `<div class="vi-title">${meta.title}</div><div class="vi-blurb">${meta.blurb}</div>
      <div class="vi-meta">${meta.cameras.length} views · ${kinds.length} camera model${kinds.length > 1 ? "s" : ""} · ${(meta.n / 1000).toFixed(0)}k points shown</div>`;
    strip.innerHTML = "";
    meta.cameras.forEach((c, k) => {
      const t = el("figure", "thumb");
      t.style.setProperty("--c", c.color);
      t.innerHTML = `<img loading="lazy" src="static/thumbs/${meta.name}_${String(k).padStart(2, "0")}.webp" alt="${c.name}">
        <figcaption>${KIND[c.kind] || c.kind} <b>${fmtFov(c.fov)}</b></figcaption>`;
      strip.appendChild(t);
    });
  }

  // views join one by one, along the camera path (all from one forward pass)
  function reveal() {
    if (!current) return;
    const n = current.cameras.length;
    viewer.playReveal(n, Math.min(7, 0.45 * n + 1.2));
  }

  index.forEach((g, i) => {
    const card = el("button", "scene-card");
    const n = g.name === "street_mixed" ? [0, 4, 7, 10] : g.name === "panocity_pinhole" ? [0, 1, 2, 3] : [0, 2, 4, 6];
    card.innerHTML = `<div class="mosaic">${n.map((k) => `<img loading="lazy" src="static/thumbs/${g.name}_${String(k).padStart(2, "0")}.webp" alt="">`).join("")}</div>
      <div class="sc-title">${g.title}</div><div class="sc-sub">${g.n_views} views</div>`;
    card.addEventListener("click", () => show(i));
    picker.appendChild(card);
  });

  $(".btn-rotate", root).addEventListener("click", (e) => {
    state.rotate = !state.rotate; viewer.setAutoRotate(state.rotate); e.currentTarget.classList.toggle("on", state.rotate);
  });
  $(".btn-cams", root).addEventListener("click", (e) => {
    state.cams = !state.cams; viewer.showCameras(state.cams); e.currentTarget.classList.toggle("on", state.cams);
  });
  $(".btn-reset", root).addEventListener("click", () => viewer.reset());
  $(".btn-replay", root).addEventListener("click", reveal);
  $(".btn-gcut", root).addEventListener("click", (e) => {
    state.cut = !state.cut; e.currentTarget.classList.toggle("on", state.cut);
    viewer.setClipQuantile(state.cut && current ? current.clip ?? 1 : 1);
  });
  $(".btn-full", root).addEventListener("click", () => {
    const box = $(".viewer-box", root);
    document.fullscreenElement ? document.exitFullscreen() : box.requestFullscreen?.();
  });
  $(".size-slider", root).addEventListener("input", (e) => viewer.setPointSize(Number(e.target.value)));
  show(Math.min(index.length - 1, Number(new URLSearchParams(location.search).get("scene") || 0)));
}

// ------------------------------------------------------------------------------ comparison
const FAMILIES = {
  gfm: { file: "cmp.json", prefix: "cmp", right: ["pi3", "mapa", "vggtomega"],
         note: "Geometry foundation models: data of the paper's eight-camera comparison figure (120k points per cloud)." },
  aware: { file: "cmpa.json", prefix: "cmpa", right: ["wid3r", "maparay", "panovggt", "omnivggt"],
           note: "Methods built for other cameras: data of the paper's camera-aware comparison figure. Wid3R receives the true ray field; MapA-ray is MapAnything with the true ray field fed to its ray encoder; greyed-out buttons mark cameras a method cannot take." },
};

async function initCompare() {
  const root = $("#compare");
  if (!root) return;
  const fams = {};
  for (const [k, f] of Object.entries(FAMILIES)) fams[k] = (await (await fetch(`static/pcd/${f.file}`)).json()).rows;
  const panes = [...root.querySelectorAll(".cmp-pane")].map((pane) => ({
    pane, viewer: new PointViewer($(".viewer-canvas", pane), { baseSize: 0.015 }),
    method: "ours", status: $(".viewer-status", pane),
  }));
  panes[0].viewer.link(panes[1].viewer);
  let fam = "gfm", row = 0, cut = true;
  const cache = new Map();
  const rowsOf = () => fams[fam];

  async function load(p) {
    const r = rowsOf()[row], m = r.methods.find((x) => x.key === p.method && !x.na);
    p.pane.querySelectorAll(".mbtn").forEach((b) => b.classList.toggle("active", b.dataset.m === p.method));
    if (!m) return;
    const key = `${fam}|${row}|${p.method}`;
    p.status.hidden = false; p.status.textContent = "Loading…";
    if (!cache.has(key)) cache.set(key, loadPoints(`static/pcd/${FAMILIES[fam].prefix}_r${r.row}_${m.key}.bin`, m.lo, m.hi));
    const pts = await cache.get(key);
    p.viewer.setPoints(pts);
    p.viewer.setClipQuantile(r.indoor && cut ? 0.7 : 1);
    p.status.hidden = true;
  }

  function buildButtons(p) {
    const box = $(".method-buttons", p.pane);
    box.innerHTML = "";
    for (const m of rowsOf()[row].methods) {
      const b = el("button", "mbtn" + (m.key === "ours" ? " ours" : m.key === "gt" ? " gt" : "") + (m.na ? " na" : ""), m.label);
      b.dataset.m = m.key;
      if (m.na) { b.disabled = true; b.title = m.na; }
      else b.addEventListener("click", () => { p.method = m.key; load(p); });
      box.appendChild(b);
    }
  }

  const chips = $(".row-chips", root), inputs = $(".cmp-inputs", root), cutBtn = $(".btn-cut", root);
  function buildChips() {
    chips.innerHTML = "";
    rowsOf().forEach((r, i) => {
      const c = el("button", "chip", `${r.label} <b>${fmtFov(r.fov)}</b><span>${r.ds}</span>`);
      c.addEventListener("click", () => select(i));
      chips.appendChild(c);
    });
  }
  cutBtn.addEventListener("click", () => {
    cut = !cut; cutBtn.classList.toggle("on", cut);
    panes.forEach((p) => p.viewer.setClipQuantile(rowsOf()[row].indoor && cut ? 0.7 : 1));
  });

  async function select(i) {
    row = i;
    chips.querySelectorAll(".chip").forEach((c, k) => c.classList.toggle("active", k === i));
    const r = rowsOf()[i];
    const avail = new Set(r.methods.filter((m) => !m.na).map((m) => m.key));
    if (!avail.has(panes[0].method)) panes[0].method = "ours";
    if (!avail.has(panes[1].method) || panes[1].method === "ours")
      panes[1].method = FAMILIES[fam].right.find((k) => avail.has(k)) || "gt";
    panes.forEach(buildButtons);
    cutBtn.hidden = !r.indoor;
    inputs.innerHTML = [0, 1, 2].map((k) => `<img src="static/thumbs/${FAMILIES[fam].prefix}_r${r.row}_${k}.webp" alt="input view ${k + 1}">`).join("");
    $(".cmp-caption", root).innerHTML = `${r.ds} · three views through a <b>${r.label} ${fmtFov(r.fov)}</b> camera`;
    $(".cmp-note", root).textContent = FAMILIES[fam].note + (r.indoor ? " Indoor clouds hide their own highest 30% of points, the same rule for every method." : "");
    const az = (225 * Math.PI) / 180, elv = ((r.indoor ? 50 : 35) * Math.PI) / 180, d = r.indoor ? 2.7 : 2.4;
    panes[0].viewer.setView([d * Math.cos(elv) * Math.sin(az), d * Math.sin(elv), d * Math.cos(elv) * Math.cos(az)], [0, 0, 0]);
    await Promise.all(panes.map(load));
  }

  function setFamily(f) {
    fam = f;
    root.querySelectorAll(".fam-tab").forEach((t) => t.classList.toggle("active", t.dataset.f === f));
    panes[1].method = "";
    buildChips();
    select(0);
  }
  root.querySelectorAll(".fam-tab").forEach((t) => t.addEventListener("click", () => setFamily(t.dataset.f)));
  root.querySelector(".btn-cmp-reset").addEventListener("click", () => panes[0].viewer.reset());
  root.querySelector(".cmp-size").addEventListener("input", (e) => panes.forEach((p) => p.viewer.setPointSize(Number(e.target.value))));
  const q = new URLSearchParams(location.search);
  fam = q.get("fam") === "aware" ? "aware" : "gfm";
  root.querySelectorAll(".fam-tab").forEach((t) => t.classList.toggle("active", t.dataset.f === fam));
  panes[1].method = "";
  buildChips();
  select(Math.min(rowsOf().length - 1, Number(q.get("row") || 0)));
}

// ------------------------------------------------------------------------------ motivation
async function initMotivation() {
  const root = $("#motivation");
  if (!root) return;
  const data = await (await fetch("static/pcd/mot.json")).json();
  const cols = ["withheld", "supplied"].map((key) => {
    const col = $(`.motiv-col.${key}`, root);
    return { key, col, rays: new PointViewer($(".sphere .viewer-canvas", col), { baseSize: 0.016 }),
             recon: new PointViewer($(".recon .viewer-canvas", col), { baseSize: 0.013 }) };
  });
  cols[0].rays.link(cols[1].rays);
  cols[0].recon.link(cols[1].recon);
  cols.forEach((c) => c.rays.sphereGuide(true));
  let row = 0, cut = true, gt = false;
  const cache = new Map();
  const get = (url, m) => { if (!cache.has(url)) cache.set(url, loadPoints(url, m.lo, m.hi)); return cache.get(url); };

  async function select(i) {
    row = i;
    const r = data.rows[i];
    root.querySelectorAll(".chip").forEach((c, k) => c.classList.toggle("active", k === i));
    $(".motiv-input", root).src = `static/thumbs/mot_r${i}_input.webp`;
    $(".motiv-caption", root).innerHTML = `${r.ds} · three views through a <b>${r.label} ${r.fov}°</b> camera; the first view is shown`;
    await Promise.all(cols.map(async (c) => {
      $(".depth", c.col).src = `static/thumbs/mot_r${i}_depth_${c.key}.webp`;
      c.rays.setPoints(await get(`static/pcd/mot_r${i}_rays_${c.key}.bin`, r.spheres[c.key]));
      const key = gt ? "gt" : c.key;
      c.recon.setPoints(await get(`static/pcd/mot_r${i}_${key}.bin`, r.clouds[key]));
      c.recon.setClipQuantile(cut ? 0.7 : 1);
    }));
    cols[0].rays.setView([1.7, 1.2, -2.2], [0, 0, 0]);
    const az = (225 * Math.PI) / 180, elv = (50 * Math.PI) / 180, d = 2.7;
    cols[0].recon.setView([d * Math.cos(elv) * Math.sin(az), d * Math.sin(elv), d * Math.cos(elv) * Math.cos(az)], [0, 0, 0]);
  }
  const chips = $(".motiv-chips", root);
  data.rows.forEach((r, i) => {
    const c = el("button", "chip", `${r.label} <b>${r.fov}°</b>`);
    c.addEventListener("click", () => select(i));
    chips.appendChild(c);
  });
  $(".btn-mcut", root).addEventListener("click", (e) => {
    cut = !cut; e.currentTarget.classList.toggle("on", cut);
    cols.forEach((c) => c.recon.setClipQuantile(cut ? 0.7 : 1));
  });
  $(".btn-mgt", root).addEventListener("click", (e) => {
    gt = !gt; e.currentTarget.classList.toggle("on", gt); select(row);
  });
  select(Math.min(data.rows.length - 1, Number(new URLSearchParams(location.search).get("mrow") || 0)));
}

// ------------------------------------------------------------------------------ tables
const CAMS = ["Pin.<i>90°</i>", "B–C<i>90°</i>", "Equi.<i>180°</i>", "ERP-C<i>180°</i>", "MEI<i>180°</i>", "KB4<i>200°</i>", "DS<i>200°</i>", "ERP<i>360°</i>"];

async function initTables() {
  const root = $("#results");
  if (!root) return;
  const T = await (await fetch("static/data/tables.json")).json();
  const box = $(".table-box", root), note = $(".table-note", root);
  let tab = "gfm", measure = 0;
  const NOTES = {
    gfm: "Geometry foundation models. Any3R* is Any3R without its ray map (it then assumes an equirectangular grid) and is not ranked.",
    camaware: "Methods built for other cameras. Wid3R receives the true ray field; MapA-ray is MapAnything with the true ray field fed to its ray encoder; – marks cameras a method cannot take.",
    zeroshot: "Real fisheye lenses, not in Any3R's training data: the ScanNet++ DSLR fisheye (validation split) and the 200° lens of the FIORD rig. † Wid3R is trained on ScanNet++. Any3R* is not ranked.",
  };
  function render() {
    root.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.t === tab));
    root.querySelectorAll(".meas").forEach((t) => t.classList.toggle("active", Number(t.dataset.m) === measure));
    $(".measure-toggle", root).hidden = tab === "zeroshot";
    const block = T[tab][tab === "zeroshot" ? 0 : measure];
    let head;
    if (tab === "zeroshot") {
      head = `<tr><th rowspan="2">Method</th><th colspan="3" class="grp">ScanNet++ (≈146° fisheye)</th><th colspan="3" class="grp">FIORD (>180° fisheye)</th></tr>
        <tr>${["Acc ↓", "AUC@15 ↑", "AUC@30 ↑", "Acc ↓", "AUC@15 ↑", "AUC@30 ↑"].map((h) => `<th>${h}</th>`).join("")}</tr>`;
    } else {
      head = `<tr><th rowspan="2">Method</th><th colspan="8" class="grp">Indoor · Matterport3D + Stanford2D3DS</th><th colspan="8" class="grp">Outdoor · PanoCity</th></tr>
        <tr>${[...CAMS, ...CAMS].map((h) => `<th class="cam">${h}</th>`).join("")}</tr>`;
    }
    const body = block.rows.map((r) => {
      const ours = r.method === "Any3R";
      return `<tr class="${ours ? "ours" : ""} ${r.method === "Any3R*" ? "ablation" : ""}"><td class="m">${ours ? "<b>Any3R</b>" : r.method}</td>${r.cells
        .map((c, j) => `<td class="${c.mark} ${tab !== "zeroshot" && j === 8 ? "sep" : ""}">${c.v === "--" ? "–" : c.v}</td>`).join("")}</tr>`;
    }).join("");
    box.innerHTML = `<table class="res"><thead>${head}</thead><tbody>${body}</tbody></table>`;
    note.textContent = NOTES[tab];
  }
  root.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => { tab = t.dataset.t; render(); }));
  root.querySelectorAll(".meas").forEach((t) => t.addEventListener("click", () => { measure = Number(t.dataset.m); render(); }));
  render();
}

// ------------------------------------------------------------------------------ misc
function initCopy() {
  const btn = $("#copy-bibtex");
  if (!btn) return;
  btn.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText($("#bibtex-text").innerText); btn.textContent = "Copied"; }
    catch { btn.textContent = "Select and copy"; }
    setTimeout(() => (btn.textContent = "Copy"), 1600);
  });
}

function lazyInit(id, fn) {
  const target = document.getElementById(id);
  if (!target) return;
  const io = new IntersectionObserver((entries) => {
    if (entries.some((e) => e.isIntersecting)) { io.disconnect(); fn(); }
  }, { rootMargin: "400px" });
  io.observe(target);
}

initGallery();
lazyInit("motivation", initMotivation);
lazyInit("compare", initCompare);
initTables();
initCopy();
