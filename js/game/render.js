// Copyright (c) 2026. MuJoCo player for game-discovery agents.
// SPDX-License-Identifier: BSD-3-Clause
//
// Minimal three.js renderer driven by a MuJoCo model/data. Builds one mesh per
// geom from model.geom_type/size/rgba, then every frame sets each mesh transform
// from data.geom_xpos (position) + data.geom_xmat (row-major 3x3 rotation).
//
// We keep MuJoCo's native Z-up coordinates (camera.up = +Z) so geom transforms
// apply with no axis juggling. Supports the geom types used by ant.xml:
// PLANE, SPHERE, CAPSULE, BOX (CYLINDER/ELLIPSOID handled too).
//
// The look mirrors the Isaac Lab game-play clips (play_hierarchical.py defaults): a flat
// off-white floor (linear 0.3 grey lit by a bright dome and a sun), a light backdrop, real
// shadows from a sun at azimuth 210 deg / elevation 50 deg, and the arena drawn as a
// translucent navy line painted on the floor instead of standing walls. For the Franka
// hockey game it also draws the play script's trails (puck + each gripper) and takes the
// Isaac ViewerCfg camera (setView) instead of the arena framing.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const GEOM = { PLANE: 0, HFIELD: 1, SPHERE: 2, CAPSULE: 3, ELLIPSOID: 4, CYLINDER: 5, BOX: 6, MESH: 7 };
const GEOM_OBJ = 5;  // mjtObj mjOBJ_GEOM (for mj_id2name)

// Isaac play_hierarchical.py defaults, all linear RGB like Isaac's --ground_color /
// --boundary_color: floor 0.3 grey, boundary line (0.002, 0.004, 0.03) at 0.6 opacity,
// 0.16 m wide (DualAntEnvCfg.boundary_line_width), a hair above the floor.
const FLOOR_RGB = 0.3;
const BACKDROP = 0xf3f4f6;
// The MJCF floor plane is finite; it is drawn this big so its edge never shows.
const FLOOR_EXTENT = 400;
const LINE_RGB = [0.002, 0.004, 0.03], LINE_OPACITY = 0.6, LINE_WIDTH = 0.16, LINE_HEIGHT = 0.01;
// Boxing-ring boundary (dual_ant_env._visualize_ring_boundary with boundary_style="ring"):
// one rope per height on each side, a post at every corner. Render-only, like the line.
const ROPE_HEIGHTS = [0.3, 0.6, 0.9], ROPE_THICKNESS = 0.04;
const POST_THICKNESS = 0.1, POST_HEIGHT = 1.0;
// Fraction of the visible viewport the framed arena fills (the rest is breathing room).
const ARENA_FILL = 0.92;
// Sun direction from Isaac's --sun_azimuth 210 (deg from +x toward +y) / --sun_elevation 50.
const SUN_AZ = (210 * Math.PI) / 180, SUN_EL = (50 * Math.PI) / 180, SUN_DIST = 20;
const SUN_DIR = [Math.cos(SUN_AZ) * Math.cos(SUN_EL), Math.sin(SUN_AZ) * Math.cos(SUN_EL), Math.sin(SUN_EL)];

export class MujocoRenderer {
  constructor(canvas, model, mujoco) {
    this.model = model;
    this.mujoco = mujoco;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(BACKDROP);
    // The floor plane is finite; fog fades its edge into the backdrop the way Isaac's
    // infinite ground fades into its dome.
    this.scene.fog = new THREE.Fog(BACKDROP, 30, 90);

    // Image-based lighting so the metallic G1 meshes have something to reflect
    // (without an env map, metalness>0 renders nearly black).
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.05, 200);
    this.camera.up.set(0, 0, 1);                 // MuJoCo Z-up
    this.camera.position.set(0, -16, 9);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, 0, 0.4);
    this.controls.enableDamping = true;

    // Dome (Isaac --dome_intensity 800) + sun (--sun_intensity 3000): the dome is a soft
    // hemisphere, the sun a shadow-casting directional light whose frustum setArena() sizes.
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0xdadce2, 0.9));
    const sun = new THREE.DirectionalLight(0xffffff, 3.0);
    sun.position.set(SUN_DIR[0] * SUN_DIST, SUN_DIR[1] * SUN_DIST, SUN_DIR[2] * SUN_DIST);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.near = 1; sun.shadow.camera.far = 60;
    sun.shadow.bias = -0.0005;
    sun.shadow.normalBias = 0.02;
    this.scene.add(sun);
    this.scene.add(sun.target);
    this.sun = sun;
    this._setShadowExtent(8);

    // "isaac" (G1: the env's flat me/foe PreviewSurface), "metal" (brushed shells) or
    // "plastic" (Franka: matte white shell).
    this.meshStyle = "isaac";
    this._leftInset = 0;        // canvas pixels hidden by the control panel (see frameCamera)
    this._viewShift = { x: 0, y: 0 };   // image translation in pixels (see _applyViewOffset)
    this._trails = [];
    this.meshes = [];
    this._buildGeoms(model);
    this._onResize();
    window.addEventListener("resize", () => this._onResize());
  }

  // Swap the MuJoCo model (game switch): drop old geom meshes, rebuild from the
  // new model. Keeps the same WebGLRenderer / scene / camera / arena.
  setModel(model, meshStyle = null) {
    if (meshStyle) this.meshStyle = meshStyle;
    for (const m of this.meshes) {
      if (!m) continue;
      this.scene.remove(m);
      if (m.geometry) m.geometry.dispose();
      if (m.material) m.material.dispose();
    }
    this.meshes = [];
    this.model = model;
    this._buildGeoms(model);
  }

  // Build a THREE.BufferGeometry for a MESH geom from the compiled MuJoCo model
  // (mesh_vert/mesh_face in the geom's local frame; positioned each frame by
  // geom_xpos/geom_xmat like every other geom). Returns null if not a mesh.
  _buildMeshGeom(model, i) {
    const dataid = model.geom_dataid[i];
    if (dataid == null || dataid < 0) return null;
    const va = model.mesh_vertadr[dataid], vn = model.mesh_vertnum[dataid];
    const fa = model.mesh_faceadr[dataid], fn = model.mesh_facenum[dataid];
    const pos = new Float32Array(vn * 3);
    for (let k = 0; k < vn * 3; k++) pos[k] = model.mesh_vert[va * 3 + k];
    const idx = new Uint32Array(fn * 3);       // face indices are local to the mesh
    for (let k = 0; k < fn * 3; k++) idx[k] = model.mesh_face[fa * 3 + k];
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    geo.computeVertexNormals();
    return geo;
  }

  _buildGeoms(model) {
    const ng = model.ngeom;
    const type = model.geom_type, size = model.geom_size, rgba = model.geom_rgba;
    for (let i = 0; i < ng; i++) {
      const t = type[i];
      // Skip geoms explicitly hidden via rgba alpha 0 (e.g. the g1 mesh model's
      // collision primitives, which collide but must not render over the meshes).
      if (rgba[4 * i + 3] <= 0) { this.meshes.push(null); continue; }
      // The MJCF's fixed boundary line (wall_*) is for the local viewer; this player draws
      // its own at the current arena size (setArena), so skip it here.
      const gname = this.mujoco ? this.mujoco.mj_id2name(model, GEOM_OBJ, i) : null;
      if (gname && gname.startsWith("wall")) { this.meshes.push(null); continue; }
      const sx = size[3 * i], sy = size[3 * i + 1], sz = size[3 * i + 2];
      let geo = null, prerotX = false;
      if (t === GEOM.MESH) {
        geo = this._buildMeshGeom(model, i);
        if (!geo) { this.meshes.push(null); continue; }
      } else if (t === GEOM.PLANE) {
        geo = new THREE.PlaneGeometry(FLOOR_EXTENT, FLOOR_EXTENT, 1, 1);   // normal +Z (matches MuJoCo)
      } else if (t === GEOM.SPHERE) {
        geo = new THREE.SphereGeometry(sx, 24, 16);
      } else if (t === GEOM.CAPSULE) {
        geo = new THREE.CapsuleGeometry(sx, 2 * sy, 6, 16); // three axis = Y
        prerotX = true;                                     // rotate Y->Z (MuJoCo)
      } else if (t === GEOM.CYLINDER) {
        geo = new THREE.CylinderGeometry(sx, sx, 2 * sy, 20);
        prerotX = true;
      } else if (t === GEOM.ELLIPSOID) {
        geo = new THREE.SphereGeometry(1, 24, 16); geo.scale(sx, sy, sz);
      } else if (t === GEOM.BOX) {
        geo = new THREE.BoxGeometry(2 * sx, 2 * sy, 2 * sz);
      } else {
        this.meshes.push(null); continue; // unsupported (e.g. mesh) -> skip
      }
      if (prerotX) geo.rotateX(Math.PI / 2);

      const r = rgba[4 * i], g = rgba[4 * i + 1], b = rgba[4 * i + 2], a = rgba[4 * i + 3];
      const isPlane = t === GEOM.PLANE;
      const isMesh = t === GEOM.MESH;
      // Colors are linear RGB (three's working space), like Isaac's diffuse_color values, so
      // the MJCF's rgba render as the same tint as the USD materials.
      const style = isMesh ? this.meshStyle : null;
      // Plastic shells (Franka) are drawn at 80% albedo: Isaac's white Panda reads as soft grey
      // gradients against the off-white backdrop, and a 1.0-albedo mesh here would vanish into it.
      const k = style === "plastic" ? 0.8 : 1.0;
      const mat = new THREE.MeshStandardMaterial({
        color: isPlane ? new THREE.Color().setRGB(FLOOR_RGB, FLOOR_RGB, FLOOR_RGB) : new THREE.Color().setRGB(k * r, k * g, k * b),
        // "isaac" (G1) is the flat PreviewSurface the env binds over the whole robot -- no
        // metalness, roughness 0.5, full albedo -- so the humanoids read as in the Isaac
        // clips; "metal" is a brushed shell, "plastic" the Panda's matte white one.
        roughness: isPlane ? 0.9 : style ? (style === "plastic" ? 0.55 : 0.5) : 0.6,
        metalness: style === "metal" ? 0.35 : 0.0,
        // decimated STL winding isn't always watertight -> render both sides so the
        // limbs read as solid (no backface-cull holes).
        side: isMesh ? THREE.DoubleSide : THREE.FrontSide,
        transparent: a < 1.0, opacity: a,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.matrixAutoUpdate = false;
      mesh.castShadow = !isPlane;
      // Robot meshes only cast: their many coincident visual shells (Franka's per-material
      // parts) self-shadow into acne when they also receive.
      mesh.receiveShadow = !isMesh;
      this.scene.add(mesh);
      this.meshes.push(mesh);
    }
    this._m4 = new THREE.Matrix4();
  }

  // Frame the camera for an arena of half-extent `half` and a robot whose action is
  // centered around height `focusZ` (ant ~0.4 m, G1 ~0.9 m). The eye keeps the clips' 3/4
  // direction and is pulled in until the arena corners (up to `topZ`, which has to clear
  // the robots and the ring posts) just fill the view, so the arena is as large as it can
  // be at any window size instead of floating in the middle of it. `leftInset` is how much
  // of the canvas the control panel covers: the arena is fitted into, and centred in, what
  // is left of it.
  frameCamera(half, focusZ = 0.4, { leftInset = 0, topZ = null } = {}) {
    this._setFov(45);
    this._leftInset = leftInset;
    const { w, h, inset } = this._viewport();
    this._viewShift = { x: inset / 2, y: 0 };   // centre the image in the visible strip
    this._applyViewOffset();

    const target = new THREE.Vector3(0, 0, focusZ);
    const dir = new THREE.Vector3(0, -1, 0.5).normalize();
    const zTop = topZ == null ? focusZ * 2 : topZ;
    const pts = [];
    for (const x of [-half, half]) for (const y of [-half, half]) for (const z of [0, zTop]) {
      pts.push(new THREE.Vector3(x, y, z));
    }
    // The arena may span ARENA_FILL of the strip's width, and of the canvas height.
    const xLimit = ARENA_FILL * (1 - inset / w), yLimit = ARENA_FILL;
    const p = new THREE.Vector3();
    // Projected bounds of the arena seen from eye distance `d`, in NDC.
    const project = (d) => {
      this.camera.position.copy(target).addScaledVector(dir, d);
      this.camera.lookAt(target);
      this.camera.updateMatrixWorld();
      const b = { x0: Infinity, x1: -Infinity, y0: Infinity, y1: -Infinity };
      for (const q of pts) {
        p.copy(q).project(this.camera);
        b.x0 = Math.min(b.x0, p.x); b.x1 = Math.max(b.x1, p.x);
        b.y0 = Math.min(b.y0, p.y); b.y1 = Math.max(b.y1, p.y);
      }
      return b;
    };
    // Smallest distance that still fits, then slide the image so the arena sits vertically
    // centred (perspective puts its centre below the point the camera aims at). Each shift
    // changes the frustum slightly, so fit and centre together, over a few rounds.
    let d = 20;
    for (let round = 0; round < 3; round++) {
      let lo = 0.5, hi = 400;
      for (let i = 0; i < 40; i++) {
        const mid = 0.5 * (lo + hi);
        const b = project(mid);
        const fills = Math.max((b.x1 - b.x0) / (2 * xLimit), (b.y1 - b.y0) / (2 * yLimit));
        if (fills > 1) lo = mid; else hi = mid;
      }
      d = hi;
      const b = project(d);
      this._viewShift.y += (b.y0 + b.y1) / 2 * h / 2;
      this._applyViewOffset();
    }
    project(d);
    this.controls.target.copy(target);
    this.controls.update();
  }

  _setFov(fov) {
    if (this.camera.fov !== fov) { this.camera.fov = fov; this.camera.updateProjectionMatrix(); }
  }

  // Half-size of the square the sun's shadow map covers, centred on the origin.
  _setShadowExtent(extent) {
    const cam = this.sun.shadow.camera;
    cam.left = -extent; cam.right = extent; cam.top = extent; cam.bottom = -extent;
    cam.updateProjectionMatrix();
  }

  // Draw a resizable square arena boundary (half-extent `half`, in meters). `style`
  // mirrors the Isaac env's boundary_style: "line" paints it on the floor (the clips'
  // look), "ring" stands it up as boxing-ring ropes and corner posts. Either way it is
  // render-only -- the boundary is a position check, not a wall.
  setArena(half, style = "line") {
    if (this._arena) { this.scene.remove(this._arena); this._arena = null; }
    if (half == null) return;   // games without an arena line (Franka hockey)
    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color().setRGB(LINE_RGB[0], LINE_RGB[1], LINE_RGB[2]),
      roughness: 0.9, metalness: 0.0,
      transparent: style !== "ring", opacity: style === "ring" ? 1.0 : LINE_OPACITY,
      envMapIntensity: 0.0,   // no room reflections: it stays the deep navy of the clips
    });
    const g = new THREE.Group();
    // A box of full size (sx, sy, sz) centred at (px, py, pz).
    const box = (px, py, pz, sx, sy, sz, shadow) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), mat);
      m.position.set(px, py, pz);
      m.castShadow = !!shadow;
      m.receiveShadow = !shadow;
      g.add(m);
    };
    const side = 2 * half;
    if (style === "ring") {
      const r = ROPE_THICKNESS;
      for (const z of ROPE_HEIGHTS) {
        for (const y of [-half, half]) box(0, y, z, side, r, r, true);   // ropes along x
        for (const x of [-half, half]) box(x, 0, z, r, side, r, true);   // ropes along y
      }
      for (const x of [-half, half]) for (const y of [-half, half]) {
        box(x, y, POST_HEIGHT / 2, POST_THICKNESS, POST_THICKNESS, POST_HEIGHT, true);
      }
    } else {
      const w = LINE_WIDTH, h = LINE_HEIGHT;
      for (const y of [-half, half]) box(0, y, h / 2, side + w, w, h, false);
      for (const x of [-half, half]) box(x, 0, h / 2, w, side + w, h, false);
    }
    this._arena = g;
    this.scene.add(g);
    this._setShadowExtent(half + 3);
  }

  // Fixed view from Isaac's ViewerCfg (the play script's follow camera keeps this eye-lookat
  // offset from the me/foe midpoint, which for fixed-base arms is the table centre). `fov` is
  // the vertical field of view: Isaac's viewport lens is long (about 14 deg vertical for the
  // hockey clips), so the same eye/lookat needs the same narrow fov to frame the table alike;
  // `zoom` scales the eye offset if a tighter/wider shot is wanted.
  setView(eye, lookat, { zoom = 1.0, fov = 45, shadowExtent = 3 } = {}) {
    this._leftInset = 0;                // a scripted camera frames itself; no offset
    this._viewShift = { x: 0, y: 0 };
    this._applyViewOffset();
    this._setFov(fov);
    const o = [eye[0] - lookat[0], eye[1] - lookat[1], eye[2] - lookat[2]];
    this.camera.position.set(lookat[0] + o[0] * zoom, lookat[1] + o[1] * zoom, lookat[2] + o[2] * zoom);
    this.controls.target.set(lookat[0], lookat[1], lookat[2]);
    this.controls.update();
    this._setShadowExtent(shadowExtent);
  }

  // Trails (play_hierarchical.py Trail): `specs` = [{color:[r,g,b] linear, radius}] ->
  // one fixed ring of `n` spheres each, ages binned into `buckets` prototypes of shrinking
  // radius (newest 1.0 -> oldest `fade`). Positions are pushed with updateTrails().
  setTrails(specs, { seconds = 0.5, hz = 30, controlDt = 1 / 60, fade = 0.5, buckets = 6 } = {}) {
    for (const tr of this._trails) { this.scene.remove(tr.group); tr.group.traverse((m) => { if (m.geometry) m.geometry.dispose(); if (m.material) m.material.dispose(); }); }
    this._trails = [];
    if (!specs || !specs.length) return;
    const stride = Math.max(1, Math.round(1 / (controlDt * hz)));
    const n = Math.max(1, Math.round(seconds * hz));
    for (const spec of specs) {
      const group = new THREE.Group();
      const mats = [];
      for (let k = 0; k < buckets; k++) {
        const fresh = 1 - k / Math.max(1, buckets - 1), scale = fade + (1 - fade) * fresh;
        mats.push({ scale, material: new THREE.MeshStandardMaterial({
          color: new THREE.Color().setRGB(spec.color[0], spec.color[1], spec.color[2]), roughness: 1.0, metalness: 0.0 }) });
      }
      const spheres = [];
      for (let i = 0; i < n; i++) {
        const m = new THREE.Mesh(new THREE.SphereGeometry(spec.radius, 12, 8), mats[0].material);
        m.visible = false; m.castShadow = false; m.receiveShadow = false;
        group.add(m); spheres.push(m);
      }
      this.scene.add(group);
      this._trails.push({ group, spheres, mats, n, stride, step: 0, history: [] });
    }
  }

  // Push the tracked points (one per trail spec, world xyz) for this control step.
  // `reset` clears the history (episode / round reset), like Trail.update(reset=True).
  updateTrails(points, reset = false) {
    for (let i = 0; i < this._trails.length; i++) {
      const tr = this._trails[i], p = points[i];
      if (reset) { tr.history.length = 0; tr.step = 0; }
      tr.step++;
      if (tr.step % tr.stride !== 0 && tr.history.length) continue;
      tr.history.push([p[0], p[1], p[2]]);
      if (tr.history.length > tr.n) tr.history.shift();
      const m = tr.history.length, B = tr.mats.length;
      for (let s = 0; s < tr.n; s++) {
        const sph = tr.spheres[s];
        if (s >= m) { sph.visible = false; continue; }
        const age = m - 1 - s;                                   // newest sample has age 0
        const b = Math.min(B - 1, Math.floor(age * B / tr.n));
        sph.visible = true;
        sph.position.set(tr.history[s][0], tr.history[s][1], tr.history[s][2]);
        sph.scale.setScalar(tr.mats[b].scale);
        sph.material = tr.mats[b].material;
      }
    }
  }

  update(data) {
    const xpos = data.geom_xpos, xmat = data.geom_xmat;
    for (let i = 0; i < this.meshes.length; i++) {
      const mesh = this.meshes[i];
      if (!mesh) continue;
      const p = 3 * i, r = 9 * i;
      // MuJoCo geom_xmat is row-major 3x3; THREE.Matrix4.set is row-major.
      this._m4.set(
        xmat[r + 0], xmat[r + 1], xmat[r + 2], xpos[p + 0],
        xmat[r + 3], xmat[r + 4], xmat[r + 5], xpos[p + 1],
        xmat[r + 6], xmat[r + 7], xmat[r + 8], xpos[p + 2],
        0, 0, 0, 1
      );
      mesh.matrix.copy(this._m4);
    }
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  _onResize() {
    const c = this.renderer.domElement;
    const w = c.clientWidth || window.innerWidth, h = c.clientHeight || window.innerHeight;
    this.renderer.setSize(w, h, false);
    this._applyViewOffset();
  }

  // Canvas size, and how much of its left edge the control panel hides.
  _viewport() {
    const c = this.renderer.domElement;
    const w = c.clientWidth || window.innerWidth || 1, h = c.clientHeight || window.innerHeight || 1;
    return { w, h, inset: Math.min(0.6 * w, Math.max(0, this._leftInset || 0)) };
  }

  // Slide the rendered image by `_viewShift` pixels (right / down positive) with an
  // off-axis frustum: the camera is told it renders a w x h window of a larger view, which
  // translates the image at every depth. Panning the camera instead would slide near
  // geometry more than far geometry and skew the framing. Used to sit the arena in the
  // middle of the strip the control panel leaves, and to centre it vertically.
  _applyViewOffset() {
    const { w, h } = this._viewport();
    const tx = Math.round(this._viewShift.x), ty = Math.round(this._viewShift.y);
    if (!tx && !ty) {
      this.camera.clearViewOffset();          // also updates the projection matrix
      this.camera.aspect = w / h;
    } else {
      const fw = w + 2 * Math.abs(tx), fh = h + 2 * Math.abs(ty);
      this.camera.aspect = fw / fh;
      this.camera.setViewOffset(fw, fh, Math.abs(tx) - tx, Math.abs(ty) - ty, w, h);
    }
    this.camera.updateProjectionMatrix();
  }
}
