// Copyright (c) 2026. MuJoCo player for game-discovery agents.
// SPDX-License-Identifier: BSD-3-Clause
//
// three.js renderer driven by a MuJoCo model/data: one mesh per geom, transforms
// copied from geom_xpos / geom_xmat every frame (MuJoCo Z-up kept). Adds a
// follow camera for the long corridors. Derived from the dual-robot player's
// renderer (web/js/render.js).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { corridorBounds, mazeBounds } from "./scenes.js";

// The default floor: a cool blue-grey (the Franka). A robot can name its own palette (the
// Ant and the G1 use ARENA_FLOOR_PALETTE, the light grey of the Game Play arena) through
// `floorPalette` in robots.js.
export const DEFAULT_FLOOR_PALETTE = { cells: ["#6f7789", "#5c6375"], line: "#3a4052" };

// 2x2 checker (two greys) with thin darker grid lines on the cell edges; tiled so that
// one cell is one metre on the floor. A palette with `line: null` is drawn flat instead
// (cells[0] over the whole tile, no lines), for scenes that want a plain floor.
function makeFloorTexture(palette = DEFAULT_FLOOR_PALETTE) {
  const size = 512, cell = size / 2, line = 3;
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d");
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
    ctx.fillStyle = !palette.line ? palette.cells[0]
      : (i + j) % 2 === 0 ? palette.cells[0] : palette.cells[1];
    ctx.fillRect(i * cell, j * cell, cell, cell);
  }
  if (palette.line) {
    ctx.fillStyle = palette.line;
    for (let k = 0; k <= 2; k++) {
      ctx.fillRect(k * cell - line / 2, 0, line, size);
      ctx.fillRect(0, k * cell - line / 2, size, line);
    }
  }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

// Fog and shadow frustum sized for the corridor scenes; the fitted views override both in
// _frameBox (the eye sits tens of metres out, and a maze spans up to 32 m).
const FOG_NEAR = 40, FOG_FAR = 90, SHADOW_EXTENT = 14;
// Backdrop and fog colour: the Game Play arena's (js/game/render.js BACKDROP).
const BACKDROP = 0xf3f4f6;
// Trail markers sit on the floor; capped so a long session cannot grow without bound.
// The robot's own trail (mazes) is red; the cube's, in the push scenes, is the same red and
// slightly bigger so it still reads from the corridor camera.
const TRAIL_RADIUS = 0.12, TRAIL_MAX = 2000;
const CUBE_TRAIL_RADIUS = 0.16;
// Maze scenes: an eye 45 deg above the -y side (not Isaac's straight-down view), pulled
// in until the maze spans this much of the canvas the control panel leaves free.
const MAZE_VIEW_DIR = [0, -1, 1], FIT_FILL = 0.94;
// Corridor scenes that ask for it (`fitView` in scenes.js) are framed the same way, from
// behind the start: 54 deg up the corridor's own axis, so the spawn sits near the bottom
// of the frame and the far leg runs up and to the left. The elevation is fitted to a
// reference screenshot; the eye is kept on the corridor's centre line (no y component,
// hence eye.y == target.y) so the first leg is symmetric rather than sheared.
const CORRIDOR_VIEW_DIR = [-1, 0, 1.4];

const GEOM = { PLANE: 0, HFIELD: 1, SPHERE: 2, CAPSULE: 3, ELLIPSOID: 4, CYLINDER: 5, BOX: 6, MESH: 7 };
const GEOM_OBJ = 5;

export class MujocoRenderer {
  constructor(canvas, model, mujoco) {
    this.model = model;
    this.mujoco = mujoco;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.scene = new THREE.Scene();
    // The light backdrop of the Game Play arena (js/game/render.js BACKDROP), so every
    // Playground scene sits under the same sky as Ant Sumo; the fog fades into it.
    this.scene.background = new THREE.Color(BACKDROP);
    this.scene.fog = new THREE.Fog(BACKDROP, FOG_NEAR, FOG_FAR);
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.05, 300);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(-6, -8, 5);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, 0, 0.4);
    this.controls.enableDamping = true;

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x33384a, 1.0));
    // Sun with a shadow camera that follows the robot (see follow()).
    const dir = new THREE.DirectionalLight(0xffffff, 1.4);
    dir.position.set(6, -4, 12);
    dir.castShadow = true;
    dir.shadow.mapSize.set(2048, 2048);
    dir.shadow.camera.near = 0.5; dir.shadow.camera.far = 60;
    this._shadowExtent = SHADOW_EXTENT;
    this._setShadowExtent(SHADOW_EXTENT, dir);
    dir.shadow.bias = -0.0005;
    dir.shadow.normalBias = 0.02;
    this.scene.add(dir);
    this.scene.add(dir.target);
    this.sun = dir;
    this._sunOffset = new THREE.Vector3(6, -4, 12);
    this._floorPalette = DEFAULT_FLOOR_PALETTE;
    this._floorTexture = makeFloorTexture(this._floorPalette);
    const fill = new THREE.DirectionalLight(0xbcd0ff, 0.5);
    fill.position.set(-8, 6, 5);
    this.scene.add(fill);

    this.meshes = [];
    this._m4 = new THREE.Matrix4();
    // Trail: one small red sphere per addTrailPoint() call (render-only, no physics).
    this._trailGeo = new THREE.SphereGeometry(TRAIL_RADIUS, 12, 8);
    this._trailMat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(0xe01b1b), emissive: new THREE.Color(0x5a0808), roughness: 0.6, metalness: 0.0,
    });
    // Cube trail: the same red, in bigger spheres, dropped where the cube has been.
    this._cubeTrailGeo = new THREE.SphereGeometry(CUBE_TRAIL_RADIUS, 12, 8);
    this._cubeTrailMat = this._trailMat;
    this._leftInset = 0;          // canvas pixels the control panel covers (maze framing)
    this._viewShift = { x: 0, y: 0 };   // image translation in pixels (_applyViewOffset)
    this._trails = [];            // franka: play-script trails (puck + gripper)
    this._trail = new THREE.Group();
    this._cubeTrail = new THREE.Group();
    this.scene.add(this._trail);
    this.scene.add(this._cubeTrail);
    this.meshStyle = "metal";
    this._buildGeoms(model);
    this._onResize();
    window.addEventListener("resize", () => this._onResize());
  }

  // `meshStyle` shades the visual meshes, exactly as in the two-player player:
  // "isaac" = the flat PreviewSurface the Isaac env binds over the whole articulation
  // (no metalness, full albedo), "metal" = a brushed shell. Sticky across setModel.
  setModel(model, meshStyle = null, floorPalette = DEFAULT_FLOOR_PALETTE) {
    if (meshStyle) this.meshStyle = meshStyle;
    if (floorPalette !== this._floorPalette) {
      this._floorPalette = floorPalette;
      this._floorTexture.dispose();
      this._floorTexture = makeFloorTexture(floorPalette);
    }
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

  _buildMeshGeom(model, i) {
    const dataid = model.geom_dataid[i];
    if (dataid == null || dataid < 0) return null;
    const va = model.mesh_vertadr[dataid], vn = model.mesh_vertnum[dataid];
    const fa = model.mesh_faceadr[dataid], fn = model.mesh_facenum[dataid];
    const pos = new Float32Array(vn * 3);
    for (let k = 0; k < vn * 3; k++) pos[k] = model.mesh_vert[va * 3 + k];
    const idx = new Uint32Array(fn * 3);
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
      if (rgba[4 * i + 3] <= 0) { this.meshes.push(null); continue; }
      const sx = size[3 * i], sy = size[3 * i + 1], sz = size[3 * i + 2];
      let geo = null, prerotX = false, planeExtent = null;
      if (t === GEOM.MESH) {
        geo = this._buildMeshGeom(model, i);
        if (!geo) { this.meshes.push(null); continue; }
      } else if (t === GEOM.PLANE) {
        const ex = sx > 0 ? sx * 2 : 80, ey = sy > 0 ? sy * 2 : 80;
        geo = new THREE.PlaneGeometry(ex, ey, 1, 1);
        planeExtent = [ex, ey];
      } else if (t === GEOM.SPHERE) {
        geo = new THREE.SphereGeometry(sx, 24, 16);
      } else if (t === GEOM.CAPSULE) {
        geo = new THREE.CapsuleGeometry(sx, 2 * sy, 6, 16); prerotX = true;
      } else if (t === GEOM.CYLINDER) {
        geo = new THREE.CylinderGeometry(sx, sx, 2 * sy, 32); prerotX = true;
      } else if (t === GEOM.ELLIPSOID) {
        geo = new THREE.SphereGeometry(1, 24, 16); geo.scale(sx, sy, sz);
      } else if (t === GEOM.BOX) {
        geo = new THREE.BoxGeometry(2 * sx, 2 * sy, 2 * sz);
      } else {
        this.meshes.push(null); continue;
      }
      if (prerotX) geo.rotateX(Math.PI / 2);

      const r = rgba[4 * i], g = rgba[4 * i + 1], b = rgba[4 * i + 2], a = rgba[4 * i + 3];
      const nm = this.mujoco ? this.mujoco.mj_id2name(model, GEOM_OBJ, i) : null;
      const isPlane = t === GEOM.PLANE, isMesh = t === GEOM.MESH, isGoal = nm === "goal_marker";
      let map = null;
      if (isPlane) {
        // 1 m checker cells with grid lines, so distance travelled is readable. The texture
        // holds a 2x2 checker, so repeat = extent / 2 gives exactly 1 m per cell.
        map = this._floorTexture.clone();
        map.repeat.set(planeExtent[0] / 2, planeExtent[1] / 2);
        map.needsUpdate = true;
      }
      const mat = new THREE.MeshStandardMaterial({
        map,
        color: isPlane ? new THREE.Color(0xffffff) : new THREE.Color().setRGB(r, g, b),
        emissive: isGoal ? new THREE.Color().setRGB(0.5 * r, 0.5 * g, 0.5 * b) : new THREE.Color(0, 0, 0),
        roughness: isMesh ? 0.5 : 0.7,
        metalness: isMesh && this.meshStyle === "metal" ? 0.35 : isMesh ? 0.0 : 0.05,
        side: isMesh ? THREE.DoubleSide : THREE.FrontSide,
        transparent: a < 1.0, opacity: a,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.matrixAutoUpdate = false;
      mesh.castShadow = !isPlane && !isGoal;
      mesh.receiveShadow = true;
      this.scene.add(mesh);
      this.meshes.push(mesh);
    }
  }

  // Place the camera behind the robot's spawn, looking along the corridor (or at the
  // robot in the arena). `heading` is the world yaw the robot faces at spawn.
  // Fixed view, identical for every scene and robot: behind and to the left of the spawn,
  // looking down +x (the direction every corridor starts in). `spec` and `heading` are
  // accepted for call compatibility but do not change the view.
  //
  // A scene with its own `camera` uses that instead, with the fog pushed back and the
  // shadow frustum widened so the whole scene is lit and sharp. A maze, or a corridor
  // with `fitView`, is framed by _frameBox from its own walls instead of either.
  frameScene(spec, focusZ, heading, leftInset = 0) {
    const fit = !spec ? null
      : spec.mazeMap ? { bounds: mazeBounds(spec), dir: MAZE_VIEW_DIR }
      : spec.fitView ? { bounds: corridorBounds(spec), dir: CORRIDOR_VIEW_DIR } : null;
    if (fit) {
      this._frameBox(fit.bounds, spec.wallHeight || 1, fit.dir, leftInset);
      this.controls.update();
      this.lightAt([this.controls.target.x, this.controls.target.y, 0]);
      return;
    }
    this._leftInset = 0; this._viewShift = { x: 0, y: 0 };
    this._applyViewOffset();
    if (spec && spec.camera) {
      const { eye, lookat } = spec.camera;
      this.camera.position.set(eye[0], eye[1], eye[2]);
      this.controls.target.set(lookat[0], lookat[1], lookat[2]);
      const dist = Math.hypot(eye[0] - lookat[0], eye[1] - lookat[1], eye[2] - lookat[2]);
      this.scene.fog.near = Math.max(FOG_NEAR, dist + 20);
      this.scene.fog.far = this.scene.fog.near + (FOG_FAR - FOG_NEAR);
      this._setShadowExtent(spec.shadowExtent || SHADOW_EXTENT);
    } else {
      this.camera.position.set(-6.8, 3.5, 5.85);
      this.controls.target.set(3.0, 0.0, focusZ);
      this.scene.fog.near = FOG_NEAR; this.scene.fog.far = FOG_FAR;
      this._setShadowExtent(SHADOW_EXTENT);
    }
    this.controls.update();
    this.lightAt([this.controls.target.x, this.controls.target.y, 0]);
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

  // Drop a trail marker at world xy (a sphere resting on the floor). The oldest one goes
  // once TRAIL_MAX are down.
  addTrailPoint(pos) {
    const m = new THREE.Mesh(this._trailGeo, this._trailMat);
    m.position.set(pos[0], pos[1], TRAIL_RADIUS);
    m.castShadow = false; m.receiveShadow = false;
    this._trail.add(m);
    if (this._trail.children.length > TRAIL_MAX) this._trail.remove(this._trail.children[0]);
  }
  clearTrail() { this._trail.clear(); this._cubeTrail.clear(); }
  trailLength() { return this._trail.children.length; }

  // The cube's own breadcrumb, same rules (floor-resting sphere, oldest dropped at TRAIL_MAX).
  addCubeTrailPoint(pos) {
    const m = new THREE.Mesh(this._cubeTrailGeo, this._cubeTrailMat);
    m.position.set(pos[0], pos[1], CUBE_TRAIL_RADIUS);
    m.castShadow = false; m.receiveShadow = false;
    this._cubeTrail.add(m);
    if (this._cubeTrail.children.length > TRAIL_MAX) this._cubeTrail.remove(this._cubeTrail.children[0]);
  }

  // Half-size of the square the sun's shadow map covers, centred on the light target.
  _setShadowExtent(extent, light = this.sun) {
    this._shadowExtent = extent;
    const cam = light.shadow.camera;
    cam.left = -extent; cam.right = extent; cam.top = extent; cam.bottom = -extent;
    cam.updateProjectionMatrix();
  }

  // Follow camera: keep the same offset from the robot as the robot moves.
  follow(pos, focusZ) {
    const t = this.controls.target;
    const dx = pos[0] - t.x, dy = pos[1] - t.y, dz = focusZ - t.z;
    this.camera.position.x += dx; this.camera.position.y += dy; this.camera.position.z += dz;
    t.set(pos[0], pos[1], focusZ);
    this.lightAt(pos);
  }

  // Keep the shadow-casting sun (and its limited shadow frustum) centred on the robot.
  lightAt(pos) {
    this.sun.target.position.set(pos[0], pos[1], 0);
    this.sun.position.set(pos[0] + this._sunOffset.x, pos[1] + this._sunOffset.y, this._sunOffset.z);
    this.sun.target.updateMatrixWorld();
  }

  update(data) {
    const xpos = data.geom_xpos, xmat = data.geom_xmat;
    for (let i = 0; i < this.meshes.length; i++) {
      const mesh = this.meshes[i];
      if (!mesh) continue;
      const p = 3 * i, r = 9 * i;
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

  // A walled scene seen from `viewDir`, as close as it can be with every wall (floor to
  // top) still inside the frame. The eye distance is solved for by projecting the box's
  // eight corners and halving the interval until they just fit, the way the Game Play
  // renderer frames its arena; the image is then slid right of the control panel and the
  // box centred vertically (perspective drops its centre below the aim point).
  _frameBox(b, zTop, viewDir, leftInset = 0) {
    const target = new THREE.Vector3(0.5 * (b.min[0] + b.max[0]), 0.5 * (b.min[1] + b.max[1]), 0);
    const pts = [];
    for (const x of [b.min[0], b.max[0]]) for (const y of [b.min[1], b.max[1]]) for (const z of [0, zTop]) {
      pts.push(new THREE.Vector3(x, y, z));
    }
    this._leftInset = leftInset;
    const { w, h, inset } = this._viewport();
    this._viewShift = { x: inset / 2, y: 0 };
    this._applyViewOffset();
    const xLimit = FIT_FILL * (1 - inset / w), yLimit = FIT_FILL;
    const dir = new THREE.Vector3(...viewDir).normalize();
    const p = new THREE.Vector3();
    const project = (d) => {
      this.camera.position.copy(target).addScaledVector(dir, d);
      this.camera.lookAt(target);
      this.camera.updateMatrixWorld();
      const box = { x0: Infinity, x1: -Infinity, y0: Infinity, y1: -Infinity };
      for (const q of pts) {
        p.copy(q).project(this.camera);
        box.x0 = Math.min(box.x0, p.x); box.x1 = Math.max(box.x1, p.x);
        box.y0 = Math.min(box.y0, p.y); box.y1 = Math.max(box.y1, p.y);
      }
      return box;
    };
    let d = 40;
    for (let round = 0; round < 3; round++) {
      let lo = 1, hi = 400;
      for (let i = 0; i < 40; i++) {
        const mid = 0.5 * (lo + hi);
        const box = project(mid);
        const fills = Math.max((box.x1 - box.x0) / (2 * xLimit), (box.y1 - box.y0) / (2 * yLimit));
        if (fills > 1) lo = mid; else hi = mid;
      }
      d = hi;
      const box = project(d);
      this._viewShift.y += (box.y0 + box.y1) / 2 * h / 2;
      this._applyViewOffset();
    }
    project(d);
    this.controls.target.copy(target);
    this.scene.fog.near = Math.max(FOG_NEAR, d + 20);
    this.scene.fog.far = this.scene.fog.near + (FOG_FAR - FOG_NEAR);
    this._setShadowExtent(0.5 * Math.max(b.max[0] - b.min[0], b.max[1] - b.min[1]) + 3);
  }

  _viewport() {
    const c = this.renderer.domElement;
    const w = c.clientWidth || window.innerWidth || 1, h = c.clientHeight || window.innerHeight || 1;
    return { w, h, inset: Math.min(0.6 * w, Math.max(0, this._leftInset || 0)) };
  }

  // Slide the rendered image by `_viewShift` pixels (right / down positive) with an
  // off-axis frustum: the camera renders a w x h window of a larger view, which translates
  // the image at every depth. Panning instead would slide near geometry more than far and
  // skew the framing.
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

  _onResize() {
    const c = this.renderer.domElement;
    const w = c.clientWidth || window.innerWidth, h = c.clientHeight || window.innerHeight;
    this.renderer.setSize(w, h, false);
    this._applyViewOffset();
  }
}
