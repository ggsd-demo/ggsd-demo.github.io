// Copyright (c) 2026. MuJoCo player for game-discovery agents.
// SPDX-License-Identifier: BSD-3-Clause
//
// Franka practice env: ONE Franka Panda arm on the air-hockey table, no opponent.
// The quaternion / IK / restitution math below is copied verbatim from the two-player
// player's web/js/franka_sim.js (itself a mirror of shared/franka_env.py); what differs
// is only what a single arm needs:
//   * one side ("me"), so the model is exports/franka/franka_single.xml (the hockey
//     scene with the foe arm stripped, assets/make_franka_single_xml.py),
//   * a 23-dim observation -- exactly the block the low-level policy conditions on
//     (policy.json: proprio_start 0, proprio_dim 23), which holds no puck and no
//     opponent state: jpos(7), jvel*0.1(7), ee_pos(3), ee_lin_vel(3), sin/cos(ee_yaw)(2),
//     ee_yaw_rate(1),
//   * no match rules: nothing ends the episode, and the puck alone is re-spawned when it
//     is scored, knocked off the table, or left idle out of the arm's reach.
// Physics (IK step, PhysX-style restitution, joint-velocity cap, PD targets) and every
// constant are the franka-hockey ones, so the arm behaves as it does in /demo/.

const OBJ_BODY = 1, OBJ_JOINT = 3, OBJ_GEOM = 5, OBJ_ACTUATOR = 19;

function nameMap(mujoco, model, objType, n) {
  const map = {};
  for (let i = 0; i < n; i++) { const nm = mujoco.mj_id2name(model, objType, i); if (nm) map[nm] = i; }
  return map;
}

// ---- quaternion helpers (w, x, y, z), matching isaacsim.core.utils.torch ----
function quatMul(a, b) {
  const [w1, x1, y1, z1] = a, [w2, x2, y2, z2] = b;
  return [
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  ];
}
function quatConj(q) { return [q[0], -q[1], -q[2], -q[3]]; }
function quatFromEulerXyz(roll, pitch, yaw) {
  const cy = Math.cos(yaw / 2), sy = Math.sin(yaw / 2), cr = Math.cos(roll / 2), sr = Math.sin(roll / 2);
  const cp = Math.cos(pitch / 2), sp = Math.sin(pitch / 2);
  return [cy * cr * cp + sy * sr * sp, cy * sr * cp - sy * cr * sp, cy * cr * sp + sy * sr * cp, sy * cr * cp - cy * sr * sp];
}
function yawFromQuat(q) { const [w, x, y, z] = q; return Math.atan2(2 * (w * z + x * y), w * w + x * x - y * y - z * z); }
function wrapAngle(a) { return Math.atan2(Math.sin(a), Math.cos(a)); }
// isaaclab.utils.math.axis_angle_from_quat
function axisAngleFromQuat(q, eps = 1e-6) {
  if (q[0] < 0) q = [-q[0], -q[1], -q[2], -q[3]];
  const mag = Math.hypot(q[1], q[2], q[3]);
  const half = Math.atan2(mag, q[0]), angle = 2 * half;
  const s = Math.abs(angle) > eps ? Math.sin(half) / angle : 0.5 - angle * angle / 48;
  return [q[1] / s, q[2] / s, q[3] / s];
}
// factory_control.get_pose_error(geometric, axis_angle) -> 6-vector [pos_err, rot_err]
function poseError(pos, quat, tpos, tquat) {
  const dot = tquat[0] * quat[0] + tquat[1] * quat[1] + tquat[2] * quat[2] + tquat[3] * quat[3];
  if (dot < 0) tquat = [-tquat[0], -tquat[1], -tquat[2], -tquat[3]];
  const norm = quatMul(quat, quatConj(quat))[0];
  const qinv = quatConj(quat).map((v) => v / norm);
  const rot = axisAngleFromQuat(quatMul(tquat, qinv));
  return [tpos[0] - pos[0], tpos[1] - pos[1], tpos[2] - pos[2], rot[0], rot[1], rot[2]];
}
// Solve the 6x6 system A x = b (Gaussian elimination with partial pivoting). A is row-major.
function solve6(A, b) {
  const n = 6, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (p !== c) { const t = M[c]; M[c] = M[p]; M[p] = t; }
    const piv = M[c][c];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / piv;
      if (f === 0) continue;
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}
// factory_control.get_delta_dof_pos(ik_method="dls"): J^T (J J^T + lam^2 I)^-1 dx, J is 6 x nJ.
function dlsDeltaDof(dx, J, nJ, lam) {
  const JJt = [];
  for (let i = 0; i < 6; i++) {
    const row = new Array(6).fill(0);
    for (let j = 0; j < 6; j++) { let s = 0; for (let k = 0; k < nJ; k++) s += J[i][k] * J[j][k]; row[j] = s; }
    row[i] += lam * lam;
    JJt.push(row);
  }
  const y = solve6(JJt, dx);
  const dq = new Array(nJ).fill(0);
  for (let k = 0; k < nJ; k++) { let s = 0; for (let i = 0; i < 6; i++) s += J[i][k] * y[i]; dq[k] = s; }
  return dq;
}


export class MujocoFrankaSingleEnv {
  constructor(mujoco, model, data, robot) {
    this.mujoco = mujoco; this.model = model; this.data = data;
    const c = this.cfg = robot.cfg;
    this.order = c.jointOrder; this.nJ = this.order.length;
    this.lo = c.jointLo; this.hi = c.jointHi;
    const jmap = nameMap(mujoco, model, OBJ_JOINT, model.njnt);
    const amap = nameMap(mujoco, model, OBJ_ACTUATOR, model.nu);
    const bmap = nameMap(mujoco, model, OBJ_BODY, model.nbody);
    const qadr = model.jnt_qposadr, dadr = model.jnt_dofadr;
    this.jid = this.order.map((n) => jmap[`me_${n}`]);
    this.jq = this.jid.map((j) => qadr[j]); this.jv = this.jid.map((j) => dadr[j]);
    this.aid = this.order.map((n) => amap[`me_${n}`]);
    const fj = c.fingerJoints.map((n) => jmap[`me_${n}`]);
    this.fq = fj.map((j) => qadr[j]); this.fv = fj.map((j) => dadr[j]);
    this.faid = c.fingerJoints.map((n) => amap[`me_${n}`]);
    this.handId = bmap[`me_${c.handBody}`];
    this.fingerIds = c.fingerBodies.map((n) => bmap[`me_${n}`]);
    this.baseId = bmap[`me_${c.baseBody}`];

    const pj = jmap["puck_root"];
    this.puckQ = qadr[pj]; this.puckV = dadr[pj]; this.puckBid = bmap["puck"];
    const gmap = nameMap(mujoco, model, OBJ_GEOM, model.ngeom);
    this.puckGid = gmap["puck"];
    this.railGids = new Set(c.railGeoms.map((n) => gmap[n]).filter((g) => g !== undefined));
    this.tableGid = gmap[c.tableGeom];
    this.pendingBounce = new Map();
    // PhysX rigid-body damping of the puck: force = -c m v, torque = -c I w.
    const pm = model.body_mass[this.puckBid], pin = model.body_inertia;
    for (let i = 0; i < 3; i++) {
      model.dof_damping[this.puckV + i] = c.puckLinearDamping * pm;
      model.dof_damping[this.puckV + 3 + i] = c.puckAngularDamping * pin[3 * this.puckBid + i];
    }
    this.lastAction = new Float64Array(c.numActions);
    this.targetPos = [0, 0, 0];
    this.targetYawLocal = 0;
    this.stepCount = 0;
    this.puckResets = 0;
    this.idleOutside = 0;
    this._obs = new Float64Array(c.obsDim);
    this.reset(true);
  }

  // ---- readers (world frame == the arm's local frame: this is the "me" side) ----
  _bodyPos(id) { const p = this.data.xpos; return [p[3 * id], p[3 * id + 1], p[3 * id + 2]]; }
  _bodyQuat(id) { const q = this.data.xquat; return [q[4 * id], q[4 * id + 1], q[4 * id + 2], q[4 * id + 3]]; }
  _bodyLinVelW(id) {
    const cv = this.data.cvel, xp = this.data.xipos, com = this.data.subtree_com;
    const root = this.model.body_rootid[id];
    const wx = cv[6 * id], wy = cv[6 * id + 1], wz = cv[6 * id + 2];
    const vx = cv[6 * id + 3], vy = cv[6 * id + 4], vz = cv[6 * id + 5];
    const rx = xp[3 * id] - com[3 * root], ry = xp[3 * id + 1] - com[3 * root + 1], rz = xp[3 * id + 2] - com[3 * root + 2];
    return [vx + wy * rz - wz * ry, vy + wz * rx - wx * rz, vz + wx * ry - wy * rx];
  }
  _bodyAngVelW(id) { const cv = this.data.cvel; return [cv[6 * id], cv[6 * id + 1], cv[6 * id + 2]]; }
  _eePos() {
    const [a, b] = this.fingerIds.map((id) => this._bodyPos(id));
    return [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])];
  }
  _eeQuat() { return this._bodyQuat(this.handId); }
  _eeLinVel() {
    const [a, b] = this.fingerIds.map((id) => this._bodyLinVelW(id));
    return [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])];
  }
  _eeAngVel() { return this._bodyAngVelW(this.handId); }
  _jointPos() { const q = this.data.qpos; return this.jq.map((a) => q[a]); }
  _jointVel() { const v = this.data.qvel; return this.jv.map((a) => v[a]); }
  _puckPos() { const q = this.data.qpos, a = this.puckQ; return [q[a], q[a + 1], q[a + 2]]; }
  _puckLinVel() { const v = this.data.qvel, a = this.puckV; return [v[a], v[a + 1], v[a + 2]]; }
  _eeYaw() { return wrapAngle(yawFromQuat(this._eeQuat())); }
  _targetEeQuatW(yaw) { return quatMul(this._bodyQuat(this.baseId), quatFromEulerXyz(Math.PI, 0, yaw)); }
  // 6 x 7 geometric Jacobian of the fingertip midpoint (world frame; rows lin, ang).
  _jacobian() {
    const ee = this._eePos(), ax = this.data.xaxis, an = this.data.xanchor;
    const J = [[], [], [], [], [], []];
    for (let k = 0; k < this.nJ; k++) {
      const j = this.jid[k];
      const a = [ax[3 * j], ax[3 * j + 1], ax[3 * j + 2]];
      const r = [ee[0] - an[3 * j], ee[1] - an[3 * j + 1], ee[2] - an[3 * j + 2]];
      J[0][k] = a[1] * r[2] - a[2] * r[1]; J[1][k] = a[2] * r[0] - a[0] * r[2]; J[2][k] = a[0] * r[1] - a[1] * r[0];
      J[3][k] = a[0]; J[4][k] = a[1]; J[5][k] = a[2];
    }
    return J;
  }

  // ---- reset ----
  // Isaac's round reset for one arm: puck to a random spot on the table, the fingertip
  // midpoint IK'd to a random pose in front of the arm. randomize=false takes the centre
  // of every range.
  reset(randomize = true) {
    const c = this.cfg, d = this.data, m = this.mujoco;
    this._randomize = randomize;
    m.mj_resetData(this.model, this.data);
    this.stepCount = 0; this.puckResets = 0; this.idleOutside = 0;
    let q = [...c.defaultJointPos];
    this._writeJoints(q); m.mj_forward(this.model, d);
    const t = [this._uni(c.resetEeXRange, randomize), this._uni(c.resetEeYRange, randomize), this._uni(c.resetEeZRange, randomize)];
    const tq = this._eeQuat();
    for (let it = 0; it < c.resetIkIters; it++) {
      const dx = poseError(this._eePos(), this._eeQuat(), t, tq);
      const dq = dlsDeltaDof(dx, this._jacobian(), this.nJ, c.ikLambda);
      q = q.map((qi, k) => Math.min(Math.max(qi + dq[k], this.lo[k]), this.hi[k]));
      this._writeJoints(q); m.mj_forward(this.model, d);
    }
    this.lastAction.fill(0);
    const jp = this._jointPos();
    for (let k = 0; k < this.nJ; k++) d.ctrl[this.aid[k]] = jp[k];
    for (const id of this.faid) d.ctrl[id] = 0;
    this._respawnPuck(randomize);
  }
  _uni(range, randomize) { return randomize ? range[0] + Math.random() * (range[1] - range[0]) : 0.5 * (range[0] + range[1]); }
  _writeJoints(q) {
    const d = this.data;
    for (let k = 0; k < this.nJ; k++) { d.qpos[this.jq[k]] = q[k]; d.qvel[this.jv[k]] = 0; }
    for (let k = 0; k < this.fq.length; k++) { d.qpos[this.fq[k]] = 0; d.qvel[this.fv[k]] = 0; }
  }
  // Puck only: the arm keeps whatever pose it is in (a practice run is never interrupted).
  _respawnPuck(randomize = true) {
    const c = this.cfg, d = this.data, a = this.puckQ, v = this.puckV;
    d.qpos[a] = this._uni(c.puckResetXRange, randomize);
    d.qpos[a + 1] = this._uni(c.puckResetYRange, randomize);
    d.qpos[a + 2] = c.puckSpawnZ;
    d.qpos[a + 3] = 1; d.qpos[a + 4] = 0; d.qpos[a + 5] = 0; d.qpos[a + 6] = 0;
    for (let i = 0; i < 6; i++) d.qvel[v + i] = 0;
    this.idleOutside = 0;
    this.pendingBounce.clear();
    this.mujoco.mj_forward(this.model, d);
  }

  // ---- observation: the 23 dims the low-level policy conditions on ----
  buildObs() {
    const c = this.cfg, out = this._obs;
    let o = 0;
    const put = (arr) => { for (const x of arr) out[o++] = x; };
    const yaw = this._eeYaw();
    put(this._jointPos());
    put(this._jointVel().map((x) => x * c.jointVelScale));
    put(this._eePos());
    put(this._eeLinVel());
    put([Math.sin(yaw), Math.cos(yaw), this._eeAngVel()[2]]);
    return Float64Array.from(out);
  }

  // ---- actions / step ----
  // action = [dx, dy, dz, dyaw]; the target built from the CURRENT ee pose is held for
  // the whole control step (_build_target_pose), then one DLS IK step per substep.
  applyAction(action) {
    const c = this.cfg, la = this.lastAction;
    for (let k = 0; k < c.numActions; k++) la[k] = Math.min(Math.max(action[k], -c.clipActions), c.clipActions);
    const ee = this._eePos();
    const p = [ee[0] + la[0] * c.posActionScale[0], ee[1] + la[1] * c.posActionScale[1], ee[2] + la[2] * c.posActionScale[2]];
    p[0] = Math.min(Math.max(p[0], c.localWorkspaceX[0]), c.localWorkspaceX[1]);
    p[1] = Math.min(Math.max(p[1], c.localWorkspaceY[0]), c.localWorkspaceY[1]);
    p[2] = Math.min(Math.max(p[2], c.localWorkspaceZ[0]), c.localWorkspaceZ[1]);
    this.targetPos = p;
    const yaw = this._eeYaw() + la[3] * c.yawActionScale;
    this.targetYawLocal = Math.min(Math.max(yaw, -c.localYawLimit), c.localYawLimit);
  }
  _writeIkCtrl() {
    const c = this.cfg, d = this.data;
    const dx = poseError(this._eePos(), this._eeQuat(), this.targetPos, this._targetEeQuatW(this.targetYawLocal));
    const dq = dlsDeltaDof(dx, this._jacobian(), this.nJ, c.ikLambda);
    const q = this._jointPos();
    for (let k = 0; k < this.nJ; k++) d.ctrl[this.aid[k]] = Math.min(Math.max(q[k] + dq[k], this.lo[k]), this.hi[k]);
    for (const id of this.faid) d.ctrl[id] = 0;
  }
  step() {
    for (let i = 0; i < this.cfg.decimation; i++) {
      this._writeIkCtrl();
      // mj_step split so the puck's contacts (and its approach speed) are read for the
      // state being integrated, before the inelastic contact resolves them.
      this.mujoco.mj_step1(this.model, this.data);
      const touching = this._puckContacts();
      this._captureBounces(touching);
      this.mujoco.mj_step2(this.model, this.data);
      this._clampJointVel();
      this._releaseBounces(touching);
    }
    // mj_step leaves xpos/cvel at the pre-integration state; refresh for the observation.
    this.mujoco.mj_forward(this.model, this.data);
    this.stepCount++;
    this._maybeRespawnPuck();
  }

  // Scored, knocked off the table, or sitting still where the arm cannot reach it: put
  // the puck back. The arm is never touched, so a skill can be watched without a break.
  _maybeRespawnPuck() {
    const c = this.cfg, p = this._puckPos(), v = this._puckLinVel();
    const scored = Math.abs(p[0]) > c.goalXThreshold && Math.abs(p[1]) <= 0.5 * c.goalWidth;
    const off = Math.abs(p[0]) > 0.5 * c.tableLength || Math.abs(p[1]) > 0.5 * c.tableWidth || p[2] < c.puckSpawnZ - 0.1;
    const reachable = p[0] >= c.localWorkspaceX[0] && p[0] <= c.localWorkspaceX[1]
      && p[1] >= c.localWorkspaceY[0] && p[1] <= c.localWorkspaceY[1];
    const still = Math.hypot(v[0], v[1]) <= c.puckIdleSpeedThreshold;
    this.idleOutside = still && !reachable ? this.idleOutside + 1 : 0;
    const idleSteps = Math.max(1, Math.round(c.puckIdleOutOfReachS / (this.model.opt.timestep * c.decimation)));
    if (scored || off || this.idleOutside >= idleSteps) { this._respawnPuck(this._randomize !== false); this.puckResets++; }
  }

  // ---- explicit restitution (PhysX has it, MuJoCo does not) ----
  _puckContacts() {
    const d = this.data, pg = this.puckGid, out = new Map();
    for (let i = 0; i < d.ncon; i++) {
      const c = d.contact.get(i), f = c.frame;
      if (c.geom1 === pg) out.set(c.geom2, [-f[0], -f[1], -f[2]]);
      else if (c.geom2 === pg) out.set(c.geom1, [f[0], f[1], f[2]]);
    }
    return out;
  }
  _partnerVel(g) {
    const b = this.model.geom_bodyid[g];
    if (this.model.body_weldid[b] === 0) return [0, 0, 0];   // static scene geometry
    return this._bodyLinVelW(b);
  }
  _restitution(g) {
    const c = this.cfg;
    return this.railGids.has(g) ? c.restitutionRail : g === this.tableGid ? c.restitutionTable : c.restitutionArm;
  }
  _puckVn(g, n) {
    const qv = this.data.qvel, a = this.puckV, pv = this._partnerVel(g);
    return (qv[a] - pv[0]) * n[0] + (qv[a + 1] - pv[1]) * n[1] + (qv[a + 2] - pv[2]) * n[2];
  }
  _captureBounces(touching) {
    for (const [g, n] of touching) {
      if (this.pendingBounce.has(g)) continue;
      const vn = this._puckVn(g, n);
      if (vn < -this.cfg.bounceThreshold) this.pendingBounce.set(g, { vin: -vn, n });
    }
  }
  _releaseBounces(touching) {
    const qv = this.data.qvel, a = this.puckV;
    for (const [g, { vin, n }] of [...this.pendingBounce]) {
      const vn = this._puckVn(g, n);
      if (vn >= -0.02 || !touching.has(g)) {
        const target = this._restitution(g) * vin;
        if (vn < target) { const k = target - vn; qv[a] += k * n[0]; qv[a + 1] += k * n[1]; qv[a + 2] += k * n[2]; }
        this.pendingBounce.delete(g);
      }
    }
  }
  // PhysX's max-joint-velocity cap, applied after integration.
  _clampJointVel() {
    const lim = this.cfg.jointVelLimits;
    if (!lim) return;
    const qv = this.data.qvel;
    for (let k = 0; k < this.nJ; k++) {
      const a = this.jv[k];
      if (qv[a] > lim[k]) qv[a] = lim[k]; else if (qv[a] < -lim[k]) qv[a] = -lim[k];
    }
  }

  // ---- the interface main.js drives every robot through ----
  rootPos() { return this._eePos(); }          // follow camera / HUD: the fingertip midpoint
  cubePos() { return this._puckPos(); }
  fallen() { return false; }                   // a bolted-down arm cannot fall
  reachedGoal() { return false; }              // practice: nothing ends the episode
  goalDistance() { return null; }
}
