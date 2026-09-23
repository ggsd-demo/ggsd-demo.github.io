// Copyright (c) 2026. MuJoCo player for game-discovery agents.
// SPDX-License-Identifier: BSD-3-Clause
//
// JS mirror of shared/franka_env.py (MujocoFrankaHockeyEnv): two Franka Panda arms play
// air hockey. Builds the SAME 61-dim observation, applies the SAME 4-dim task-space
// action (end-effector delta in the side's local frame -> workspace clamp -> one damped
// least-squares IK step per physics substep -> PD joint targets), steps the SAME
// decimation and applies the SAME goal / round / idle-puck rules. Keep in lockstep with
// the Python side.
//
// Observation (per side, local frame = world for "me", x/y negated for "foe"):
//   [ jpos(7), jvel*0.1(7), ee_pos(3), ee_lin_vel(3), sin/cos(ee_yaw)(2), ee_yaw_rate(1) |
//     puck_pos(3), puck_lin_vel(3), puck_ang_vel(3), puck-ee(3), goal-puck xy(2),
//     opp_ee_pos(3), opp_ee_lin_vel(3), opp_jpos(7), opp_jvel*0.1(7),
//     my_score/2, opp_score/2, round_progress(0), match_progress ]

import { FRANKA_JOINT_ORDER, FRANKA_FINGER_JOINTS, FRANKA_JOINT_LO, FRANKA_JOINT_HI, FRANKA_DEFAULT_JOINT_POS, FRANKA_JOINT_VEL_LIMITS, FRANKA } from "./registry.js";

const OBJ_BODY = 1, OBJ_JOINT = 3, OBJ_GEOM = 5, OBJ_ACTUATOR = 19;
const FACING_YAW = { me: 0, foe: Math.PI };
const SIGN = { me: 1, foe: -1 };

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

export class MujocoFrankaHockeyEnv {
  constructor(mujoco, model, data, cfg) {
    this.mujoco = mujoco; this.model = model; this.data = data; this.cfg = cfg;
    this.sides = ["me", "foe"];
    this.order = FRANKA_JOINT_ORDER; this.nJ = this.order.length;
    this.lo = FRANKA_JOINT_LO; this.hi = FRANKA_JOINT_HI;
    const jmap = nameMap(mujoco, model, OBJ_JOINT, model.njnt);
    const amap = nameMap(mujoco, model, OBJ_ACTUATOR, model.nu);
    const bmap = nameMap(mujoco, model, OBJ_BODY, model.nbody);
    const qadr = model.jnt_qposadr, dadr = model.jnt_dofadr;
    this.jid = {}; this.jq = {}; this.jv = {}; this.aid = {}; this.fq = {}; this.fv = {}; this.faid = {};
    this.handId = {}; this.fingerIds = {}; this.baseId = {};
    for (const s of this.sides) {
      this.jid[s] = this.order.map((n) => jmap[`${s}_${n}`]);
      this.jq[s] = this.jid[s].map((j) => qadr[j]); this.jv[s] = this.jid[s].map((j) => dadr[j]);
      this.aid[s] = this.order.map((n) => amap[`${s}_${n}`]);
      const fj = FRANKA_FINGER_JOINTS.map((n) => jmap[`${s}_${n}`]);
      this.fq[s] = fj.map((j) => qadr[j]); this.fv[s] = fj.map((j) => dadr[j]);
      this.faid[s] = FRANKA_FINGER_JOINTS.map((n) => amap[`${s}_${n}`]);
      this.handId[s] = bmap[`${s}_${cfg.handBody}`];
      this.fingerIds[s] = cfg.fingerBodies.map((n) => bmap[`${s}_${n}`]);
      this.baseId[s] = bmap[`${s}_${cfg.baseBody}`];
    }
    const pj = jmap["puck_root"];
    this.puckQ = qadr[pj]; this.puckV = dadr[pj]; this.puckBid = bmap["puck"];
    const gmap = nameMap(mujoco, model, OBJ_GEOM, model.ngeom);
    this.puckGid = gmap["puck"]; this.railGids = new Set(cfg.railGeoms.map((n) => gmap[n])); this.tableGid = gmap[cfg.tableGeom];
    this.pendingBounce = new Map();   // other geom id -> {vin, n}
    // PhysX rigid-body damping of the puck: force = -c m v, torque = -c I w.
    const pm = model.body_mass[this.puckBid], pin = model.body_inertia;
    for (let i = 0; i < 3; i++) {
      model.dof_damping[this.puckV + i] = cfg.puckLinearDamping * pm;
      model.dof_damping[this.puckV + 3 + i] = cfg.puckAngularDamping * pin[3 * this.puckBid + i];
    }
    this.lastAction = { me: new Float64Array(cfg.numActions), foe: new Float64Array(cfg.numActions) };
    this.targetPos = { me: [0, 0, 0], foe: [0, 0, 0] };
    this.targetYawLocal = { me: 0, foe: 0 };
    this.score = { me: 0, foe: 0 };
    this.stepCount = 0;
    this.idleInside = 0; this.idleOutside = 0; this.idleTimeout = false;
    this.hp = null;
    this.lastEvent = ""; this.roundResetFlag = false;
    this._obs = new Float64Array(cfg.obsDim);
    this.reset(true);
  }

  // ---- readers ----
  _bodyPos(id) { const p = this.data.xpos; return [p[3 * id], p[3 * id + 1], p[3 * id + 2]]; }
  _bodyQuat(id) { const q = this.data.xquat; return [q[4 * id], q[4 * id + 1], q[4 * id + 2], q[4 * id + 3]]; }
  // World linear velocity of a body's centre of mass (== mj_objectVelocity(mjOBJ_BODY) lin part).
  _bodyLinVelW(id) {
    const cv = this.data.cvel, xp = this.data.xipos, com = this.data.subtree_com;
    const root = this.model.body_rootid[id];
    const wx = cv[6 * id], wy = cv[6 * id + 1], wz = cv[6 * id + 2];
    const vx = cv[6 * id + 3], vy = cv[6 * id + 4], vz = cv[6 * id + 5];
    const rx = xp[3 * id] - com[3 * root], ry = xp[3 * id + 1] - com[3 * root + 1], rz = xp[3 * id + 2] - com[3 * root + 2];
    return [vx + wy * rz - wz * ry, vy + wz * rx - wx * rz, vz + wx * ry - wy * rx];
  }
  _bodyAngVelW(id) { const cv = this.data.cvel; return [cv[6 * id], cv[6 * id + 1], cv[6 * id + 2]]; }
  _rootPos(side) { return this._bodyPos(this.baseId[side]); }
  _eePos(side) {
    const [a, b] = this.fingerIds[side].map((id) => this._bodyPos(id));
    return [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])];
  }
  _eeQuat(side) { return this._bodyQuat(this.handId[side]); }
  _eeLinVel(side) {
    const [a, b] = this.fingerIds[side].map((id) => this._bodyLinVelW(id));
    return [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])];
  }
  _eeAngVel(side) { return this._bodyAngVelW(this.handId[side]); }
  _jointPos(side) { const q = this.data.qpos; return this.jq[side].map((a) => q[a]); }
  _jointVel(side) { const v = this.data.qvel; return this.jv[side].map((a) => v[a]); }
  _puckPos() { const q = this.data.qpos, a = this.puckQ; return [q[a], q[a + 1], q[a + 2]]; }
  _puckLinVel() { const v = this.data.qvel, a = this.puckV; return [v[a], v[a + 1], v[a + 2]]; }
  // World angular velocity: the free joint's qvel[3:6] is in the body frame -> rotate by xmat.
  _puckAngVel() {
    const v = this.data.qvel, a = this.puckV, R = this.data.xmat, o = 9 * this.puckBid;
    const wl = [v[a + 3], v[a + 4], v[a + 5]];
    return [R[o] * wl[0] + R[o + 1] * wl[1] + R[o + 2] * wl[2],
            R[o + 3] * wl[0] + R[o + 4] * wl[1] + R[o + 5] * wl[2],
            R[o + 6] * wl[0] + R[o + 7] * wl[1] + R[o + 8] * wl[2]];
  }
  // 6 x 7 geometric Jacobian of the fingertip midpoint (world frame; rows lin, ang).
  _jacobian(side) {
    const ee = this._eePos(side), ax = this.data.xaxis, an = this.data.xanchor;
    const J = [[], [], [], [], [], []];
    for (let k = 0; k < this.nJ; k++) {
      const j = this.jid[side][k];
      const a = [ax[3 * j], ax[3 * j + 1], ax[3 * j + 2]];
      const r = [ee[0] - an[3 * j], ee[1] - an[3 * j + 1], ee[2] - an[3 * j + 2]];
      J[0][k] = a[1] * r[2] - a[2] * r[1]; J[1][k] = a[2] * r[0] - a[0] * r[2]; J[2][k] = a[0] * r[1] - a[1] * r[0];
      J[3][k] = a[0]; J[4][k] = a[1]; J[5][k] = a[2];
    }
    return J;
  }
  _toLocal(v, sign) { return [sign * v[0], sign * v[1], v[2]]; }
  _localEeYaw(side) { return wrapAngle(yawFromQuat(this._eeQuat(side)) - FACING_YAW[side]); }
  _targetEeQuatW(side, yawLocal) { return quatMul(this._bodyQuat(this.baseId[side]), quatFromEulerXyz(Math.PI, 0, yawLocal)); }

  // ---- reset ----
  // Match reset: scores to zero, then a round reset. randomize=false takes the centre of
  // every reset range (deterministic replays / parity).
  reset(randomize = true) {
    this.score.me = 0; this.score.foe = 0;
    this.stepCount = 0; this.lastEvent = "";
    this._randomize = randomize;   // round resets inside the match follow the match reset
    this.mujoco.mj_resetData(this.model, this.data);
    this._resetRound(randomize);
  }
  _uni(lo, hi, randomize) { return randomize ? lo + Math.random() * (hi - lo) : 0.5 * (lo + hi); }
  _writeJoints(side, q) {
    const d = this.data;
    for (let k = 0; k < this.nJ; k++) { d.qpos[this.jq[side][k]] = q[k]; d.qvel[this.jv[side][k]] = 0; }
    for (let k = 0; k < this.fq[side].length; k++) { d.qpos[this.fq[side][k]] = 0; d.qvel[this.fv[side][k]] = 0; }
  }
  _resetRound(randomize) {
    const c = this.cfg, d = this.data, m = this.mujoco;
    const a = this.puckQ, v = this.puckV;
    d.qpos[a] = this._uni(c.puckResetXRange[0], c.puckResetXRange[1], randomize);
    d.qpos[a + 1] = this._uni(c.puckResetYRange[0], c.puckResetYRange[1], randomize);
    d.qpos[a + 2] = c.puckSpawnZ;
    d.qpos[a + 3] = 1; d.qpos[a + 4] = 0; d.qpos[a + 5] = 0; d.qpos[a + 6] = 0;
    for (let i = 0; i < 6; i++) d.qvel[v + i] = 0;
    // me: default joints, then IK the fingertip midpoint to a random pose, orientation kept
    let q = [...FRANKA_DEFAULT_JOINT_POS];
    this._writeJoints("me", q); m.mj_forward(this.model, d);
    const tl = [this._uni(c.resetEeXRange[0], c.resetEeXRange[1], randomize),
                this._uni(c.resetEeYRange[0], c.resetEeYRange[1], randomize),
                this._uni(c.resetEeZRange[0], c.resetEeZRange[1], randomize)];
    const tw = this._toLocal(tl, SIGN.me), tq = this._eeQuat("me");
    for (let it = 0; it < c.resetIkIters; it++) {
      const dx = poseError(this._eePos("me"), this._eeQuat("me"), tw, tq);
      const dq = dlsDeltaDof(dx, this._jacobian("me"), this.nJ, c.ikLambda);
      q = q.map((qi, k) => Math.min(Math.max(qi + dq[k], this.lo[k]), this.hi[k]));
      this._writeJoints("me", q); m.mj_forward(this.model, d);
    }
    this._writeJoints("foe", q);   // foe copies me's solved joints (mirrored by its base rotation)
    for (const s of this.sides) {
      this.lastAction[s].fill(0);
      const jp = this._jointPos(s);
      for (let k = 0; k < this.nJ; k++) d.ctrl[this.aid[s][k]] = jp[k];
      for (const id of this.faid[s]) d.ctrl[id] = 0;
    }
    this.idleInside = 0; this.idleOutside = 0; this.idleTimeout = false;
    this.pendingBounce.clear();
    this.roundResetFlag = true;
    m.mj_forward(this.model, d);
  }

  // ---- observation ----
  buildObs(selfSide, oppSide) {
    const c = this.cfg, out = this._obs, sign = SIGN[selfSide];
    let o = 0;
    const put = (arr) => { for (const x of arr) out[o++] = x; };
    const ownEe = this._toLocal(this._eePos(selfSide), sign);
    const puck = this._toLocal(this._puckPos(), sign);
    const yaw = this._localEeYaw(selfSide);
    put(this._jointPos(selfSide));
    put(this._jointVel(selfSide).map((x) => x * c.jointVelScale));
    put(ownEe);
    put(this._toLocal(this._eeLinVel(selfSide), sign));
    put([Math.sin(yaw), Math.cos(yaw), this._eeAngVel(selfSide)[2]]);
    put(puck);
    put(this._toLocal(this._puckLinVel(), sign));
    put(this._toLocal(this._puckAngVel(), sign));
    put([puck[0] - ownEe[0], puck[1] - ownEe[1], puck[2] - ownEe[2]]);
    put([0.5 * c.tableLength - puck[0], 0 - puck[1]]);
    put(this._toLocal(this._eePos(oppSide), sign));
    put(this._toLocal(this._eeLinVel(oppSide), sign));
    put(this._jointPos(oppSide));
    put(this._jointVel(oppSide).map((x) => x * c.jointVelScale));
    const stw = Math.max(1, c.scoreToWin);
    put([this.score[selfSide] / stw, this.score[oppSide] / stw, 0, this.stepCount / Math.max(1, c.maxEpisodeSteps)]);
    return Float64Array.from(out);   // a copy: me's and foe's observations must not alias
  }

  // ---- actions / step ----
  // actions: {side: [dx, dy, dz, dyaw]} in the side's local frame; the target built here
  // from the CURRENT ee pose is held for the whole control step (_build_target_pose).
  applyActions(actions) {
    const c = this.cfg;
    for (const side of Object.keys(actions)) {
      const a = actions[side], la = this.lastAction[side];
      for (let k = 0; k < c.numActions; k++) la[k] = Math.min(Math.max(a[k], -c.clipActions), c.clipActions);
      const sign = SIGN[side], ee = this._toLocal(this._eePos(side), sign);
      const l = [ee[0] + la[0] * c.posActionScale[0], ee[1] + la[1] * c.posActionScale[1], ee[2] + la[2] * c.posActionScale[2]];
      l[0] = Math.min(Math.max(l[0], c.localWorkspaceX[0]), c.localWorkspaceX[1]);
      l[1] = Math.min(Math.max(l[1], c.localWorkspaceY[0]), c.localWorkspaceY[1]);
      l[2] = Math.min(Math.max(l[2], c.localWorkspaceZ[0]), c.localWorkspaceZ[1]);
      this.targetPos[side] = this._toLocal(l, sign);
      const yaw = this._localEeYaw(side) + la[3] * c.yawActionScale;
      this.targetYawLocal[side] = Math.min(Math.max(yaw, -c.localYawLimit), c.localYawLimit);
    }
  }
  // One DLS step toward the held target -> joint position targets (_apply_robot_action).
  _writeIkCtrl(side) {
    const c = this.cfg, d = this.data;
    const dx = poseError(this._eePos(side), this._eeQuat(side), this.targetPos[side], this._targetEeQuatW(side, this.targetYawLocal[side]));
    const dq = dlsDeltaDof(dx, this._jacobian(side), this.nJ, c.ikLambda);
    const q = this._jointPos(side);
    for (let k = 0; k < this.nJ; k++) d.ctrl[this.aid[side][k]] = Math.min(Math.max(q[k] + dq[k], this.lo[k]), this.hi[k]);
    for (const id of this.faid[side]) d.ctrl[id] = 0;
  }
  // FrankaArmSoccerEnv._maybe_unstick_puck (evaluated before the physics step).
  _updateIdle() {
    const c = this.cfg, p = this._puckPos(), v = this._puckLinVel();
    const speed = Math.hypot(v[0], v[1]);
    const inside = c.localWorkspaceX[0] <= -Math.abs(p[0]) && -Math.abs(p[0]) <= c.localWorkspaceX[1]
      && c.localWorkspaceY[0] <= p[1] && p[1] <= c.localWorkspaceY[1];
    const still = speed <= c.puckIdleSpeedThreshold;
    this.idleInside = inside && still ? this.idleInside + 1 : 0;
    this.idleOutside = !inside && still ? this.idleOutside + 1 : 0;
    const dt = this.model.opt.timestep * c.decimation;
    const inThr = Math.max(1, Math.round(c.puckIdleInsideS / dt)), outThr = Math.max(1, Math.round(c.puckIdleOutsideS / dt));
    if (this.idleInside >= inThr || this.idleOutside >= outThr) this.idleTimeout = true;
  }
  step() {
    this.roundResetFlag = false;
    this._updateIdle();
    for (let i = 0; i < this.cfg.decimation; i++) {
      for (const s of this.sides) { if (this.cfg.noFoe && s === "foe") continue; this._writeIkCtrl(s); }
      // mj_step split so the puck's contacts (and its approach speed) are read for the state
      // being integrated, before the inelastic contact resolves them.
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
  }

  // ---- explicit restitution (PhysX has it, MuJoCo does not); mirror of franka_env.py ----
  // {other geom id: unit normal from the other geom toward the puck}
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
  // Once a captured contact stops closing (or is gone), set the puck's separating normal speed
  // to e * approach speed -- the bounce PhysX's restitution would have produced.
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

  // PhysX's max-joint-velocity cap, applied after integration (mirror of _clamp_joint_vel).
  _clampJointVel() {
    const lim = this.cfg.jointVelLimits;
    if (!lim) return;
    const qv = this.data.qvel;
    for (const s of this.sides) for (let k = 0; k < this.nJ; k++) {
      const a = this.jv[s][k];
      if (qv[a] > lim[k]) qv[a] = lim[k]; else if (qv[a] < -lim[k]) qv[a] = -lim[k];
    }
  }

  // ---- rules (FrankaArmSoccerEnv._get_dones + the round bookkeeping of _reset_idx) ----
  // A non-final goal (or, with the play rules, an idle puck) resets the round here and
  // returns no done; the match ends on the score, a puck off the table, the time-out or
  // (training rules) an idle puck. winner: 1 me / -1 foe / 0 draw.
  checkDone() {
    const c = this.cfg, p = this._puckPos();
    const inGoalY = Math.abs(p[1]) <= 0.5 * c.goalWidth;
    const meScored = p[0] > c.goalXThreshold && inGoalY, foeScored = p[0] < -c.goalXThreshold && inGoalY;
    const out = Math.abs(p[0]) > 0.5 * c.tableLength || Math.abs(p[1]) > 0.5 * c.tableWidth;
    const idle = this.idleTimeout;
    this.score.me += meScored ? 1 : 0; this.score.foe += foeScored ? 1 : 0;
    const matchWon = meScored && this.score.me >= c.scoreToWin, matchLost = foeScored && this.score.foe >= c.scoreToWin;
    const decided = matchWon || matchLost;
    const nonfinalGoal = (meScored || foeScored) && !decided;
    const terminated = decided || out;
    const episodeTimeout = this.stepCount >= c.maxEpisodeSteps - 1;
    let idleMatchTimeout, idleRoundReset;
    if (c.puckIdleEndsMatch) { idleMatchTimeout = idle; idleRoundReset = false; }
    else { idleMatchTimeout = false; idleRoundReset = idle && !terminated && !episodeTimeout; }
    const roundReset = nonfinalGoal || idleRoundReset;
    const timeOut = episodeTimeout || idleMatchTimeout || roundReset;
    const timeoutWin = timeOut && this.score.me > this.score.foe && !roundReset;
    const timeoutLoss = timeOut && this.score.me < this.score.foe && !roundReset;
    const matchEnded = terminated || episodeTimeout || idleMatchTimeout;
    const win = (matchWon && !matchLost) || timeoutWin, lose = (matchLost && !matchWon) || timeoutLoss;
    if (meScored) this.lastEvent = "goal me"; else if (foeScored) this.lastEvent = "goal foe";
    if (matchEnded) {
      const winner = win ? 1 : lose ? -1 : 0;
      let reason = matchWon ? "match won" : matchLost ? "match lost" : out ? "puck off table"
        : idleMatchTimeout ? "idle puck" : "timeout, score " + (winner ? "lead" : "tied");
      reason += ` (${this.score.me}-${this.score.foe})`;
      return { terminated, timeout: timeOut && !terminated, winner, reason };
    }
    if (roundReset) {
      this.lastEvent = (meScored ? "goal me" : foeScored ? "goal foe" : "idle puck") + ", new round";
      this._resetRound(this._randomize !== false);
    }
    return { terminated: false, timeout: false, winner: 0, reason: "" };
  }
}

// Build the franka_hockey config (mirror of games/franka_hockey.make_config) from FRANKA.
// rules: "train" (12 s, an idle puck ends the match) or "play" (the *-Play cfg the clips
// were recorded with: 15 s, an idle puck only resets the round).
export function frankaHockeyConfig(rules = "play") {
  const F = FRANKA, r = rules === "train" ? F.rulesTrain : F.rulesPlay;
  return {
    obsLayout: "franka", controlMode: "ik_pose",
    jointOrder: FRANKA_JOINT_ORDER, jointLo: FRANKA_JOINT_LO, jointHi: FRANKA_JOINT_HI, defaultJointPos: FRANKA_DEFAULT_JOINT_POS,
    jointVelLimits: FRANKA_JOINT_VEL_LIMITS,
    numActions: F.numActions, decimation: F.decimation, clipActions: F.clipActions,
    posActionScale: F.posActionScale, yawActionScale: F.yawActionScale,
    localWorkspaceX: F.localWorkspaceX, localWorkspaceY: F.localWorkspaceY, localWorkspaceZ: F.localWorkspaceZ,
    localYawLimit: F.localYawLimit, ikLambda: F.ikLambda, jointVelScale: F.jointVelScale, obsDim: F.obsDim,
    handBody: F.handBody, fingerBodies: F.fingerBodies, baseBody: F.baseBody,
    tableLength: F.tableLength, tableWidth: F.tableWidth, goalWidth: F.goalWidth, goalXThreshold: F.goalXThreshold,
    scoreToWin: F.scoreToWin, maxEpisodeSteps: r.maxEpisodeSteps,
    puckIdleSpeedThreshold: F.puckIdleSpeedThreshold, puckIdleInsideS: r.puckIdleInsideS, puckIdleOutsideS: r.puckIdleOutsideS,
    puckIdleEndsMatch: r.puckIdleEndsMatch,
    puckResetXRange: F.puckResetXRange, puckResetYRange: F.puckResetYRange, puckSpawnZ: F.puckSpawnZ,
    resetEeXRange: F.resetEeXRange, resetEeYRange: F.resetEeYRange, resetEeZRange: F.resetEeZRange, resetIkIters: F.resetIkIters,
    puckLinearDamping: F.puckLinearDamping, puckAngularDamping: F.puckAngularDamping,
    restitutionRail: F.restitutionRail, restitutionTable: F.restitutionTable, restitutionArm: F.restitutionArm,
    bounceThreshold: F.bounceThreshold, railGeoms: F.railGeoms, tableGeom: F.tableGeom,
    // shared player switches (no arena for this game; kept so the UI code can read them)
    boundaryMin: [-0.5 * F.tableLength, -0.5 * F.tableWidth], boundaryMax: [0.5 * F.tableLength, 0.5 * F.tableWidth],
    randomizeSpawnPositions: true, randomizeSpawnYaw: false, noFoe: false,
    rules,
  };
}
