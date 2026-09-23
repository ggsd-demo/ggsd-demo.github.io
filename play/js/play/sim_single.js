// Copyright (c) 2026. MuJoCo player for game-discovery agents.
// SPDX-License-Identifier: BSD-3-Clause
//
// JS mirror of shared/single_env.py (MujocoSingleEnv): one robot, one scene.
// Builds the SAME low-level observation the Isaac play tasks feed the policy,
// applies actions with the SAME clipping / joint mapping, steps the SAME
// decimation, and reports fall / goal the same way. Keep in lockstep with Python.
//
//   "ant35": [ z, lin_b(3), ang_b(3), yaw, proj_grav(3), jpos_norm(8), jvel*0.2(8), last_action(8) ]
//   "g1_86": [ yaw, z, lin_b(3), ang_b(3), proj_grav(3), hands(6), jpos_raw(23), jvel*0.2(23), last_action(23) ]
//
// controlMode "pd_implicit" (G1) reproduces PhysX's implicit joint drive: once per
// physxDt the torque clip(kp (target - q_next) - kv qd_next) is found with a few
// Newton steps around a trial rollout (diagonal effective-inertia model from
// mj_solveM), then held for that step's substeps. rootVelAtCom makes lin_b the
// root COM velocity, which is what Isaac's root_lin_vel_b reports.

const OBJ_BODY = 1, OBJ_JOINT = 3, OBJ_ACTUATOR = 19;

export function yawFromQuat(q) {
  const [w, x, y, z] = q;
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}

// Rotate v by the inverse of quaternion q (w,x,y,z). Matches mj_env.quat_apply_inverse.
export function quatApplyInverse(q, v) {
  const [w, x, y, z] = q;
  const ux = -x, uy = -y, uz = -z;
  const uvx = uy * v[2] - uz * v[1], uvy = uz * v[0] - ux * v[2], uvz = ux * v[1] - uy * v[0];
  const uuvx = uy * uvz - uz * uvy, uuvy = uz * uvx - ux * uvz, uuvz = ux * uvy - uy * uvx;
  return [v[0] + 2 * (w * uvx + uuvx), v[1] + 2 * (w * uvy + uuvy), v[2] + 2 * (w * uvz + uuvz)];
}

function nameMap(mujoco, model, objType, n) {
  const map = {};
  for (let i = 0; i < n; i++) {
    const nm = mujoco.mj_id2name(model, objType, i);
    if (nm) map[nm] = i;
  }
  return map;
}

export class MujocoSingleEnv {
  constructor(mujoco, model, data, robot, scene) {
    this.mujoco = mujoco; this.model = model; this.data = data;
    this.robot = robot; this.scene = scene;
    this.order = robot.jointOrder; this.lo = robot.jointLo; this.hi = robot.jointHi;
    this.nJ = this.order.length;
    const jmap = nameMap(mujoco, model, OBJ_JOINT, model.njnt);
    const amap = nameMap(mujoco, model, OBJ_ACTUATOR, model.nu);
    const bmap = nameMap(mujoco, model, OBJ_BODY, model.nbody);
    const rj = jmap["root"];
    this.rootQ = model.jnt_qposadr[rj]; this.rootV = model.jnt_dofadr[rj];
    this.rootBodyId = bmap[robot.rootBody];
    this.termBodyId = robot.terminationBody ? bmap[robot.terminationBody] : this.rootBodyId;
    this.gloveIds = robot.gloveBodies.map((n) => bmap[n]);
    this.jq = []; this.jv = []; this.aid = [];
    for (const n of this.order) {
      const j = jmap[n];
      if (j == null || amap[n] == null) throw new Error(`joint/actuator '${n}' missing in model`);
      this.jq.push(model.jnt_qposadr[j]); this.jv.push(model.jnt_dofadr[j]); this.aid.push(amap[n]);
    }
    const cj = jmap["cube_root"];
    this.cubeQ = cj == null ? null : model.jnt_qposadr[cj];
    this.cubeV = cj == null ? null : model.jnt_dofadr[cj];
    this.lastAction = new Float64Array(robot.numActions);
    this._obs = new Float64Array(robot.obsDim);
    this.stepCount = 0;
    this.comOffset = robot.rootVelAtCom
      ? [model.body_ipos[3 * this.rootBodyId], model.body_ipos[3 * this.rootBodyId + 1], model.body_ipos[3 * this.rootBodyId + 2]]
      : null;
    if (robot.controlMode === "pd_implicit") {
      this.kp = Float64Array.from(robot.pdKp); this.kv = Float64Array.from(robot.pdKv);
      this.frc = Float64Array.from(this.aid, (a) => model.actuator_forcerange[2 * a + 1]);
      // The MJCF ships <position> actuators; the implicit drive hands them its own torque,
      // so make them pure-force motors (gain 1, no bias).
      const NPRM = 10, gain = model.actuator_gainprm, bias = model.actuator_biasprm;
      for (const a of this.aid) { gain[NPRM * a] = 1; bias[NPRM * a] = 0; bias[NPRM * a + 1] = 0; bias[NPRM * a + 2] = 0; }
      this.nSub = Math.max(1, Math.round(robot.physxDt / robot.simDt));
      this.nPhysx = Math.max(1, Math.round(robot.decimation / this.nSub));
      this.target = new Float64Array(this.nJ);
      this._tau = new Float64Array(this.nJ); this._c = new Float64Array(this.nJ);
      this._qpos0 = new Float64Array(model.nq); this._qvel0 = new Float64Array(model.nv);
      this._warm0 = new Float64Array(model.nv); this._act0 = new Float64Array(model.na);
    }
    this.reset(false);
  }

  reset(randomize = false) {
    const { mujoco, model, data: d, robot, scene } = this;
    mujoco.mj_resetData(model, d);
    const a = this.rootQ;
    d.qpos[a] = robot.spawnPos[0]; d.qpos[a + 1] = robot.spawnPos[1]; d.qpos[a + 2] = robot.spawnPos[2];
    const yaw = scene.spawnYaw;
    d.qpos[a + 3] = Math.cos(yaw / 2); d.qpos[a + 4] = 0; d.qpos[a + 5] = 0; d.qpos[a + 6] = Math.sin(yaw / 2);
    for (let i = 0; i < 6; i++) d.qvel[this.rootV + i] = 0;
    for (let k = 0; k < this.nJ; k++) {
      let q = robot.defaultJointPos[k];
      if (randomize && robot.resetJointPosNoise > 0) q += (Math.random() * 2 - 1) * robot.resetJointPosNoise;
      d.qpos[this.jq[k]] = Math.min(Math.max(q, this.lo[k]), this.hi[k]);
      d.qvel[this.jv[k]] = 0;
    }
    if (this.cubeQ != null) {
      const c = this.cubeQ;
      d.qpos[c] = scene.cubePos[0]; d.qpos[c + 1] = scene.cubePos[1]; d.qpos[c + 2] = 0.5 * scene.cubeSize;
      d.qpos[c + 3] = 1; d.qpos[c + 4] = 0; d.qpos[c + 5] = 0; d.qpos[c + 6] = 0;
      for (let i = 0; i < 6; i++) d.qvel[this.cubeV + i] = 0;
    }
    this.lastAction.fill(0);
    this.stepCount = 0;
    mujoco.mj_forward(model, d);
  }

  rootPos() { const a = this.rootQ, q = this.data.qpos; return [q[a], q[a + 1], q[a + 2]]; }
  rootQuat() { const a = this.rootQ, q = this.data.qpos; return [q[a + 3], q[a + 4], q[a + 5], q[a + 6]]; }
  yaw() { return yawFromQuat(this.rootQuat()); }
  bodyPos(id) { const p = this.data.xpos; return [p[3 * id], p[3 * id + 1], p[3 * id + 2]]; }
  cubePos() { if (this.cubeQ == null) return null; const c = this.cubeQ, q = this.data.qpos; return [q[c], q[c + 1], q[c + 2]]; }

  bodyVelLocal() {
    const v = this.rootV, qv = this.data.qvel;
    const lin_b = quatApplyInverse(this.rootQuat(), [qv[v], qv[v + 1], qv[v + 2]]);
    const ang_b = [qv[v + 3], qv[v + 4], qv[v + 5]];
    if (this.comOffset) {
      const r = this.comOffset;
      lin_b[0] += ang_b[1] * r[2] - ang_b[2] * r[1];
      lin_b[1] += ang_b[2] * r[0] - ang_b[0] * r[2];
      lin_b[2] += ang_b[0] * r[1] - ang_b[1] * r[0];
    }
    return [ang_b, lin_b];
  }

  handPosObs(out, off) {
    const root = this.rootPos(), yaw = this.yaw(), c = Math.cos(yaw), s = Math.sin(yaw);
    for (const id of this.gloveIds) {
      const p = this.bodyPos(id);
      const dx = p[0] - root[0], dy = p[1] - root[1];
      out[off++] = c * dx + s * dy;
      out[off++] = -s * dx + c * dy;
      out[off++] = p[2];
    }
    return off;
  }

  buildObs() {
    const out = this._obs, r = this.robot;
    const pos = this.rootPos(), quat = this.rootQuat();
    const [ang_b, lin_b] = this.bodyVelLocal();
    const pg = quatApplyInverse(quat, [0, 0, -1]);
    const yaw = yawFromQuat(quat);
    const q = this.data.qpos, qv = this.data.qvel, nJ = this.nJ;
    let o = 0;
    if (r.obsLayout === "ant35") {
      out[o++] = pos[2];
      out[o++] = lin_b[0]; out[o++] = lin_b[1]; out[o++] = lin_b[2];
      out[o++] = ang_b[0]; out[o++] = ang_b[1]; out[o++] = ang_b[2];
      out[o++] = yaw;
      out[o++] = pg[0]; out[o++] = pg[1]; out[o++] = pg[2];
      for (let k = 0; k < nJ; k++) out[o++] = 2 * (q[this.jq[k]] - this.lo[k]) / (this.hi[k] - this.lo[k]) - 1;
    } else if (r.obsLayout === "g1_86") {
      out[o++] = yaw;
      out[o++] = pos[2];
      out[o++] = lin_b[0]; out[o++] = lin_b[1]; out[o++] = lin_b[2];
      out[o++] = ang_b[0]; out[o++] = ang_b[1]; out[o++] = ang_b[2];
      out[o++] = pg[0]; out[o++] = pg[1]; out[o++] = pg[2];
      o = this.handPosObs(out, o);
      for (let k = 0; k < nJ; k++) out[o++] = q[this.jq[k]];
    } else {
      throw new Error(`unknown obs layout ${r.obsLayout}`);
    }
    for (let k = 0; k < nJ; k++) out[o++] = qv[this.jv[k]] * r.jointVelScale;
    for (let k = 0; k < r.numActions; k++) out[o++] = this.lastAction[k];
    if (o !== r.obsDim) throw new Error(`obs is ${o}-dim, expected ${r.obsDim}`);
    return out;
  }

  applyAction(action) {
    const r = this.robot, ctrl = this.data.ctrl;
    for (let k = 0; k < r.numActions; k++) {
      const x = Math.min(Math.max(action[k], -r.clipActions), r.clipActions);
      this.lastAction[k] = x;
      if (r.controlMode === "pd_implicit") this.target[k] = r.actionScale[k] * x;
      else ctrl[this.aid[k]] = r.controlMode === "pd_position" ? r.actionScale[k] * x : x * r.actionTorqueScale;
    }
  }

  step() {
    if (this.robot.controlMode === "pd_implicit") {
      for (let p = 0; p < this.nPhysx; p++) this._implicitPdStep();
    } else {
      for (let i = 0; i < this.robot.decimation; i++) this.mujoco.mj_step(this.model, this.data);
    }
    this.stepCount++;
  }

  // 1 / (M^-1)_jj for one dof: solves M x = e_dof with mj_solveM on the factorisation of the
  // last mj_forward. The WASM build only accepts heap-backed arrays, so two mjData vectors that
  // the next forward pass overwrites anyway (qfrc_inverse, qacc_smooth) serve as scratch.
  _effectiveInertia(dof) {
    const x = this.data.qfrc_inverse, y = this.data.qacc_smooth;
    y.fill(0); y[dof] = 1;
    this.mujoco.mj_solveM(this.model, this.data, x, y);
    return 1.0 / x[dof];
  }

  _rollout(tau) {
    const ctrl = this.data.ctrl;
    for (let k = 0; k < this.nJ; k++) ctrl[this.aid[k]] = tau[k];
    for (let i = 0; i < this.nSub; i++) this.mujoco.mj_step(this.model, this.data);
  }

  _implicitPdStep() {
    const { mujoco, model, data: d, robot } = this;
    const q = d.qpos, qd = d.qvel, kp = this.kp, kv = this.kv, frc = this.frc, tau = this._tau, c = this._c, tgt = this.target;
    this._qpos0.set(q); this._qvel0.set(qd); this._warm0.set(d.qacc_warmstart);
    if (model.na > 0) this._act0.set(d.act);
    const time0 = d.time;
    mujoco.mj_forward(model, d);
    const h = robot.physxDt;
    for (let k = 0; k < this.nJ; k++) {
      const I = this._effectiveInertia(this.jv[k]);
      c[k] = kp[k] * (h * h / (2.0 * I)) + kv[k] * (h / I);
      const t = kp[k] * (tgt[k] - q[this.jq[k]]) - kv[k] * qd[this.jv[k]];
      tau[k] = Math.min(Math.max(t, -frc[k]), frc[k]);
    }
    for (let it = 0; it < robot.implicitIters; it++) {
      this._rollout(tau);
      for (let k = 0; k < this.nJ; k++) {
        const tauPd = kp[k] * (tgt[k] - q[this.jq[k]]) - kv[k] * qd[this.jv[k]];
        const t = (tauPd + c[k] * tau[k]) / (1.0 + c[k]);
        tau[k] = Math.min(Math.max(t, -frc[k]), frc[k]);
      }
      q.set(this._qpos0); qd.set(this._qvel0); d.qacc_warmstart.set(this._warm0);
      if (model.na > 0) d.act.set(this._act0);
      try { d.time = time0; } catch (e) { /* read-only in some builds; time is not used by the physics */ }
    }
    this._rollout(tau);
  }

  fallen() { return this.bodyPos(this.termBodyId)[2] < this.robot.terminationHeight; }

  // Metres to the goal: to the goal point, or, for a goal line, the distance still to
  // travel along the corridor before crossing it (<= 0 once the target's centre is past it).
  goalDistance() {
    const gp = this.scene.goalPos;
    if (!gp) return null;
    const p = this.scene.goalTarget === "cube" ? this.cubePos() : this.rootPos();
    const dir = this.scene.goalDir;
    if (dir) return -((p[0] - gp[0]) * dir[0] + (p[1] - gp[1]) * dir[1]);
    return Math.hypot(p[0] - gp[0], p[1] - gp[1]);
  }

  reachedGoal() {
    const d = this.goalDistance();
    if (d == null) return false;
    return this.scene.goalDir ? d <= 0 : d < this.scene.goalRadius;
  }
}

