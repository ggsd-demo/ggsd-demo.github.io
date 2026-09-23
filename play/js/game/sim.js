// Copyright (c) 2026. MuJoCo player for game-discovery agents.
// SPDX-License-Identifier: BSD-3-Clause
//
// JS mirror of shared/mj_env.py (MujocoDualEnv) + games/*.py. Builds the SAME
// observation, applies actions with the SAME clipping/joint mapping, steps the
// SAME decimation, and uses the SAME push/fall/out-of-bounds win-lose. Keep in
// lockstep with the Python side.
//
// Two layouts, selected by cfg.obsLayout:
//   "ant" (79 + cfg.highLevelOnlyObsDim): [ self_xy(2) | proprio(36) | rel(3) | opp_xy(2) | opp_proprio(36) | hl(12) ]
//        hl(12) = [ |rel_xy|, rel_pos_b xy, rel_vel_b xy, wall distances / arena size (4),
//                   sin/cos(opp_yaw - own_yaw), step / max_steps ]   (DualAntEnv._compute_high_level_observations)
//        proprio(36) = [ h, lin_b(3), ang_b(3), yaw_roll(2), proj_grav(3),
//                        jpos_norm(8), jvel*0.2(8), last_action(8) ]
//   "g1" (194-dim): [ self(88) | rel_yaw_aligned(3) | opp_hands(12) | opp(88) | match(3) ]
//        self(88) = [ x, y | yaw, z, lin_b(3), ang_b(3), proj_grav(3), hands(6),
//                     jpos_raw(23), jvel*0.2(23), last_action(23) ]
//        hands = own glove positions in the heading frame (rotated by -yaw, z absolute);
//        opp_hands = opponent glove positions then velocities, both relative to MY
//        torso_link and rotated into its frame; match = [own_hp, opp_hp, time_left] in [0,1].
//        lin_b is the pelvis COM velocity (Isaac root_lin_vel_b). Punch damage and HP
//        follow DualG1Env (hand_vel_perp_sq): see _applyPunchDamage.
// Two control modes (cfg.controlMode): "effort" (ant, <motor> torque) or
// "pd_position" (g1, <position> target; PD lives in the MJCF).

import {
  ISAAC_JOINT_ORDER, JOINT_LO, JOINT_HI, DEFAULT_JOINT_POS, TRANSFER,
  G1_JOINT_ORDER, G1_JOINT_LO, G1_JOINT_HI, G1_DEFAULT_JOINT_POS, G1_ACTION_SCALE, G1,
} from "./registry.js";

const OBJ_BODY = 1, OBJ_JOINT = 3, OBJ_ACTUATOR = 19;

function yawRollFromQuat(q) {
  const [w, x, y, z] = q;
  const roll = Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
  const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  return [yaw, roll];
}

// Rotate v by inverse of quaternion q (w,x,y,z). Matches mj_env.quat_apply_inverse.
function quatApplyInverse(q, v) {
  const [w, x, y, z] = q;
  const ux = -x, uy = -y, uz = -z;
  const uvx = uy * v[2] - uz * v[1];
  const uvy = uz * v[0] - ux * v[2];
  const uvz = ux * v[1] - uy * v[0];
  const uuvx = uy * uvz - uz * uvy;
  const uuvy = uz * uvx - ux * uvz;
  const uuvz = ux * uvy - uy * uvx;
  return [
    v[0] + 2 * (w * uvx + uuvx),
    v[1] + 2 * (w * uvy + uuvy),
    v[2] + 2 * (w * uvz + uuvz),
  ];
}

// Rotate v by quaternion q (w,x,y,z). Matches isaaclab quat_apply.
function quatApply(q, v) {
  const [w, x, y, z] = q;
  const uvx = y * v[2] - z * v[1];
  const uvy = z * v[0] - x * v[2];
  const uvz = x * v[1] - y * v[0];
  const uuvx = y * uvz - z * uvy;
  const uuvy = z * uvx - x * uvz;
  const uuvz = x * uvy - y * uvx;
  return [
    v[0] + 2 * (w * uvx + uuvx),
    v[1] + 2 * (w * uvy + uuvy),
    v[2] + 2 * (w * uvz + uuvz),
  ];
}
function nameMap(mujoco, model, objType, n) {
  const map = {};
  for (let i = 0; i < n; i++) {
    const nm = mujoco.mj_id2name(model, objType, i);
    if (nm) map[nm] = i;
  }
  return map;
}

export class MujocoDualEnv {
  constructor(mujoco, model, data, cfg) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.cfg = cfg;
    this.sides = ["me", "foe"];
    // per-game joint arrays live on the cfg (ant vs g1)
    this.order = cfg.jointOrder;
    this.lo = cfg.jointLo;
    this.hi = cfg.jointHi;
    this.defaultJointPos = cfg.defaultJointPos;
    this.actionScale = cfg.actionScale || null;
    this.nJ = this.order.length;

    const jmap = nameMap(mujoco, model, OBJ_JOINT, model.njnt);
    const amap = nameMap(mujoco, model, OBJ_ACTUATOR, model.nu);
    const bmap = nameMap(mujoco, model, OBJ_BODY, model.nbody);
    const jnt_qposadr = model.jnt_qposadr, jnt_dofadr = model.jnt_dofadr;

    this.rootQ = {}; this.rootV = {}; this.bodyId = {};
    this.jq = {}; this.jv = {}; this.aid = {};
    for (const s of this.sides) {
      const rj = jmap[`${s}_root`];
      this.rootQ[s] = jnt_qposadr[rj];
      this.rootV[s] = jnt_dofadr[rj];
      this.bodyId[s] = bmap[`${s}_${cfg.rootBody || "torso"}`];
      this.jq[s] = []; this.jv[s] = []; this.aid[s] = [];
      for (const n of this.order) {
        const j = jmap[`${s}_${n}`];
        this.jq[s].push(jnt_qposadr[j]);
        this.jv[s].push(jnt_dofadr[j]);
        this.aid[s].push(amap[`${s}_${n}`]);
      }
    }
    this.lastAction = { me: new Float64Array(cfg.numActions), foe: new Float64Array(cfg.numActions) };
    this.stepCount = 0;
    this._obs = new Float64Array(cfg.obsDim);
    this.hp = null;
    if (cfg.obsLayout === "g1") {
      const mass = model.body_mass;
      this.gloveIds = {}; this.targetIds = {}; this.torsoId = {}; this.termId = {};
      this.armIds = {}; this.armW = {}; this.comOffset = {}; this.prevStrike = {};
      for (const s of this.sides) {
        this.gloveIds[s] = cfg.gloveBodies.map((n) => bmap[`${s}_${n}`]);
        this.targetIds[s] = cfg.targetBodies.map((n) => bmap[`${s}_${n}`]);
        this.torsoId[s] = bmap[`${s}_${cfg.torsoBody}`];
        this.termId[s] = bmap[`${s}_${cfg.terminationBody || cfg.rootBody}`];
        // punch_vel_arm_weighted: glove velocity = mass-weighted mean over the arm chain
        this.armIds[s] = cfg.gloveBodies.map((g) =>
          cfg.armChainLinks.map((l) => bmap[`${s}_${g.replace("wrist_yaw_link", l)}`]).filter((id) => id !== undefined));
        this.armW[s] = this.armIds[s].map((ids) => {
          const w = ids.map((id) => mass[id]); const t = w.reduce((a, b) => a + b, 0);
          return w.map((x) => x / t);
        });
        const b = this.bodyId[s];
        this.comOffset[s] = cfg.rootVelAtCom
          ? [model.body_ipos[3 * b], model.body_ipos[3 * b + 1], model.body_ipos[3 * b + 2]] : null;
        this.prevStrike[s] = new Float64Array(this.gloveIds[s].length * this.targetIds[s].length);
      }
      this.isHead = cfg.targetBodies.map((n) => cfg.headBodies.includes(n));
      this.hp = { me: cfg.initialHp, foe: cfg.initialHp };
    }
    this.reset(true);
  }

  // ---- reset ----
  reset(randomize = true) {
    this.mujoco.mj_resetData(this.model, this.data);
    const c = this.cfg;
    let mePos = c.meSpawnPos, foePos = c.foeSpawnPos, meYaw = c.meSpawnYaw, foeYaw = c.foeSpawnYaw;
    if (c.randomizeSpawnPositions) {
      const s = this._sampleSpawn();
      mePos = s.mePos; foePos = s.foePos; meYaw = s.meYaw; foeYaw = s.foeYaw;
    }
    if (c.noFoe) {
      foePos = [c.boundaryMax[0] + 50, c.boundaryMax[1] + 50, c.foeSpawnPos[2]];
      foeYaw = 0;
    }
    this._setAgent("me", mePos, meYaw, randomize);
    this._setAgent("foe", foePos, foeYaw, randomize);
    for (const s of this.sides) this.lastAction[s].fill(0);
    this.stepCount = 0;
    if (this.hp) {
      this.hp.me = c.initialHp; this.hp.foe = c.initialHp;
      for (const s of this.sides) this.prevStrike[s].fill(0);
    }
    this.mujoco.mj_forward(this.model, this.data);
  }

  _sampleSpawn() {
    const c = this.cfg;
    const uni = (lo, hi) => lo + Math.random() * (hi - lo);
    const margin = Math.max(0, c.spawnBoundaryMargin || 0);
    const xLo = c.boundaryMin[0] + margin, xHi = c.boundaryMax[0] - margin;
    const yLo = c.boundaryMin[1] + margin, yHi = c.boundaryMax[1] - margin;
    let mx = uni(xLo, xHi), my = uni(yLo, yHi);
    let fx = uni(xLo, xHi), fy = uni(yLo, yHi);
    const minSep = Math.max(0, c.spawnMinSeparation || 0);
    if (minSep > 0) {
      const sepSq = minSep * minSep;
      const maxA = Math.max(1, c.spawnSeparationMaxAttempts || 1);
      for (let i = 0; i < maxA; i++) {
        const dx = fx - mx, dy = fy - my;
        if (dx * dx + dy * dy >= sepSq) break;
        fx = uni(xLo, xHi); fy = uni(yLo, yHi);
      }
      const dx = fx - mx, dy = fy - my;
      if (dx * dx + dy * dy < sepSq) {
        const th = uni(-Math.PI, Math.PI);
        fx = Math.min(Math.max(mx + Math.cos(th) * minSep, xLo), xHi);
        fy = Math.min(Math.max(my + Math.sin(th) * minSep, yLo), yHi);
      }
    }
    let meYaw, foeYaw;
    const rx = fx - mx, ry = fy - my, off = c.frontYawOffset || 0;
    meYaw = Math.atan2(ry, rx) + off; foeYaw = Math.atan2(-ry, -rx) + off;
    if (c.randomizeSpawnYaw) {
      if (c.spawnYawRangeDeg != null) {
        const h = c.spawnYawRangeDeg * Math.PI / 180;
        meYaw += uni(-h, h); foeYaw += uni(-h, h);
      } else { meYaw = uni(-Math.PI, Math.PI); foeYaw = uni(-Math.PI, Math.PI); }
    }
    const z = c.meSpawnPos[2];
    return { mePos: [mx, my, z], foePos: [fx, fy, z], meYaw, foeYaw };
  }

  _setAgent(side, pos, yaw, randomize) {
    const d = this.data, c = this.cfg;
    const a = this.rootQ[side];
    let px = pos[0], py = pos[1];
    if (randomize && c.resetRootXyNoise > 0) {
      px += (Math.random() * 2 - 1) * c.resetRootXyNoise;
      py += (Math.random() * 2 - 1) * c.resetRootXyNoise;
    }
    d.qpos[a] = px; d.qpos[a + 1] = py; d.qpos[a + 2] = pos[2];
    d.qpos[a + 3] = Math.cos(yaw / 2); d.qpos[a + 4] = 0; d.qpos[a + 5] = 0;
    d.qpos[a + 6] = Math.sin(yaw / 2);
    const v = this.rootV[side];
    for (let i = 0; i < 6; i++) d.qvel[v + i] = 0;
    for (let k = 0; k < this.nJ; k++) {
      let q = this.defaultJointPos[k];
      if (randomize && c.resetJointPosNoise > 0)
        q += (Math.random() * 2 - 1) * c.resetJointPosNoise;
      q = Math.min(Math.max(q, this.lo[k]), this.hi[k]);
      d.qpos[this.jq[side][k]] = q;
      let vel = 0;
      if (randomize && c.resetJointVelNoise > 0)
        vel = (Math.random() * 2 - 1) * c.resetJointVelNoise;
      d.qvel[this.jv[side][k]] = vel;
    }
  }

  // ---- readers ----
  _rootPos(side) { const a = this.rootQ[side]; const q = this.data.qpos; return [q[a], q[a + 1], q[a + 2]]; }
  _rootQuat(side) { const a = this.rootQ[side]; const q = this.data.qpos; return [q[a + 3], q[a + 4], q[a + 5], q[a + 6]]; }
  _yaw(side) { return yawRollFromQuat(this._rootQuat(side))[0]; }

  _bodyVelLocal(side) {
    const v = this.rootV[side], qv = this.data.qvel;
    const quat = this._rootQuat(side);
    const lin_b = quatApplyInverse(quat, [qv[v], qv[v + 1], qv[v + 2]]);
    const ang_b = [qv[v + 3], qv[v + 4], qv[v + 5]];
    const r = this.comOffset && this.comOffset[side];
    if (r) {
      lin_b[0] += ang_b[1] * r[2] - ang_b[2] * r[1];
      lin_b[1] += ang_b[2] * r[0] - ang_b[0] * r[2];
      lin_b[2] += ang_b[0] * r[1] - ang_b[1] * r[0];
    }
    return [ang_b, lin_b];
  }
  // World linear velocity of the root (Isaac root_lin_vel_w; free-joint qvel[0:3]).
  _rootLinVelW(side) { const v = this.rootV[side], qv = this.data.qvel; return [qv[v], qv[v + 1], qv[v + 2]]; }
  _bodyPos(id) { const p = this.data.xpos; return [p[3 * id], p[3 * id + 1], p[3 * id + 2]]; }
  _bodyQuat(id) { const q = this.data.xquat; return [q[4 * id], q[4 * id + 1], q[4 * id + 2], q[4 * id + 3]]; }
  // World linear velocity of a body's centre of mass (Isaac body_lin_vel_w is
  // body_com_lin_vel_w): cvel is the spatial velocity at the kinematic tree's
  // subtree COM, shifted to the body's COM (xipos). Same as mj_objectVelocity(mjOBJ_BODY).
  _bodyLinVelW(id) {
    const cv = this.data.cvel, xp = this.data.xipos, com = this.data.subtree_com;
    const root = this.model.body_rootid[id];
    const wx = cv[6 * id], wy = cv[6 * id + 1], wz = cv[6 * id + 2];
    const vx = cv[6 * id + 3], vy = cv[6 * id + 4], vz = cv[6 * id + 5];
    const rx = xp[3 * id] - com[3 * root], ry = xp[3 * id + 1] - com[3 * root + 1], rz = xp[3 * id + 2] - com[3 * root + 2];
    return [vx + wy * rz - wz * ry, vy + wz * rx - wx * rz, vz + wx * ry - wy * rx];
  }

  // ---- ant observation (79-dim) ----
  _writeProprio(side, out, off) {
    const c = this.cfg;
    const pos = this._rootPos(side);
    const quat = this._rootQuat(side);
    const [ang_b, lin_b] = this._bodyVelLocal(side);
    const [yaw, roll] = yawRollFromQuat(quat);
    const pg = quatApplyInverse(quat, [0, 0, -1]);
    out[off + 0] = pos[2];
    out[off + 1] = lin_b[0]; out[off + 2] = lin_b[1]; out[off + 3] = lin_b[2];
    out[off + 4] = ang_b[0]; out[off + 5] = ang_b[1]; out[off + 6] = ang_b[2];
    out[off + 7] = yaw; out[off + 8] = roll;
    out[off + 9] = pg[0]; out[off + 10] = pg[1]; out[off + 11] = pg[2];
    const q = this.data.qpos, qv = this.data.qvel;
    for (let k = 0; k < 8; k++) {
      const qn = 2 * (q[this.jq[side][k]] - this.lo[k]) / (this.hi[k] - this.lo[k]) - 1;
      out[off + 12 + k] = qn;
    }
    for (let k = 0; k < 8; k++) out[off + 20 + k] = qv[this.jv[side][k]] * c.jointVelScale;
    const la = this.lastAction[side];
    for (let k = 0; k < 8; k++) out[off + 28 + k] = la[k];
    return 36;
  }

  _buildObsAnt(selfSide, oppSide) {
    const out = this._obs;
    const sp = this._rootPos(selfSide), op = this._rootPos(oppSide);
    out[0] = sp[0]; out[1] = sp[1];
    this._writeProprio(selfSide, out, 2);       // 2..38
    out[38] = op[0] - sp[0]; out[39] = op[1] - sp[1]; out[40] = op[2] - sp[2];
    out[41] = op[0]; out[42] = op[1];
    this._writeProprio(oppSide, out, 43);       // 43..79
    if (this.cfg.highLevelOnlyObsDim > 0) this._writeHighLevelObs(selfSide, oppSide, out, 79);
    return Float64Array.from(out);
  }

  // The 12 strategic features appended for the high-level net only (mirror of
  // mj_env._high_level_obs): planar distance to the opponent, its position and relative
  // velocity in my body frame (xy), my four wall distances over the arena size, sin/cos of
  // the heading difference, and the episode progress.
  _writeHighLevelObs(selfSide, oppSide, out, off) {
    const c = this.cfg;
    if (c.highLevelOnlyObsDim !== 12) throw new Error(`unsupported highLevelOnlyObsDim ${c.highLevelOnlyObsDim}`);
    const own = this._rootPos(selfSide), opp = this._rootPos(oppSide), q = this._rootQuat(selfSide);
    const relW = [opp[0] - own[0], opp[1] - own[1], opp[2] - own[2]];
    const relB = quatApplyInverse(q, relW);
    const vo = this._rootLinVelW(selfSide), vp = this._rootLinVelW(oppSide);
    const relVelB = quatApplyInverse(q, [vp[0] - vo[0], vp[1] - vo[1], vp[2] - vo[2]]);
    const [xMin, yMin] = c.boundaryMin, [xMax, yMax] = c.boundaryMax;
    const w = xMax - xMin, h = yMax - yMin;
    const dYaw = this._yaw(oppSide) - this._yaw(selfSide);
    let o = off;
    out[o++] = Math.hypot(relW[0], relW[1]);
    out[o++] = relB[0]; out[o++] = relB[1];
    out[o++] = relVelB[0]; out[o++] = relVelB[1];
    out[o++] = (own[0] - xMin) / w; out[o++] = (xMax - own[0]) / w;
    out[o++] = (own[1] - yMin) / h; out[o++] = (yMax - own[1]) / h;
    out[o++] = Math.sin(dYaw); out[o++] = Math.cos(dYaw);
    out[o++] = this.stepCount / c.maxEpisodeSteps;
    return o - off;
  }

  // ---- g1 observation (194-dim) ----
  // Writes the 88-dim self block [x, y | yaw, z, lin_b, ang_b, proj_grav, hands(6),
  // jpos(23), jvel(23), last_action(23)] at offset off.
  _writeSelfBlockG1(side, out, off) {
    const c = this.cfg;
    const pos = this._rootPos(side);
    const quat = this._rootQuat(side);
    const [ang_b, lin_b] = this._bodyVelLocal(side);
    const pg = quatApplyInverse(quat, [0, 0, -1]);
    const yaw = yawRollFromQuat(quat)[0];
    let o = off;
    out[o++] = pos[0]; out[o++] = pos[1]; out[o++] = yaw;
    out[o++] = pos[2];
    out[o++] = lin_b[0]; out[o++] = lin_b[1]; out[o++] = lin_b[2];
    out[o++] = ang_b[0]; out[o++] = ang_b[1]; out[o++] = ang_b[2];
    out[o++] = pg[0]; out[o++] = pg[1]; out[o++] = pg[2];
    const cs = Math.cos(yaw), sn = Math.sin(yaw);
    for (const id of this.gloveIds[side]) {                          // own hands, heading frame
      const p = this._bodyPos(id);
      const dx = p[0] - pos[0], dy = p[1] - pos[1];
      out[o++] = cs * dx + sn * dy; out[o++] = -sn * dx + cs * dy; out[o++] = p[2];
    }
    const q = this.data.qpos, qv = this.data.qvel, nJ = this.nJ;
    for (let k = 0; k < nJ; k++) out[o++] = q[this.jq[side][k]];              // raw jpos
    for (let k = 0; k < nJ; k++) out[o++] = qv[this.jv[side][k]] * c.jointVelScale;
    const la = this.lastAction[side];
    for (let k = 0; k < nJ; k++) out[o++] = la[k];
    return o - off; // == 88 for nJ=23
  }

  // Opponent glove positions (6) then velocities (6) in MY torso_link frame.
  _writeOppHandsG1(selfSide, oppSide, out, off) {
    const t = this.torsoId[selfSide];
    const tp = this._bodyPos(t), tq = this._bodyQuat(t), tv = this._bodyLinVelW(t);
    let o = off;
    for (const id of this.gloveIds[oppSide]) {
      const p = this._bodyPos(id);
      const r = quatApplyInverse(tq, [p[0] - tp[0], p[1] - tp[1], p[2] - tp[2]]);
      out[o++] = r[0]; out[o++] = r[1]; out[o++] = r[2];
    }
    for (const id of this.gloveIds[oppSide]) {
      const v = this._bodyLinVelW(id);
      const r = quatApplyInverse(tq, [v[0] - tv[0], v[1] - tv[1], v[2] - tv[2]]);
      out[o++] = r[0]; out[o++] = r[1]; out[o++] = r[2];
    }
    return o - off; // 12
  }

  _buildObsG1(selfSide, oppSide) {
    const out = this._obs, c = this.cfg;
    let o = this._writeSelfBlockG1(selfSide, out, 0);                 // 0..88
    // rel = opponent position in self's yaw-aligned frame
    const sp = this._rootPos(selfSide), op = this._rootPos(oppSide);
    const yaw = this._yaw(selfSide), cs = Math.cos(yaw), sn = Math.sin(yaw);
    const dx = op[0] - sp[0], dy = op[1] - sp[1], dz = op[2] - sp[2];
    out[o++] = cs * dx + sn * dy;
    out[o++] = -sn * dx + cs * dy;
    out[o++] = dz;                                                    // 88..91
    o += this._writeOppHandsG1(selfSide, oppSide, out, o);            // 91..103
    o += this._writeSelfBlockG1(oppSide, out, o);                     // 103..191
    const inv = 1 / c.initialHp;
    out[o++] = this.hp[selfSide] * inv;
    out[o++] = this.hp[oppSide] * inv;
    out[o++] = Math.min(1, Math.max(0, 1 - this.stepCount / c.maxEpisodeSteps)); // 191..194
    return Float64Array.from(out);   // a copy: me's and foe's observations must not alias
  }

  buildObs(selfSide, oppSide) {
    return this.cfg.obsLayout === "g1"
      ? this._buildObsG1(selfSide, oppSide)
      : this._buildObsAnt(selfSide, oppSide);
  }

  // ---- actions / step ----
  applyActions(actions) {
    const c = this.cfg;
    if (c.controlMode === "pd_position") {
      const scale = this.actionScale;
      for (const side of Object.keys(actions)) {
        const a = actions[side], clipped = this.lastAction[side];
        for (let k = 0; k < c.numActions; k++) {
          const x = Math.min(Math.max(a[k], -c.clipActions), c.clipActions);
          clipped[k] = x;
          this.data.ctrl[this.aid[side][k]] = scale[k] * x;   // PD position target
        }
      }
      return;
    }
    for (const side of Object.keys(actions)) {
      const a = actions[side], clipped = this.lastAction[side];
      for (let k = 0; k < c.numActions; k++) {
        let x = Math.min(Math.max(a[k], -c.clipActions), c.clipActions);
        clipped[k] = x;
        let tq = x * c.actionTorqueScale;
        if (c.effortLimit != null) tq = Math.min(Math.max(tq, -c.effortLimit), c.effortLimit);
        this.data.ctrl[this.aid[side][k]] = tq;
      }
    }
  }

  step() {
    for (let i = 0; i < this.cfg.decimation; i++) this.mujoco.mj_step(this.model, this.data);
    // mj_step leaves xpos/cvel/contacts at the pre-integration state; refresh them so
    // the observation and damage see the post-step state, as Isaac does.
    if (this.hp) this.mujoco.mj_forward(this.model, this.data);
    this.stepCount++;
    if (this.hp) this._applyPunchDamage();
  }

  // DualG1Env punch damage, hand_vel_perp_sq: per (glove, target) the closing speed
  // of the attacker's arm-chain velocity along the target's outward face normal,
  // max over this and the previous control step, capped, squared, times the scale
  // (head targets x headDamageMultiplier). A pair scores only while its bodies are
  // in contact (punch_hit_force_threshold 0). Both sides' drains come from the
  // pre-step HP, as in Isaac.
  _applyPunchDamage() {
    // model/data array views are re-read each call: Emscripten heap views go stale
    // when the wasm memory grows, so caching them at construction silently breaks.
    const c = this.cfg, d = this.data, gb = this.model.geom_bodyid;
    const touching = new Set();
    for (let i = 0; i < d.ncon; i++) {
      const ct = d.contact.get(i);
      const b1 = gb[ct.geom1], b2 = gb[ct.geom2];
      touching.add(b1 + ":" + b2); touching.add(b2 + ":" + b1);
    }
    const dealt = { me: 0, foe: 0 };
    for (const [atk, def] of [["me", "foe"], ["foe", "me"]]) {
      if (c.noFoe) break;
      const gl = this.gloveIds[atk], tg = this.targetIds[def], prev = this.prevStrike[atk];
      const nT = tg.length;
      const gv = gl.map((_, gi) => {
        const ids = this.armIds[atk][gi], w = this.armW[atk][gi], v = [0, 0, 0];
        for (let k = 0; k < ids.length; k++) { const u = this._bodyLinVelW(ids[k]); v[0] += w[k] * u[0]; v[1] += w[k] * u[1]; v[2] += w[k] * u[2]; }
        return v;
      });
      const tv = tg.map((id) => this._bodyLinVelW(id));
      const tn = tg.map((id) => quatApply(this._bodyQuat(id), c.targetFaceNormalLocal));
      let total = 0;
      for (let gi = 0; gi < gl.length; gi++) {
        for (let ti = 0; ti < nT; ti++) {
          const rx = gv[gi][0] - tv[ti][0], ry = gv[gi][1] - tv[ti][1], rz = gv[gi][2] - tv[ti][2];
          const cur = Math.max(0, -(rx * tn[ti][0] + ry * tn[ti][1] + rz * tn[ti][2]));
          const k = gi * nT + ti;
          const v = Math.min(Math.max(cur, prev[k]), c.punchVelCap);
          prev[k] = cur;
          if (touching.has(gl[gi] + ":" + tg[ti])) {
            total += v * v * c.punchVelSqDamageScale * (this.isHead[ti] ? c.headDamageMultiplier : 1);
          }
        }
      }
      dealt[def] = total;
    }
    this.hp.me = Math.max(0, this.hp.me - Math.min(dealt.me, this.hp.me));
    this.hp.foe = Math.max(0, this.hp.foe - Math.min(dealt.foe, this.hp.foe));
  }

  // ---- termination ----
  _outOfBounds(side) {
    const p = this._rootPos(side), bn = this.cfg.boundaryMin, bx = this.cfg.boundaryMax;
    return p[0] < bn[0] || p[0] > bx[0] || p[1] < bn[1] || p[1] > bx[1];
  }
  _fallen(side) {
    const z = this.termId ? this._bodyPos(this.termId[side])[2] : this._rootPos(side)[2];
    return z < this.cfg.terminationHeight;
  }

  // Ant: a fall/out is a loss. G1 (HP present): DualG1Env semantics -- HP-out or the
  // HP lead at timeout decides; a fall/out with no HP-out only stops the episode.
  checkDone() {
    const c = this.cfg;
    const meArena = this._outOfBounds("me") || this._fallen("me");
    const foeArena = !c.noFoe && (this._outOfBounds("foe") || this._fallen("foe"));
    if (!this.hp) {
      const timeout = this.stepCount >= c.maxEpisodeSteps;
      const terminated = meArena || foeArena;
      const winner = foeArena && !meArena ? 1 : meArena && !foeArena ? -1 : 0;
      return { terminated, timeout, winner, reason: terminated ? "fall/out" : "timeout" };
    }
    const meHp = this.hp.me <= 0, foeHp = !c.noFoe && this.hp.foe <= 0;
    const terminated = meArena || foeArena || meHp || foeHp;
    const timeout = !terminated && this.stepCount >= c.maxEpisodeSteps;
    let winner = 0, reason = "";
    if (foeHp && !meHp) { winner = 1; reason = "foe HP out"; }
    else if (meHp && !foeHp) { winner = -1; reason = "me HP out"; }
    else if (meHp && foeHp) { reason = "both HP out"; }
    else if (terminated) { reason = meArena && foeArena ? "both fell/out" : meArena ? "me fell/out" : "foe fell/out"; }
    else if (timeout) {
      winner = this.hp.me > this.hp.foe ? 1 : this.hp.me < this.hp.foe ? -1 : 0;
      reason = "timeout, HP " + (winner ? "lead" : "tied");
    }
    return { terminated, timeout, winner, reason };
  }
}

// Build the ant_sumo GameConfig (mirror of games/ant_sumo.make_config) from TRANSFER.
export function antSumoConfig() {
  const T = TRANSFER;
  return {
    obsLayout: "ant",
    controlMode: "effort",
    jointOrder: ISAAC_JOINT_ORDER, jointLo: JOINT_LO, jointHi: JOINT_HI,
    defaultJointPos: DEFAULT_JOINT_POS, actionScale: null, rootBody: "torso",
    numActions: T.numActions,
    decimation: T.decimation,
    clipActions: T.clipActions,
    actionTorqueScale: T.actionTorqueScale,
    effortLimit: T.effortLimit,
    jointVelScale: T.jointVelScale,
    obsDim: T.obsDim,
    highLevelOnlyObsDim: T.highLevelOnlyObsDim,
    frontYawOffset: T.frontYawOffset,
    boundaryMin: T.boundaryMin,
    boundaryMax: T.boundaryMax,
    terminationHeight: T.terminationHeight,
    maxEpisodeSteps: T.maxEpisodeSteps,
    meSpawnPos: T.meSpawnPos,
    foeSpawnPos: T.foeSpawnPos,
    meSpawnYaw: T.meSpawnYaw,
    foeSpawnYaw: T.foeSpawnYaw,
    resetJointPosNoise: T.resetJointPosNoise,
    resetJointVelNoise: T.resetJointVelNoise,
    resetRootXyNoise: 0,
    spawnBoundaryMargin: 0.2,
    spawnMinSeparation: 0.8,
    spawnSeparationMaxAttempts: 20,
    randomizeSpawnPositions: false,
    randomizeSpawnYaw: false,
  };
}

// Build the g1_boxing GameConfig (mirror of games/g1_boxing.make_config) from G1.
export function g1BoxingConfig() {
  return {
    obsLayout: "g1",
    controlMode: "pd_position",
    jointOrder: G1_JOINT_ORDER, jointLo: G1_JOINT_LO, jointHi: G1_JOINT_HI,
    defaultJointPos: G1_DEFAULT_JOINT_POS, actionScale: G1_ACTION_SCALE, rootBody: G1.rootBody,
    numActions: G1.numActions,
    decimation: G1.decimation,
    clipActions: G1.clipActions,
    actionTorqueScale: 1.0,
    effortLimit: null,
    jointVelScale: G1.jointVelScale,
    obsDim: G1.obsDim,
    rootVelAtCom: G1.rootVelAtCom,
    gloveBodies: G1.gloveBodies,
    torsoBody: G1.torsoBody,
    terminationBody: G1.terminationBody,
    targetBodies: G1.targetBodies,
    headBodies: G1.headBodies,
    targetFaceNormalLocal: G1.targetFaceNormalLocal,
    armChainLinks: G1.armChainLinks,
    initialHp: G1.initialHp,
    punchVelCap: G1.punchVelCap,
    punchVelSqDamageScale: G1.punchVelSqDamageScale,
    headDamageMultiplier: G1.headDamageMultiplier,
    boundaryMin: G1.boundaryMin,
    boundaryMax: G1.boundaryMax,
    terminationHeight: G1.terminationHeight,
    maxEpisodeSteps: G1.maxEpisodeSteps,
    meSpawnPos: G1.meSpawnPos,
    foeSpawnPos: G1.foeSpawnPos,
    meSpawnYaw: G1.meSpawnYaw,
    foeSpawnYaw: G1.foeSpawnYaw,
    resetJointPosNoise: G1.resetJointPosNoise,
    resetJointVelNoise: G1.resetJointVelNoise,
    resetRootXyNoise: G1.resetRootXyNoise,
    spawnBoundaryMargin: 0.2,
    spawnMinSeparation: 0.8,
    spawnSeparationMaxAttempts: 20,
    randomizeSpawnPositions: false,
    randomizeSpawnYaw: false,
  };
}
