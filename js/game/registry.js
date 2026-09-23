// Copyright (c) 2026. MuJoCo player for game-discovery agents.
// SPDX-License-Identifier: BSD-3-Clause
//
// JS mirror of shared/transfer_cfg.py + shared/registry.py. These numbers MUST
// stay in lockstep with the Python side -- they define the policy<->sim contract
// (joint order, limits, torque scale, obs slicing, arena, spawn). Only changes
// to transfer_cfg.py during calibration need to be copied here.

const D30 = Math.PI / 6;            // 30 deg
const D40 = 2 * Math.PI / 9;        // 40 deg
const D100 = 5 * Math.PI / 9;       // 100 deg

// Isaac DoF order (per side, without me_/foe_ prefix). Calibration knob #1.
export const ISAAC_JOINT_ORDER = [
  "front_left_leg", "front_right_leg", "left_back_leg", "right_back_leg",
  "front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot",
];

// Joint normalization limits (radians) in ISAAC_JOINT_ORDER: Isaac's ant_instanceable.usd
// limits (legs +-40 deg, feet 30..100 deg), which the env normalizes joint positions with.
export const JOINT_LO = [-D40, -D40, -D40, -D40,  D30, -D100, -D100,  D30];
export const JOINT_HI = [ D40,  D40,  D40,  D40,  D100, -D30,  -D30,  D100];

// Default joint angles at reset, in ISAAC_JOINT_ORDER.
export const DEFAULT_JOINT_POS = [0, 0, 0, 0, 0.785398, -0.785398, -0.785398, 0.785398];

export const TRANSFER = {
  numJoints: 8,
  numActions: 8,
  // control / physics
  simDt: 1 / 120,
  decimation: 2,            // physics substeps per control step (control = 1/60 s)
  // action -> torque (calibration knob #2; ant.xml uses gear=1)
  actionScale: 1.0,
  clipActions: 10.0,
  actionTorqueScale: 1.0,   // CALIBRATE
  effortLimit: null,        // CALIBRATE
  // observation (91-dim): [ me_xy(2) | me_proprio(36) | rel(3) | foe_xy(2) | foe_proprio(36) | hl(12) ]
  // The low-level actor reads the me block minus roll: obs[2:38] without index 10 (35 dims);
  // the 12 high-level-only features are DualAntEnv._compute_high_level_observations.
  jointVelScale: 0.2,
  proprioStart: 2,
  proprioDim: 35,
  lowLevelObsIndices: [...Array(36).keys()].map((i) => i + 2).filter((i) => i !== 10),
  obsDim: 91,
  highLevelOnlyObsDim: 12,
  // arena / termination
  boundaryMin: [-4.0, -4.0],
  boundaryMax: [4.0, 4.0],
  terminationHeight: 0.31,
  episodeLengthS: 10.0,
  maxEpisodeSteps: 600,     // 10 / (1/60)
  // spawn (randomize_spawn_* = False in the play cfg)
  meSpawnPos: [-2.0, 0.0, 0.5],
  foeSpawnPos: [2.0, 0.0, 0.5],
  meSpawnYaw: 0.0,
  foeSpawnYaw: Math.PI,
  // Random spawn with yaw not randomized points each ant AT the other plus this offset
  // (DualAntEnvCfg.front_yaw_offset): the ant's front feet are on its local +y side.
  frontYawOffset: -Math.PI / 2,
  resetJointPosNoise: 0.2,
  resetJointVelNoise: 0.1,
};

// Actual TRAINING env of the deployed checkpoint (DualAnt-Hierarchical-Sumo ->
// DualAntHierarchicalEnvCfg, run ant_sumo_facing): a ±4 arena, random spawn positions
// (margin 0.3, min separation 0.8), the ants facing each other (yaw not randomized).
// TRANSFER above holds the *play* cfg (fixed ±2 spawn) that antSumoConfig()/the parity
// test use; the web player defaults to THESE training values.
export const TRAIN = {
  boundaryMin: [-4.0, -4.0],
  boundaryMax: [4.0, 4.0],
  meSpawnPos: [-1.0, 0.0, 0.5],   // fixed-spawn face-off (randomize off)
  foeSpawnPos: [1.0, 0.0, 0.5],
  spawnBoundaryMargin: 0.3,
  spawnMinSeparation: 0.8,
  spawnSeparationMaxAttempts: 20,
  randomizeSpawnPositions: true,
  randomizeSpawnYaw: false,
};

// ===========================================================================
// G1 boxing — JS mirror of shared/g1_transfer_cfg.py. Two Unitree G1 humanoids
// box in a ±5 ring: 23 DoF (wrists welded), PD position control, 194-dim
// observation, 6 skills, HP from punch damage. Matches the v22 training env
// (tag v22.3, DualG1EnvCfg) the deployed checkpoints come from.
// ===========================================================================
// Isaac articulation DoF order (breadth-first by depth, NOT alphabetical -- dumped
// from env.me.joint_names). Joints are mapped BY NAME so g1_boxing.xml tree order
// is irrelevant; this list defines the policy<->sim contract.
export const G1_JOINT_ORDER = [
  "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
  "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
  "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
  "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
  "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
  "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
  "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
  "left_elbow_joint", "right_elbow_joint",
];
export const G1_JOINT_LO = [-2.5307, -2.5307, -2.618, -0.5236, -2.9671, -0.52, -2.7576, -2.7576, -0.52, -0.087267, -0.087267, -3.0892, -3.0892, -0.87267, -0.87267, -1.5882, -2.2515, -0.2618, -0.2618, -2.618, -2.618, -1.0472, -1.0472];
export const G1_JOINT_HI = [2.8798, 2.8798, 2.618, 2.9671, 0.5236, 0.52, 2.7576, 2.7576, 0.52, 2.8798, 2.8798, 2.6704, 2.6704, 0.5236, 0.5236, 2.2515, 1.5882, 0.2618, 0.2618, 2.618, 2.618, 2.0944, 2.0944];
// Elbows bent 1.57; shoulders rolled 0.25 rad out (idx 15/16) so the gloves clear the
// hips at spawn. At 0 each glove starts ~10 cm inside its own hip capsule, opening
// every episode in self-collision. Matches G1_INITIAL_JOINT_POS in dual_g1_env_cfg.py.
export const G1_DEFAULT_JOINT_POS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, -0.25, 0, 0, 0, 0, 1.57, 1.57];
// action -> PD target scale = 1.4 * max(|lo|,|hi|) (mimic_zero_center, offset 0)
export const G1_ACTION_SCALE = G1_JOINT_LO.map((lo, i) => 1.4 * Math.max(Math.abs(lo), Math.abs(G1_JOINT_HI[i])));

export const G1 = {
  numJoints: 23,
  numActions: 23,
  // control / physics: 30 Hz control, but 4x finer physics substeps (dt=1/480,
  // decimation=16) so MuJoCo's explicit <position> PD is stable (see g1_transfer_cfg.py).
  simDt: 1 / 480,
  decimation: 16,
  controlMode: "pd_position",
  clipActions: 1.0,
  // observation (194-dim g1 layout): self(88) | rel(3) | opp_hands(12) | opp(88) | match(3)
  obsLayout: "g1",
  jointVelScale: 0.2,
  proprioStart: 2,
  proprioDim: 86,
  obsDim: 194,
  rootBody: "pelvis",
  rootVelAtCom: true,          // obs lin_vel_b = pelvis COM velocity (Isaac root_lin_vel_b)
  gloveBodies: ["left_wrist_yaw_link", "right_wrist_yaw_link"],
  torsoBody: "torso_link",     // opponent-hand obs frame
  terminationBody: "torso_link",
  // punch damage (punch_damage_source = hand_vel_perp_sq): closing speed of the
  // arm chain's mass-weighted velocity along the target's outward normal, max over
  // this and the previous control step, capped, squared, times the scale; head hits
  // x1.5. Any glove-target contact counts (punch_hit_force_threshold 0).
  targetBodies: ["torso_link", "head_link"],
  headBodies: ["head_link"],
  targetFaceNormalLocal: [1, 0, 0],
  armChainLinks: ["shoulder_yaw_link", "elbow_link", "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link"],
  initialHp: 100000,
  punchVelCap: 12.0,
  punchVelSqDamageScale: 150.0,
  headDamageMultiplier: 1.5,
  // arena / termination
  boundaryMin: [-5.0, -5.0],
  boundaryMax: [5.0, 5.0],
  terminationHeight: 0.65,     // torso_link z
  episodeLengthS: 15.0,
  maxEpisodeSteps: 450,
  // spawn (fixed face-off; the player defaults to the random-spawn training env below)
  meSpawnPos: [-1.2, 0.0, 1.0],
  foeSpawnPos: [1.2, 0.0, 1.0],
  meSpawnYaw: 0.0,
  foeSpawnYaw: Math.PI,
  // No joint reset noise: reset_joint_pos/vel_noise_range are (0, 0) in the G1 training
  // env, which turned off the dual-ant defaults of 0.2 rad / 0.1 rad/s deliberately --
  // with the elbows at 1.57 rad, 0.2 rad opens some episodes with the arms against the
  // torso. The ant above keeps them, since its own training env does.
  resetJointPosNoise: 0.0,
  resetJointVelNoise: 0.0,
  resetRootXyNoise: 0.1,
  // look (play_hierarchical.py defaults, linear RGB): --ee_trail draws a trail behind
  // every glove body of both boxers, radius 0.025 (the G1 default; Franka grippers use
  // 0.015), me blue / foe red. The robots' own colour is baked into the MJCF instead
  // (g1_boxing.xml SIDE_RGBA = Isaac's flat me/foe PreviewSurface).
  trail: { seconds: 0.5, hz: 30, fade: 0.5, buckets: 6, gloveRadius: 0.025,
           meColor: [0.05, 0.15, 0.9], foeColor: [0.9, 0.08, 0.05] },
};

// Training env for g1_boxing: random spawn in the ±5 ring, yaw = facing the
// opponent ± spawnYawRangeDeg.
export const G1_TRAIN = {
  boundaryMin: [-5.0, -5.0],
  boundaryMax: [5.0, 5.0],
  meSpawnPos: [-1.2, 0.0, 1.0],
  foeSpawnPos: [1.2, 0.0, 1.0],
  spawnBoundaryMargin: 0.3,
  spawnMinSeparation: 2.0,
  spawnSeparationMaxAttempts: 20,
  randomizeSpawnPositions: true,
  randomizeSpawnYaw: true,
  spawnYawRangeDeg: 120.0,
};

// ===========================================================================
// Franka air hockey — JS mirror of shared/franka_transfer_cfg.py. Two Panda arms on a
// 1.8 x 1.0 m table (top z 0.75), bases at x = -+0.92, first to 2 goals. Task-space
// action (ee position / yaw deltas in the side's local frame, one DLS IK step per
// physics substep into kp 400 / kd 80 PD), 61-dim observation, 5 skills.
// ===========================================================================
export const FRANKA_JOINT_ORDER = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"];
export const FRANKA_FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"];
export const FRANKA_JOINT_LO = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973];
export const FRANKA_JOINT_HI = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973];
export const FRANKA_DEFAULT_JOINT_POS = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741];
// PhysX caps joint velocities at the USD limits (Isaac reference p99: 2.17 / 2.62 rad/s); the env clamps qvel per substep.
export const FRANKA_JOINT_VEL_LIMITS = [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61];

export const FRANKA = {
  numActions: 4,
  simDt: 1 / 120,
  decimation: 2,            // 60 Hz control, IK + PD every physics substep
  clipActions: 10.0,        // rsl_rl clip_actions of the runner cfg; NOT +-1 (deltas run 3-10x the scale)
  posActionScale: [0.12, 0.12, 0.04],
  yawActionScale: 0.30,
  localWorkspaceX: [-0.90, -0.10],
  localWorkspaceY: [-0.45, 0.45],
  localWorkspaceZ: [0.77, 0.90],
  localYawLimit: 0.85,
  ikLambda: 0.1,
  jointVelScale: 0.1,
  obsDim: 61,
  proprioStart: 0, proprioDim: 23,
  handBody: "panda_hand", fingerBodies: ["panda_leftfinger", "panda_rightfinger"], baseBody: "panda_link0",
  tableLength: 1.8, tableWidth: 1.0, tableTopZ: 0.75, goalWidth: 0.4,
  goalXThreshold: 0.47 * 1.8 - 0.07,     // _compute_round_events
  scoreToWin: 2,
  puckIdleSpeedThreshold: 0.01,
  // Training cfg (FrankaArmSoccerHierarchicalEnvCfg) vs the *-Play cfg the clips use.
  rulesTrain: { maxEpisodeSteps: 720, puckIdleInsideS: 3.0, puckIdleOutsideS: 0.5, puckIdleEndsMatch: true },
  rulesPlay: { maxEpisodeSteps: 900, puckIdleInsideS: 1.0, puckIdleOutsideS: 10.0, puckIdleEndsMatch: false },
  puckResetXRange: [-0.55, 0.55], puckResetYRange: [-0.2, 0.2], puckSpawnZ: 0.75 + 0.015,
  resetEeXRange: [-0.75, -0.6], resetEeYRange: [-0.3, 0.3], resetEeZRange: [0.78, 0.80], resetIkIters: 3,
  puckLinearDamping: 0.25, puckAngularDamping: 2.0,
  // explicit PhysX-style restitution of the puck (see franka_transfer_cfg.py): e per partner + bounce threshold
  restitutionRail: 0.9, restitutionTable: 0.225, restitutionArm: 0.35, bounceThreshold: 0.2,
  railGeoms: ["rail_left", "rail_right", "rail_me_top", "rail_me_bottom", "rail_foe_top", "rail_foe_bottom"], tableGeom: "table",
  // look (play_hierarchical.py defaults, linear RGB): trails and the camera
  trail: { seconds: 0.5, hz: 30, fade: 0.5, buckets: 6, puckRadius: 0.02, eeRadius: 0.015,
           puckColor: [0.95, 0.75, 0.0], meColor: [0.05, 0.15, 0.9], foeColor: [0.9, 0.08, 0.05] },
  cameraEye: [5.8, 5.2, 4.0], cameraLookat: [0.0, 0.0, 0.75],
  cameraFov: 14,            // vertical deg; Isaac's viewport lens frames the table this tightly from that eye
};

// Playable games. ant_sumo, g1_boxing and franka_hockey are implemented.
export const GAMES = {
  ant_sumo: {
    title: "Ant Sumo",
    robot: "ant",
    implemented: true,
    mjcf: "exports/ant_sumo/ant.xml",
    policy: "exports/ant_sumo/ckpts/ant_sumo_facing_60000/policy.json",
    // Selectable runs: full hierarchical exports (export_policy.py --game ant_sumo) under
    // exports/ant_sumo/ckpts/<key>/. The same checkpoint whose low level drives /downstream/.
    runs: [
      { key: "ant_sumo_facing_60000", label: "ant_sumo_facing · iter 60000",
        checkpoint: "dual_ant_hierarchical/ant_sumo_facing/model_60000.pt" },
    ],
    numSkills: 5,
    // Shown next to each skill button in the panel. Free text -- edit these.
    skillNames: ["Move Forward-Right", "Sharp Turn Left", "Turn Left", "No-Op", "Push"],
    // Number key -> skill index (key 1 is entry 0). Reorders the buttons and the
    // keyboard without touching the policy's own skill numbering above.
    keySkills: [2, 1, 0, 3, 4],
    tips: [
      // Tips for this task go here, one string per line.
    ],
    description: "Two ants sumo: push the opponent out of the ring or topple it.",
  },
  ant_boxing: { title: "Ant Fencing", robot: "ant", implemented: false,
    description: "Two ants fence (not implemented yet)." },
  franka_hockey: {
    title: "Franka AirHockey",
    robot: "franka",
    implemented: true,
    mjcf: "exports/franka_hockey/franka_hockey.xml",
    policy: "exports/franka_hockey/ckpts/franka_hockey_seed2_70000/policy.json",
    // Selectable runs: full hierarchical exports (export_policy.py --game franka_hockey)
    // under exports/franka_hockey/ckpts/<key>/.
    runs: [
      { key: "franka_hockey_seed2_70000", label: "franka_hockey seed2 · iter 70000",
        checkpoint: "franka_arm_hierarchical/franka_hockey_seed2/model_70000.pt" },
    ],
    numSkills: 5,
    // Shown next to each skill button in the panel. Free text -- edit these.
    skillNames: ["Bottom Left", "Bottom Right", "Top Left", "Top Right", "Center"],
    // Number keys are the default; this robot uses these letters instead (slot i <- keys[i]).
    skillKeys: ["q", "e", "s", "z", "c"],
    keySkills: [2, 3, 4, 0, 1],
    // Reset (and a fresh load) goes back to this button slot -- the Q one.
    resetSkill: 0,
    tips: [
      // Tips for this task go here, one string per line.
    ],
    description: "Two Franka arms play air hockey: first to two goals wins.",
  },
  g1_boxing: {
    title: "G1 Boxing",
    robot: "g1",
    implemented: true,
    mjcf: "exports/g1_boxing/g1_boxing.xml",
    policy: "exports/g1_boxing/ckpts/v21.2_sky_60000/policy.json",
    // The run the page serves: a full hierarchical export (export_policy.py --game
    // g1_boxing) under exports/g1_boxing/ckpts/<key>/. One entry hides the panel's Run
    // row (game/mode.js runsOf); the two v22.3 DR exports it used to offer are gone.
    runs: [
      { key: "v21.2_sky_60000", label: "v21.2 sky · iter 60000",
        checkpoint: "dual_g1_boxing_hierarchical/g1_boxing_v21.2_PerpSq_R125_FacingVel5x_RSI_Yaw120_OldHitbox_sky/model_60000.pt" },
    ],
    numSkills: 6,
    // Shown next to each skill button in the panel. Free text -- edit these.
    skillNames: ["Turn Right", "Walk Forward", "Turn Left", "Punch", "Guard / Hard Punch from Lean", "Lean Backward"],
    // Number key -> skill index (key 1 is entry 0). Reorders the buttons and the
    // keyboard without touching the policy's own skill numbering above.
    keySkills: [2, 1, 0, 3, 4, 5],
    tips: [
      // Tips for this task go here, one string per line.
    ],
    description: "Two G1 humanoids box: knock the opponent down or out of the ring.",
  },
};
