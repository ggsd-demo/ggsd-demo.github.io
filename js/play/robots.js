// Copyright (c) 2026. MuJoCo player for game-discovery agents.
// SPDX-License-Identifier: BSD-3-Clause
//
// JS mirror of shared/downstream_robots.py: the policy<->sim contract for the two
// downstream robots (joint order, limits, action mapping, control rate, spawn,
// fall check). Numbers come from the training env.yaml + the Isaac reference
// dumps. Keep in lockstep with the Python side.

import { GREY_WALL_COLOR } from "./scenes.js";

const D = Math.PI / 180;

// The Ant and the G1 keep the 1 m checker but wear the light grey of the Ant's Game Play
// arena (js/game/render.js FLOOR_RGB): the two cell greys average to that tone under the
// Playground renderer's dimmer lights, and the grid lines are a shade below them.
// The corridor / maze walls go grey with it: the default off-white vanishes on that floor.
const ARENA_FLOOR_PALETTE = { cells: ["#aeaeae", "#9a9a9a"], line: "#7c7c7c" };

// ---------------------------------------------------------------------------
// Ant (ant_sumo_facing low level): 8 DoF, joint effort, 35-dim obs, 60 Hz.
// ---------------------------------------------------------------------------
export const ANT_JOINT_ORDER = [
  "front_left_leg", "front_right_leg", "left_back_leg", "right_back_leg",
  "front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot",
];
// Isaac's ant_instanceable.usd limits: legs +-40 deg, feet 30..100 deg (mirrored right).
export const ANT_JOINT_LO = [-40 * D, -40 * D, -40 * D, -40 * D, 30 * D, -100 * D, -100 * D, 30 * D];
export const ANT_JOINT_HI = [ 40 * D,  40 * D,  40 * D,  40 * D, 100 * D, -30 * D, -30 * D, 100 * D];
export const ANT_DEFAULT_JOINT_POS = [0, 0, 0, 0, 0.785398, -0.785398, -0.785398, 0.785398];

export const ANT = {
  key: "ant",
  title: "Ant",
  mjcf: "exports/ant/ant_single.xml",
  policy: "exports/ant/policy.json",
  jointOrder: ANT_JOINT_ORDER, jointLo: ANT_JOINT_LO, jointHi: ANT_JOINT_HI,
  defaultJointPos: ANT_DEFAULT_JOINT_POS,
  numActions: 8,
  simDt: 1 / 120,
  decimation: 2,             // 60 Hz control
  controlMode: "effort",
  clipActions: 10.0,
  actionTorqueScale: 1.0,
  actionScale: null,
  obsLayout: "ant35",
  obsDim: 35,
  jointVelScale: 0.2,
  rootBody: "torso",
  gloveBodies: [],
  terminationBody: "torso",
  terminationHeight: 0.31,
  spawnPos: [0, 0, 0.5],
  rootVelAtCom: false,
  floorFriction: 0.75,        // cube-floor only: Isaac terrain 1.0 averaged with the cube's
                              // 0.5 default material (CuboidCfg binds none, and AntEnvCfg
                              // leaves SimulationCfg.physics_material at its 0.5 default)
  baseHeading: Math.PI / 2,   // the Isaac ant asset faces +y at yaw 0
  focusZ: 0.4,
  numSkills: 5,
  // Shown next to each skill button in the panel. Free text -- edit these.
  skillNames: ["Move Forward-Right", "Sharp Turn Left", "Turn Left", "No-Op", "Push"],
  floorPalette: ARENA_FLOOR_PALETTE,
  wallColor: GREY_WALL_COLOR,
  // Number key -> skill index (key 1 is entry 0). Reorders the buttons and the
  // keyboard without touching the policy's own skill numbering above.
  keySkills: [2, 1, 0, 3, 4],
  checkpoint: "dual_ant_hierarchical/ant_sumo_facing/model_60000.pt",
  transferNotes: [
    ["Engine", "MuJoCo (WebAssembly) replaces PhysX. Same dt = 1/120 s, 2 substeps per 60 Hz control step, joint-effort actions (clip ±10)."],
    ["Joint limits", "Isaac's ant USD limits (legs ±40°, feet 30°..100°) drive the normalised joint-position observation."],
    ["Joint dynamics", "Armature 0.01, damping 0 (PhysX ignores joint damping); masses match Isaac (0.911 kg)."],
    ["Floor friction", "Robot-floor 1.0 (Isaac terrain 1.0). The pushed cube gets 0.75: Isaac leaves the cuboid on the 0.5 default material, which PhysX averages with the terrain."],
    ["Arena spawn", "Turned to face +x like the corridor tasks (Isaac SingleAnt-Play leaves the ant facing +y), so every scene opens with the same view."],
    ["Not re-tuned", "The ant was validated against an Isaac reference (obs 2e-7, action 1e-6) but its physics was not re-tuned in the G1 pass."],
  ],
};

// ---------------------------------------------------------------------------
// G1 (g1_boxing_v21.2 low level): 23 DoF (wrists welded), implicit PD, 86-dim
// obs, 30 Hz control = 4 PhysX-sized steps (1/120) of 4 MuJoCo substeps (1/480).
// ---------------------------------------------------------------------------
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
export const G1_DEFAULT_JOINT_POS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.57, 1.57];
// PD target = 1.4 * max(|lo|,|hi|) * clip(action, +-1)  (mimic_zero_center_action)
export const G1_ACTION_SCALE = G1_JOINT_LO.map((lo, i) => 1.4 * Math.max(Math.abs(lo), Math.abs(G1_JOINT_HI[i])));
// MimicKit PD gains (Isaac ImplicitActuatorCfg G1_MIMICKIT_BODY_ACTUATOR), applied by
// sim_single.js as a PhysX-style implicit drive. Effort limits come from the MJCF motors.
const G1_GAINS = [  // [name fragment, kp, kv]
  ["hip_pitch", 40.179238471, 2.557889765], ["hip_yaw", 40.179238471, 2.557889765],
  ["hip_roll", 99.098427777, 6.308801853], ["knee", 99.098427777, 6.308801853],
  ["ankle", 28.501246196, 1.814445686], ["waist_yaw", 40.179238471, 2.557889765],
  ["waist_roll", 28.501246196, 1.814445686], ["waist_pitch", 28.501246196, 1.814445686],
  ["shoulder", 14.250623098, 0.907222843], ["elbow", 14.250623098, 0.907222843],
];
const g1Gain = (name) => G1_GAINS.find((g) => name.includes(g[0]));
export const G1_PD_KP = G1_JOINT_ORDER.map((n) => g1Gain(n)[1]);
export const G1_PD_KV = G1_JOINT_ORDER.map((n) => g1Gain(n)[2]);

export const G1 = {
  key: "g1",
  title: "G1 Humanoid",
  mjcf: "exports/g1/g1_single.xml",
  mjcfMesh: "exports/g1/g1_single_mesh.xml",
  meshDir: "exports/g1/meshes",
  // Same flat Isaac shading as the two-player player's G1, so the robot looks identical
  // in both pages (the MJCF already paints every visual geom "me" blue).
  meshStyle: "isaac",
  policy: "exports/g1/ckpts/v21.2_sky_60000/policy.json",
  jointOrder: G1_JOINT_ORDER, jointLo: G1_JOINT_LO, jointHi: G1_JOINT_HI,
  defaultJointPos: G1_DEFAULT_JOINT_POS,
  numActions: 23,
  simDt: 1 / 480,
  decimation: 16,            // 30 Hz control (4 MuJoCo substeps per 1/120 s PhysX step)
  controlMode: "pd_implicit",  // the Isaac-like implicit drive on top of the MJCF's <position> actuators
  physxDt: 1 / 120,          // the implicit drive (pd_implicit) is solved once per Isaac physics step
  implicitIters: 2,
  pdKp: G1_PD_KP, pdKv: G1_PD_KV,
  rootVelAtCom: true,        // obs lin_vel_b = pelvis COM velocity (Isaac root_lin_vel_b)
  clipActions: 1.0,
  actionTorqueScale: 1.0,
  actionScale: G1_ACTION_SCALE,
  obsLayout: "g1_86",
  obsDim: 86,
  jointVelScale: 0.2,
  rootBody: "pelvis",
  gloveBodies: ["left_wrist_yaw_link", "right_wrist_yaw_link"],
  terminationBody: "torso_link",
  terminationHeight: 0.5,     // play reset; Isaac boxing terminates at 0.65
  spawnPos: [0, 0, 1.0],
  floorFriction: 0.75,        // cube-floor only: Isaac GroundPlaneCfg 0.5 averaged with the
                              // cube's 1.0 default material (DualAntEnvCfg sets
                              // SimulationCfg.physics_material to 1.0 and the whole G1 chain
                              // inherits it; the ant's own AntEnvCfg does not)
  baseHeading: 0,             // the G1 faces +x at yaw 0
  focusZ: 0.8,
  floorPalette: ARENA_FLOOR_PALETTE,   // same floor and walls as the Ant's scenes
  wallColor: GREY_WALL_COLOR,
  numSkills: 6,
  // Shown next to each skill button in the panel. Free text -- edit these.
  skillNames: ["Turn Right", "Walk Forward", "Turn Left", "Punch", "Guard / Hard Punch from Lean", "Lean Backward"],
  // Number key -> skill index (key 1 is entry 0). Reorders the buttons and the
  // keyboard without touching the policy's own skill numbering above.
  keySkills: [2, 1, 0, 3, 4, 5],
  checkpoint: "dual_g1_boxing_hierarchical/g1_boxing_v21.2_PerpSq_R125_FacingVel5x_RSI_Yaw120_OldHitbox_sky/model_60000.pt",
  // The run the page serves: a low-level-only export of one boxing checkpoint
  // (export_policy.py --game g1 --low_level_only) under exports/g1/ckpts/<key>/.
  // The iteration is part of the key, so a deployed policy is always traceable to one
  // model_*.pt. A single entry hides the panel's Run row (mode.js runsOf); the two
  // v22.3 DR exports it used to offer are gone.
  runs: [
    { key: "v21.2_sky_60000", label: "v21.2 sky · iter 60000",
      checkpoint: "dual_g1_boxing_hierarchical/g1_boxing_v21.2_PerpSq_R125_FacingVel5x_RSI_Yaw120_OldHitbox_sky/model_60000.pt" },
  ],
  // Everything that differs from the Isaac SingleG1-Play task, kept here as reference
  // documentation (the player no longer shows it). Tuned with
  // mujoco_play/tune_single_physics.py against Isaac reference rollouts.
  transferNotes: [
    ["Engine", "MuJoCo (WebAssembly) replaces PhysX TGS (8 position / 4 velocity iterations). Isaac steps at 1/120 s with 4 substeps per 30 Hz control step; here each 1/120 s step is 4 MuJoCo substeps of 1/480 s (implicitfast integrator)."],
    ["PD drive", "PhysX solves the joint drive implicitly. Emulated on top of the MJCF's <position> actuators (turned into pure-force motors): torque = clip(kp·(target − q_next) − kv·qd_next, ±effort limit) is solved once per 1/120 s step with 2 Newton iterations around a trial rollout, then held over that step. Gains, limits and the 1.4·max|limit| target scale are Isaac's."],
    ["Self-collision", "On, as in the training and play tasks (enabled_self_collisions=True). MuJoCo skips parent–child pairs like PhysX."],
    ["Joint limits", "PhysX limits are hard. MuJoCo limits use solref 0.005 s / solimp 0.99–0.9999 (default 0.02 s let the knee and elbow overshoot by up to 0.27 rad under the PD torque); residual overshoot < 0.01 rad."],
    ["Contacts", "Stiff MuJoCo contacts (solref 0.001 s, solimp 0.8/0.95/0.4 mm, pyramidal cone, impratio 1, sliding friction only) stand in for PhysX rigid contacts (contact offset 2 cm, rest offset 0, restitution 0). Friction 0.75 everywhere on the floor = PhysX average of the ground (0.5) and the robot (1.0)."],
    ["Hitboxes", "Torso and head are one box each; Isaac splits each into a front slab + back box welded together (same union). Glove = one r 70 mm sphere; Isaac adds a second r 72.1 mm sphere 4 mm behind on a welded link. Both variants were tested: no measurable difference."],
    ["Observation", "Root linear velocity is the pelvis centre-of-mass velocity, as Isaac's root_lin_vel_b (MuJoCo's free joint reports the frame origin). Yaw, height, angular velocity, projected gravity, glove positions, joint state and last action use Isaac's definitions."],
    ["Spawn / reset", "Dropped from z = 1.0 m in the default pose like Isaac; here the episode resets when torso_link drops below 0.5 m (Isaac play never resets on a fall; the boxing task terminates at 0.65 m)."],
    ["Residual gap", "One-step joint error vs Isaac ≈ 0.017 rad RMS (0.022 before tuning). Hopping skills 4 and 6 still fall in some MuJoCo runs (about 1 in 5 within 3 s, half within 10 s) while Isaac never fell in the reference runs."],
  ],
};

// ---------------------------------------------------------------------------
// Franka (franka_hockey seed2 low level): one 7-DoF Panda, task-space IK control,
// 23-dim proprio obs, 60 Hz. Same checkpoint and same scene as /demo/'s Franka
// Hockey with the opponent arm deleted (assets/make_franka_single_xml.py); every
// constant below mirrors shared/franka_transfer_cfg.py (via web/js/registry.js).
// ---------------------------------------------------------------------------
export const FRANKA_JOINT_ORDER = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"];
export const FRANKA_FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"];
export const FRANKA_JOINT_LO = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973];
export const FRANKA_JOINT_HI = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973];
export const FRANKA_DEFAULT_JOINT_POS = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741];
// PhysX caps joint velocities at the USD limits (Isaac reference p99: 2.17 / 2.62 rad/s).
export const FRANKA_JOINT_VEL_LIMITS = [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61];

export const FRANKA = {
  key: "franka",
  title: "Franka Arm",
  kind: "franka",              // franka_single.js env instead of the legged sim_single.js
  mjcf: "exports/franka/franka_single.xml",
  meshDir: "exports/franka/meshes",
  policy: "exports/franka/policy.json",
  numActions: 4,
  simDt: 1 / 120,
  decimation: 2,               // 60 Hz control, IK + PD every physics substep
  obsDim: 23,
  numSkills: 5,
  // Shown next to each skill button in the panel. Free text -- edit these.
  skillNames: ["Bottom Left", "Bottom Right", "Top Left", "Top Right", "Center"],
  // Number keys are the default; this robot uses these letters instead (slot i <- keys[i]).
  skillKeys: ["q", "e", "s", "z", "c"],
  keySkills: [2, 3, 4, 0, 1],
  // Reset (and a fresh load) goes back to this button slot -- the Q one.
  resetSkill: 0,
  checkpoint: "franka_arm_hierarchical/franka_hockey_seed2/model_70000.pt",
  // play_hierarchical.py --ee_trail, as /demo/ draws it: a yellow puck trail and a
  // blue gripper trail, 0.5 s of history sampled at 30 Hz, fading to half size.
  trail: { seconds: 0.5, hz: 30, fade: 0.5, buckets: 6, puckRadius: 0.02, eeRadius: 0.015,
           puckColor: [0.95, 0.75, 0.0], eeColor: [0.05, 0.15, 0.9] },
  cfg: {
    jointOrder: FRANKA_JOINT_ORDER, fingerJoints: FRANKA_FINGER_JOINTS,
    jointLo: FRANKA_JOINT_LO, jointHi: FRANKA_JOINT_HI,
    defaultJointPos: FRANKA_DEFAULT_JOINT_POS, jointVelLimits: FRANKA_JOINT_VEL_LIMITS,
    numActions: 4, decimation: 2,
    clipActions: 10.0,         // the runner's clip_actions; NOT +-1 (deltas run 3-10x the scale)
    posActionScale: [0.12, 0.12, 0.04], yawActionScale: 0.30,
    localWorkspaceX: [-0.90, -0.10], localWorkspaceY: [-0.45, 0.45], localWorkspaceZ: [0.77, 0.90],
    localYawLimit: 0.85, ikLambda: 0.1, jointVelScale: 0.1, obsDim: 23,
    handBody: "panda_hand", fingerBodies: ["panda_leftfinger", "panda_rightfinger"], baseBody: "panda_link0",
    tableLength: 1.8, tableWidth: 1.0, goalWidth: 0.4,
    goalXThreshold: 0.47 * 1.8 - 0.07,
    // Isaac spawns the puck anywhere on the table (x -0.55..0.55) because a second arm
    // covers the far half; with one arm it would sit there unreachable, so practice
    // spawns it inside this arm's workspace (x -0.90..-0.10).
    puckResetXRange: [-0.55, -0.15], puckResetYRange: [-0.2, 0.2], puckSpawnZ: 0.75 + 0.015,
    resetEeXRange: [-0.75, -0.6], resetEeYRange: [-0.3, 0.3], resetEeZRange: [0.78, 0.80], resetIkIters: 3,
    puckIdleSpeedThreshold: 0.01,
    puckIdleOutOfReachS: 2.0,  // practice-only: a still puck the arm cannot reach comes back
    puckLinearDamping: 0.25, puckAngularDamping: 2.0,
    restitutionRail: 0.9, restitutionTable: 0.225, restitutionArm: 0.35, bounceThreshold: 0.2,
    railGeoms: ["rail_left", "rail_right", "rail_me_top", "rail_me_bottom", "rail_foe_top", "rail_foe_bottom"],
    tableGeom: "table",
  },
  transferNotes: [
    ["Scene", "The franka-hockey table, rails, goals and puck of /demo/, with the opponent arm deleted (assets/make_franka_single_xml.py). The low-level policy conditions on this arm's own 23 dims only, so removing the opponent changes nothing it sees."],
    ["Rules", "None: nothing ends the episode. Only the puck is re-spawned - when it is scored, knocked off the table, or left still for 2 s where the arm cannot reach it. R resets the arm and the puck."],
    ["Puck spawn", "Isaac spawns the puck anywhere on the table (x -0.55..0.55), where the second arm covers the far half. With one arm it spawns inside this arm's workspace (x -0.55..-0.15, y +-0.2) so every skill has something to hit."],
    ["Action", "Task-space: end-effector position deltas x (0.12, 0.12, 0.04) plus a yaw delta x 0.30, clamped to the workspace; one damped-least-squares IK step (lambda 0.1) per physics substep drives the kp 400 / kd 80 PD."],
    ["Engine", "MuJoCo (WebAssembly) replaces PhysX. dt 1/120 s, 2 substeps per 60 Hz control step, implicitfast integration, gravity-compensated arm links, no self-collision (as in Isaac)."],
    ["Puck physics", "MuJoCo has no restitution, so PhysX's is applied explicitly (rails 0.9, table 0.225, arm 0.35, bounce threshold 0.2 m/s); sliding friction 0.013 and the free-joint damping reproduce Isaac's free-slide deceleration; joint velocity is capped at 2.175 / 2.61 rad/s after every substep."],
    ["Residual gap", "Calibrated against an Isaac reference rollout: end-effector speed matches (0.91 vs 0.97 m/s mean), the puck still runs slower than Isaac's (mean 1.0 vs 1.6 m/s, p90 2.4 vs 2.9)."],
  ],
};

export const ROBOTS = { ant: ANT, franka: FRANKA, g1: G1 };
