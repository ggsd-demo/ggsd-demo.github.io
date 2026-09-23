// Copyright (c) 2026. MuJoCo player for game-discovery agents.
// SPDX-License-Identifier: BSD-3-Clause
//
// Playground mode of the web player: one robot (ant, G1 or Franka arm) driven by the
// LOW-LEVEL policy of a hierarchical self-play checkpoint, with the skill chosen by the
// keyboard, in the Isaac play scenes (practice arena / cube push straight / cube push
// with a right turn / walled mazes seen from above; the Franka practises on its hockey
// table with no opponent, see franka_single.js). Same loop as
// scripts/reinforcement_learning/rsl_rl/play_*_skill_policy.py
// --playable: the chosen skill is held skill_duration control steps, then re-read.
//
// ../main.js owns the page: the MuJoCo module, the Robot / Task / Run dropdowns, the
// Play / Reset buttons, the keyboard and the animation loop. It drives this mode through
// the PlayMode object at the bottom (activate / deactivate / frame / ...); everything else
// in here is the scene loop and its own panel rows (Follow camera, Reset on fall, Trail).

import { ROBOTS } from "./robots.js";
import { SCENES, getScene, injectScene } from "./scenes.js";
import { MujocoSingleEnv } from "./sim_single.js";
import { MujocoFrankaSingleEnv } from "./franka_single.js";
import { MujocoRenderer, DEFAULT_FLOOR_PALETTE } from "./render.js";
import { loadPolicy } from "./policy.js";

const els = {};
const $ = (id) => document.getElementById(id);
async function fetchText(url) { const r = await fetch(url); if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`); return r.text(); }

// One camera for both robots: the same focus height and the same world-frame framing,
// whatever way the robot faces at spawn (robots.js focusZ / baseHeading are unused).
const CAMERA_FOCUS_Z = 0.6;
// Maze trail: one marker per this much simulated time.
const TRAIL_SECONDS = 0.5;
// Cube-push scenes: one red marker on the cube's path per this much simulated time,
// plus one at t = 0 (seedCubeTrail) so the path starts where the cube spawned.
const CUBE_TRAIL_SECONDS = 1.0;
// Canvas width the control panel covers, so the maze camera frames the maze in what is
// left of it instead of behind the panel (the Game Play mode does the same).
function panelInset() {
  const p = $("panel");
  return p ? p.getBoundingClientRect().right + 12 : 0;
}

const state = {
  mujoco: null, robotKey: "ant", runKey: null, sceneKey: "arena", robot: null, scene: null,
  robotXml: null, model: null, data: null, env: null, policy: null, renderer: null,
  running: false, loading: false, follow: false, resetOnFall: true,
  trail: true,               // a marker where the robot's root is (mazes) or where the cube is (push scenes)
  puckResets: 0, trailReset: true,   // franka: clear the puck/gripper trails on a respawn
  kbSkill: 0, skill: 0, ctr: 0, controlDt: 1 / 60, acc: 0, last: 0,
  // Success / Fail (a fall or a time-out) / Total, plus the summed sim time of the successes
  // for the average the HUD shows. Reset with the robot and with the scene.
  stats: { success: 0, fail: 0, total: 0, successTime: 0 }, pendingReset: null,
};

function setStatus(msg) { if (els.status) els.status.textContent = msg; }

// Write the STL meshes an MJCF references into the MuJoCo WASM filesystem.
async function loadMeshes(xml, dir) {
  try { state.mujoco.FS.mkdir("/working/meshes"); } catch (e) { /* exists */ }
  const files = [...new Set([...xml.matchAll(/file="([^"]+\.STL)"/gi)].map((m) => m[1]))];
  await Promise.all(files.map(async (f) => {
    const buf = new Uint8Array(await (await fetch(`${dir}/${f}`)).arrayBuffer());
    state.mujoco.FS.writeFile(`/working/meshes/${f}`, buf);
  }));
}

async function loadRobot(key) {
  const robot = ROBOTS[key];
  state.loading = true;
  const useMesh = key === "g1";        // the G1 always renders its visual meshes
  const meshNote = useMesh ? " (meshes ~8 MB)" : robot.kind === "franka" ? " (meshes ~3.6 MB)" : "";
  setStatus(`Loading ${robot.title} model + policy…${meshNote}`);
  try {
    let xml;
    if (useMesh) {
      xml = await fetchText(robot.mjcfMesh);
      await loadMeshes(xml, robot.meshDir);
    } else {
      xml = await fetchText(robot.mjcf);
      if (robot.kind === "franka") await loadMeshes(xml, robot.meshDir);   // the Panda is meshes only
    }
    const run = runFor(robot, state.robotKey === key ? state.runKey : null);
    const policy = await loadPolicy(run.policy, 12345);
    checkPolicyFits(policy, robot);
    state.robotKey = key; state.robot = robot; state.robotXml = xml; state.policy = policy; state.runKey = run.key;
    populateRunSelect(robot);
    state.controlDt = robot.simDt * robot.decimation;
    state.kbSkill = Math.min(state.kbSkill, policy.numSkills - 1);
    if (robot.resetSkill != null) state.kbSkill = robot.resetSkill;
    state.stats = freshStats();
    if (!SCENES[key].some((s) => s.key === state.sceneKey)) state.sceneKey = SCENES[key][0].key;
    buildSkillButtons();
    await buildScene(state.sceneKey);
  } catch (err) {
    console.error(err);
    setStatus("Error loading robot: " + err.message + " (see console)");
  } finally {
    state.loading = false;
    state.last = performance.now();
  }
}

async function buildScene(key) {
  const { mujoco, robot } = state;
  const scene = getScene(state.robotKey, key);
  state.loading = true;
  const oldModel = state.model, oldData = state.data;
  try {
    const franka = robot.kind === "franka";
    if (franka) { state.follow = false; els.followCam.checked = false; }   // fixed view, always
    const xml = franka ? state.robotXml
                       : injectScene(state.robotXml, scene, robot.floorFriction, robot.wallColor);
    mujoco.FS.writeFile("/working/scene.xml", xml);
    const model = mujoco.MjModel.loadFromXML("/working/scene.xml");
    const data = new mujoco.MjData(model);
    const env = franka ? new MujocoFrankaSingleEnv(mujoco, model, data, robot)
                       : new MujocoSingleEnv(mujoco, model, data, robot, scene);
    state.sceneKey = key; state.scene = scene;
    state.model = model; state.data = data; state.env = env;
    if (!state.renderer) state.renderer = new MujocoRenderer(els.canvas, model, mujoco);
    state.renderer.setModel(model, robot.meshStyle || "metal",
                            scene.floorPalette || robot.floorPalette || DEFAULT_FLOOR_PALETTE);
    state.renderer.frameScene(scene, CAMERA_FOCUS_Z, 0, panelInset());
    state.renderer.clearTrail();
    seedCubeTrail();
    state.ctr = 0; state.pendingReset = null;
    state.stats = freshStats();
    const slot = resetSlot();                 // the scene's startKey, else the robot's resetSkill
    if (slot != null) state.kbSkill = slot;
    refreshSkillButtons();
    els.hudResult.textContent = ""; els.hudResult.className = "result";
    state.renderer.update(data);
    // The trail row drives both trails: the robot's in the mazes, the cube's in the push scenes.
    const trailLabel = els.trailRow.querySelector("span");
    if (scene.mazeMap) trailLabel.textContent = `Trail: a marker every ${TRAIL_SECONDS} s of sim time`;
    else if (scene.cubeSize != null) trailLabel.textContent = `Trail: a red marker on the cube's path every ${CUBE_TRAIL_SECONDS} s of sim time`;
    els.trailRow.style.display = scene.mazeMap || scene.cubeSize != null ? "" : "none";
    setTrails(franka, robot);
    els.followRow.style.display = franka ? "none" : "";     // the arm never moves: fixed view
    els.fallRow.style.display = franka ? "none" : "";        // a bolted-down arm cannot fall
    els.hudStats.style.display = franka ? "none" : "";       // the Franka keeps no Success / Fail tally
    els.hudScene.textContent = `${robot.title} · ${scene.title}`;
    setStatus("Ready. Space for Play/Pause.");
    if (oldData && oldData !== data && oldData.delete) { try { oldData.delete(); } catch (e) {} }
    if (oldModel && oldModel !== model && oldModel.delete) { try { oldModel.delete(); } catch (e) {} }
  } catch (err) {
    console.error(err);
    setStatus("Error building scene: " + err.message + " (see console)");
  } finally {
    state.loading = false;
    state.last = performance.now();
  }
}

// Franka: the puck and gripper trails of /demo/ (play_hierarchical.py --ee_trail).
// The legged robots use the maze marker trail instead, so they get none.
function setTrails(franka, robot) {
  const T = robot.trail;
  if (!franka || !T) { state.renderer.setTrails([]); return; }
  state.renderer.setTrails(
    [{ color: T.puckColor, radius: T.puckRadius }, { color: T.eeColor, radius: T.eeRadius }],
    { seconds: T.seconds, hz: T.hz, controlDt: state.controlDt, fade: T.fade, buckets: T.buckets });
  state.puckResets = 0; state.trailReset = true;
}

// The t = 0 marker of the cube trail, at the cube's spawn (or wherever it is when the
// trail is switched back on). Every clearTrail() on a cube scene is followed by this.
function seedCubeTrail() {
  if (!state.trail || !state.renderer || !state.env) return;
  if (!state.scene || state.scene.cubeSize == null || !state.env.cubePos) return;
  const cp = state.env.cubePos();
  if (cp) state.renderer.addCubeTrailPoint(cp);
}

function controlStep(now) {
  const { env, policy } = state;
  if (state.pendingReset != null) {
    if (now >= state.pendingReset) {
      env.reset(state.robot.kind === "franka");
      state.renderer.clearTrail();
      seedCubeTrail();
      state.trailReset = true;
      state.ctr = 0; state.pendingReset = null;
      els.hudResult.textContent = ""; els.hudResult.className = "result";
    }
    return;
  }
  const obs = env.buildObs();
  if (state.ctr <= 0) { state.skill = skillOfKey(state.kbSkill); state.ctr = policy.skillDuration; }
  state.ctr--;
  env.applyAction(policy.act(obs, state.skill, true));   // always the mean action
  env.step();
  // Trail (mazes only): one marker every TRAIL_SECONDS of simulated time (controlDt is
  // 1/60 s on the ant, 1/30 s on the G1).
  if (state.trail && state.scene.mazeMap && env.stepCount % Math.round(TRAIL_SECONDS / state.controlDt) === 0) {
    state.renderer.addTrailPoint(env.rootPos());
  }
  // Cube trail (push scenes): the same, every CUBE_TRAIL_SECONDS, at the cube instead.
  if (state.trail && state.scene.cubeSize != null && env.cubePos
      && env.stepCount % Math.round(CUBE_TRAIL_SECONDS / state.controlDt) === 0) {
    const cp = env.cubePos();
    if (cp) state.renderer.addCubeTrailPoint(cp);
  }
  if (state.robot.kind === "franka") {
    // A re-spawned puck teleports, so its trail starts over (as a round reset does in /demo/).
    const clear = state.trailReset || env.puckResets !== state.puckResets;
    state.puckResets = env.puckResets; state.trailReset = false;
    state.renderer.updateTrails([env.cubePos(), env._bodyPos(env.handId)], clear);
  }
  if (env.reachedGoal()) endEpisode("goal", now);
  else if (state.resetOnFall && env.fallen()) endEpisode("fall", now);
  else if (state.scene.timeLimit != null && elapsed() >= state.scene.timeLimit) endEpisode("timeout", now);
}

// Simulated seconds since the episode started.
const elapsed = () => (state.env ? state.env.stepCount * state.controlDt : 0);
const freshStats = () => ({ success: 0, fail: 0, total: 0, successTime: 0 });

// kind: "goal" (a Success), "fall" or "timeout" (both a Fail).
function endEpisode(kind, now) {
  const s = state.stats; s.total++;
  const label = { goal: "GOAL!", fall: "Fell", timeout: "Time out" }[kind];
  if (kind === "goal") { s.success++; s.successTime += elapsed(); els.hudResult.className = "result win"; }
  else { s.fail++; els.hudResult.className = "result lose"; }
  els.hudResult.textContent = label;
  els.flashText.textContent = label.toUpperCase();
  els.flash.classList.toggle("goal", kind === "goal");
  els.flash.classList.remove("on"); void els.flash.offsetWidth;   // restart the animation
  els.flash.classList.add("on");
  state.pendingReset = now + 1500;
}

// One animation frame (called by main.js while this mode is active).
function frame(now) {
  const dt = Math.min((now - state.last) / 1000, 0.1);
  state.last = now;
  if (state.running && !state.loading && state.env) {
    state.acc += dt;
    let n = 0;
    while (state.acc >= state.controlDt && n < 8) { controlStep(now); state.acc -= state.controlDt; n++; }
    if (state.acc > state.controlDt) state.acc = 0;   // dropped frames: do not run ahead
    state.renderer.update(state.data);
  }
  if (state.renderer && state.env) {
    if (state.follow) state.renderer.follow(state.env.rootPos(), CAMERA_FOCUS_Z);
    state.renderer.render();
  }
  updateHud();
}

// ---------------- UI ----------------
// Shared panel / HUD elements (main.js owns their layout) plus this mode's own rows and
// its own canvas.
function cacheEls() {
  for (const id of ["status", "play", "followCam", "resetOnFall", "trail", "trailRow",
                    "runSelect", "runRow",
                    "skillBtns", "skillLabel", "fallRow", "followRow",
                    "hudScene", "hudSkill", "hudCtr", "hudTime",
                    "hudGoalLine", "hudGoal", "hudResult", "hudStats",
                    "flash", "flashText"]) els[id] = $(id);
  els.canvas = $("canvasPlay");
}

const nSkills = () => (state.policy ? state.policy.numSkills : state.robot.numSkills);
// Number key slot (0-based) -> policy skill index, via robots.js `keySkills`; identity
// when the robot defines none. `state.kbSkill` is always the key slot.
const skillOfKey = (k) => {
  const m = state.robot && state.robot.keySkills;
  return m && m[k] != null ? m[k] : k;
};
// The key that drives each button slot: `skillKeys` from robots.js when the robot names
// its own (the Franka uses letters), else the number keys 1-N.
const keyLabels = () => {
  const k = state.robot && state.robot.skillKeys;
  return k && k.length >= nSkills() ? k.slice(0, nSkills())
                                    : Array.from({ length: nSkills() }, (_, i) => `${i + 1}`);
};
const keyLabel = (i) => keyLabels()[i] || `${i + 1}`;
// The button a fresh scene and Reset start on: the scene's own `startKey` (the push and
// maze scenes open on the robot's forward gait), else the robot's `resetSkill` (the Franka
// starts on Q); null leaves whatever was picked.
const resetSlot = () => {
  const sk = state.scene && state.scene.startKey;     // 1-based number key -> 0-based slot
  if (sk != null && sk - 1 < nSkills()) return sk - 1;
  const rs = state.robot && state.robot.resetSkill;
  return rs != null && rs < nSkills() ? rs : null;
};
// "keys 1-5" for digits, "keys Q E S Z C" for letters.
const keyHint = () => {
  const ls = keyLabels();
  return state.robot && state.robot.skillKeys ? `keys ${ls.map((s) => s.toUpperCase()).join(" ")}`
                                              : `keys 1–${ls.length}`;
};

// Runs a robot can load: robots.js `runs`, or the single committed export.
function runsOf(robot) {
  const runs = robot.runs && robot.runs.length ? robot.runs
    : [{ key: "default", label: "Committed export", checkpoint: robot.checkpoint || "policy.json" }];
  return runs.map((r) => ({ ...r, policy: robot.runs ? `exports/${robot.key}/ckpts/${r.key}/policy.json` : robot.policy }));
}
function runFor(robot, key) {
  const runs = runsOf(robot);
  return runs.find((r) => r.key === key) || runs[0];
}
function populateRunSelect(robot) {
  const runs = runsOf(robot);
  els.runSelect.innerHTML = "";
  for (const r of runs) {
    const o = document.createElement("option");
    o.value = r.key; o.textContent = r.label;
    if (r.key === state.runKey) o.selected = true;
    els.runSelect.appendChild(o);
  }
  els.runRow.style.display = runs.length > 1 ? "" : "none";
}
// The player feeds the policy exactly robot.obsDim proprio dims and reads numActions back.
function checkPolicyFits(policy, robot) {
  if (policy.obsDim !== robot.obsDim) throw new Error(`policy expects ${policy.obsDim}-dim obs, ${robot.title} provides ${robot.obsDim}`);
  if (policy.numActions !== robot.numActions) throw new Error(`policy has ${policy.numActions} actions, ${robot.title} has ${robot.numActions}`);
}
function applyPolicy(policy) {
  state.policy = policy;
  state.kbSkill = Math.min(state.kbSkill, policy.numSkills - 1);
  buildSkillButtons();
  resetEpisode();
}

// Swap only the policy (the model, meshes and scene stay).
async function loadRun(runKey) {
  const robot = state.robot;
  const run = runFor(robot, runKey);
  state.loading = true;
  setStatus(`Loading ${run.label}…`);
  try {
    const policy = await loadPolicy(run.policy, 12345);
    checkPolicyFits(policy, robot);
    state.runKey = run.key;
    els.runSelect.value = run.key;
    applyPolicy(policy);
    setStatus(`Loaded ${run.label}.`);
  } catch (err) {
    console.error(err);
    els.runSelect.value = state.runKey;
    setStatus("Error loading run: " + err.message);
  } finally {
    state.loading = false;
    state.last = performance.now();
  }
}

// One row per number key: the 1-N key as a badge, then the description of the skill
// that key drives (robots.js `keySkills` picks the skill, `skillNames` describes it).
function buildSkillButtons() {
  els.skillBtns.innerHTML = "";
  const names = state.robot.skillNames || [];
  for (let i = 0; i < nSkills(); i++) {
    const s = skillOfKey(i);
    const b = document.createElement("button");
    b.className = "skillbtn";
    const n = document.createElement("span"); n.className = "n";
    n.textContent = keyLabel(i).toUpperCase();
    const d = document.createElement("span"); d.className = "d"; d.textContent = names[s] || `Skill ${s}`;
    b.append(n, d);
    b.addEventListener("click", () => pickSkill(i));
    els.skillBtns.appendChild(b);
  }
  els.skillLabel.textContent = `Skill (${keyHint()})`;
  refreshSkillButtons();
}
function pickSkill(i) { state.kbSkill = i; state.ctr = 0; refreshSkillButtons(); }
function refreshSkillButtons() {
  [...els.skillBtns.children].forEach((b, i) => b.classList.toggle("active", i === state.kbSkill));
}

function resetEpisode() {
  if (!state.env) return;
  state.env.reset(state.robot.kind === "franka");
  if (state.renderer) state.renderer.clearTrail();
  seedCubeTrail();
  state.trailReset = true;
  state.ctr = 0; state.pendingReset = null;
  const rs = resetSlot();
  if (rs != null) { state.kbSkill = rs; refreshSkillButtons(); }
  els.hudResult.textContent = ""; els.hudResult.className = "result";
}

// This mode's own controls (Follow camera, Reset on fall, Trail). The shared ones --
// Play, Reset, the dropdowns and the keyboard -- are wired by main.js.
function wireControls() {
  els.followCam.addEventListener("change", (e) => {
    state.follow = e.target.checked;
    if (!state.follow && state.renderer) state.renderer.frameScene(state.scene, CAMERA_FOCUS_Z, 0, panelInset());
  });
  els.resetOnFall.addEventListener("change", (e) => { state.resetOnFall = e.target.checked; });
  els.trail.addEventListener("change", (e) => {
    state.trail = e.target.checked;
    if (!state.trail && state.renderer) state.renderer.clearTrail();
    else seedCubeTrail();
  });
}

let _lastHud = 0;
function updateHud() {
  const now = performance.now();
  if (now - _lastHud < 80 || !state.env) return;
  _lastHud = now;
  els.hudSkill.textContent = keyLabel(state.kbSkill).toUpperCase();
  els.hudCtr.textContent = `${Math.max(0, state.ctr)}`;
  const limit = state.scene && state.scene.timeLimit;
  els.hudTime.textContent = `${elapsed().toFixed(1)}s` + (limit != null ? ` / ${limit}s` : "");
  const d = state.env.goalDistance();
  els.hudGoalLine.style.display = d == null ? "none" : "";
  if (d != null) els.hudGoal.textContent = Math.max(0, d).toFixed(2);   // a crossed goal line reads 0, not negative
  const s = state.stats;
  const avg = s.success ? `${(s.successTime / s.success).toFixed(1)}s` : "–";
  els.hudStats.textContent = `Success ${s.success} · Fail ${s.fail} · Total ${s.total} · Avg success time ${avg}`;
}

// ---------------- the interface main.js drives ----------------
export const PlayMode = {
  get robotKey() { return state.robotKey; },
  get sceneKey() { return state.sceneKey; },
  get running() { return state.running; },
  get numSkills() { return state.env ? nSkills() : 0; },
  get skillKeys() { return state.env && state.robot.skillKeys ? keyLabels() : null; },

  // Once, after the MuJoCo module is up (before any activate).
  init({ mujoco }) {
    state.mujoco = mujoco;
    cacheEls();
    wireControls();
  },

  // Show a robot in a scene (a fresh model + policy + scene). The previous mode has been
  // deactivated and this mode's rows / canvas are being shown by main.js.
  async activate({ robotKey, sceneKey }) {
    if (sceneKey) state.sceneKey = sceneKey;     // loadRobot falls back to the robot's first scene
    await loadRobot(robotKey);
    state.running = true;
    els.play.textContent = "Pause";
    state.last = performance.now();
  },

  // Leaving for Game Play: stop stepping and free the MuJoCo model / data (the renderer
  // keeps its last meshes; the next activate rebuilds them).
  deactivate() {
    state.running = false;
    const { model, data } = state;
    state.env = null; state.model = null; state.data = null;
    if (data && data.delete) { try { data.delete(); } catch (e) {} }
    if (model && model.delete) { try { model.delete(); } catch (e) {} }
  },

  // Same mode: another robot (its scene picked by main.js) or another scene of this robot.
  async loadRobot(robotKey, sceneKey) {
    if (sceneKey) state.sceneKey = sceneKey;
    await loadRobot(robotKey);
  },
  buildScene,
  loadRun,
  frame,
  togglePlay() {
    state.running = !state.running;
    els.play.textContent = state.running ? "Pause" : "Play";
    state.last = performance.now();
  },
  reset: resetEpisode,
  pickSkill(i) { if (state.env) pickSkill(i); },
};
