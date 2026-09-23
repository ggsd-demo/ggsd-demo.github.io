// Copyright (c) 2026. MuJoCo player for game-discovery agents.
// SPDX-License-Identifier: BSD-3-Clause
//
// Game Play mode of the web player: the two-player games. Runs the SAME hierarchical
// skill loop as local/run_local.py: every skill_duration control steps the high-level
// net picks "me"'s skill (or the keyboard in playable mode); "foe" runs the same net on
// its mirrored observation with an independent skill counter. Physics is MuJoCo WASM;
// rendering is three.js on its own canvas (#canvasGame).
//
// ../main.js owns the page: the MuJoCo module, the Robot / Task / Run dropdowns, the
// Play / Reset buttons, the keyboard and the animation loop. It drives this mode through
// the GameMode object at the bottom (activate / deactivate / frame / ...); everything
// else in here is the game loop and its own panel rows (Max seconds, Auto / Playable).

import { GAMES, TRANSFER, TRAIN, G1, G1_TRAIN, FRANKA } from "./registry.js";
import { MujocoRenderer } from "./render.js";
import { HierarchicalPolicy } from "./policy.js";
import * as antSumo from "./games/ant_sumo.js";
import * as g1Boxing from "./games/g1_boxing.js";
import * as frankaHockey from "./games/franka_hockey.js";

// Per-game module + the training-env defaults the player boots into.
// simDt: physics step of the game's MJCF; focusZ: camera look-at height for frameCamera.
const GAME_MODULES = {
  ant_sumo: { mod: antSumo, train: TRAIN, simDt: TRANSFER.simDt, focusZ: 0.4, meshStyle: "metal",
              arena: "line", topZ: 0.8 },
  // the boxers get the Isaac env's boxing-ring boundary (boundary_style="ring")
  g1_boxing: { mod: g1Boxing, train: G1_TRAIN, simDt: G1.simDt, focusZ: 0.9, meshStyle: "isaac",
               arena: "ring", topZ: 1.6 },
  // no training-env override: the page always runs the clips' *-Play env (RULES)
  franka_hockey: { mod: frankaHockey, train: {}, simDt: FRANKA.simDt, focusZ: 0.75, meshStyle: "plastic" },
};
// franka_hockey always runs the *-Play env the clips were recorded with (15 s match,
// an idle puck only resets the round), never the training env.
const RULES = "play";
// Every game runs the same match length here, whatever its training env used; the
// "Max seconds" box overrides it. Steps = seconds / control dt of the active game.
const EPISODE_SEC = 30;
const isFranka = (key) => key === "franka_hockey";
const isG1 = (key) => key === "g1_boxing";

const els = {};
function $(id) { return document.getElementById(id); }
// Width of the canvas the control panel covers, so the camera can frame the arena in what
// is left of it rather than behind the panel.
function panelInset() {
  const panel = $("panel");
  return panel ? panel.getBoundingClientRect().right + 12 : 0;
}
async function fetchText(url) { const r = await fetch(url); if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`); return r.text(); }

const state = {
  mujoco: null, env: null, model: null, data: null, policy: null, foePolicy: null,
  renderer: null, running: false,
  // "me"'s driver: Auto (the trained high level picks its skill) or Playable (you do).
  playable: false,
  // no UI switch: the foe only stands still when the high-level net cannot run
  foeNoop: false,
  gameKey: "ant_sumo", gameMod: null,   // active game
  runKey: null,              // selected committed run (GAMES[key].runs)
  obsDim: TRANSFER.obsDim, numActions: TRANSFER.numActions, controlDt: 1 / 60,
  loading: false,
  exportData: null,          // current {meta, buffer} of the committed run
  llSlice: { start: 0, end: 0 },  // low-level obs slice the active checkpoint reports
  hlUsable: true,            // high-level net matches this env's obs -> can pick skills
  skillDuration: TRANSFER.maxEpisodeSteps, numSkills: 5,
  me: { skill: 0, ctr: 0 }, foe: { skill: 0, ctr: 0 },
  kbSkill: 0,
  stats: { win: 0, lose: 0, draw: 0, ep: 0 },
  acc: 0, last: 0,
};

function setStatus(msg) { if (els.status) els.status.textContent = msg; }

// (Re)load a game: build its MuJoCo model + env, default to its training env,
// load its policy, and (re)build the renderer geoms. Safe to call on game switch.
async function loadGame(key) {
  const entry = GAME_MODULES[key];
  if (!entry) { setStatus(`Game '${key}' not implemented.`); return; }
  state.loading = true;
  const wasRunning = state.running;
  state.running = false;
  // G1 always renders its visual meshes (the collision-primitive build stays in the repo
  // for local debugging, but nobody wants the capsule look in the browser).
  const useMesh = key === "g1_boxing";
  setStatus(`Loading ${GAMES[key].title} model + policy…${useMesh ? " (meshes ~8 MB)" : ""}`);
  const oldModel = state.model, oldData = state.data;
  try {
    // Build the NEW env first; only free the old model/data after it succeeds, so a
    // failed load leaves the current game intact (and never reads a deleted object).
    const { model, data, env } = await entry.mod.makeEnv(state.mujoco, fetchText, { mesh: useMesh, rules: RULES });
    Object.assign(env.cfg, entry.train);   // default to the real training env
    // ...except the spawn: the player always starts from the fixed face-off.
    if (!isFranka(key)) {
      env.cfg.randomizeSpawnPositions = false;
      env.cfg.randomizeSpawnYaw = false;
      env.cfg.noFoe = false;
    }
    state.gameKey = key; state.gameMod = entry.mod;
    state.model = model; state.data = data; state.env = env;
    state.obsDim = env.cfg.obsDim;
    state.numActions = env.cfg.numActions;
    state.controlDt = env.cfg.decimation * entry.simDt;

    const run = runsOf(key).find((r) => r.key === state.runKey) || runsOf(key)[0] || null;
    state.runKey = run ? run.key : null;
    const exp = await entry.mod.makeExport(state.runKey);
    state.stats = { win: 0, lose: 0, draw: 0, ep: 0 };
    populateRunSelect(key);
    applyExport(exp);

    if (!state.renderer) { state.renderer = new MujocoRenderer(els.canvas, model, state.mujoco); state.renderer.setModel(model, entry.meshStyle); }
    else state.renderer.setModel(model, entry.meshStyle);
    frameGame();
    syncSettingsUI();

    env.reset(true);
    state.me.ctr = 0; state.foe.ctr = 0;
    state.renderer.update(data);
    els.hudResult.textContent = ""; els.hudResult.className = "result";
    setStatus("Ready. Space for Play/Pause.");

    // new game is live -> free the previous model/data (if a different object)
    if (oldData && oldData !== data && oldData.delete) { try { oldData.delete(); } catch (e) {} }
    if (oldModel && oldModel !== model && oldModel.delete) { try { oldModel.delete(); } catch (e) {} }
  } catch (err) {
    console.error(err);
    setStatus("Error loading game: " + err.message + " (see console)");
  } finally {
    state.loading = false;
    state.running = wasRunning || true;
    state.last = performance.now();
  }
}

// Camera, arena line and trails for the active game, following play_hierarchical.py's
// --ee_trail: Franka hockey gets no arena line, Isaac's ViewerCfg camera and the puck +
// gripper trails (yellow puck, blue me gripper, red foe); G1 boxing gets one trail per
// glove, so both fists of both boxers leave the punch arc the Isaac clips show.
function frameGame() {
  const r = state.renderer, c = state.env.cfg, key = state.gameKey;
  if (isFranka(key)) {
    r.setArena(null);
    r.setView(FRANKA.cameraEye, FRANKA.cameraLookat, { fov: FRANKA.cameraFov, shadowExtent: 3 });
    const T = FRANKA.trail;
    r.setTrails([
      { color: T.puckColor, radius: T.puckRadius },
      { color: T.meColor, radius: T.eeRadius },
      { color: T.foeColor, radius: T.eeRadius },
    ], { seconds: T.seconds, hz: T.hz, controlDt: state.controlDt, fade: T.fade, buckets: T.buckets });
    return;
  }
  const entry = GAME_MODULES[key];
  r.setArena(c.boundaryMax[0], entry.arena || "line");
  r.frameCamera(c.boundaryMax[0], entry.focusZ, { leftInset: panelInset(), topZ: entry.topZ });
  if (isG1(key)) {
    const T = G1.trail;
    r.setTrails(gloveTrailOrder().map(([side]) => ({
      color: side === "me" ? T.meColor : T.foeColor, radius: T.gloveRadius,
    })), { seconds: T.seconds, hz: T.hz, controlDt: state.controlDt, fade: T.fade, buckets: T.buckets });
  } else {
    r.setTrails([]);
  }
}
// The traced G1 bodies, as [side, bodyId] pairs: every glove of both boxers, in a fixed
// order so the trail specs and the pushed points line up.
function gloveTrailOrder() {
  const e = state.env;
  return ["me", "foe"].flatMap((side) => (e.gloveIds?.[side] || []).map((id) => [side, id]));
}
function pushTrails(reset) {
  const e = state.env;
  if (isFranka(state.gameKey)) {
    state.renderer.updateTrails([e._puckPos(), e._bodyPos(e.handId.me), e._bodyPos(e.handId.foe)], reset);
  } else if (isG1(state.gameKey)) {
    state.renderer.updateTrails(gloveTrailOrder().map(([, id]) => e._bodyPos(id)), reset);
  }
}

// ------------- checkpoint / hierarchical-slice handling -------------

// (Re)build both policies from state.exportData. The low-level obs slice comes
// from the checkpoint itself; the one-hot skill size is derived from it:
// num_skills = ll_input - slice_len.
function applyExport(exp) {
  state.exportData = exp;
  const base = exp.meta;
  const llIn = base.low_level_input_dim;
  // A checkpoint with explicit low-level input indices (newer ant runs: the me block minus
  // roll) uses them as is; the slice inputs only apply to contiguous-slice checkpoints.
  const idx = Array.isArray(base.low_level_obs_indices) && base.low_level_obs_indices.length ? base.low_level_obs_indices : null;

  let start, end, proprioDim;
  if (idx) {
    start = idx[0]; end = idx[idx.length - 1] + 1; proprioDim = idx.length;
    if (idx.some((i) => i < 0 || i >= state.obsDim)) {
      setStatus(`Checkpoint's low-level obs indices exceed this env's ${state.obsDim}-dim obs.`);
      return;
    }
  } else {
    start = Math.round(base.proprio_start);
    end = start + Math.round(base.proprio_dim);
    if (!Number.isFinite(start)) start = 0;
    if (!Number.isFinite(end)) end = start + 1;
    start = Math.min(Math.max(start, 0), state.obsDim - 1);
    end = Math.min(Math.max(end, start + 1), Math.min(state.obsDim, start + llIn));
    proprioDim = end - start;
  }
  state.llSlice = { start, end };

  const numSkills = llIn - proprioDim;
  if (numSkills < 1) {
    setStatus(`Slice too wide: low-level input is ${llIn}, so slice length must be < ${llIn}.`);
    return;
  }

  const meta = { ...base, proprio_start: start, proprio_dim: proprioDim, num_skills: numSkills, low_level_obs_indices: idx };
  state.policy = new HierarchicalPolicy(meta, exp.buffer, 12345);
  state.foePolicy = new HierarchicalPolicy(meta, exp.buffer, 99999); // independent sampling
  state.skillDuration = meta.skill_duration;
  state.numSkills = numSkills;
  state.kbSkill = Math.min(state.kbSkill, numSkills - 1);
  const rs0 = GAMES[state.gameKey] && GAMES[state.gameKey].resetSkill;
  if (rs0 != null && rs0 < numSkills) state.kbSkill = rs0;
  state.me.ctr = 0; state.foe.ctr = 0;

  // High-level is usable only if it exists, eats exactly this env's obs, and
  // its skill count matches the derived one-hot size.
  const hlLayers = base.num_high_level_layers ?? (base.tensors.some((t) => t.name === "hl_w0") ? 1 : 0);
  let reason = null;
  if (!hlLayers) reason = "checkpoint has no high-level net";
  else if (base.obs_dim !== state.obsDim) reason = `high-level expects ${base.obs_dim}-dim obs, env provides ${state.obsDim}`;
  else if (base.num_skills !== numSkills) reason = `high-level outputs ${base.num_skills} skills but slice implies ${numSkills}`;
  setHlUsable(!reason, reason);

  buildSkillButtons();
  if (base.has_obs_normalizer) {
    setStatus("Warning: checkpoint has an obs normalizer, which the player ignores — actions will be off.");
  }
}

// When the high-level net can't run: you drive "me"'s skills (playable) and the
// foe just stands (no-op torques).
function setHlUsable(ok, reason) {
  state.hlUsable = ok;
  if (!ok) {
    state.playable = true;
    state.foeNoop = true;
    els.modePlayable.checked = true; els.modeAuto.disabled = true;
    setStatus(`High-level disabled (${reason}). You pick skills; foe stands.`);
  } else {
    els.modeAuto.disabled = false;
    state.playable = els.modePlayable.checked;
    state.foeNoop = false;
  }
  refreshSkillButtons();
}

function controlStep() {
  const { env, policy, foePolicy } = state;
  // ---- me ----
  const obsMe = env.buildObs("me", "foe");
  if (state.me.ctr <= 0) {
    if (state.playable || !state.hlUsable) state.me.skill = skillOfKey(state.kbSkill);
    else state.me.skill = policy.selectSkill(obsMe, true);
    state.me.ctr = state.skillDuration;
  }
  state.me.ctr--;
  // Auto: the policy as trained -- sampled actions. Playable: you picked the skill, so the
  // low level runs its mean, which reads as a clean, repeatable version of that skill.
  const meA = policy.act(obsMe, state.me.skill, state.playable);

  // ---- foe ----
  let foeA;
  if (env.cfg.noFoe || state.foeNoop || !state.hlUsable) {
    foeA = new Float64Array(state.numActions);
  } else {
    const obsFoe = env.buildObs("foe", "me");
    // the foe is always the trained policy, sampled at both levels
    if (state.foe.ctr <= 0) { state.foe.skill = foePolicy.selectSkill(obsFoe, true); state.foe.ctr = state.skillDuration; }
    state.foe.ctr--;
    foeA = foePolicy.act(obsFoe, state.foe.skill, false);
  }

  env.applyActions({ me: meA, foe: foeA });
  env.step();

  const { terminated, timeout, winner, reason } = env.checkDone();
  if (terminated || timeout) endEpisode(winner, reason);
  else {
    // A round reset inside a match (franka: non-final goal / idle puck) restarts the
    // trails and the skill holds, like the Isaac play script's per-reset hooks.
    const roundReset = !!env.roundResetFlag;
    if (roundReset) { state.me.ctr = 0; state.foe.ctr = 0; if (env.lastEvent) { els.hudResult.textContent = env.lastEvent; els.hudResult.className = "result draw"; } }
    pushTrails(roundReset);
  }
}

function endEpisode(winner, reason) {
  const s = state.stats; s.ep++;
  if (winner === 1) s.win++; else if (winner === -1) s.lose++; else s.draw++;
  state.env.reset(true);
  state.me.ctr = 0; state.foe.ctr = 0;
  pushTrails(true);
  updateHud(reason, winner);
}

function resetEpisode() {
  if (!state.env) return;
  state.env.reset(true); state.me.ctr = 0; state.foe.ctr = 0; pushTrails(true);
  // Games that name a default button start every match on it (the Franka: Q). Only the
  // Reset button comes through here; a match that ends on its own keeps the pick.
  const rs = GAMES[state.gameKey] && GAMES[state.gameKey].resetSkill;
  if (rs != null && rs < state.numSkills) { state.kbSkill = rs; refreshSkillButtons(); }
}

// One animation frame (called by main.js while this mode is active).
function frame(now) {
  const dt = Math.min((now - state.last) / 1000, 0.1);
  state.last = now;
  if (!state.env) return;
  if (state.running && !state.loading && state.renderer) {
    state.acc += dt;
    let n = 0;
    while (state.acc >= state.controlDt && n < 8) { controlStep(); state.acc -= state.controlDt; n++; }
    state.renderer.update(state.data);
  }
  if (state.renderer) state.renderer.render();
  updateHud();
}

// ---------------- UI ----------------
// Shared panel / HUD elements (main.js owns their layout) plus this mode's own rows and
// its own canvas.
function cacheEls() {
  for (const id of ["status", "play", "modeAuto", "modePlayable",
                    "hudLeft", "hudMax", "hudResult", "hudStats", "skillBtns",
                    "runSelect", "runRow", "envHint", "hudHp", "hudHpLine", "skillLabel",
                    "hudScoreLine", "hudScore"]) els[id] = $(id);
  els.canvas = $("canvasGame");
}

// Set the match length in seconds (the env counts steps) and show it as the clock's
// full time. Called on every game load; EPISODE_SEC is the only length the page offers.
function setEpisodeSeconds(sec) {
  const s = Math.min(300, Math.max(1, Math.round(sec || 0))) || EPISODE_SEC;
  state.env.cfg.maxEpisodeSteps = Math.max(1, Math.round(s / state.controlDt));
  els.hudMax.textContent = `${s}`;
  return s;
}

// Push current env cfg into the settings inputs (called after every game load).
function syncSettingsUI() {
  const c = state.env.cfg, franka = isFranka(state.gameKey);
  const sec = setEpisodeSeconds(EPISODE_SEC);
  els.hudScoreLine.style.display = franka ? "" : "none";
  if (franka) {
    els.envHint.textContent =
      `Play env (as in the clips): 1.8 x 1.0 m table, first to ${c.scoreToWin}, ${sec} s match, an idle puck resets the round.`;
    return;
  }
  els.envHint.textContent = `Training env: arena ±${c.boundaryMax[0]}, ${sec} s match, fixed face-off spawn.`;
}

// Number key slot (0-based) -> policy skill index, via registry.js `keySkills`; identity
// when the game defines none. `state.kbSkill` is always the key slot, while
// `state.me.skill` uses the policy's own numbering.
function skillOfKey(k) {
  const m = GAMES[state.gameKey] && GAMES[state.gameKey].keySkills;
  return m && m[k] != null ? m[k] : k;
}
// The key that drives each button slot: `skillKeys` from registry.js when the game names
// its own (the Franka uses letters), else the number keys 1-N.
function keyLabels() {
  const n = state.numSkills;
  const k = GAMES[state.gameKey] && GAMES[state.gameKey].skillKeys;
  return k && k.length >= n ? k.slice(0, n) : Array.from({ length: n }, (_, i) => `${i + 1}`);
}
const keyLabel = (i) => keyLabels()[i] || `${i + 1}`;
// "keys 1-5" for digits, "keys Q E S Z C" for letters.
function keyHint() {
  const ls = keyLabels();
  const own = GAMES[state.gameKey] && GAMES[state.gameKey].skillKeys;
  return own ? `keys ${ls.map((s) => s.toUpperCase()).join(" ")}` : `keys 1–${ls.length}`;
}
function keyOfSkill(s) {
  const m = GAMES[state.gameKey] && GAMES[state.gameKey].keySkills;
  if (!m) return s;
  const k = m.indexOf(s);
  return k >= 0 ? k : s;
}

// One row per number key: the 1-N key as a badge, then the description of the skill
// that key drives (registry.js `keySkills` picks the skill, `skillNames` describes it).
function buildSkillButtons() {
  els.skillBtns.innerHTML = "";
  const names = (GAMES[state.gameKey] && GAMES[state.gameKey].skillNames) || [];
  for (let i = 0; i < state.numSkills; i++) {
    const s = skillOfKey(i);
    const b = document.createElement("button");
    b.className = "skillbtn"; b.dataset.skill = s;
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
// A skill button or number key hands "me" to the keyboard: Playable switches on by
// itself, so the press always does something visible.
function pickSkill(i) {
  state.kbSkill = i;
  if (!state.playable) {
    state.playable = true;
    els.modePlayable.checked = true;
    setStatus(`Playable on: you pick "me"'s skill (${keyHint()}).`);
  }
  state.me.ctr = 0;
  refreshSkillButtons();
}
function refreshSkillButtons() {
  const picked = state.playable ? state.kbSkill : -1;
  const running = state.playable ? -1 : keyOfSkill(state.me.skill);
  [...els.skillBtns.children].forEach((b, i) => {
    b.classList.toggle("active", i === picked);
    b.classList.toggle("auto", i === running);
  });
}

function runsOf(gameKey) { return (GAMES[gameKey] && GAMES[gameKey].runs) || []; }
function populateRunSelect(gameKey) {
  const runs = runsOf(gameKey);
  els.runSelect.innerHTML = "";
  for (const r of runs) {
    const o = document.createElement("option");
    o.value = r.key; o.textContent = r.label;
    if (r.key === state.runKey) o.selected = true;
    els.runSelect.appendChild(o);
  }
  els.runRow.style.display = runs.length > 1 ? "" : "none";
}
// Swap only the policy export (model and env stay).
async function loadRun(runKey) {
  const run = runsOf(state.gameKey).find((r) => r.key === runKey);
  if (!run) return;
  state.loading = true;
  setStatus(`Loading ${run.label}…`);
  try {
    const exp = await state.gameMod.makeExport(run.key);
    state.runKey = run.key;
    state.stats = { win: 0, lose: 0, draw: 0, ep: 0 };
    applyExport(exp);
    state.env.reset(true);
    if (state.hlUsable) setStatus(`Loaded ${run.label}.`);
  } catch (err) {
    console.error(err);
    els.runSelect.value = state.runKey;
    setStatus("Error loading run: " + err.message);
  } finally {
    state.loading = false;
    state.last = performance.now();
  }
}

// This mode's own controls (Auto / Playable, Max seconds). The shared ones -- Play,
// Reset, the dropdowns and the keyboard -- are wired by main.js.
function wireControls() {
  for (const el of [els.modeAuto, els.modePlayable]) {
    el.addEventListener("change", () => {
      state.playable = els.modePlayable.checked;
      state.me.ctr = 0;
      refreshSkillButtons();
    });
  }
}

let _lastHud = 0;
function updateHud(reason, winner) {
  if (!state.env) return;
  const now = performance.now();
  if (reason === undefined && now - _lastHud < 80) return;
  _lastHud = now;
  refreshSkillButtons();      // in Auto the high level changes skill on its own
  const left = (state.env.cfg.maxEpisodeSteps - state.env.stepCount) * state.controlDt;
  els.hudLeft.textContent = `${Math.max(0, Math.ceil(left))}`;
  const hp = state.env.hp;
  els.hudHpLine.style.display = hp ? "" : "none";
  if (state.env.score) els.hudScore.textContent = `${state.env.score.me} – ${state.env.score.foe}`;
  if (hp) { const ih = state.env.cfg.initialHp; els.hudHp.textContent = `${(100 * hp.me / ih).toFixed(1)}% / ${(100 * hp.foe / ih).toFixed(1)}%`; }
  const s = state.stats;
  els.hudStats.textContent = `Episodes ${s.ep} · W ${s.win} L ${s.lose} D ${s.draw}`;
  if (winner !== undefined) {
    els.hudResult.textContent = (winner === 1 ? "ME wins!" : winner === -1 ? "FOE wins!" : "Draw") + (reason ? ` (${reason})` : "");
    els.hudResult.className = "result " + (winner === 1 ? "win" : winner === -1 ? "lose" : "draw");
  }
}

// ---------------- the interface main.js drives ----------------
export const GameMode = {
  get gameKey() { return state.gameKey; },
  get running() { return state.running; },
  get numSkills() { return state.env ? state.numSkills : 0; },
  get skillKeys() {
    const own = GAMES[state.gameKey] && GAMES[state.gameKey].skillKeys;
    return state.env && own ? keyLabels() : null;
  },

  // Once, after the MuJoCo module is up (before any activate).
  init({ mujoco }) {
    state.mujoco = mujoco;
    cacheEls();
    wireControls();
  },

  // Show a game (a fresh model + env + policy). The previous mode has been deactivated
  // and this mode's rows / canvas are being shown by main.js.
  async activate({ gameKey }) {
    els.hudStats.style.display = "";                   // the Playground hides it for the Franka
    await loadGame(gameKey);
    if (state.renderer) state.renderer.render();
    state.running = true;
    els.play.textContent = "Pause";
    state.last = performance.now();
  },

  // Leaving for the Playground: stop stepping and free the MuJoCo model / data (the
  // renderer keeps its last meshes; the next activate rebuilds them).
  deactivate() {
    state.running = false;
    const { model, data } = state;
    state.env = null; state.model = null; state.data = null;
    if (data && data.delete) { try { data.delete(); } catch (e) {} }
    if (model && model.delete) { try { model.delete(); } catch (e) {} }
  },

  loadGame,           // same mode, another game (the Robot dropdown while a game is up)
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
