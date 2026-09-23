// Copyright (c) 2026. MuJoCo player for game-discovery agents.
// SPDX-License-Identifier: BSD-3-Clause
//
// Boot + page-level UI of the web player. One page, two modes that share the panel,
// the HUD and the MuJoCo WASM module but keep their own sim / renderer / policy code:
//
//   Playground (js/play/)  one robot, low-level policy, the Isaac play scenes
//   Game Play  (js/game/)  two players, full hierarchical policy, the self-play games
//
// The Task dropdown lists the robot's scenes and, last, "Game Play: <its game>". Picking
// a task of the other mode deactivates the current one (frees its model) and activates
// the other; each mode renders on its own canvas (#canvasPlay / #canvasGame) so neither
// three.js renderer has to be torn down. body[data-mode] shows the active mode's canvas
// and panel / HUD rows (see style.css .only-play / .only-game).
//
// URL: ?robot=ant|g1|franka&scene=<scene key>|game opens straight into one task, and
// ?game=ant_sumo|g1_boxing|franka_hockey into one game (the old /game/ links).

import load_mujoco from "../lib/mujoco_wasm.js";
import { ROBOTS } from "./play/robots.js";
import { SCENES } from "./play/scenes.js";
import { GAMES } from "./game/registry.js";
import { PlayMode } from "./play/mode.js";
import { GameMode } from "./game/mode.js";

// Task dropdown value of a robot's game (the scenes use their own keys).
const GAME_TASK = "game";

const els = {};
const $ = (id) => document.getElementById(id);

const state = {
  mujoco: null,
  active: null,              // PlayMode | GameMode, once its first activate is done
  robotKey: "ant", taskKey: "arena",
  switching: false,          // a task / robot change is in flight: ignore further ones
};

function setStatus(msg) { if (els.status) els.status.textContent = msg; }

// The (implemented) game a robot plays, if any.
function gameOf(robotKey) {
  const hit = Object.entries(GAMES).find(([, g]) => g.implemented && g.robot === robotKey);
  return hit ? hit[0] : null;
}
// The task to open on a robot when the current one is not available there.
function taskFor(robotKey, taskKey) {
  if (taskKey === GAME_TASK) return gameOf(robotKey) ? GAME_TASK : SCENES[robotKey][0].key;
  return SCENES[robotKey].some((s) => s.key === taskKey) ? taskKey : SCENES[robotKey][0].key;
}

function readUrlParams() {
  const q = new URLSearchParams(window.location.search);
  const robot = q.get("robot"), scene = q.get("scene"), game = q.get("game");
  if (game && GAMES[game] && GAMES[game].implemented) { state.robotKey = GAMES[game].robot; state.taskKey = GAME_TASK; return; }
  if (robot && ROBOTS[robot]) state.robotKey = robot;
  if (scene) state.taskKey = taskFor(state.robotKey, scene);
}

async function boot() {
  cacheEls();
  readUrlParams();
  populateRobotSelect();
  populateTaskSelect();
  renderTips();
  setStatus("Loading MuJoCo WASM (~11 MB)…");
  state.mujoco = await load_mujoco();
  try { state.mujoco.FS.mkdir("/working"); } catch (e) { /* exists */ }
  try { state.mujoco.FS.mount(state.mujoco.MEMFS, { root: "." }, "/working"); } catch (e) { /* mounted */ }
  PlayMode.init({ mujoco: state.mujoco });
  GameMode.init({ mujoco: state.mujoco });
  wireControls();
  await selectTask(state.robotKey, state.taskKey);
  requestAnimationFrame(frame);
}

// Open a task on a robot: a scene (Playground) or the robot's game (Game Play). Switches
// mode when needed; inside a mode it hands over to that mode's own loaders, which keep
// the current model up until the new one is built.
async function selectTask(robotKey, taskKey) {
  if (state.switching) return;
  state.switching = true;
  const wantGame = taskKey === GAME_TASK;
  const target = wantGame ? GameMode : PlayMode;
  try {
    if (state.active !== target) {
      if (state.active) state.active.deactivate();
      state.active = null;
      if (wantGame) await GameMode.activate({ gameKey: gameOf(robotKey) });
      else await PlayMode.activate({ robotKey, sceneKey: taskKey });
      state.active = target;
      document.body.dataset.mode = wantGame ? "game" : "play";
      window.dispatchEvent(new Event("resize"));   // the canvas was hidden while it was sized
    } else if (wantGame) {
      const gameKey = gameOf(robotKey);
      if (gameKey !== GameMode.gameKey) await GameMode.loadGame(gameKey);
    } else if (robotKey !== PlayMode.robotKey) {
      await PlayMode.loadRobot(robotKey, taskKey);
    } else if (taskKey !== PlayMode.sceneKey) {
      await PlayMode.buildScene(taskKey);
    }
  } finally {
    state.switching = false;
    syncSelection();
  }
}

// The dropdowns follow what is actually up (a failed load leaves the previous task).
function syncSelection() {
  if (state.active === GameMode) {
    state.robotKey = GAMES[GameMode.gameKey].robot; state.taskKey = GAME_TASK;
  } else if (state.active === PlayMode) {
    state.robotKey = PlayMode.robotKey; state.taskKey = PlayMode.sceneKey;
  }
  els.robotSelect.value = state.robotKey;
  populateTaskSelect();
  renderTips();
}

// ---------------- Tips ----------------
// The panel's Tips row opens a card holding the selected task's own tips: `tips` on the
// scene (play/scenes.js) or on the game (game/registry.js), with `description` as the
// fallback. syncSelection() refreshes it after every robot / task change.
function taskEntry(robotKey, taskKey) {
  if (taskKey === GAME_TASK) { const g = gameOf(robotKey); return g ? GAMES[g] : null; }
  return SCENES[robotKey].find((s) => s.key === taskKey) || null;
}

function renderTips() {
  const entry = taskEntry(state.robotKey, state.taskKey);
  const lines = !entry ? [] : (entry.tips && entry.tips.length ? entry.tips
                                                              : entry.description ? [entry.description] : []);
  els.tipsBox.innerHTML = "";
  const h = document.createElement("h3");
  h.textContent = entry ? entry.title : "Tips";
  els.tipsBox.appendChild(h);
  const ul = document.createElement("ul");
  for (const line of lines) {
    const li = document.createElement("li");
    li.textContent = line;
    ul.appendChild(li);
  }
  els.tipsBox.appendChild(ul);
  if (!els.tipsBox.hidden) placeTips();     // open while the task changed: re-fit it
}

// The card sits to the right of the panel, level with the row; on a narrow window (the
// panel is full width there) it drops below the row instead. Both are clamped to the
// viewport, so it can never open off-screen.
function placeTips() {
  const row = els.tipsRow.getBoundingClientRect(), panel = els.panel.getBoundingClientRect();
  const w = els.tipsBox.offsetWidth, h = els.tipsBox.offsetHeight;
  const margin = 8;
  const beside = panel.right + 12;
  const fits = beside + w <= window.innerWidth - margin;
  const left = fits ? beside : Math.max(margin, Math.min(row.left, window.innerWidth - w - margin));
  const wanted = fits ? row.top - 10 : row.bottom + 8;
  const top = Math.max(margin, Math.min(wanted, window.innerHeight - h - margin));
  els.tipsBox.style.left = `${left}px`;
  els.tipsBox.style.top = `${top}px`;
}

function showTips(on) {
  els.tipsBox.hidden = !on;
  els.tipsRow.classList.toggle("open", on);
  els.tipsRow.setAttribute("aria-expanded", on ? "true" : "false");
  if (on) placeTips();      // measured only once it is laid out
}

// Hover on a mouse, tap or keyboard focus everywhere else.
function wireTips() {
  els.tipsRow.addEventListener("mouseenter", () => showTips(true));
  els.tipsRow.addEventListener("mouseleave", () => showTips(false));
  els.tipsRow.addEventListener("focus", () => showTips(true));
  els.tipsRow.addEventListener("blur", () => showTips(false));
  // Tap-to-toggle only where there is no hover to open it (phones, tablets).
  els.tipsRow.addEventListener("click", (e) => {
    if (window.matchMedia("(hover: hover)").matches) return;
    e.preventDefault(); showTips(els.tipsBox.hidden);
  });
  els.tipsRow.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); showTips(els.tipsBox.hidden); }
    else if (e.key === "Escape") showTips(false);
  });
  window.addEventListener("resize", () => showTips(false));
  els.panel.addEventListener("scroll", () => { if (!els.tipsBox.hidden) placeTips(); });
}

function frame(now) {
  if (state.active) state.active.frame(now);
  requestAnimationFrame(frame);
}

// ---------------- UI ----------------
function cacheEls() {
  for (const id of ["status", "play", "reset", "robotSelect", "sceneSelect", "runSelect",
                    "panel", "tipsRow", "tipsBox"]) els[id] = $(id);
}

function populateRobotSelect() {
  els.robotSelect.innerHTML = "";
  for (const [key, r] of Object.entries(ROBOTS)) {
    const o = document.createElement("option");
    o.value = key; o.textContent = r.title;
    if (key === state.robotKey) o.selected = true;
    els.robotSelect.appendChild(o);
  }
}

// The robot's scenes, then its game as the last entry.
function populateTaskSelect() {
  els.sceneSelect.innerHTML = "";
  for (const s of SCENES[state.robotKey]) {
    const o = document.createElement("option");
    o.value = s.key; o.textContent = s.title; o.title = s.description;
    if (s.key === state.taskKey) o.selected = true;
    els.sceneSelect.appendChild(o);
  }
  const gameKey = gameOf(state.robotKey);
  if (gameKey) {
    const g = GAMES[gameKey];
    const o = document.createElement("option");
    o.value = GAME_TASK; o.textContent = `Game Play: ${g.title}`; o.title = g.description;
    if (state.taskKey === GAME_TASK) o.selected = true;
    els.sceneSelect.appendChild(o);
  }
}

function wireControls() {
  wireTips();
  els.play.addEventListener("click", () => { if (state.active) state.active.togglePlay(); });
  els.reset.addEventListener("click", () => { if (state.active) state.active.reset(); });
  els.robotSelect.addEventListener("change", (e) => {
    const key = e.target.value;
    e.target.blur();
    if (key === state.robotKey || state.switching) { e.target.value = state.robotKey; return; }
    selectTask(key, taskFor(key, state.taskKey));   // keep the kind of task: same scene, or the game
  });
  els.sceneSelect.addEventListener("change", (e) => {
    const key = e.target.value;
    e.target.blur();
    if (key === state.taskKey || state.switching) { e.target.value = state.taskKey; return; }
    selectTask(state.robotKey, key);
  });
  els.runSelect.addEventListener("change", (e) => {
    e.target.blur();
    if (state.switching || !state.active) return;
    state.active.loadRun(e.target.value);
  });
  // Keyboard: the skill keys pick the skill, R resets, Space pauses. Works whatever the page
  // element in focus (a dropdown or button that was just clicked included); only a text field
  // keeps its keys.
  window.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const t = e.target;
    if (t && t.tagName === "INPUT" && !["checkbox", "radio", "file", "button"].includes(t.type)) return;
    const blur = () => { if (t && t.blur && t !== document.body) t.blur(); };
    const mode = state.active;
    const slot = mode ? skillSlotFor(e, mode) : -1;
    if (slot >= 0) { e.preventDefault(); blur(); mode.pickSkill(slot); }
    else if (e.key === "r" || e.key === "R" || e.code === "KeyR") { e.preventDefault(); blur(); if (mode) mode.reset(); }
    else if (e.key === " " || e.code === "Space") { e.preventDefault(); blur(); els.play.click(); }
  });
}

// Which skill slot a key press picks, or -1. A mode that names its own keys (the Franka
// uses Q E S Z C) is matched on the letter -- e.code first, so the physical key works on
// any layout, then e.key. Otherwise the number keys 1-N, read from e.code so the number
// row and the numpad both work.
function skillSlotFor(e, mode) {
  const n = mode.numSkills;
  if (!n) return -1;
  const keys = mode.skillKeys;
  if (keys) {
    const code = (e.code || "").toLowerCase(), ch = (e.key || "").toLowerCase();
    const i = keys.findIndex((k) => code === `key${k.toLowerCase()}` || ch === k.toLowerCase());
    return i >= 0 && i < n ? i : -1;
  }
  const digit = /^(Digit|Numpad)([1-9])$/.exec(e.code || "");
  const d = digit ? +digit[2] : (/^[1-9]$/.test(e.key) ? +e.key : 0);
  return d && d <= n ? d - 1 : -1;
}

boot().catch((e) => { console.error(e); setStatus("Error: " + e.message + " (see console)"); });
