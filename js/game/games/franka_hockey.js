// Copyright (c) 2026. MuJoCo player for game-discovery agents.
// SPDX-License-Identifier: BSD-3-Clause
//
// JS mirror of shared/games/franka_hockey.py: binds the franka-hockey game (MJCF + Panda
// meshes, policy, 5 skills, task-space IK control, 61-dim obs, goals) for the web player.

import { GAMES } from "../registry.js";
import { MujocoFrankaHockeyEnv, frankaHockeyConfig } from "../franka_sim.js";
import { loadPolicy, fetchExport } from "../policy.js";

export const SPEC = GAMES.franka_hockey;
const MESH_DIR = "exports/franka/meshes";   // shared with the Playground Franka (identical STLs)

// Loads the MJCF and writes the Panda STL meshes it references into the MuJoCo WASM
// filesystem (/working/meshes) before compiling. opts.rules: "play" (default) | "train".
export async function makeEnv(mujoco, fetchText, opts = {}) {
  try { mujoco.FS.mkdir("/working"); } catch (e) { /* already exists */ }
  try { mujoco.FS.mount(mujoco.MEMFS, { root: "." }, "/working"); } catch (e) { /* already mounted */ }
  const xml = await fetchText(SPEC.mjcf);
  try { mujoco.FS.mkdir("/working/meshes"); } catch (e) { /* exists */ }
  const files = [...new Set([...xml.matchAll(/file="([^"]+\.stl)"/gi)].map((m) => m[1]))];
  await Promise.all(files.map(async (f) => {
    const buf = new Uint8Array(await (await fetch(`${MESH_DIR}/${f}`)).arrayBuffer());
    mujoco.FS.writeFile(`/working/meshes/${f}`, buf);
  }));
  mujoco.FS.writeFile("/working/franka_hockey.xml", xml);
  const model = mujoco.MjModel.loadFromXML("/working/franka_hockey.xml");
  const data = new mujoco.MjData(model);
  const env = new MujocoFrankaHockeyEnv(mujoco, model, data, frankaHockeyConfig(opts.rules || "play"));
  return { model, data, env };
}

// One of SPEC.runs by key (default: the first), with its policy.json path.
export function runFor(runKey) {
  const runs = SPEC.runs || [];
  const r = runs.find((x) => x.key === runKey) || runs[0];
  return r ? { ...r, policy: `exports/franka_hockey/ckpts/${r.key}/policy.json` } : { key: null, label: "committed", policy: SPEC.policy };
}

export async function makePolicy(seed = 12345, runKey = null) {
  return loadPolicy(runFor(runKey).policy, seed);
}

export async function makeExport(runKey = null) {
  return fetchExport(runFor(runKey).policy);
}
