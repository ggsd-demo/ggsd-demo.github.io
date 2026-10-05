// Copyright (c) 2026. MuJoCo player for game-discovery agents.
// SPDX-License-Identifier: BSD-3-Clause
//
// Binds the ant-fencing game (MJCF, policy, skills, config) for the web player: the
// ant_sumo model and loop with the fencing contact rule (sim.js antFencingConfig).

import { GAMES } from "../registry.js";
import { MujocoDualEnv, antFencingConfig } from "../sim.js";
import { loadPolicy, fetchExport } from "../policy.js";

export const SPEC = GAMES.ant_fencing;

// Load mujoco model + data + env from the vendored engine.
export async function makeEnv(mujoco, fetchText) {
  const xml = await fetchText(SPEC.mjcf);
  try { mujoco.FS.mkdir("/working"); } catch (e) { /* already exists */ }
  try { mujoco.FS.mount(mujoco.MEMFS, { root: "." }, "/working"); } catch (e) { /* already mounted (game switch) */ }
  mujoco.FS.writeFile("/working/ant_fencing.xml", xml);
  const model = mujoco.MjModel.loadFromXML("/working/ant_fencing.xml");
  const data = new mujoco.MjData(model);
  const env = new MujocoDualEnv(mujoco, model, data, antFencingConfig());
  return { model, data, env };
}

// One of SPEC.runs by key (default: the first), with its policy.json path.
export function runFor(runKey) {
  const runs = SPEC.runs || [];
  const r = runs.find((x) => x.key === runKey) || runs[0];
  return r ? { ...r, policy: `exports/ant_fencing/ckpts/${r.key}/policy.json` } : { key: null, label: "committed", policy: SPEC.policy };
}

export async function makePolicy(seed = 12345, runKey = null) {
  return loadPolicy(runFor(runKey).policy, seed);
}

// Raw {meta, buffer} of a committed export (for slice re-instantiation).
export async function makeExport(runKey = null) {
  return fetchExport(runFor(runKey).policy);
}
