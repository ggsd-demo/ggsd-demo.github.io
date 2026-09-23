// Copyright (c) 2026. MuJoCo player for game-discovery agents.
// SPDX-License-Identifier: BSD-3-Clause
//
// JS mirror of shared/games/g1_boxing.py: binds the g1-boxing game (MJCF, policy,
// 6 skills, PD/194-dim config, HP) for the web player.

import { GAMES } from "../registry.js";
import { MujocoDualEnv, g1BoxingConfig } from "../sim.js";
import { loadPolicy, fetchExport } from "../policy.js";

export const SPEC = GAMES.g1_boxing;
const MESH_MJCF = "exports/g1_boxing/g1_boxing_mesh.xml";
const MESH_DIR = "exports/g1/meshes";       // shared with the Playground G1 (identical STLs)

// opts.mesh=true loads the high-detail visual-mesh model (writes the STLs into the
// MuJoCo WASM filesystem first); otherwise the lightweight primitive-collider model.
export async function makeEnv(mujoco, fetchText, opts = {}) {
  try { mujoco.FS.mkdir("/working"); } catch (e) { /* already exists */ }
  try { mujoco.FS.mount(mujoco.MEMFS, { root: "." }, "/working"); } catch (e) { /* already mounted */ }
  let xmlFile;
  if (opts.mesh) {
    const xml = await fetchText(MESH_MJCF);
    // ship the STLs the model references into /working/meshes
    try { mujoco.FS.mkdir("/working/meshes"); } catch (e) { /* exists */ }
    const files = [...new Set([...xml.matchAll(/file="([^"]+\.STL)"/gi)].map((m) => m[1]))];
    await Promise.all(files.map(async (f) => {
      const buf = new Uint8Array(await (await fetch(`${MESH_DIR}/${f}`)).arrayBuffer());
      mujoco.FS.writeFile(`/working/meshes/${f}`, buf);
    }));
    mujoco.FS.writeFile("/working/g1_boxing_mesh.xml", xml);
    xmlFile = "/working/g1_boxing_mesh.xml";
  } else {
    mujoco.FS.writeFile("/working/g1_boxing.xml", await fetchText(SPEC.mjcf));
    xmlFile = "/working/g1_boxing.xml";
  }
  const model = mujoco.MjModel.loadFromXML(xmlFile);
  const data = new mujoco.MjData(model);
  const env = new MujocoDualEnv(mujoco, model, data, g1BoxingConfig());
  return { model, data, env };
}

// One of SPEC.runs by key (default: the first), with its policy.json path.
export function runFor(runKey) {
  const runs = SPEC.runs || [];
  const r = runs.find((x) => x.key === runKey) || runs[0];
  return r ? { ...r, policy: `exports/g1_boxing/ckpts/${r.key}/policy.json` } : { key: null, label: "committed", policy: SPEC.policy };
}

export async function makePolicy(seed = 12345, runKey = null) {
  return loadPolicy(runFor(runKey).policy, seed);
}

export async function makeExport(runKey = null) {
  return fetchExport(runFor(runKey).policy);
}
