// Copyright (c) 2026. MuJoCo player for game-discovery agents.
// SPDX-License-Identifier: BSD-3-Clause
//
// JS port of shared/policy.py -- the hierarchical actor-critic INFERENCE path.
// Identical math to the numpy version (verified to ~1e-5 against torch):
//   selectSkill(obs, stochastic) = argmax high_level_policy(obs), or a sample from
//                      softmax(logits) when stochastic  (every skill_duration steps)
//   act(obs, skill)  = low_level_policy([obs[low_level_obs_indices] (else obs[start:start+dim]), onehot(skill)]) mean,
//                      then sample Normal(mean, exp(log_std)) unless deterministic.
// Weights are loaded from policy.json (layer table) + weights.bin (float32).

function elu(x) { return x > 0 ? x : Math.exp(Math.min(x, 0)) - 1; }
const ACT = { elu, relu: (x) => Math.max(0, x), tanh: Math.tanh };

// Gaussian (Box-Muller). Optional seeded RNG for reproducibility.
function makeGaussian(rng) {
  return function gauss() {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}
// Mulberry32 seeded PRNG (so web replays are reproducible if desired).
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

class Layer {
  constructor(w, b, outDim, inDim) { this.w = w; this.b = b; this.out = outDim; this.in = inDim; }
  // y = x @ W^T + b  ; W stored row-major (out, in)
  forward(x, y) {
    const { w, b, out, in: din } = this;
    for (let o = 0; o < out; o++) {
      let s = b[o];
      const base = o * din;
      for (let i = 0; i < din; i++) s += w[base + i] * x[i];
      y[o] = s;
    }
    return y;
  }
}

export class HierarchicalPolicy {
  constructor(meta, buffer, seed = 12345) {
    this.meta = meta;
    this.numSkills = meta.num_skills;
    this.numActions = meta.num_actions;
    this.proprioStart = meta.proprio_start;
    this.proprioDim = meta.proprio_dim;
    // Explicit low-level input indices (newer ant runs: me block minus roll) win over the
    // contiguous obs[start:start+dim] slice.
    this.obsIndices = Array.isArray(meta.low_level_obs_indices) && meta.low_level_obs_indices.length
      ? Int32Array.from(meta.low_level_obs_indices) : null;
    if (this.obsIndices) this.proprioDim = this.obsIndices.length;
    this.skillDuration = meta.skill_duration;
    this._actfn = ACT[meta.activation] || elu;
    this.tanhMean = !!meta.low_level_tanh_mean;   // mean squashed to [-1, 1] before sampling

    const f32 = new Float32Array(buffer);
    const named = {};
    for (const t of meta.tensors) {
      named[t.name] = f32.subarray(t.offset / 4, t.offset / 4 + t.count);
    }
    const stack = (prefix) => {
      const layers = [];
      let i = 0;
      while (named[`${prefix}_w${i}`]) {
        const wt = meta.tensors.find((t) => t.name === `${prefix}_w${i}`);
        layers.push(new Layer(named[`${prefix}_w${i}`], named[`${prefix}_b${i}`],
                              wt.shape[0], wt.shape[1]));
        i++;
      }
      return layers;
    };
    this.hl = stack("hl");
    this.ll = stack("ll");
    this.logStd = named["low_level_log_std"];
    this.std = new Float32Array(this.logStd.length);
    for (let i = 0; i < this.logStd.length; i++) this.std[i] = Math.exp(this.logStd[i]);

    this._rng = mulberry32(seed);
    this._gauss = makeGaussian(this._rng);
    // scratch buffers
    this._buf = [];
  }

  _mlp(x, layers) {
    let cur = x;
    for (let k = 0; k < layers.length; k++) {
      const L = layers[k];
      let out = this._buf[k];
      if (!out || out.length !== L.out) out = this._buf[k] = new Float64Array(L.out);
      L.forward(cur, out);
      if (k < layers.length - 1) for (let i = 0; i < L.out; i++) out[i] = this._actfn(out[i]);
      cur = out;
    }
    return cur;
  }

  highLevelLogits(obs) { return this._mlp(obs, this.hl); }

  // Default: argmax, the act_inference choice. `stochastic` samples the categorical the
  // high level actually is (softmax over the logits) -- what the Isaac play script does
  // without --argmax_skill, and what the web player always uses.
  selectSkill(obs, stochastic = false) {
    const logits = this.highLevelLogits(obs);
    const n = logits.length;
    let best = 0, bv = logits[0];
    for (let i = 1; i < n; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
    if (!stochastic) return best;
    // softmax, shifted by the max for numerical safety, then one uniform draw
    let sum = 0;
    const probs = new Float64Array(n);
    for (let i = 0; i < n; i++) { probs[i] = Math.exp(logits[i] - bv); sum += probs[i]; }
    let u = this._rng() * sum;
    for (let i = 0; i < n; i++) { u -= probs[i]; if (u <= 0) return i; }
    return n - 1;
  }

  actionMean(obs, skill) {
    const inp = new Float64Array(this.proprioDim + this.numSkills);
    if (this.obsIndices) for (let i = 0; i < this.proprioDim; i++) inp[i] = obs[this.obsIndices[i]];
    else for (let i = 0; i < this.proprioDim; i++) inp[i] = obs[this.proprioStart + i];
    inp[this.proprioDim + skill] = 1.0;
    // mlp buffers for hl/ll overlap; use a separate buffer set for ll
    return this._mlpLow(inp);
  }

  _mlpLow(x) {
    let cur = x;
    for (let k = 0; k < this.ll.length; k++) {
      const L = this.ll[k];
      let out = (this._lbuf || (this._lbuf = []))[k];
      if (!out || out.length !== L.out) out = this._lbuf[k] = new Float64Array(L.out);
      L.forward(cur, out);
      if (k < this.ll.length - 1) for (let i = 0; i < L.out; i++) out[i] = this._actfn(out[i]);
      else if (this.tanhMean) for (let i = 0; i < L.out; i++) out[i] = Math.tanh(out[i]);
      cur = out;
    }
    return cur;
  }

  act(obs, skill, deterministic = false) {
    const mean = this.actionMean(obs, skill);
    const a = new Float64Array(this.numActions);
    for (let i = 0; i < this.numActions; i++) {
      a[i] = deterministic ? mean[i] : mean[i] + this.std[i] * this._gauss();
    }
    return a;
  }
}

// Fetch policy.json + weights.bin (paths relative to the page) as a raw
// {meta, buffer} export (export_policy.py writes the pair).
export async function fetchExport(policyJsonUrl) {
  const meta = await (await fetch(policyJsonUrl)).json();
  const base = policyJsonUrl.substring(0, policyJsonUrl.lastIndexOf("/") + 1);
  const buffer = await (await fetch(base + meta.weights_file)).arrayBuffer();
  return { meta, buffer };
}

export async function loadPolicy(policyJsonUrl, seed = 12345) {
  const { meta, buffer } = await fetchExport(policyJsonUrl);
  return new HierarchicalPolicy(meta, buffer, seed);
}
