// Copyright (c) 2026. MuJoCo player for game-discovery agents.
// SPDX-License-Identifier: BSD-3-Clause
//
// Low-level actor of a hierarchical checkpoint (JS mirror of shared/policy.py
// LowLevelPolicy): act(obs_ll, skill) = MLP([obs_ll, onehot(skill)]) mean,
// tanh-squashed when the checkpoint set low_level_tanh_mean, then sampled
// Normal(mean, exp(log_std)) unless deterministic. Weights: policy.json + weights.bin.

function elu(x) { return x > 0 ? x : Math.exp(Math.min(x, 0)) - 1; }
const ACT = { elu, relu: (x) => Math.max(0, x), tanh: Math.tanh };

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function makeGaussian(rng) {
  return function () {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

class Layer {
  constructor(w, b, outDim, inDim) { this.w = w; this.b = b; this.out = outDim; this.in = inDim; }
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

export class LowLevelPolicy {
  constructor(meta, buffer, seed = 12345) {
    this.meta = meta;
    this.numSkills = meta.num_skills;
    this.numActions = meta.num_actions;
    this.obsDim = meta.proprio_dim;
    this.skillDuration = meta.skill_duration;
    this.tanhMean = !!meta.low_level_tanh_mean;
    this._actfn = ACT[meta.activation] || elu;
    const f32 = new Float32Array(buffer);
    const named = {};
    for (const t of meta.tensors) named[t.name] = f32.subarray(t.offset / 4, t.offset / 4 + t.count);
    this.ll = [];
    for (let i = 0; named[`ll_w${i}`]; i++) {
      const wt = meta.tensors.find((t) => t.name === `ll_w${i}`);
      this.ll.push(new Layer(named[`ll_w${i}`], named[`ll_b${i}`], wt.shape[0], wt.shape[1]));
    }
    if (this.ll[0].in !== this.obsDim + this.numSkills) {
      throw new Error(`low-level input ${this.ll[0].in} != obs ${this.obsDim} + skills ${this.numSkills}`);
    }
    this.logStd = named["low_level_log_std"];
    this.std = Float64Array.from(this.logStd, Math.exp);
    this._gauss = makeGaussian(mulberry32(seed));
    this._buf = [];
    this._inp = new Float64Array(this.obsDim + this.numSkills);
  }

  actionMean(obs, skill) {
    if (obs.length !== this.obsDim) throw new Error(`obs is ${obs.length}-dim, policy expects ${this.obsDim}`);
    const inp = this._inp;
    inp.fill(0);
    for (let i = 0; i < this.obsDim; i++) inp[i] = obs[i];
    inp[this.obsDim + skill] = 1.0;
    let cur = inp;
    for (let k = 0; k < this.ll.length; k++) {
      const L = this.ll[k];
      let out = this._buf[k];
      if (!out || out.length !== L.out) out = this._buf[k] = new Float64Array(L.out);
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
    for (let i = 0; i < this.numActions; i++) a[i] = deterministic ? mean[i] : mean[i] + this.std[i] * this._gauss();
    return a;
  }
}

export async function fetchExport(policyJsonUrl) {
  const meta = await (await fetch(policyJsonUrl)).json();
  const base = policyJsonUrl.substring(0, policyJsonUrl.lastIndexOf("/") + 1);
  const buffer = await (await fetch(base + meta.weights_file)).arrayBuffer();
  return { meta, buffer };
}

export async function loadPolicy(policyJsonUrl, seed = 12345) {
  const { meta, buffer } = await fetchExport(policyJsonUrl);
  return new LowLevelPolicy(meta, buffer, seed);
}
