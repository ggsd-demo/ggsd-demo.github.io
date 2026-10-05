# Vendored MuJoCo WASM

`mujoco_wasm.js` is `mujoco-js@0.0.7` (`dist/mujoco_wasm.js`) from npm:
https://www.npmjs.com/package/mujoco-js  (mirror of zalo/mujoco_wasm, MuJoCo 3.3.x).

Why this build:
- self-contained: the WASM binary is embedded (base64) -- no separate .wasm to host
- single-threaded: no SharedArrayBuffer, so it runs on GitHub Pages with NO
  COOP/COEP headers / no service-worker hack required.

Refresh with:
  curl -sL -o mujoco_wasm.js https://cdn.jsdelivr.net/npm/mujoco-js@0.0.7/dist/mujoco_wasm.js
