# render — p5.brush render pipeline (code → PNG)

Turns a p5.brush JavaScript sketch (string) into a rendered PNG in headless
Chrome. This is the standalone rendering infra (plan Phase 1) that both the
reference-pool bootstrap (Phase 4) and the GRPO reward loop (Phase 5) build on.

## Stack (pinned locally, offline/deterministic)
- **p5.js 2.3.2** — `vendor/p5.min.js`
- **p5.brush 2.2.2** (the *p5 build*, requires p5 2.x + WEBGL) — `vendor/p5.brush.js`
- **Puppeteer** headless Chrome, WebGL2 via ANGLE + SwiftShader (no GPU needed)

## Files
- `template.html` — HTML harness: loads vendored libs, captures load/runtime
  errors on `window.__renderError`, flags `window.__renderDone` after one frame.
- `render.js`     — `renderSketch(code, opts)` → `{ ok, png, outPath, error, ms, console }`.
  Reuses one browser across calls. Also a CLI (see below).
- `samples/`      — hand-authored reference sketches.
- `render.test.js` — unit + integration tests (`npm test`, Node's built-in runner).
- `test_pipeline.js` — quick visual smoke test that writes viewable PNGs to `out/`.

## Sketch contract
A sketch is a full global-mode p5 program defining `setup()` and `draw()`:
```js
function setup(){ createCanvas(600,600,WEBGL); angleMode(DEGREES); brush.scaleBrushes(3); }
function draw(){ translate(-width/2,-height/2); background("#f4e8c1"); /* brush.* */ noLoop(); }
```
`brush` is a global. The model's code is injected as an **external** `<script src>`,
so a stray `</script>` can't break rendering.

## Usage
CLI:
```
node render.js samples/hibiscus.js out/hibiscus.png [width] [height]
```
Programmatic:
```js
const { renderSketch, closeBrowser } = require('./render');
const res = await renderSketch(codeString, { outPath: 'out/x.png', width: 600, height: 600, timeoutMs: 20000 });
// res.ok === false on any compile/runtime/timeout error; res.error has the message.
await closeBrowser();
```

## Testing
```
npm test          # node --test — 14 tests, ~13 s
```
Covers `fileUrl` (pure), the happy path (PNG magic bytes, dimensions, disk
output), all gate paths (syntax/runtime/missing-setup/missing-draw), the
timeout/no-hang path, and render stability.

## Behavior / timing
- First-ever render on a machine: ~13 s (one-time SwiftShader shader compile,
  then cached on disk). Cold render on a warm cache: ~1–2 s.
- **Warm render (browser reused): ~0.5–1 s** — the real per-rollout cost.
- Any syntax error, runtime error, missing `setup()/draw()`, or timeout returns
  `ok:false` with a captured `error` — it never throws. This IS the compile gate.

## Two things to know for training
- **Rendering is perceptually stable but NOT byte-deterministic.** p5.brush uses
  its own internal RNG for stroke/bleed jitter (p5's `randomSeed`/`noiseSeed`
  don't cover it) and SwiftShader adds FP noise, so identical code renders the
  same *picture* with slightly different grain. → reward signals are stable per
  rollout, but you cannot byte-cache a reward keyed on source code.
- **A JS-wedged sketch (`while(true)` in draw) times out to `ok:false`, but its
  renderer process survives** `page.close()` (Chrome can't reap a spinning
  renderer). Each such timeout leaks ~1 Chrome process. `closeBrowser()`
  force-kills the whole tree (`taskkill /T`), so the pipeline always exits
  cleanly. Mitigation for long runs: recycle the browser
  (`closeBrowser()` + let it relaunch) every N steps to reap any leaks.
