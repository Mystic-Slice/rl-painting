// Unit + integration tests for the p5.brush render pipeline.
// Run with:  npm test   (uses Node's built-in test runner, no extra deps)
//
// The browser is launched once (shared singleton in render.js) and torn down
// in the final after() hook. The first render pays the cold-start cost, so
// browser-backed tests get a generous per-test timeout.

const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { renderSketch, closeBrowser, fileUrl } = require('./render');

const BROWSER_TIMEOUT = 45000; // cold start (~13s) + render, with headroom
let TMP;

before(() => {
  TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'render-test-'));
});
after(async () => {
  await closeBrowser();
  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch (_) {}
});

// --- helpers ---------------------------------------------------------------
const PNG_MAGIC = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
function isPng(buf) {
  return Buffer.isBuffer(buf) && buf.length > 8 && buf.subarray(0, 8).equals(PNG_MAGIC);
}
// PNG IHDR: width/height are big-endian uint32 at byte offsets 16 and 20.
function pngSize(buf) {
  return { width: buf.readUInt32BE(16), height: buf.readUInt32BE(20) };
}
// A minimal, valid p5.brush sketch at a chosen canvas size.
function simpleSketch(size = 200) {
  return `function setup(){ createCanvas(${size},${size},WEBGL); angleMode(DEGREES); brush.scaleBrushes(1); }
function draw(){
  translate(-width/2,-height/2);
  background("#ffffff");
  brush.set("HB","#204a7a",1);
  brush.line(20,20,${size - 20},${size - 20});
  brush.fill("#4a90d9",60); brush.noStroke();
  brush.circle(${size / 2},${size / 2},${size / 4});
  noLoop();
}`;
}

// ===========================================================================
// fileUrl() -- pure, no browser
// ===========================================================================
test('fileUrl: produces a file:// URL with forward slashes', () => {
  const u = fileUrl('C:\\Users\\me\\sketch.js');
  assert.ok(u.startsWith('file:///'), `expected file:/// prefix, got ${u}`);
  assert.ok(!u.includes('\\'), 'should not contain backslashes');
  assert.match(u, /\/Users\/me\/sketch\.js$/);
});

test('fileUrl: drive-letter absolute path keeps the C: segment', () => {
  const u = fileUrl('C:\\tmp\\a.js');
  assert.match(u, /^file:\/\/\/C:\/tmp\/a\.js$/);
});

test('fileUrl: percent-encodes spaces', () => {
  const u = fileUrl('C:\\my folder\\a b.js');
  assert.ok(u.includes('%20'), `expected %20 in ${u}`);
  assert.ok(!/ /.test(u), 'URL should contain no literal spaces');
});

test('fileUrl: resolves a relative path to an absolute file URL', () => {
  const u = fileUrl('render.js');
  assert.ok(u.startsWith('file:///'));
  assert.ok(u.endsWith('/render.js'));
});

// ===========================================================================
// renderSketch() -- happy path
// ===========================================================================
test('renders a valid sketch to a PNG buffer', { timeout: BROWSER_TIMEOUT }, async () => {
  const res = await renderSketch(simpleSketch(200), { width: 200, height: 200 });
  assert.equal(res.ok, true, `expected ok, got error: ${res.error}`);
  assert.ok(isPng(res.png), 'png should have PNG magic bytes');
  assert.equal(typeof res.ms, 'number');
  assert.ok(Array.isArray(res.console));
});

test('PNG dimensions match the sketch canvas size', { timeout: BROWSER_TIMEOUT }, async () => {
  const res = await renderSketch(simpleSketch(256), { width: 256, height: 256 });
  assert.equal(res.ok, true, res.error);
  assert.deepEqual(pngSize(res.png), { width: 256, height: 256 });
});

test('non-square canvas dimensions are honored', { timeout: BROWSER_TIMEOUT }, async () => {
  const code = `function setup(){ createCanvas(320,180,WEBGL); }
    function draw(){ translate(-160,-90); background("#eee");
      brush.set("HB","#333",1); brush.line(10,10,310,170); noLoop(); }`;
  const res = await renderSketch(code, { width: 320, height: 180 });
  assert.equal(res.ok, true, res.error);
  assert.deepEqual(pngSize(res.png), { width: 320, height: 180 });
});

test('writes the PNG to outPath when provided', { timeout: BROWSER_TIMEOUT }, async () => {
  const out = path.join(TMP, 'written.png');
  const res = await renderSketch(simpleSketch(200), { width: 200, height: 200, outPath: out });
  assert.equal(res.ok, true, res.error);
  assert.equal(res.outPath, path.resolve(out));
  assert.ok(fs.existsSync(out), 'file should exist on disk');
  assert.ok(isPng(fs.readFileSync(out)), 'written file should be a PNG');
});

// ===========================================================================
// renderSketch() -- the compile/usage gate: failures return ok:false, no throw
// ===========================================================================
test('syntax error -> ok:false with captured error (no throw)', { timeout: BROWSER_TIMEOUT }, async () => {
  const bad = `function setup(){ createCanvas(200,200,WEBGL); }
    function draw(){ background("#fff"); brush.line(0,0,10,10  /* missing ) */ noLoop(); }`;
  const res = await renderSketch(bad, { width: 200, height: 200, timeoutMs: 10000 });
  assert.equal(res.ok, false);
  assert.equal(res.png, undefined);
  assert.match(res.error, /SyntaxError/);
});

test('runtime error in draw -> ok:false', { timeout: BROWSER_TIMEOUT }, async () => {
  const bad = `function setup(){ createCanvas(200,200,WEBGL); }
    function draw(){ background("#fff"); doesNotExist.foo(); noLoop(); }`;
  const res = await renderSketch(bad, { width: 200, height: 200, timeoutMs: 10000 });
  assert.equal(res.ok, false);
  assert.match(res.error, /is not defined|doesNotExist/);
});

test('missing draw() -> ok:false with explanatory message', { timeout: BROWSER_TIMEOUT }, async () => {
  const res = await renderSketch(`function setup(){ createCanvas(200,200,WEBGL); }`,
    { width: 200, height: 200, timeoutMs: 10000 });
  assert.equal(res.ok, false);
  assert.match(res.error, /draw\(\)/);
});

test('missing setup() -> ok:false with explanatory message', { timeout: BROWSER_TIMEOUT }, async () => {
  const res = await renderSketch(`function draw(){ background("#fff"); noLoop(); }`,
    { width: 200, height: 200, timeoutMs: 10000 });
  assert.equal(res.ok, false);
  assert.match(res.error, /setup\(\)/);
});

// A sketch that wedges the JS thread must NOT hang the pipeline: it must return
// ok:false promptly after timeoutMs (regression test for the catch-block fix).
test('infinite loop in draw times out cleanly (no hang)', { timeout: BROWSER_TIMEOUT }, async () => {
  const wedged = `function setup(){ createCanvas(200,200,WEBGL); }
    function draw(){ while(true){} }`;
  const t0 = Date.now();
  const res = await renderSketch(wedged, { width: 200, height: 200, timeoutMs: 2500 });
  const elapsed = Date.now() - t0;
  assert.equal(res.ok, false);
  assert.ok(res.error && res.error.length > 0, 'should report a timeout error');
  assert.ok(elapsed < 20000, `should return shortly after timeout, took ${elapsed}ms`);
});

// ===========================================================================
// Render stability. NOTE (verified 2026-08-26): rendering is NOT byte
// deterministic — p5.brush uses its own internal RNG for stroke/bleed jitter
// (p5's randomSeed/noiseSeed don't cover it), and SwiftShader adds FP noise, so
// two renders of identical code differ at the pixel-grain level. They are,
// however, perceptually identical (same composition/hatching/bleed). Reward
// signals (HPS/CLIP/VLM) are therefore stable per rollout, but a reward cannot
// be byte-cached by source code. This test pins the property we DO rely on:
// identical code yields valid PNGs of stable dimensions.
// ===========================================================================
test('identical code renders a valid PNG of stable size (content not byte-exact)',
  { timeout: BROWSER_TIMEOUT }, async () => {
  const code = `function setup(){ createCanvas(200,200,WEBGL); angleMode(DEGREES);
      randomSeed(7); noiseSeed(7); brush.scaleBrushes(1); }
    function draw(){ translate(-100,-100); background("#f4e8c1");
      brush.set("HB","#333",0.6); brush.fill("#b66a5e",50); brush.fillBleed(0.2,"out");
      brush.hatch(5,45,{rand:0}); brush.circle(100,100,60); noLoop(); }`;
  const a = await renderSketch(code, { width: 200, height: 200 });
  const b = await renderSketch(code, { width: 200, height: 200 });
  assert.equal(a.ok, true, a.error);
  assert.equal(b.ok, true, b.error);
  assert.deepEqual(pngSize(a.png), pngSize(b.png), 'dimensions should be stable');
  assert.deepEqual(pngSize(a.png), { width: 200, height: 200 });
});
