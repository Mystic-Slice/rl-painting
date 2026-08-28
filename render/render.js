// render.js — p5.brush render pipeline: sketch code (string) -> PNG.
//
// Design notes:
//  * The model's sketch is written to its OWN .js file and referenced via
//    <script src>, so a stray "</script>" in generated code can't break HTML.
//  * The vendored p5 / p5.brush live in ./vendor and are referenced by absolute
//    file:// URL; Chrome is launched with --allow-file-access-from-files.
//  * One shared headless Chrome instance is reused across renders for speed.
//  * Any load/compile/runtime error is captured (window.__renderError) and
//    returned as { ok:false, error } instead of crashing — this IS the compile
//    gate the training loop relies on.

const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');
const puppeteer = require('puppeteer');

const HERE = __dirname;
const VENDOR = path.join(HERE, 'vendor');
const TEMPLATE = fs.readFileSync(path.join(HERE, 'template.html'), 'utf8');

// file:// URL for a local path (Windows-safe: backslashes -> slashes).
function fileUrl(p) {
  let u = path.resolve(p).split('\\').join('/');
  if (!u.startsWith('/')) u = '/' + u; // drive-letter paths -> /C:/...
  return 'file://' + encodeURI(u);
}

let _browserPromise = null;
function getBrowser() {
  if (!_browserPromise) {
    _browserPromise = puppeteer.launch({
      headless: 'new',
      // Puppeteer injects --disable-gpu for headless on some platforms; strip it
      // so Chrome can bind the real GPU.
      ignoreDefaultArgs: ['--disable-gpu'],
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--allow-file-access-from-files',
        // Hardware WebGL2 via ANGLE's Direct3D11 backend on the real (NVIDIA) GPU.
        // (Previously forced to CPU SwiftShader; p5.brush's watercolour fills are
        // shader-bound, so hardware ANGLE is the offload target.)
        '--use-gl=angle',
        '--use-angle=d3d11',
        '--enable-gpu',
        '--enable-webgl',
        '--ignore-gpu-blocklist',
        '--disable-dev-shm-usage',
      ],
    });
  }
  return _browserPromise;
}

// Force-kill a browser process tree. A JS-wedged renderer (e.g. `while(true)`
// in draw) can survive Puppeteer's graceful close on Windows AND inherits the
// CDP pipe FDs, keeping Node's event loop alive so the process never exits.
// This must run while the PARENT is still alive: once the parent is killed the
// renderer is orphaned and `taskkill /T` can no longer walk the tree. So we kill
// the whole tree first, then let browser.close() clean up the Node-side handles.
function killTree(proc) {
  if (!proc || !proc.pid || proc.exitCode !== null) return;
  if (process.platform === 'win32') {
    try {
      execFileSync('taskkill', ['/pid', String(proc.pid), '/T', '/F'],
        { stdio: 'ignore' });
    } catch (_) {}
  } else {
    try { process.kill(-proc.pid, 'SIGKILL'); } catch (_) {
      try { proc.kill('SIGKILL'); } catch (_) {}
    }
  }
}

async function closeBrowser() {
  if (_browserPromise) {
    const b = await _browserPromise;
    _browserPromise = null;
    const proc = b.process();
    killTree(proc);              // kill tree while parent still alive
    await b.close().catch(() => {}); // then release Node-side transport handles
  }
}

/**
 * Render a p5.brush sketch string to a PNG.
 * @param {string} code  Full sketch defining global setup()/draw().
 * @param {object} opts  { outPath?, width?, height?, timeoutMs?, seed? }
 * @returns {Promise<{ok:boolean, outPath?:string, png?:Buffer, width?:number,
 *                     height?:number, error?:string, ms:number, console:string[]}>}
 */
async function renderSketch(code, opts = {}) {
  const width = opts.width || 600;
  const height = opts.height || 600;
  const timeoutMs = opts.timeoutMs || 20000;
  const started = Date.now();

  // Per-render scratch dir with the sketch + assembled html.
  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), 'p5render-'));
  const sketchPath = path.join(workDir, 'sketch.js');
  const htmlPath = path.join(workDir, 'index.html');

  // Optional deterministic seeding: prepend seed calls into setup via a shim.
  let sketchCode = code;
  if (opts.seed != null) {
    sketchCode =
      `/* injected seed */ (function(){var _s=${JSON.stringify(opts.seed)};` +
      `window.__seed=_s;})();\n` + code;
  }
  fs.writeFileSync(sketchPath, sketchCode, 'utf8');

  const html = TEMPLATE
    .replace('__P5__', fileUrl(path.join(VENDOR, 'p5.min.js')))
    .replace('__P5BRUSH__', fileUrl(path.join(VENDOR, 'p5.brush.js')))
    // sketch injected as external script src (escaping-proof):
    .replace(
      /<script id="user-sketch">[\s\S]*?<\/script>/,
      `<script id="user-sketch" src="${fileUrl(sketchPath)}"></script>`
    );
  fs.writeFileSync(htmlPath, html, 'utf8');

  const browser = await getBrowser();
  const page = await browser.newPage();
  const consoleLines = [];
  page.on('console', (m) => consoleLines.push(`[${m.type()}] ${m.text()}`));
  page.on('pageerror', (e) => consoleLines.push(`[pageerror] ${e.message}`));

  const result = { ok: false, console: consoleLines, width, height };
  try {
    await page.setViewport({ width, height, deviceScaleFactor: 1 });
    await page.goto(fileUrl(htmlPath), { waitUntil: 'load', timeout: timeoutMs });

    // Wait until either a frame painted or an error was captured.
    await page.waitForFunction(
      'window.__renderDone === true || window.__renderError !== null',
      { timeout: timeoutMs, polling: 50 }
    );

    const errStr = await page.evaluate('window.__renderError');
    if (errStr) {
      result.error = String(errStr);
    } else {
      // small settle for any GPU flush, then snapshot the canvas element
      const canvas = await page.$('canvas');
      if (!canvas) {
        result.error = 'No <canvas> element was created (sketch never ran?).';
      } else {
        const buf = await canvas.screenshot({ type: 'png' });
        result.ok = true;
        result.png = buf;
        if (opts.outPath) {
          fs.mkdirSync(path.dirname(path.resolve(opts.outPath)), { recursive: true });
          fs.writeFileSync(opts.outPath, buf);
          result.outPath = path.resolve(opts.outPath);
        }
      }
    }
  } catch (e) {
    // Timeout or navigation failure -> treat as a failed (gated) render.
    // NOTE: don't call page.evaluate() here — on a timeout the page's JS thread
    // may be wedged (e.g. a `while(true)` in draw), and evaluate() would hang
    // until the CDP protocol timeout. If __renderError had been set, the
    // waitForFunction above would have resolved rather than thrown.
    result.error = e && e.message ? e.message : String(e);
  } finally {
    await page.close().catch(() => {});
    result.ms = Date.now() - started;
    try { fs.rmSync(workDir, { recursive: true, force: true }); } catch (_) {}
  }
  return result;
}

// Probe the actual WebGL backend Chrome bound (hardware vs software). Useful to
// confirm GPU offload took: hardware reports e.g. "ANGLE (NVIDIA ... Direct3D11)",
// software reports "... SwiftShader ...". Uses a throwaway context so it never
// disturbs the sketch's canvas.
async function glRenderer() {
  const browser = await getBrowser();
  const page = await browser.newPage();
  try {
    await page.goto('about:blank');
    return await page.evaluate(() => {
      const c = document.createElement('canvas');
      const gl = c.getContext('webgl2') || c.getContext('webgl');
      if (!gl) return 'no-webgl-context';
      const ext = gl.getExtension('WEBGL_debug_renderer_info');
      return String(ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)
                        : gl.getParameter(gl.RENDERER));
    });
  } catch (e) {
    return 'probe-failed: ' + (e && e.message ? e.message : e);
  } finally {
    await page.close().catch(() => {});
  }
}

module.exports = { renderSketch, getBrowser, closeBrowser, fileUrl, glRenderer };

// ---- CLI: node render.js <sketch.js> <out.png> [width] [height] -----------
if (require.main === module) {
  (async () => {
    const [, , sketchArg, outArg, wArg, hArg, tArg] = process.argv;
    if (!sketchArg || !outArg) {
      console.error('usage: node render.js <sketch.js> <out.png> [width] [height] [timeoutMs]');
      process.exit(2);
    }
    const code = fs.readFileSync(sketchArg, 'utf8');
    const res = await renderSketch(code, {
      outPath: outArg,
      width: wArg ? parseInt(wArg, 10) : 600,
      height: hArg ? parseInt(hArg, 10) : 600,
      timeoutMs: tArg ? parseInt(tArg, 10) : 20000,
    });
    await closeBrowser();
    if (res.ok) {
      console.log(`OK  ${res.outPath}  (${res.width}x${res.height}, ${res.ms}ms)`);
      process.exit(0);
    } else {
      console.error(`FAIL (${res.ms}ms): ${res.error}`);
      if (res.console.length) console.error(res.console.join('\n'));
      process.exit(1);
    }
  })();
}
