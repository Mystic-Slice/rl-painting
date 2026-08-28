// worker.js — persistent render worker wrapping renderSketch() from render.js.
//
// Speaks a JSON-lines protocol over stdin/stdout so a long-lived Python process
// (reward/render_bridge.py) can dispatch many renders without paying a fresh
// Chrome launch per call (the render.js CLI does). One shared headless Chrome is
// reused across all requests; each render gets its own page, so overlapping
// requests are serviced concurrently and responses come back out-of-order.
//
// PROTOCOL
//   stdin  : one JSON object per line. Two shapes:
//     render : {"id": str, "code": str, "outPath": str,
//               "width": 600, "height": 600, "timeoutMs": 45000, "seed": int|null}
//     op     : {"id": str, "op": "recycle"|"shutdown"|"ping"}
//   stdout : one JSON object per line, echoing "id":
//     render : {"id", "ok", "outPath"|null, "error"|null, "ms", "console": [...]}
//     op     : {"id", "ok": true, "op": <op>}
//     ready  : {"ready": true, "pid": <int>}   (emitted once at startup)
//
// Only protocol JSON is ever written to stdout. Any incidental logging goes to
// stderr. On stdin EOF the worker closes the browser and exits.

const readline = require('readline');
const { renderSketch, closeBrowser } = require('./render.js');

// The PNG bytes never travel over the pipe — renderSketch writes the file to
// outPath and we return the path. This keeps the protocol lines small.
const CONSOLE_CAP = 4000;         // truncate captured console text per response
const RECYCLE_EVERY = parseInt(process.env.RENDER_RECYCLE_EVERY || '50', 10);

let rendersSinceRecycle = 0;
let recycling = null;             // in-flight recycle promise (serializes recycles)

function send(obj) {
  process.stdout.write(JSON.stringify(obj) + '\n');
}

function logErr(msg) {
  process.stderr.write('[worker] ' + msg + '\n');
}

// Recycle the shared browser between requests. Clears the leaked renderer
// process a JS-wedged sketch (e.g. while(true) in draw) leaves behind, which
// render.js's closeBrowser() reaps via taskkill /T. Serialized so concurrent
// triggers don't double-close.
async function recycleBrowser() {
  if (recycling) return recycling;
  recycling = (async () => {
    try {
      await closeBrowser();
    } catch (e) {
      logErr('recycle closeBrowser error: ' + (e && e.message ? e.message : e));
    } finally {
      rendersSinceRecycle = 0;
      recycling = null;
    }
  })();
  return recycling;
}

async function handleRender(msg) {
  // If a recycle is in flight, wait for it so we don't grab a closing browser.
  if (recycling) await recycling;
  const started = Date.now();
  let res;
  try {
    res = await renderSketch(msg.code, {
      outPath: msg.outPath,
      width: msg.width || 600,
      height: msg.height || 600,
      timeoutMs: msg.timeoutMs || 45000,
      seed: msg.seed != null ? msg.seed : undefined,
    });
  } catch (e) {
    // renderSketch is documented never to throw, but fail closed regardless.
    res = { ok: false, error: 'worker: ' + (e && e.message ? e.message : String(e)),
            ms: Date.now() - started, console: [] };
  }

  let consoleText = Array.isArray(res.console) ? res.console.join('\n') : '';
  if (consoleText.length > CONSOLE_CAP) consoleText = consoleText.slice(0, CONSOLE_CAP);

  send({
    id: msg.id,
    ok: !!res.ok,
    outPath: res.ok ? (res.outPath || msg.outPath) : null,
    error: res.ok ? null : (res.error || 'render failed'),
    ms: res.ms != null ? res.ms : (Date.now() - started),
    console: consoleText,
  });

  rendersSinceRecycle += 1;
  if (rendersSinceRecycle >= RECYCLE_EVERY) {
    // Respond first (already sent), then recycle before servicing more.
    await recycleBrowser();
  }
}

async function handleLine(line) {
  line = line.trim();
  if (!line) return;
  let msg;
  try {
    msg = JSON.parse(line);
  } catch (e) {
    send({ id: null, ok: false, error: 'worker: bad JSON: ' + (e && e.message ? e.message : e) });
    return;
  }

  if (msg.op) {
    if (msg.op === 'shutdown') {
      await closeBrowser().catch(() => {});
      send({ id: msg.id, ok: true, op: 'shutdown' });
      process.exit(0);
    } else if (msg.op === 'recycle') {
      await recycleBrowser();
      send({ id: msg.id, ok: true, op: 'recycle' });
    } else if (msg.op === 'ping') {
      send({ id: msg.id, ok: true, op: 'ping' });
    } else {
      send({ id: msg.id, ok: false, error: 'worker: unknown op ' + msg.op });
    }
    return;
  }

  // A render request — do NOT await here, so concurrent requests overlap.
  handleRender(msg).catch((e) => {
    send({ id: msg.id, ok: false, error: 'worker: ' + (e && e.message ? e.message : e),
           ms: 0, console: '' });
  });
}

const rl = readline.createInterface({ input: process.stdin });
rl.on('line', (line) => { handleLine(line); });
rl.on('close', async () => {
  await closeBrowser().catch(() => {});
  process.exit(0);
});

process.on('uncaughtException', (e) => {
  logErr('uncaughtException: ' + (e && e.stack ? e.stack : e));
});

send({ ready: true, pid: process.pid });
