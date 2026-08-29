"""reward/render_bridge.py -- async client for the persistent Node render worker.

Wraps render/worker.js (which wraps render/render.js `renderSketch`) so the RL
loop can render many p5.brush sketches to PNGs without paying a fresh Chrome
launch per render. One long-lived worker process services overlapping requests
over a JSON-lines pipe.

Windows-safe transport: a blocking `subprocess.Popen` plus a daemon reader
thread, NOT `asyncio.create_subprocess_exec` (whose behaviour depends on the
Proactor-vs-Selector event-loop policy the cookbook's `asyncio.run` installs).
The reader thread resolves per-request `asyncio.Future`s via
`loop.call_soon_threadsafe`.

The bridge NEVER raises into the training loop: every failure path (worker crash,
timeout, wedged sketch, malformed line) resolves the pending render as
`RenderResult(ok=False, error=...)`, which the reward code treats as a failed
compile gate.

Usage
-----
    bridge = get_render_bridge()            # lazy module singleton (per event loop)
    res = await bridge.render(code, out_path)
    if res.ok: ...                          # res.out_path holds the PNG
    await bridge.close()                    # at shutdown

Do not store the bridge on a picklable object (e.g. an EnvGroupBuilder). Fetch it
lazily inside async code via `get_render_bridge()`.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RENDER_DIR = ROOT / "render"
WORKER_JS = RENDER_DIR / "worker.js"


@dataclass
class RenderResult:
    ok: bool
    out_path: str | None = None
    error: str | None = None
    ms: int = 0
    console: str = ""


class RenderBridge:
    """Async front-end to a single persistent Node render worker.

    A RenderBridge is bound to the event loop that first drives it. All public
    methods are coroutines and must be awaited from that loop.
    """

    def __init__(
        self,
        node_path: str = "node",
        worker_js: Path = WORKER_JS,
        cwd: Path = RENDER_DIR,
        max_concurrent: int = 3,
        recycle_every: int = 50,
        default_timeout_ms: int = 45000,
        watchdog_pad_s: float = 20.0,
    ):
        self.node_path = node_path
        self.worker_js = Path(worker_js)
        self.cwd = Path(cwd)
        self.max_concurrent = max_concurrent
        self.recycle_every = recycle_every
        self.default_timeout_ms = default_timeout_ms
        self.watchdog_pad_s = watchdog_pad_s

        self._proc: subprocess.Popen | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._sem: asyncio.Semaphore | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._ready: asyncio.Future | None = None
        self._stdin_lock = threading.Lock()
        self._start_lock: asyncio.Lock | None = None
        self._reader_thread: threading.Thread | None = None
        self._closing = False

    # --- lifecycle --------------------------------------------------------- #
    async def _ensure_started(self) -> None:
        if self._start_lock is None:
            self._start_lock = asyncio.Lock()
        async with self._start_lock:
            if self._proc is not None and self._proc.poll() is None:
                return
            await self._spawn()

    async def _spawn(self) -> None:
        if not self.worker_js.exists():
            raise FileNotFoundError(f"render worker not found: {self.worker_js}")
        self._loop = asyncio.get_running_loop()
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.max_concurrent)
        self._ready = self._loop.create_future()
        env = dict(os.environ, RENDER_RECYCLE_EVERY=str(self.recycle_every))
        # text mode, line-buffered, utf-8; stderr piped to a drain thread.
        self._proc = subprocess.Popen(
            [self.node_path, str(self.worker_js)],
            cwd=str(self.cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
        )
        self._closing = False
        self._reader_thread = threading.Thread(
            target=self._read_stdout, args=(self._proc,), daemon=True
        )
        self._reader_thread.start()
        threading.Thread(
            target=self._drain_stderr, args=(self._proc,), daemon=True
        ).start()
        # Wait for the worker's {"ready": true} line before dispatching renders.
        try:
            await asyncio.wait_for(self._ready, timeout=30.0)
        except asyncio.TimeoutError:
            raise RuntimeError("render worker did not report ready within 30s")

    def _read_stdout(self, proc: subprocess.Popen) -> None:
        """Daemon thread: parse worker stdout, resolve futures on the loop."""
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("ready"):
                    self._resolve_ready()
                    continue
                rid = obj.get("id")
                if rid is not None:
                    self._resolve(rid, obj)
        except Exception:
            pass
        finally:
            # stdout closed => worker exited. Fail everything still pending.
            self._fail_all("render worker exited")

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        try:
            for line in proc.stderr:  # type: ignore[union-attr]
                if line.strip():
                    sys.stderr.write("[render-worker] " + line)
        except Exception:
            pass

    def _resolve_ready(self) -> None:
        loop, fut = self._loop, self._ready
        if loop and fut and not fut.done():
            loop.call_soon_threadsafe(lambda: fut.done() or fut.set_result(True))

    def _resolve(self, rid: str, obj: dict) -> None:
        loop = self._loop
        if not loop:
            return

        def _set():
            fut = self._pending.pop(rid, None)
            if fut and not fut.done():
                fut.set_result(obj)

        loop.call_soon_threadsafe(_set)

    def _fail_all(self, reason: str) -> None:
        loop = self._loop
        if not loop:
            return

        def _fail():
            for rid, fut in list(self._pending.items()):
                if not fut.done():
                    fut.set_result({"id": rid, "ok": False, "error": reason,
                                    "ms": 0, "console": ""})
                self._pending.pop(rid, None)

        loop.call_soon_threadsafe(_fail)

    # --- rendering --------------------------------------------------------- #
    async def render(
        self,
        code: str,
        out_path: str | os.PathLike,
        seed: int | None = None,
        timeout_ms: int | None = None,
    ) -> RenderResult:
        await self._ensure_started()
        assert self._sem is not None and self._loop is not None
        timeout_ms = timeout_ms or self.default_timeout_ms
        rid = uuid.uuid4().hex
        out_path = str(Path(out_path).resolve())

        async with self._sem:
            fut = self._loop.create_future()
            self._pending[rid] = fut
            req = {
                "id": rid,
                "code": code,
                "outPath": out_path,
                "width": 600,
                "height": 600,
                "timeoutMs": timeout_ms,
                "seed": seed,
            }
            try:
                self._write_line(json.dumps(req))
            except Exception as e:
                self._pending.pop(rid, None)
                return RenderResult(ok=False, error=f"bridge write failed: {e}")

            watchdog = timeout_ms / 1000.0 + self.watchdog_pad_s
            try:
                obj = await asyncio.wait_for(fut, timeout=watchdog)
            except asyncio.TimeoutError:
                # Worker is wedged (e.g. a while(true) sketch survived). Hard-kill
                # the process tree and respawn lazily on the next render.
                self._pending.pop(rid, None)
                await self._hard_restart()
                return RenderResult(ok=False, error="bridge_timeout", ms=timeout_ms)
            except Exception as e:
                self._pending.pop(rid, None)
                return RenderResult(ok=False, error=f"bridge error: {e}")

        return RenderResult(
            ok=bool(obj.get("ok")),
            out_path=obj.get("outPath"),
            error=obj.get("error"),
            ms=int(obj.get("ms") or 0),
            console=obj.get("console") or "",
        )

    def _write_line(self, line: str) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            raise RuntimeError("render worker not running")
        with self._stdin_lock:
            proc.stdin.write(line + "\n")
            proc.stdin.flush()

    async def _hard_restart(self) -> None:
        """Kill the worker process tree (Windows: taskkill /T) and drop state.

        The next render() call will lazily respawn via _ensure_started().
        """
        proc, self._proc = self._proc, None
        self._fail_all("render worker restarted")
        if proc is None:
            return
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True,
                )
            else:
                proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass

    async def close(self) -> None:
        self._closing = True
        proc = self._proc
        if proc is None or proc.poll() is not None:
            self._proc = None
            return
        try:
            self._write_line(json.dumps({"id": "shutdown", "op": "shutdown"}))
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            await self._hard_restart()
        finally:
            self._proc = None


# --- module singleton (lazy, per process) --------------------------------- #
_BRIDGE: RenderBridge | None = None


def get_render_bridge(**kwargs) -> RenderBridge:
    """Return the process-wide RenderBridge, creating it on first use.

    kwargs are applied only on first construction. Safe to call from any async
    context; the bridge binds to the running loop the first time render() runs.
    """
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = RenderBridge(**kwargs)
    return _BRIDGE


async def close_render_bridge() -> None:
    global _BRIDGE
    if _BRIDGE is not None:
        await _BRIDGE.close()
        _BRIDGE = None
