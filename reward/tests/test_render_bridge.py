"""Phase 1 manual test for the render bridge (no ML, no Tinker).

Run:  .venv/Scripts/python.exe reward/tests/test_render_bridge.py

Checks:
  1. Several known-good sketches render concurrently → ok=True, PNG on disk.
  2. A syntax-error sketch → ok=False with a captured error (compile gate).
  3. An infinite-loop sketch → times out → worker hard-restart → next render OK.
"""

import asyncio
import glob
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from reward.render_bridge import RenderBridge  # noqa: E402

SCRATCH = ROOT / "render" / "out" / "bridge_test"
SCRATCH.mkdir(parents=True, exist_ok=True)

GOOD_GLOB = str(ROOT / "datagen" / "out" / "*" / "sketches" / "*.js")

SYNTAX_ERR = "function setup( { createCanvas(600,600,WEBGL); }\nfunction draw(){}"
INFINITE = (
    "function setup(){ createCanvas(600,600,WEBGL); angleMode(DEGREES); brush.scaleBrushes(3); }\n"
    "function draw(){ translate(-width/2,-height/2); background('#fff'); while(true){} noLoop(); }"
)


async def main():
    bridge = RenderBridge(max_concurrent=3, recycle_every=50, default_timeout_ms=45000)
    ok_all = True

    # 1. concurrent good renders
    good = sorted(glob.glob(GOOD_GLOB))[:3]
    if not good:
        print("!! no known-good sketches found under datagen/out; skipping good-render test")
    else:
        codes = [Path(p).read_text(encoding="utf-8") for p in good]
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *[bridge.render(c, SCRATCH / f"good_{i}.png") for i, c in enumerate(codes)]
        )
        dt = time.perf_counter() - t0
        for i, (p, r) in enumerate(zip(good, results)):
            tag = Path(p).name
            print(f"[good {i}] ok={r.ok} ms={r.ms} err={r.error} :: {tag}")
            if not r.ok:
                ok_all = False
            elif not Path(r.out_path).exists():
                print(f"   !! out_path missing: {r.out_path}")
                ok_all = False
        print(f"   ({len(good)} concurrent renders in {dt:.1f}s wall)")

    # 2. syntax error → compile gate
    r = await bridge.render(SYNTAX_ERR, SCRATCH / "syntax.png")
    print(f"[syntax] ok={r.ok} err={(r.error or '')[:80]}")
    if r.ok:
        print("   !! expected ok=False for a syntax error")
        ok_all = False

    # 3. infinite loop → timeout → restart → recover
    r = await bridge.render(INFINITE, SCRATCH / "inf.png", timeout_ms=6000)
    print(f"[infinite] ok={r.ok} err={r.error} (expect ok=False, error=bridge_timeout)")
    if r.ok:
        ok_all = False
    # worker should have restarted; a fresh good render must still work
    if good:
        r = await bridge.render(codes[0], SCRATCH / "recover.png")
        print(f"[recover] ok={r.ok} ms={r.ms} err={r.error}")
        if not r.ok:
            print("   !! worker did not recover after timeout restart")
            ok_all = False

    await bridge.close()
    print("\nRESULT:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
