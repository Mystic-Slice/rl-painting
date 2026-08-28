#!/usr/bin/env python
"""
datagen/promptopt/compare.py

Render a before/after comparison of two system prompts across every combo in
prompts.json: for each scene, generate a sketch with the SEED prompt and with the
OPTIMIZED prompt, render both, and score both with the same 5-criterion judge.

Writes PNGs + a manifest.json that build_report.py turns into a visual grid.

    python datagen/promptopt/compare.py \
        --optimized datagen/promptopt/runs/20260827_153429/best_system_prompt.txt
"""
import argparse
import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
DATAGEN = HERE.parent
ROOT = DATAGEN.parent
sys.path.insert(0, str(DATAGEN))
sys.path.insert(0, str(HERE))

from generate import load_env, extract_code, user_prompt_for      # noqa: E402
from refine import chat, render                                    # noqa: E402
from optimize_prompt import judge_image, CRITERIA, Cfg, Tracker    # noqa: E402


def gen_render_score(client, cfg, system_prompt, scene, out_png):
    """Generate -> render -> judge one (prompt, scene). Returns a result dict."""
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_for(scene)}]
    content, gm, _ = chat(client, cfg.base_url, cfg.key, cfg.generator, messages,
                          cfg.gen_temperature, cfg.gen_max_tokens)
    cfg.tracker.record_metrics("generation", gm)
    if gm.get("error"):
        return {"ok": False, "stage": "generation", "error": gm["error"], "score": None}
    code = extract_code(content)
    if not code.strip():
        return {"ok": False, "stage": "generation", "error": "no code", "score": None}

    js_path = out_png.with_suffix(".js")
    js_path.write_text(code, encoding="utf-8")
    ok, rms, rerr = render(js_path, out_png, cfg.render_timeout)
    cfg.tracker.add("render", time_s=(rms or 0) / 1000.0, error=not ok)
    if not ok:
        return {"ok": False, "stage": "render", "error": rerr, "score": None,
                "code_lines": code.count("\n") + 1}

    composite, scores, _ = judge_image(client, cfg, scene, out_png)
    return {"ok": True, "stage": "judged", "score": composite,
            "scores": {k: scores[k] for k in CRITERIA if scores.get(k) is not None},
            "png": str(out_png), "code_lines": code.count("\n") + 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-prompt", default=str(DATAGEN / "system_prompt.txt"))
    ap.add_argument("--optimized", required=True, help="path to optimized system prompt")
    ap.add_argument("--generator", default="openai/gpt-5.6-sol")
    ap.add_argument("--judge", default="google/gemini-3.7-flash")
    ap.add_argument("--gen-temperature", type=float, default=0.85)
    ap.add_argument("--gen-max-tokens", type=int, default=16000)
    ap.add_argument("--render-timeout", type=int, default=90000)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--out", default=str(HERE / "compare"))
    args = ap.parse_args()

    key, base_url = load_env()
    seed_prompt = Path(args.seed_prompt).read_text(encoding="utf-8")
    opt_prompt = Path(args.optimized).read_text(encoding="utf-8")
    combos = json.loads((DATAGEN / "prompts.json").read_text(encoding="utf-8"))["combos"]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out) / stamp
    (out_root / "seed").mkdir(parents=True, exist_ok=True)
    (out_root / "opt").mkdir(parents=True, exist_ok=True)

    client = httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0))
    tracker = Tracker(save_path=out_root / "tracker.json")
    cfg = Cfg()
    cfg.key, cfg.base_url = key, base_url
    cfg.generator, cfg.judge = args.generator, args.judge
    cfg.gen_temperature, cfg.gen_max_tokens = args.gen_temperature, args.gen_max_tokens
    cfg.render_timeout, cfg.tracker = args.render_timeout, tracker

    # one job per (combo, variant)
    jobs = []
    for c in combos:
        for variant, sp in (("seed", seed_prompt), ("opt", opt_prompt)):
            png = out_root / variant / (c["id"] + ".png")
            jobs.append((c, variant, sp, png))

    print("Comparing %d combos x 2 prompts = %d renders  (generator=%s, judge=%s)"
          % (len(combos), len(jobs), args.generator, args.judge))
    print("out: %s\n" % out_root)

    results = {}   # (combo_id, variant) -> result
    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(gen_render_score, client, cfg, sp, c["prompt"], png):
                (c["id"], variant) for (c, variant, sp, png) in jobs}
        for fut in as_completed(futs):
            cid, variant = futs[fut]
            try:
                results[(cid, variant)] = fut.result()
            except Exception as e:
                results[(cid, variant)] = {"ok": False, "stage": "exception",
                                           "error": str(e), "score": None}
            done += 1
            r = results[(cid, variant)]
            sc = ("%.3f" % r["score"]) if isinstance(r.get("score"), (int, float)) else \
                 ("FAIL:" + str(r.get("stage")))
            print("  [%2d/%d] %-24s %-4s  %s" % (done, len(jobs), cid, variant, sc), flush=True)

    # assemble manifest
    entries = []
    s_tot = o_tot = s_n = o_n = 0.0
    for c in combos:
        s = results.get((c["id"], "seed"), {})
        o = results.get((c["id"], "opt"), {})
        if isinstance(s.get("score"), (int, float)):
            s_tot += s["score"]; s_n += 1
        if isinstance(o.get("score"), (int, float)):
            o_tot += o["score"]; o_n += 1
        entries.append({"id": c["id"], "animal": c["animal"], "prompt": c["prompt"],
                        "seed": s, "opt": o})
    manifest = {
        "generated_at": stamp,
        "generator": args.generator, "judge": args.judge,
        "seed_prompt": str(args.seed_prompt), "optimized_prompt": str(args.optimized),
        "seed_mean_score": round(s_tot / s_n, 4) if s_n else None,
        "opt_mean_score": round(o_tot / o_n, 4) if o_n else None,
        "n": len(entries), "entries": entries,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rep, _, total = tracker.report()
    print("\n" + rep)
    print("\nseed mean: %s   opt mean: %s   (n combos scored)"
          % (manifest["seed_mean_score"], manifest["opt_mean_score"]))
    print("manifest -> %s" % (out_root / "manifest.json"))
    client.close()


if __name__ == "__main__":
    main()
