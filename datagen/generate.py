#!/usr/bin/env python
"""
datagen/generate.py -- generate p5.brush sketches from several models via OpenRouter,
recording time / cost / tokens per generation.

Default run (the smoke test): one prompt ("koi fish in a small pond on a sunny day")
across every model in MODELS, one sample each.

    python datagen/generate.py

Useful flags:
    --prompt "..."         override the task prompt (default: koi smoke-test prompt)
    --combo koi_pond_sunny use a combo id from prompts.json instead of --prompt
    --all-combos           run every combo in prompts.json (dataset generation)
    --models a,b,c         comma list overriding MODELS
    --runs N               samples per (model, prompt)  [default 1]
    --temperature 0.85     sampling temperature
    --max-tokens 8000      completion cap
    --render               after generating, render each sketch via ../render/render.js
    --out datagen/out      output root
    --resume <run_dir>     continue an existing run dir: skip generations that already
                           succeeded, retry missing/errored ones, and (with --render)
                           re-render successful sketches whose render is missing/failed
                           without re-calling the model

Outputs, per run, under <out>/<timestamp>/ :
    sketches/<combo>__<model>__r<k>.js   extracted JS
    raw/<combo>__<model>__r<k>.json      full OpenRouter response
    results.jsonl                        one metrics row per generation
    summary.txt                          human-readable table
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# The initial model set (expand later). Exact OpenRouter slugs as requested.
MODELS = [
    # "qwen/qwen3.8-max",
    # "z-ai/glm-5.3",
    "openai/gpt-5.6-sol",
    # "anthropic/claude-opus-5",
    # "x-ai/grok-4.6",
    "google/gemini-3.1-pro-preview",
]

DEFAULT_PROMPT = "koi fish in a small pond on a sunny day"


def load_env():
    """Minimal .env parser -> returns (api_key, base_url)."""
    env = {}
    envfile = ROOT / ".env"
    if envfile.exists():
        for line in envfile.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    key = env.get("OPENROUTER_KEY") or os.environ.get("OPENROUTER_KEY")
    url = env.get("OPENROUTER_URL") or "https://openrouter.ai/api/v1"
    if not key:
        sys.exit("ERROR: OPENROUTER_KEY not found in .env or environment.")
    return key, url.rstrip("/")


def extract_code(text):
    """Strip markdown fences / stray prose, return the JS body."""
    if not text:
        return ""
    t = text.strip()
    # Prefer a fenced block if present.
    m = re.search(r"```(?:javascript|js)?\s*\n(.*?)```", t, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Otherwise drop a leading lone fence and trailing fence.
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _metrics(model, elapsed, **kw):
    d = {
        "model": model,
        "time_s": round(elapsed, 3),
        "error": None,
        "finish_reason": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "cost_usd": None,
        "provider": None,
        "response_id": None,
        "served_model": None,
    }
    d.update(kw)
    return d


def call_model(client, base_url, key, model, system_prompt, user_prompt,
               temperature, max_tokens):
    """One chat.completions call. Returns (metrics_dict, raw_json_or_None)."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "usage": {"include": True},   # ask OpenRouter to return cost + token usage
    }
    headers = {
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json",
    }
    t0 = time.perf_counter()
    try:
        r = client.post(base_url + "/chat/completions", headers=headers, json=body)
        elapsed = time.perf_counter() - t0
        if r.status_code != 200:
            return _metrics(model, elapsed,
                            error="HTTP %d: %s" % (r.status_code, r.text[:500])), None
        data = r.json()
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return _metrics(model, elapsed, error="%s: %s" % (type(e).__name__, e)), None

    # API-level error object.
    if isinstance(data, dict) and data.get("error"):
        return _metrics(model, elapsed, error=str(data["error"])[:500]), data

    try:
        choice = data["choices"][0]
        content = choice["message"]["content"] or ""
        finish = choice.get("finish_reason")
    except Exception as e:
        return _metrics(model, elapsed, error="bad response shape: %s" % e), data

    usage = data.get("usage") or {}
    code = extract_code(content)
    m = _metrics(
        model, elapsed,
        error=None,
        finish_reason=finish,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        cost_usd=usage.get("cost"),
        provider=data.get("provider"),
        response_id=data.get("id"),
        served_model=data.get("model"),
    )
    m["code"] = code
    m["code_chars"] = len(code)
    m["code_lines"] = code.count("\n") + 1 if code else 0
    m["raw_content_chars"] = len(content)
    return m, data


def maybe_render(js_path, out_png, render_timeout_ms=90000):
    """Render a sketch via the existing node pipeline; returns (ok, ms, error).

    p5.brush watercolour fills are expensive under headless software WebGL, so a
    rich sketch can take 40-60s. We pass the per-render cap through to render.js
    and give the subprocess a generous buffer on top of it.
    """
    render_js = ROOT / "render" / "render.js"
    if not render_js.exists():
        return None, None, "render/render.js not found"
    # render.js runs with cwd=render/, so relative sketch/png paths would resolve
    # against that dir. Pass absolute paths so a relative --out still works.
    js_path = Path(js_path).resolve()
    out_png = Path(out_png).resolve()
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            ["node", str(render_js), str(js_path), str(out_png),
             "600", "600", str(render_timeout_ms)],
            cwd=str(ROOT / "render"), capture_output=True, text=True,
            timeout=render_timeout_ms / 1000.0 + 30,
        )
        ms = int((time.perf_counter() - t0) * 1000)
        if proc.returncode == 0:
            return True, ms, None
        return False, ms, (proc.stderr or proc.stdout or "render failed").strip()[:500]
    except Exception as e:
        return False, int((time.perf_counter() - t0) * 1000), "%s: %s" % (type(e).__name__, e)


def user_prompt_for(p):
    return ("Paint this as a loose watercolour: %s.\n"
            "Write the complete p5.brush sketch now, following every harness rule." % p)


def tag_for(combo_id, model, k):
    return "%s__%s__r%d" % (combo_id, model.replace("/", "_"), k)


def load_existing_rows(results_path):
    """Read a prior results.jsonl into {tag: row}, latest line winning per tag.

    Rows written before resume support may lack a "tag" field; reconstruct it from
    combo_id/model/run so old runs remain resumable.
    """
    by_tag = {}
    if not results_path.exists():
        return by_tag
    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        tag = row.get("tag")
        if not tag:
            tag = tag_for(row.get("combo_id", "adhoc"),
                          row.get("model", "?"), row.get("run", 0))
            row["tag"] = tag
        by_tag[tag] = row
    return by_tag


def resolve_tasks(args, combos_data):
    """Returns [(combo_id, prompt), ...]."""
    if args.all_combos:
        return [(c["id"], c["prompt"]) for c in combos_data["combos"]]
    if args.combo:
        match = [c for c in combos_data["combos"] if c["id"] == args.combo]
        if not match:
            sys.exit("combo id '%s' not in prompts.json" % args.combo)
        return [(match[0]["id"], match[0]["prompt"])]
    return [("adhoc", args.prompt or DEFAULT_PROMPT)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--combo", default=None, help="combo id from prompts.json")
    ap.add_argument("--all-combos", action="store_true")
    ap.add_argument("--models", default=None, help="comma list overriding MODELS")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--max-tokens", type=int, default=64000)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--render-timeout", type=int, default=90000,
                    help="per-render cap in ms (watercolour fills are slow; default 90000)")
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--resume", default=None,
                    help="existing run dir to continue (skips already-successful generations)")
    args = ap.parse_args()

    key, base_url = load_env()
    system_prompt = (HERE / "system_prompt.txt").read_text(encoding="utf-8")
    models = [m.strip() for m in args.models.split(",")] if args.models else MODELS
    combos_data = json.loads((HERE / "prompts.json").read_text(encoding="utf-8"))
    tasks = resolve_tasks(args, combos_data)

    if args.resume:
        run_dir = Path(args.resume)
        if not run_dir.exists():
            sys.exit("--resume dir does not exist: %s" % run_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.out) / stamp
    (run_dir / "sketches").mkdir(parents=True, exist_ok=True)
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    if args.render:
        (run_dir / "png").mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "results.jsonl"
    by_tag = load_existing_rows(results_path) if args.resume else {}

    planned = len(tasks) * len(models) * args.runs
    done_ok = sum(1 for r in by_tag.values() if not r.get("error"))
    if args.resume:
        print("Resuming: %s" % run_dir)
        print("  existing rows: %d  (successful: %d)" % (len(by_tag), done_ok))
    else:
        print("Run dir: %s" % run_dir)
    print("Models (%d): %s" % (len(models), ", ".join(models)))
    print("Tasks: %d  x  runs: %d  = %d generations planned\n"
          % (len(tasks), args.runs, planned))

    def append_row(row):
        by_tag[row["tag"]] = row
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def do_render(tag, js_path, metrics):
        png = run_dir / "png" / (tag + ".png")
        ok, ms, rerr = maybe_render(js_path, png, args.render_timeout)
        metrics["render_ok"] = ok
        metrics["render_ms"] = ms
        metrics["render_error"] = rerr
        metrics["png_file"] = str(png.relative_to(run_dir)) if ok else None
        return ok

    n_gen = n_skip = n_rerender = 0
    timeout = httpx.Timeout(args.timeout, connect=30.0)
    with httpx.Client(timeout=timeout) as client:
        for combo_id, prompt in tasks:
            up = user_prompt_for(prompt)
            for model in models:
                for k in range(args.runs):
                    tag = tag_for(combo_id, model, k)
                    prev = by_tag.get(tag)

                    # Already generated successfully: skip, or cheaply re-render.
                    if prev is not None and not prev.get("error"):
                        needs_render = (
                            args.render and prev.get("render_ok") is not True
                            and prev.get("sketch_file"))
                        js_path = (run_dir / prev["sketch_file"]) if prev.get("sketch_file") else None
                        if needs_render and js_path and js_path.exists():
                            print("  -> %-32s [%s r%d] re-render ..." % (model, combo_id, k),
                                  end="", flush=True)
                            ok = do_render(tag, js_path, prev)
                            append_row(prev)
                            n_rerender += 1
                            print("  render:" + ("ok" if ok else "FAIL"))
                        else:
                            n_skip += 1
                            print("  -- %-32s [%s r%d] skip (done)" % (model, combo_id, k))
                        continue

                    print("  -> %-32s [%s r%d] ..." % (model, combo_id, k),
                          end="", flush=True)
                    metrics, raw = call_model(
                        client, base_url, key, model, system_prompt, up,
                        args.temperature, args.max_tokens,
                    )
                    metrics["tag"] = tag
                    metrics["combo_id"] = combo_id
                    metrics["prompt"] = prompt
                    metrics["run"] = k

                    if raw is not None:
                        (run_dir / "raw" / (tag + ".json")).write_text(
                            json.dumps(raw, indent=2), encoding="utf-8")

                    code = metrics.pop("code", "")
                    if code:
                        js_path = run_dir / "sketches" / (tag + ".js")
                        js_path.write_text(code, encoding="utf-8")
                        metrics["sketch_file"] = str(js_path.relative_to(run_dir))
                        if args.render:
                            do_render(tag, js_path, metrics)
                    else:
                        metrics["sketch_file"] = None

                    append_row(metrics)
                    n_gen += 1

                    if metrics["error"]:
                        print(" ERROR (%.1fs): %s"
                              % (metrics["time_s"], metrics["error"][:80]))
                    else:
                        cost = metrics["cost_usd"]
                        cost_s = ("$%.4f" % cost) if isinstance(cost, (int, float)) else "n/a"
                        extra = ""
                        if args.render:
                            extra = "  render:" + ("ok" if metrics.get("render_ok") else "FAIL")
                        print(" %6.1fs  %s tok  %s  %d lines%s"
                              % (metrics["time_s"],
                                 metrics["completion_tokens"] or "?",
                                 cost_s, metrics["code_lines"], extra))

    print("\nThis invocation: %d generated, %d re-rendered, %d skipped."
          % (n_gen, n_rerender, n_skip))
    rows = list(by_tag.values())
    write_summary(run_dir, rows, args)
    write_manifest(run_dir, rows, args)
    print("\nDone. Metrics: %s" % results_path)
    print("Summary: %s" % (run_dir / "summary.txt"))
    print("Manifest: %s" % (run_dir / "manifest.json"))


def write_manifest(run_dir, rows, args):
    """A clean, loadable index mapping every output image -> model + prompt + metrics."""
    entries = []
    for r in rows:
        entries.append({
            "png_file": r.get("png_file"),          # rendered image (None if not rendered/failed)
            "sketch_file": r.get("sketch_file"),    # the p5.brush .js
            "model": r.get("model"),                # requested OpenRouter slug
            "served_model": r.get("served_model"),  # what OpenRouter actually served
            "provider": r.get("provider"),
            "combo_id": r.get("combo_id"),
            "prompt": r.get("prompt"),              # the scene prompt used
            "run": r.get("run"),
            "time_s": r.get("time_s"),
            "cost_usd": r.get("cost_usd"),
            "prompt_tokens": r.get("prompt_tokens"),
            "completion_tokens": r.get("completion_tokens"),
            "code_lines": r.get("code_lines"),
            "render_ok": r.get("render_ok"),
            "render_error": r.get("render_error"),
            "error": r.get("error"),
        })
    manifest = {
        "run": run_dir.name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "params": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "runs": args.runs,
            "rendered": bool(args.render),
        },
        "count": len(entries),
        "entries": entries,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")


def write_summary(run_dir, rows, args):
    lines = []
    lines.append("Run: %s" % run_dir.name)
    lines.append("temperature=%s  max_tokens=%s  runs=%s  render=%s"
                 % (args.temperature, args.max_tokens, args.runs, args.render))
    lines.append("")
    hdr = ("%-32s %-18s %-3s %7s %6s %6s %9s %6s"
           % ("model", "combo", "ok", "time_s", "ptok", "ctok", "cost$", "lines"))
    lines.append(hdr)
    lines.append("-" * len(hdr))
    tot_cost, tot_time = 0.0, 0.0
    for r in rows:
        ok = "ERR" if r["error"] else "ok"
        cost = r["cost_usd"] if isinstance(r["cost_usd"], (int, float)) else 0.0
        tot_cost += cost or 0.0
        tot_time += r["time_s"] or 0.0
        cost_s = ("%.4f" % cost) if r["cost_usd"] is not None else "n/a"
        lines.append("%-32s %-18s %-3s %7.1f %6s %6s %9s %6s"
                     % (r["model"][:32], str(r.get("combo_id"))[:18], ok,
                        r["time_s"], str(r["prompt_tokens"] or "-"),
                        str(r["completion_tokens"] or "-"), cost_s,
                        str(r["code_lines"] or "-")))
        if r["error"]:
            lines.append("    error: %s" % r["error"][:160])
    lines.append("-" * len(hdr))
    lines.append("TOTAL cost $%.4f   wall(sum of calls) %.1fs   n=%d"
                 % (tot_cost, tot_time, len(rows)))
    (run_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
