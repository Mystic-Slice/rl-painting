#!/usr/bin/env python
"""
datagen/refine.py -- AutoResearch-style iterative refinement of a single p5.brush
painting, driven by a VLM judge (the blog's "iterate against a critic" loop).

For one prompt:

    editor model writes an initial sketch
        -> render.js renders it to a PNG
        -> a VLM judge (Gemini Flash) scores it 0-10 and writes concrete feedback
        -> the editor SEES its own render + the critique and rewrites the sketch
        -> repeat, keeping the best-scoring iteration.

Each editor turn is STATELESS: no message history accumulates. The initial sketch
is a plain (contract + paint task) call, and every revision is rebuilt fresh from
the CURRENT code + its render + the CURRENT critique only — nothing from earlier
iterations. The judge is reference-free by default (Gemini judges from its own
knowledge of the subject); pass --reference IMG to also show it a reference photo,
and --editor-reference to additionally ground the EDITOR on that same photo.

    python datagen/refine.py --prompt "a koi fish in a small pond on a sunny day"

Useful flags:
    --prompt "..."            the scene to paint  (default: the koi smoke-test prompt)
    --combo koi_pond_sunny    use a combo id from prompts.json instead of --prompt
    --iters 6                 max refine iterations (renders/critiques)  [default 6]
    --threshold 8.5           stop early once the judge score reaches this  [default 8.5]
    --editor openai/gpt-5.6-sol            code-editing agent (OpenRouter slug)
    --judge  google/gemini-3.7-flash       VLM judge (OpenRouter slug)
    --reference path.jpg      optional reference photo shown to the JUDGE
    --editor-reference        also show the --reference photo to the EDITOR (grounding)
    --temperature 0.85        editor sampling temperature
    --max-tokens 64000        editor completion cap
    --out datagen/out_refine  output root

Outputs, per run, under <out>/<timestamp>/ :
    iters/i0.js, i0.png, i1.js, i1.png, ...   every iteration's code + render
    best.js, best.png                          the highest-scoring iteration
    log.jsonl                                  one row per iteration (score, feedback, cost)
    summary.txt                                human-readable score trajectory
"""
import argparse
import base64
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

# Reuse the plumbing from the one-shot generator so there's a single source of truth.
from generate import load_env, extract_code, user_prompt_for

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


# --------------------------------------------------------------------------- #
# OpenRouter chat (multimodal-capable: content may be a string or a parts list)
# --------------------------------------------------------------------------- #
def chat(client, base_url, key, model, messages, temperature, max_tokens):
    """One chat.completions call over an arbitrary message list.

    Returns (content_str, metrics_dict, raw_json_or_None). Never raises.
    """
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "usage": {"include": True},
    }
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    t0 = time.perf_counter()
    try:
        r = client.post(base_url + "/chat/completions", headers=headers, json=body)
        elapsed = time.perf_counter() - t0
        if r.status_code != 200:
            return "", {"model": model, "time_s": round(elapsed, 3),
                        "error": "HTTP %d: %s" % (r.status_code, r.text[:400]),
                        "cost_usd": None, "completion_tokens": None}, None
        data = r.json()
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return "", {"model": model, "time_s": round(elapsed, 3),
                    "error": "%s: %s" % (type(e).__name__, e),
                    "cost_usd": None, "completion_tokens": None}, None

    if isinstance(data, dict) and data.get("error"):
        return "", {"model": model, "time_s": round(elapsed, 3),
                    "error": str(data["error"])[:400],
                    "cost_usd": None, "completion_tokens": None}, data

    try:
        content = data["choices"][0]["message"]["content"] or ""
    except Exception as e:
        return "", {"model": model, "time_s": round(elapsed, 3),
                    "error": "bad response shape: %s" % e,
                    "cost_usd": None, "completion_tokens": None}, data

    usage = data.get("usage") or {}
    metrics = {
        "model": model,
        "time_s": round(elapsed, 3),
        "error": None,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cost_usd": usage.get("cost"),
        "served_model": data.get("model"),
    }
    return content, metrics, data


def img_part(png_path):
    """A base64 data-URL image content-part for a chat message."""
    b64 = base64.b64encode(Path(png_path).read_bytes()).decode("ascii")
    return {"type": "image_url",
            "image_url": {"url": "data:image/png;base64," + b64}}


def reference_parts(ref_path):
    """Labeled reference-photo content-parts for the EDITOR (grounding, not a target
    to copy). Kept visually distinct from the editor's own previous render so a
    stateless turn can't confuse the two images."""
    return [
        {"type": "text", "text":
            "REFERENCE PHOTOGRAPH of the subject — use it to get the subject's form, "
            "proportions and colours right, but interpret it as a LOOSE watercolour; "
            "do not trace or copy it literally:"},
        img_part(ref_path),
    ]


# --------------------------------------------------------------------------- #
# Render (reuse the existing node pipeline)
# --------------------------------------------------------------------------- #
def render(js_path, out_png, timeout_ms=90000):
    """Render a sketch via render/render.js. Returns (ok, ms, error)."""
    render_js = ROOT / "render" / "render.js"
    if not render_js.exists():
        return False, None, "render/render.js not found"
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            ["node", str(render_js), str(js_path), str(out_png),
             "600", "600", str(timeout_ms)],
            cwd=str(ROOT / "render"), capture_output=True, text=True,
            timeout=timeout_ms / 1000.0 + 30,
        )
        ms = int((time.perf_counter() - t0) * 1000)
        if proc.returncode == 0:
            return True, ms, None
        return False, ms, (proc.stderr or proc.stdout or "render failed").strip()[:400]
    except Exception as e:
        return False, int((time.perf_counter() - t0) * 1000), \
            "%s: %s" % (type(e).__name__, e)


# --------------------------------------------------------------------------- #
# VLM judge (reference-free by default)
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM = (
    "You are a meticulous, fair-but-harsh art critic evaluating WATERCOLOUR "
    "paintings that were produced procedurally by code. You will be shown one "
    "painting and told what it is meant to depict. Judge it as a watercolour "
    "painting on: (1) subject recognisability, (2) composition and focal clarity, "
    "(3) colour harmony, (4) painterly looseness — soft bleeding washes and "
    "confident layered shapes, NOT stiff outlines or muddy blobs, (5) overall "
    "aesthetic appeal. Most first attempts are mediocre; reserve 8+ for genuinely "
    "accomplished paintings.\n\n"
    "Respond with ONLY a JSON object, no prose, no code fence:\n"
    '{"score": <number 0-10, one decimal>, '
    '"feedback": "<2-4 sentences of concrete, actionable critique: name the '
    'specific weaknesses and say how to fix them in the next revision>"}'
)


def judge(client, base_url, key, model, prompt, png_path, reference=None):
    """Score + critique one render. Returns (score_or_None, feedback, metrics)."""
    parts = [{"type": "text",
              "text": "This painting is meant to depict: %s.\nEvaluate it." % prompt}]
    if reference and Path(reference).exists():
        parts.append({"type": "text", "text": "Reference photograph of the subject:"})
        parts.append(img_part(reference))
        parts.append({"type": "text", "text": "The painting to evaluate:"})
    parts.append(img_part(png_path))

    messages = [{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": parts}]
    content, metrics, _ = chat(client, base_url, key, model, messages,
                               temperature=0.2, max_tokens=1200)
    if metrics.get("error"):
        return None, "JUDGE ERROR: " + metrics["error"], metrics

    score, feedback = parse_judgement(content)
    return score, feedback, metrics


def parse_judgement(text):
    """Pull {score, feedback} out of the judge's reply, tolerating stray prose."""
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            score = obj.get("score")
            score = float(score) if score is not None else None
            return score, str(obj.get("feedback", "")).strip()
        except Exception:
            pass
    return None, (text or "").strip()  # unparseable: keep raw text as feedback


# --------------------------------------------------------------------------- #
# The refine loop
# --------------------------------------------------------------------------- #
def refine(client, base_url, key, args, prompt, run_dir, system_prompt):
    iters_dir = run_dir / "iters"
    iters_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"

    best = {"i": None, "score": -1.0, "js": None, "png": None}
    tot_cost = 0.0

    def add_cost(m):
        nonlocal tot_cost
        c = m.get("cost_usd")
        if isinstance(c, (int, float)):
            tot_cost += c

    # Whether the editor is grounded on the reference photo too (separate from the
    # judge's --reference). Needs a reference image to actually be present.
    editor_ref = bool(args.editor_reference and args.reference
                      and Path(args.reference).exists())

    # --- iteration 0: initial generation -----------------------------------
    # Every editor turn is STATELESS: no history accumulates. The initial sketch
    # is a plain (contract + paint task) call; later revisions are rebuilt fresh.
    print("  [i0] generating initial sketch (%s)%s ..."
          % (args.editor, " +ref" if editor_ref else ""), flush=True)
    if editor_ref:
        init_user = {"role": "user",
                     "content": [{"type": "text", "text": user_prompt_for(prompt)}]
                                + reference_parts(args.reference)}
    else:
        init_user = {"role": "user", "content": user_prompt_for(prompt)}
    init_messages = [{"role": "system", "content": system_prompt}, init_user]
    content, gm, _ = chat(client, base_url, key, args.editor, init_messages,
                          args.temperature, args.max_tokens)
    add_cost(gm)
    if gm.get("error"):
        sys.exit("initial generation failed: " + gm["error"])
    code = extract_code(content)

    for i in range(args.iters):
        js_path = iters_dir / ("i%d.js" % i)
        png_path = iters_dir / ("i%d.png" % i)
        js_path.write_text(code, encoding="utf-8")

        ok, rms, rerr = render(js_path, png_path, args.render_timeout)
        if ok:
            score, feedback, jm = judge(client, base_url, key, args.judge,
                                        prompt, png_path, args.reference)
            add_cost(jm)
        else:
            # A broken render can't be judged; feed the error back as the critique.
            score, feedback, jm = None, ("Render FAILED: %s\nFix the code so it "
                                         "renders under the harness contract." % rerr), {}

        row = {
            "iter": i, "score": score, "feedback": feedback,
            "render_ok": ok, "render_ms": rms, "render_error": rerr,
            "code_lines": code.count("\n") + 1 if code else 0,
            "editor_time_s": gm.get("time_s"), "editor_cost": gm.get("cost_usd"),
            "editor_ctok": gm.get("completion_tokens"),
            "judge_time_s": jm.get("time_s"), "judge_cost": jm.get("cost_usd"),
            "js_file": str(js_path.relative_to(run_dir)),
            "png_file": str(png_path.relative_to(run_dir)) if ok else None,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        sc = ("%.1f" % score) if isinstance(score, (int, float)) else "n/a"
        print("  [i%d] render:%s  score:%s  %s"
              % (i, "ok" if ok else "FAIL", sc, (feedback or "")[:100]), flush=True)

        if ok and isinstance(score, (int, float)) and score > best["score"]:
            best = {"i": i, "score": score, "js": str(js_path), "png": str(png_path)}

        # Stop conditions.
        if ok and isinstance(score, (int, float)) and score >= args.threshold:
            print("  reached threshold %.1f at i%d." % (args.threshold, i))
            break
        if i == args.iters - 1:
            break

        # --- editor revision turn: fresh context, no history -----------------
        # Rebuilt from scratch each turn: the harness contract, the CURRENT code,
        # its render, and the CURRENT critique. Nothing from earlier iterations.
        refine_text = (
            "You are revising a p5.brush watercolour sketch. It must depict: %s.\n\n"
            "This is the CURRENT sketch source:\n\n```js\n%s\n```\n\n"
            "It renders to the image shown below. A critic scored it %s/10.\n"
            "Critic feedback: %s\n\n"
            "Produce a REVISED, complete p5.brush sketch that directly addresses "
            "this feedback. Keep the harness contract EXACTLY. Output ONLY the "
            "JavaScript source." % (prompt, code, sc, feedback)
        )
        user_parts = [{"type": "text", "text": refine_text}]
        if ok:
            user_parts.append(img_part(png_path))   # the "image shown below"
        if editor_ref:
            user_parts += reference_parts(args.reference)
        revise_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_parts},
        ]

        print("  [i%d] revising ..." % (i + 1), flush=True)
        content, gm, _ = chat(client, base_url, key, args.editor, revise_messages,
                              args.temperature, args.max_tokens)
        add_cost(gm)
        if gm.get("error"):
            print("  editor error at i%d: %s -- stopping." % (i + 1, gm["error"]))
            break
        code = extract_code(content)

    # --- persist best + summary --------------------------------------------
    if best["js"]:
        (run_dir / "best.js").write_text(
            Path(best["js"]).read_text(encoding="utf-8"), encoding="utf-8")
        (run_dir / "best.png").write_bytes(Path(best["png"]).read_bytes())
    write_summary(run_dir, prompt, best, tot_cost, args)
    return best, tot_cost


def write_summary(run_dir, prompt, best, tot_cost, args):
    rows = [json.loads(l) for l in
            (run_dir / "log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    lines = []
    lines.append("Refine run: %s" % run_dir.name)
    lines.append("prompt: %s" % prompt)
    lines.append("editor=%s  judge=%s  iters=%d  threshold=%.1f  reference=%s%s"
                 % (args.editor, args.judge, args.iters, args.threshold,
                    args.reference or "none",
                    "  (editor-grounded)" if args.editor_reference else ""))
    lines.append("")
    lines.append("%-5s %-7s %-8s %-8s  %s" % ("iter", "render", "score", "lines", "feedback"))
    lines.append("-" * 78)
    for r in rows:
        sc = ("%.1f" % r["score"]) if isinstance(r["score"], (int, float)) else "n/a"
        lines.append("%-5d %-7s %-8s %-8s  %s"
                     % (r["iter"], "ok" if r["render_ok"] else "FAIL", sc,
                        str(r["code_lines"]), (r["feedback"] or "")[:90]))
    lines.append("-" * 78)
    bi = best["i"]
    lines.append("BEST: i%s  score %.1f -> best.png"
                 % (bi, best["score"]) if bi is not None else "BEST: none (all renders failed)")
    lines.append("TOTAL cost $%.4f" % tot_cost)
    (run_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))


def resolve_prompt(args):
    if args.combo:
        combos = json.loads((HERE / "prompts.json").read_text(encoding="utf-8"))
        match = [c for c in combos["combos"] if c["id"] == args.combo]
        if not match:
            sys.exit("combo id '%s' not in prompts.json" % args.combo)
        return match[0]["prompt"]
    return args.prompt or "a koi fish in a small pond on a sunny day"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--combo", default=None, help="combo id from prompts.json")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--threshold", type=float, default=8.5)
    ap.add_argument("--editor", default="google/gemini-3.1-pro-preview")
    ap.add_argument("--judge", default="google/gemini-3.1-pro-preview")
    ap.add_argument("--reference", default=None, help="optional reference photo for the judge")
    ap.add_argument("--editor-reference", action="store_true",
                    help="also show the --reference photo to the editor (grounding), not just the judge")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=64000)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--render-timeout", type=int, default=90000)
    ap.add_argument("--out", default=str(HERE / "out_refine"))
    args = ap.parse_args()

    if args.editor_reference and not (args.reference and Path(args.reference).exists()):
        sys.exit("--editor-reference requires --reference to point at an existing image.")

    key, base_url = load_env()
    system_prompt = (HERE / "system_prompt.txt").read_text(encoding="utf-8")
    prompt = resolve_prompt(args)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out) / stamp

    print("Run dir: %s" % run_dir)
    print("Prompt : %s" % prompt)
    print("Editor : %s   Judge: %s   iters=%d threshold=%.1f\n"
          % (args.editor, args.judge, args.iters, args.threshold))

    timeout = httpx.Timeout(args.timeout, connect=30.0)
    with httpx.Client(timeout=timeout) as client:
        best, tot_cost = refine(client, base_url, key, args, prompt, run_dir, system_prompt)

    print("\nDone. Best: %s" % (run_dir / "best.png" if best["js"] else "none"))
    print("Log: %s" % (run_dir / "log.jsonl"))


if __name__ == "__main__":
    main()
