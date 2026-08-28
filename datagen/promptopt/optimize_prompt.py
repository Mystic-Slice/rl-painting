#!/usr/bin/env python
"""
datagen/promptopt/optimize_prompt.py

GEPA (optimize_anything) optimization of datagen/system_prompt.txt.

The pipeline, per evaluation:

    candidate system prompt
        -> GENERATOR model writes a complete p5.brush sketch (OpenRouter)
        -> render/render.js renders it to a 600x600 PNG (headless Chrome)
        -> a Gemini-Flash VLM JUDGE rates the PNG on several criteria (OpenRouter)
        -> gated composite score in [0,1]  (render/compile failure = 0.0)

GEPA's reflection ("proposer") LM then reads the judge's per-criterion feedback +
the code the prompt produced, and rewrites the system prompt. Over many rounds it
evolves a prompt that makes the generator paint better watercolours.

    CANDIDATE  : the system prompt (a single string; seed = datagen/system_prompt.txt)
    DATASET    : scene prompts from datagen/prompts.json  (train / val split)
    SCORE      : weighted mean of the judge's 5 criteria, gated on a successful render
    OBJECTIVES : the 5 criteria are also returned under info["scores"] for GEPA's
                 objective-level Pareto frontier
    JUDGE      : google/gemini-3.7-flash (VLM), via OpenRouter
    REFLECTION : the proposer LM, via OpenRouter (a custom LM shim on your one key)

Everything routes through OpenRouter using the single key in ../.env.

Usage
-----
    # 0. one-eval self-test of the generate->render->judge chain (no GEPA, ~1 call):
    python datagen/promptopt/optimize_prompt.py --test-eval

    # 1. smoke test: tiny budget, validates the whole GEPA loop end-to-end:
    python datagen/promptopt/optimize_prompt.py --smoke

    # 2. real run (size max-evals ~= 15-20 x len(valset)):
    python datagen/promptopt/optimize_prompt.py --max-evals 200
"""
import argparse
import json
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent          # datagen/promptopt
DATAGEN = HERE.parent                             # datagen
ROOT = DATAGEN.parent                             # rl-painting

# Reuse datagen's plumbing as the single source of truth for OpenRouter + render.
sys.path.insert(0, str(DATAGEN))
from generate import load_env, extract_code, user_prompt_for   # noqa: E402
from refine import chat, render, img_part                      # noqa: E402

from gepa.optimize_anything import optimize_anything, OptimizeAnythingConfig  # noqa: E402
from gepa.optimize_anything import log as oa_log                              # noqa: E402


# --------------------------------------------------------------------------- #
# Cost / time tracker -- accumulates per pipeline stage across all threads
# --------------------------------------------------------------------------- #
class Tracker:
    """Thread-safe accumulator of calls / wall-time / $ / tokens per stage.

    Stages: 'generation' and 'judge' are eval-pipeline spend (OpenRouter);
    'render' is local compute (no $, time only); 'reflection' is GEPA's own
    proposer-LM spend.
    """
    STAGES = ("generation", "render", "judge", "reflection")

    def __init__(self, save_path=None, save_interval=3.0):
        self._lock = threading.Lock()
        self._d = {s: {"calls": 0, "time_s": 0.0, "cost_usd": 0.0,
                       "tokens": 0, "errors": 0} for s in self.STAGES}
        self.save_path = Path(save_path) if save_path else None
        self._save_interval = save_interval
        self._last_save = 0.0

    def load(self):
        """Seed counters from an existing snapshot (for --resume). Returns True if loaded."""
        if not (self.save_path and self.save_path.exists()):
            return False
        try:
            data = json.loads(self.save_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        with self._lock:
            for s in self.STAGES:
                if isinstance(data.get(s), dict):
                    for k in self._d[s]:
                        self._d[s][k] = data[s].get(k, self._d[s][k])
        return True

    def add(self, stage, time_s=0.0, cost=None, tokens=None, error=False):
        with self._lock:
            d = self._d[stage]
            d["calls"] += 1
            d["time_s"] += float(time_s or 0.0)
            d["cost_usd"] += float(cost or 0.0)
            d["tokens"] += int(tokens or 0)
            d["errors"] += 1 if error else 0
            self._maybe_save_locked()

    def record_metrics(self, stage, metrics):
        """Record an OpenRouter chat() metrics dict."""
        self.add(stage, time_s=metrics.get("time_s"), cost=metrics.get("cost_usd"),
                 tokens=metrics.get("completion_tokens"),
                 error=bool(metrics.get("error")))

    def _write_locked(self):
        snap = {s: dict(v) for s, v in self._d.items()}
        tmp = Path(str(self.save_path) + ".tmp")
        tmp.write_text(json.dumps(snap, indent=2), encoding="utf-8")
        tmp.replace(self.save_path)   # atomic on the same filesystem

    def _maybe_save_locked(self):
        if not self.save_path:
            return
        now = time.time()
        if now - self._last_save < self._save_interval:
            return
        self._last_save = now
        try:
            self._write_locked()
        except Exception:
            pass   # cost bookkeeping must never break the run

    def save(self):
        """Force a snapshot to disk (e.g. at end of run)."""
        if not self.save_path:
            return
        with self._lock:
            self._last_save = time.time()
            try:
                self._write_locked()
            except Exception:
                pass

    def snapshot(self):
        with self._lock:
            return {s: dict(v) for s, v in self._d.items()}

    def report(self):
        snap = self.snapshot()
        pipeline_cost = sum(snap[s]["cost_usd"] for s in ("generation", "judge"))
        total_cost = pipeline_cost + snap["reflection"]["cost_usd"]
        lines = []
        hdr = "%-12s %6s %10s %9s %11s %9s %7s" % (
            "stage", "calls", "time_s", "avg_s", "cost_usd", "tokens", "errors")
        lines.append(hdr)
        lines.append("-" * len(hdr))
        for s in self.STAGES:
            d = snap[s]
            avg = d["time_s"] / d["calls"] if d["calls"] else 0.0
            lines.append("%-12s %6d %10.1f %9.2f %11.4f %9d %7d" % (
                s, d["calls"], d["time_s"], avg, d["cost_usd"], d["tokens"], d["errors"]))
        lines.append("-" * len(hdr))
        lines.append("pipeline (generation+judge) OpenRouter cost: $%.4f" % pipeline_cost)
        lines.append("reflection (GEPA proposer) cost:             $%.4f"
                     % snap["reflection"]["cost_usd"])
        lines.append("GRAND TOTAL OpenRouter cost:                 $%.4f" % total_cost)
        return "\n".join(lines), snap, total_cost


# --------------------------------------------------------------------------- #
# Multi-criteria VLM judge -- ONE focused API call per criterion
# --------------------------------------------------------------------------- #
# Rather than overwhelm the judge with a 5-part rubric in a single call, each
# criterion gets its own dedicated, single-focus judge call (run concurrently).
# The composite is the weighted mean of the per-criterion 0-10 ratings, normalised
# to [0,1]. Weights mirror the reward intent in docs/plan.md: subject fidelity +
# painterly watercolour quality dominate, composition / colour / overall fill in.
CRITERIA_WEIGHTS = {
    "subject_recognisability": 0.35,   # is it clearly the requested animal + scene?
    "painterly_looseness":     0.15,   # soft bleeding washes, layered & confident (NOT stiff/muddy)
    "composition":             0.15,   # focal clarity, balance, use of the frame
    "colour_harmony":          0.15,   # a harmonious, intentional limited palette
    "overall_aesthetic":       0.20,   # does it read at a glance as an accomplished painting?
}
CRITERIA = list(CRITERIA_WEIGHTS)

# Each criterion's dedicated instruction: the judge sees ONLY this one question.
CRITERION_GUIDANCE = {
    "subject_recognisability":
        "SUBJECT RECOGNISABILITY: how clearly and correctly does the painting depict the "
        "requested animal AND scene? A viewer who was not told the subject should still be "
        "able to name the animal. Penalise wrong/ambiguous species, missing scene elements, "
        "and shapes that don't read as the animal at all.",
    "painterly_looseness":
        "PAINTERLY LOOSENESS: judge ONLY the watercolour quality of the mark-making. Reward "
        "soft bleeding washes, confident layered translucent shapes, and organic edges. "
        "Penalise stiff mechanical outlines, flat vector geometry, hard pencil linework, and "
        "muddy undifferentiated blobs. Ignore whether the subject is correct here.",
    "composition":
        "COMPOSITION: judge ONLY focal clarity, balance, and use of the frame. Is there a "
        "clear focal subject well placed in the 600x600 canvas, with a considered relationship "
        "between subject and negative space? Penalise empty, cramped, or aimless layouts.",
    "colour_harmony":
        "COLOUR HARMONY: judge ONLY the palette. Reward a harmonious, intentional, limited set "
        "of colours that work together for the scene. Penalise garish, muddy, random, or "
        "clashing colour choices.",
    "overall_aesthetic":
        "OVERALL AESTHETIC: your holistic gut reaction. Does this read at a glance as an "
        "accomplished, appealing watercolour painting that someone would be happy to have made?",
}

JUDGE_ONE_SYSTEM = (
    "You are a meticulous, fair-but-harsh art critic evaluating a WATERCOLOUR painting that "
    "was produced procedurally by code. You are shown ONE painting, told what it is meant to "
    "depict, and asked about ONE specific quality only. Focus solely on that quality.\n\n"
    "{guidance}\n\n"
    "Most procedurally-generated attempts are mediocre: reserve 8+ for genuinely accomplished "
    "work on THIS quality, and do not be shy about low scores.\n\n"
    "Respond with ONLY a JSON object, no prose, no code fence:\n"
    '{{"score": <number 0-10, one decimal>, "feedback": "<1-3 sentences of concrete, '
    "actionable critique on THIS quality, aimed at whoever writes the generator's "
    'INSTRUCTIONS: what to change so the next painting scores higher here>"}}'
)


def _parse_one(text):
    """Pull (score, feedback) from a single-criterion judge reply.

    Tolerates truncated JSON (a verbose reply cut off at the token cap): first try
    strict JSON, then salvage the score number + whatever feedback text is present.
    """
    t = text or ""
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return max(0.0, min(10.0, float(obj["score"]))), str(obj.get("feedback", "")).strip()
        except Exception:
            pass
    sm = re.search(r'"?score"?\s*[:=]\s*(-?\d+(?:\.\d+)?)', t, re.IGNORECASE)
    if not sm:
        return None, t.strip()
    score = max(0.0, min(10.0, float(sm.group(1))))
    fm = re.search(r'"feedback"\s*:\s*"(.*)', t, re.IGNORECASE | re.DOTALL)
    fb = fm.group(1).strip().rstrip("}").rstrip('"').strip() if fm else t.strip()
    return score, fb


def judge_criterion(client, cfg, scene_prompt, png_path, criterion):
    """One dedicated judge call for a single criterion. Returns (score_0to10|None, feedback)."""
    system = JUDGE_ONE_SYSTEM.format(guidance=CRITERION_GUIDANCE[criterion])
    parts = [
        {"type": "text",
         "text": "This painting is meant to depict: %s.\nRate it on the quality described."
                 % scene_prompt},
        img_part(png_path),
    ]
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": parts}]
    content, metrics, _ = chat(client, cfg.base_url, cfg.key, cfg.judge, messages,
                               temperature=0.2, max_tokens=2000)
    cfg.tracker.record_metrics("judge", metrics)
    if metrics.get("error"):
        return None, "JUDGE ERROR: " + str(metrics["error"])
    return _parse_one(content)


def judge_image(client, cfg, scene_prompt, png_path):
    """Rate one render across all criteria, one focused call each (concurrent).

    Returns (composite_0to1|None, scores_0to10 dict, per_criterion_feedback dict).
    If some criteria fail, the composite renormalises over those that succeeded;
    only if ALL fail is the composite None.
    """
    scores, feedback = {}, {}
    with ThreadPoolExecutor(max_workers=len(CRITERIA)) as pool:
        futs = {pool.submit(judge_criterion, client, cfg, scene_prompt, png_path, c): c
                for c in CRITERIA}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                s, fb = fut.result()
            except Exception as e:
                s, fb = None, "judge exception: %s" % e
            scores[c] = s
            feedback[c] = fb

    got = {c: scores[c] for c in CRITERIA if scores[c] is not None}
    if not got:
        return None, scores, feedback
    wsum = sum(CRITERIA_WEIGHTS[c] for c in got)
    composite = sum(CRITERIA_WEIGHTS[c] * got[c] / 10.0 for c in got) / wsum
    return composite, scores, feedback


# --------------------------------------------------------------------------- #
# OpenRouter reflection ("proposer") LM  -- a GEPA LM-protocol callable
# --------------------------------------------------------------------------- #
class OpenRouterLM:
    """GEPA reflection LM over OpenRouter. __call__(prompt) -> completion text.

    `prompt` may be a plain string or an already-formed messages list; both are
    accepted so this works with any GEPA reflection-prompt shape.
    """
    def __init__(self, client, base_url, key, model, tracker,
                 temperature=1.0, max_tokens=32000):
        self.client, self.base_url, self.key = client, base_url, key
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.tracker = tracker

    def __call__(self, prompt):
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        content, metrics, _ = chat(self.client, self.base_url, self.key, self.model,
                                   messages, self.temperature, self.max_tokens)
        self.tracker.record_metrics("reflection", metrics)
        if metrics.get("error"):
            raise RuntimeError("reflection LM error: " + str(metrics["error"]))
        return content


# --------------------------------------------------------------------------- #
# The evaluator: candidate prompt + scene -> (score, feedback)
# --------------------------------------------------------------------------- #
def _code_excerpt(code, head=1600, tail=800):
    if len(code) <= head + tail:
        return code
    return code[:head] + "\n\n/* ...(%d chars elided)... */\n\n" % (len(code) - head - tail) + code[-tail:]


def make_evaluator(client, cfg):
    counter = {"n": 0}
    lock = threading.Lock()

    def evaluate(candidate, example):
        scene = example["prompt"]
        cid = example["id"]
        with lock:
            counter["n"] += 1
            n = counter["n"]
        uid = "%s_%s" % (cid, uuid.uuid4().hex[:8])

        base = {"combo": cid, "scene": scene}

        # 1. GENERATE -------------------------------------------------------- #
        messages = [{"role": "system", "content": candidate},
                    {"role": "user", "content": user_prompt_for(scene)}]
        content, gm, _ = chat(client, cfg.base_url, cfg.key, cfg.generator,
                              messages, cfg.gen_temperature, cfg.gen_max_tokens)
        cfg.tracker.record_metrics("generation", gm)
        if gm.get("error"):
            return 0.0, {**base, "score": 0.0, "stage": "generation",
                         "error": gm["error"],
                         "note": "The generator call failed; no image to judge."}
        code = extract_code(content)
        if not code.strip():
            return 0.0, {**base, "score": 0.0, "stage": "generation",
                         "error": "generator returned no JavaScript code",
                         "raw_head": (content or "")[:600],
                         "note": "The prompt must make the model output ONLY a p5.brush sketch."}

        # 2. RENDER (this is the compile gate) ------------------------------- #
        js_path = cfg.renders_dir / (uid + ".js")
        png_path = cfg.renders_dir / (uid + ".png")
        js_path.write_text(code, encoding="utf-8")
        ok, rms, rerr = render(js_path, png_path, cfg.render_timeout)
        cfg.tracker.add("render", time_s=(rms or 0) / 1000.0, error=not ok)
        if not ok:
            # A broken render can't be judged. Score 0 and feed the error + code back:
            # this is exactly the signal GEPA needs to tighten the harness contract.
            return 0.0, {**base, "score": 0.0, "stage": "render",
                         "render_error": rerr,
                         "code_excerpt": _code_excerpt(code),
                         "note": "The sketch failed to render under the harness contract. "
                                 "The system prompt should prevent this class of error."}

        # 3. JUDGE (one focused call per criterion) -------------------------- #
        composite, scores, crit_fb = judge_image(client, cfg, scene, png_path)
        if composite is None:
            return 0.0, {**base, "score": 0.0, "stage": "judge",
                         "error": "; ".join("%s: %s" % (k, crit_fb.get(k)) for k in CRITERIA),
                         "png": str(png_path),
                         "note": "Rendered OK but every criterion judge call failed."}

        # A combined, per-criterion critique is the richest ASI for the proposer.
        critique = "\n".join(
            "- %s (%s/10): %s"
            % (k, ("%.1f" % scores[k]) if scores[k] is not None else "n/a", crit_fb.get(k, ""))
            for k in CRITERIA)

        oa_log("[eval %d] %-22s render:%dms  score:%.3f  (%s)"
               % (n, cid, rms, composite,
                  " ".join("%s=%s" % (k[:4], ("%.1f" % scores[k]) if scores[k] is not None else "-")
                           for k in CRITERIA)))

        info = {
            **base,
            "score": composite,
            "stage": "judged",
            # per-criterion metrics (0-1) -> GEPA objective-level Pareto frontier
            "scores": {k: scores[k] / 10.0 for k in CRITERIA if scores[k] is not None},
            "criterion_ratings_0to10": scores,
            "critique": critique,
            "render_ms": rms,
            "code_lines": code.count("\n") + 1,
            "code_excerpt": _code_excerpt(code),
            "png": str(png_path),
        }
        return composite, info

    return evaluate


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def load_examples():
    data = json.loads((DATAGEN / "prompts.json").read_text(encoding="utf-8"))
    return [{"id": c["id"], "animal": c["animal"], "prompt": c["prompt"]}
            for c in data["combos"]]


def split_examples(examples, n_val, n_test=0, seed=0):
    """Deterministic shuffle, then carve off test (reporting-only) + val (selection);
    the remainder is train. Returns (train, val, test)."""
    import random
    idx = list(range(len(examples)))
    random.Random(seed).shuffle(idx)
    shuffled = [examples[i] for i in idx]
    test = shuffled[:n_test]
    val = shuffled[n_test:n_test + n_val]
    train = shuffled[n_test + n_val:]
    return train, val, test


# --------------------------------------------------------------------------- #
# Config object threaded into the evaluator/judge
# --------------------------------------------------------------------------- #
class Cfg:
    pass


def build_cfg(args, key, base_url, renders_dir, tracker):
    c = Cfg()
    c.key, c.base_url = key, base_url
    c.generator = args.generator
    c.judge = args.judge
    c.gen_temperature = args.gen_temperature
    c.gen_max_tokens = args.gen_max_tokens
    c.render_timeout = args.render_timeout
    c.renders_dir = renders_dir
    c.tracker = tracker
    return c


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # models (all OpenRouter slugs)
    ap.add_argument("--generator", default="openai/gpt-5.6-sol",
                    help="model whose system prompt is optimized (writes the sketches)")
    ap.add_argument("--judge", default="google/gemini-3.7-flash",
                    help="VLM judge that rates the renders")
    ap.add_argument("--reflection", default="openai/gpt-5.6-sol",
                    help="GEPA proposer LM that rewrites the prompt")
    # generation params
    ap.add_argument("--gen-temperature", type=float, default=0.85)
    ap.add_argument("--gen-max-tokens", type=int, default=16000)
    ap.add_argument("--render-timeout", type=int, default=90000)
    # optimization budget / shape
    ap.add_argument("--max-evals", type=int, default=None,
                    help="server-side eval-call cap (default: 200 real, 12 smoke)")
    ap.add_argument("--max-concurrency", type=int, default=None)
    ap.add_argument("--n-val", type=int, default=None,
                    help="held-out combos used for candidate selection (default: 8 real, 2 smoke)")
    ap.add_argument("--n-test", type=int, default=None,
                    help="combos held out for reporting-only before/after scoring "
                         "(default: 6 real, 0 smoke)")
    ap.add_argument("--reflection-minibatch", type=int, default=None)
    ap.add_argument("--reflection-max-tokens", type=int, default=32000)
    ap.add_argument("--stop-at-score", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    # modes
    ap.add_argument("--smoke", action="store_true",
                    help="tiny budget end-to-end validation of the GEPA loop")
    ap.add_argument("--test-eval", action="store_true",
                    help="run ONE evaluation of the seed prompt (no GEPA) and print it")
    ap.add_argument("--resume", default=None,
                    help="resume an interrupted run: path to an existing runs/<...> dir "
                         "(GEPA reloads its state; cost totals continue from tracker.json). "
                         "Re-pass the SAME --smoke / budget / model flags as the original run.")
    ap.add_argument("--out", default=str(HERE / "runs"))
    args = ap.parse_args()

    key, base_url = load_env()
    seed_prompt = (DATAGEN / "system_prompt.txt").read_text(encoding="utf-8")
    examples = load_examples()

    resuming = bool(args.resume)
    if resuming:
        run_root = Path(args.resume)
        if not run_root.exists():
            sys.exit("--resume path does not exist: %s" % run_root)
        stamp = run_root.name.replace("smoke_", "")
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = Path(args.out) / (("smoke_" if args.smoke else "") + stamp)
    renders_dir = run_root / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    # generous, shared, thread-safe httpx client (renders dominate wall-time anyway)
    client = httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0))
    # persistent, resume-aware cost/time tracker (autosaves to run_root/tracker.json)
    tracker = Tracker(save_path=run_root / "tracker.json")
    if resuming and tracker.load():
        print("RESUME: reloaded cost totals from %s" % (run_root / "tracker.json"))
    cfg = build_cfg(args, key, base_url, renders_dir, tracker)
    evaluate = make_evaluator(client, cfg)

    # --- one-eval self-test: prove the chain before spending on GEPA -------- #
    if args.test_eval:
        ex = examples[0]
        print("TEST EVAL  generator=%s  judge=%s" % (args.generator, args.judge))
        print("scene: %s\n(generating -> rendering -> judging; this takes a minute)\n"
              % ex["prompt"])
        score, info = evaluate(seed_prompt, ex)
        print("score: %.3f   stage: %s" % (score, info.get("stage")))
        if info.get("criterion_ratings_0to10"):
            print("criteria:", json.dumps(info["criterion_ratings_0to10"]))
        for k in ("render_error", "error", "critique", "png"):
            if info.get(k):
                print("%s: %s" % (k, str(info[k])[:400]))
        rep, _, _ = tracker.report()
        print("\n--- cost/time for this single eval ---\n" + rep)
        client.close()
        return

    # --- budget / split defaults per mode ----------------------------------- #
    if args.smoke:
        n_val = args.n_val if args.n_val is not None else 2
        n_test = args.n_test if args.n_test is not None else 0
        max_evals = args.max_evals if args.max_evals is not None else 12
        max_conc = args.max_concurrency if args.max_concurrency is not None else 2
    else:
        n_val = args.n_val if args.n_val is not None else 8
        n_test = args.n_test if args.n_test is not None else 6
        max_evals = args.max_evals if args.max_evals is not None else 160
        max_conc = args.max_concurrency if args.max_concurrency is not None else 6

    train, val, test = split_examples(examples, n_val, n_test, seed=args.seed)
    if args.smoke:
        train = train[:2]   # keep the smoke tiny on the train side too
    minibatch = args.reflection_minibatch or min(3, len(train))

    reflection_lm = OpenRouterLM(client, base_url, key, args.reflection, tracker,
                                 temperature=1.0, max_tokens=args.reflection_max_tokens)

    print("=" * 72)
    print("GEPA system-prompt optimization%s" % ("  [RESUMING]" if resuming else ""))
    print("  generator : %s" % args.generator)
    print("  judge     : %s  (criteria: %s)" % (args.judge, ", ".join(CRITERIA)))
    print("  reflection: %s" % args.reflection)
    print("  train=%d  val=%d  max_evals=%d  concurrency=%d  minibatch=%d"
          % (len(train), len(val), max_evals, max_conc, minibatch))
    print("  train combos: %s" % ", ".join(e["id"] for e in train))
    print("  val   combos: %s" % ", ".join(e["id"] for e in val))
    print("  test  combos: %s" % (", ".join(e["id"] for e in test) if test else "(none)"))
    print("  run dir   : %s" % run_root)
    if resuming:
        print("  RESUME    : GEPA reloads gepa_state.bin from run dir; budget/split flags "
              "must match the original run.")
    else:
        print("  (interrupt-safe: re-run with  --resume %s  to continue; or drop a "
              "'gepa.stop' file in %s/gepa to stop gracefully)" % (run_root, run_root))
    print("=" * 72)

    config = OptimizeAnythingConfig(
        engine="gepa",
        name=("smoke_" if args.smoke else "") + "p5brush_prompt_" + stamp,
        max_evals=max_evals,
        max_concurrency=max_conc,
        stop_at_score=args.stop_at_score,
        run_dir=str(run_root / "gepa"),
        output_dir=str(run_root / "evalserver"),
        engine_config={
            "reflection": {
                "reflection_lm": reflection_lm,          # our OpenRouter shim
                "reflection_minibatch_size": minibatch,
            },
            "engine": {
                "max_workers": max_conc,
                "seed": args.seed,
                "raise_on_exception": False,             # a bad eval -> score 0, never abort
            },
        },
    )

    t0 = time.time()
    result = None
    try:
      result = optimize_anything(
        seed_candidate=seed_prompt,
        evaluator=evaluate,
        dataset=train,
        valset=val,
        test_set=(test or None),
        objective=("Rewrite the SYSTEM PROMPT given to a code model so that the p5.brush "
                   "JavaScript sketches it produces render into loose, expressive, "
                   "recognisable WATERCOLOUR paintings of the requested animal and scene. "
                   "Maximise the judge's composite score across the criteria."),
        background=("The system prompt is injected into a fixed headless-Chrome p5.brush "
                    "harness. The model must output ONLY a self-contained sketch defining "
                    "global setup()/draw() and obeying the harness contract, or the render "
                    "fails and scores 0. Feedback includes the judge's per-criterion ratings, "
                    "its written critique, and (on failures) the render error and the code the "
                    "prompt produced. Keep the strict harness rules intact while improving the "
                    "artistic guidance."),
        config=config,
      )
    finally:
        # Always persist + report cost/time, even if the run errored partway through.
        tracker.save()
        wall_s = time.time() - t0
        rep, snap, total_cost = tracker.report()
        print("\n" + "=" * 72)
        print("PIPELINE COST / TIME BREAKDOWN   (wall %.1fs = %.1f min)"
              % (wall_s, wall_s / 60.0))
        print(rep)
        print("=" * 72)

    if result is None:
        client.close()
        return

    # --- persist the winner ------------------------------------------------- #
    best_path = run_root / "best_system_prompt.txt"
    best_path.write_text(result.best_candidate, encoding="utf-8")
    summary = {
        "best_score": result.best_score,
        "total_evals": result.total_evals,
        "n_candidates": len(result.candidates),
        "generator": args.generator, "judge": args.judge, "reflection": args.reflection,
        "train": [e["id"] for e in train], "val": [e["id"] for e in val],
        "test": [e["id"] for e in test],
        "seed_test_score": result.metadata.get("baseline_test_score"),
        "best_test_score": result.metadata.get("test_score"),
        "seed_test_scores": result.metadata.get("baseline_test_scores"),
        "best_test_scores": result.metadata.get("test_scores"),
        "gepa_metadata": {k: result.metadata.get(k) for k in
                          ("total_cost", "adapter_cost", "wall_time", "engine", "output_dir")},
        "wall_time_s": round(wall_s, 1),
        "cost_time_by_stage": snap,
        "total_openrouter_cost_usd": round(total_cost, 4),
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDONE.  best val score: %.4f   over %d candidates, %d evals"
          % (result.best_score, len(result.candidates), result.total_evals))
    seed_t, best_t = result.metadata.get("baseline_test_score"), result.metadata.get("test_score")
    if seed_t is not None and best_t is not None:
        print("HELD-OUT TEST:  seed %.4f  ->  optimized %.4f   (%+.4f)"
              % (seed_t, best_t, best_t - seed_t))
    print("Seed prompt kept?  %s" % (result.best_candidate.strip() == seed_prompt.strip()))
    print("Best system prompt -> %s" % best_path)
    print("Summary + costs    -> %s" % (run_root / "summary.json"))
    print("GEPA artifacts     -> %s" % (run_root / "gepa"))
    print("Renders            -> %s" % renders_dir)
    client.close()


if __name__ == "__main__":
    main()
