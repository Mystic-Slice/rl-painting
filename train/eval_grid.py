"""train/eval_grid.py -- offline before/after eval on the held-out test combos.

Samples one sketch per test combo from a base model or a trained checkpoint,
renders each, and scores it with the existing 5-criterion GEPA judge (eval-only,
never part of the RL reward). Writes PNGs + a scores.json + a printed summary so a
checkpoint can be compared against the base model.

Examples
--------
Base model:
    .venv/Scripts/python.exe -m train.eval_grid model_name=Qwen/Qwen3.5-4B out=runs/eval/base
Trained checkpoint (tinker sampler path from a saved checkpoint):
    .venv/Scripts/python.exe -m train.eval_grid model_name=Qwen/Qwen3.5-4B \
        checkpoint_path=tinker://... out=runs/eval/step50
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import chz
import httpx
import tinker

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinker_cookbook import model_info  # noqa: E402
from tinker_cookbook.completers import TinkerMessageCompleter  # noqa: E402
from tinker_cookbook.renderers import get_renderer  # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402

from reward.compat import (  # noqa: E402
    CRITERIA_WEIGHTS,
    extract_code,
    judge_image,
    load_env,
    load_examples,
    split_examples,
    user_prompt_for,
)
from reward.render_bridge import close_render_bridge, get_render_bridge  # noqa: E402
from reward.tracking import get_tracker  # noqa: E402
from train.env import DEFAULT_SYSTEM_PROMPT  # noqa: E402


@chz.chz
class EvalConfig:
    model_name: str = "Qwen/Qwen3.5-9B"
    checkpoint_path: str | None = None      # tinker sampler path; None -> base model
    renderer_name: str | None = None
    out: str | None = None
    n_test: int = 6
    seed: int = 0
    max_tokens: int = 16384
    temperature: float = 1.0
    judge_model: str = "google/gemini-3.7-flash"
    system_prompt_path: str = DEFAULT_SYSTEM_PROMPT
    base_url: str | None = None


def _ensure_tinker_key() -> None:
    import os
    if os.environ.get("TINKER_API_KEY"):
        return
    envfile = ROOT / ".env"
    if not envfile.exists():
        return
    for line in envfile.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            if k.strip() in ("TINKER_API_KEY", "TINKER_KEY") and v.strip():
                os.environ["TINKER_API_KEY"] = v.strip().strip('"').strip("'")
                return


async def main(cfg: EvalConfig):
    _ensure_tinker_key()
    key, base_url = load_env()
    combos = load_examples()
    _train, _val, test = split_examples(combos, n_val=0, n_test=cfg.n_test, seed=cfg.seed)
    if not test:
        print("no test combos (n_test=0)")
        return

    out_dir = Path(cfg.out or ("runs/eval/%s" % datetime.now().strftime("%Y%m%d_%H%M%S")))
    out_dir.mkdir(parents=True, exist_ok=True)

    renderer_name = cfg.renderer_name or model_info.get_recommended_renderer_name(cfg.model_name)
    tokenizer = get_tokenizer(cfg.model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    service = tinker.ServiceClient(base_url=cfg.base_url)
    if cfg.checkpoint_path:
        sampling_client = service.create_sampling_client(model_path=cfg.checkpoint_path)
    else:
        sampling_client = service.create_sampling_client(base_model=cfg.model_name)
    completer = TinkerMessageCompleter(sampling_client, renderer, max_tokens=cfg.max_tokens,
                                       temperature=cfg.temperature)

    system_prompt = Path(cfg.system_prompt_path).read_text(encoding="utf-8")
    http = httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0))
    jcfg = SimpleNamespace(base_url=base_url, key=key, judge=cfg.judge_model,
                           tracker=get_tracker())
    bridge = get_render_bridge()

    async def eval_one(combo: dict) -> dict:
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_for(combo["prompt"])}]
        msg = await completer(messages)
        content = msg["content"] if isinstance(msg["content"], str) else \
            "".join(p.get("text", "") for p in msg["content"] if isinstance(p, dict))
        code = extract_code(content)
        row = {"id": combo["id"], "prompt": combo["prompt"], "code_lines": code.count("\n") + 1}
        if not code.strip():
            row.update(score=0.0, stage="generation", note="no code")
            return row
        png = out_dir / ("%s.png" % combo["id"])
        (out_dir / ("%s.js" % combo["id"])).write_text(code, encoding="utf-8")
        res = await bridge.render(code, png)
        row["render_ok"] = res.ok
        if not res.ok:
            row.update(score=0.0, stage="render", render_error=res.error)
            return row
        composite, scores, _fb = await asyncio.to_thread(
            judge_image, http, jcfg, combo["prompt"], str(png))
        row.update(score=composite if composite is not None else 0.0,
                   stage="judged", criteria=scores, png=str(png))
        return row

    rows = await asyncio.gather(*[eval_one(c) for c in test])
    mean = sum(r.get("score") or 0.0 for r in rows) / len(rows)

    (out_dir / "scores.json").write_text(
        json.dumps({"model": cfg.model_name, "checkpoint": cfg.checkpoint_path,
                    "mean_score": mean, "criteria_weights": CRITERIA_WEIGHTS,
                    "rows": rows}, indent=2), encoding="utf-8")
    http.close()
    await close_render_bridge()

    print("\nEval grid: %s   (checkpoint=%s)" % (cfg.model_name, cfg.checkpoint_path or "base"))
    for r in rows:
        print("  %-24s score=%.3f  %s" % (r["id"], r.get("score") or 0.0, r.get("stage")))
    print("  MEAN composite score: %.4f" % mean)
    print("  -> %s" % (out_dir / "scores.json"))


if __name__ == "__main__":
    import os
    if sys.platform == "win32" and os.environ.get("PYTHONUTF8") != "1":
        os.environ["PYTHONUTF8"] = "1"
        os.execv(sys.executable, [sys.executable, "-X", "utf8",
                                  os.path.abspath(__file__), *sys.argv[1:]])
    cfg = chz.entrypoint(EvalConfig)
    asyncio.run(main(cfg))
