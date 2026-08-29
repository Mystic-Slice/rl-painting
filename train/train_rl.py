"""train/train_rl.py -- GRPO training driver for p5.brush watercolour code gen.

Adapts the cookbook's math_rl CLI pattern: a chz CLIConfig is turned into an
`rl.train.Config` wired to our PaintRLDatasetBuilder, then `rl.train.main(config)`
runs the loop (it handles checkpoint resume from log_path automatically).

Reward = compile gate + length penalty (Env.step) + holistic aesthetic judge +
pairwise judge vs references (compute_group_rewards). Cost is tracked to
<log_path>/tracker.json and the render bridge / OpenRouter client are closed on
exit.

Examples
--------
Smoke (no judge spend, tiny, 3 steps):
    .venv/Scripts/python.exe -m train.train_rl smoke=True log_path=runs/smoke

MVP (holistic + pairwise, needs refpool filled):
    .venv/Scripts/python.exe -m train.train_rl log_path=runs/mvp \
        group_size=8 groups_per_batch=8 max_steps=50

Resume: re-run with the SAME log_path.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

import chz

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinker_cookbook import checkpoint_utils, cli_utils, hyperparam_utils  # noqa: E402
from tinker_cookbook.rl.train import Config, KLReferenceConfig, main  # noqa: E402

from reward.orclient import close_orclient  # noqa: E402
from reward.render_bridge import close_render_bridge  # noqa: E402
from reward.tracking import init_tracker  # noqa: E402
from train.env import (  # noqa: E402
    DEFAULT_FEWSHOT,
    DEFAULT_REFPOOL,
    DEFAULT_SYSTEM_PROMPT,
    PaintRLDatasetBuilder,
)

logger = logging.getLogger(__name__)


def _ensure_tinker_key() -> None:
    """The Tinker SDK reads TINKER_API_KEY; our .env stores it as TINKER_KEY.

    Bridge the two (without overriding an already-set TINKER_API_KEY) so the
    ServiceClient authenticates from the project's single .env.
    """
    import os

    if os.environ.get("TINKER_API_KEY"):
        return
    envfile = ROOT / ".env"
    if not envfile.exists():
        return
    for line in envfile.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k in ("TINKER_API_KEY", "TINKER_KEY") and v:
            os.environ["TINKER_API_KEY"] = v
            return


@chz.chz
class CLIConfig:
    # model / run
    model_name: str = "Qwen/Qwen3.5-9B"
    lora_rank: int = 32
    # Thinking re-enabled after run 3 (run 1, the only thinking-on run, had by far
    # the best aesthetic; the run-3 collapse was a buggy-pattern lock-in that a
    # reasoning pass can catch). NOTE: with the cap still at 8k, long thinking can
    # truncate (-> r_fail); watch env/all/format_ok.
    # Pass renderer_name=qwen3_5_disable_thinking to turn thinking back off.
    renderer_name: str | None = "qwen3_5"
    load_checkpoint_path: str | None = None
    log_path: str | None = None
    seed: int = 0

    # training hyperparameters
    group_size: int = 8
    groups_per_batch: int = 8
    # ~1/3 of hyperparam_utils.get_lr's 4.7e-4: every run at the default LR showed
    # wholesale policy shifts within 5-10 steps (run 3: one buggy code pattern took
    # over 48/64 rollouts in ~3 steps). None -> get_lr(model) default.
    learning_rate: float | None = 1.5e-4
    max_tokens: int = 8192
    temperature: float = 1.0
    max_steps: int | None = 50
    num_substeps: int = 1
    # Small KL-to-base anchor against mode collapse / competence loss (0 disables).
    kl_penalty_coef: float = 0.01

    # reward knobs
    scorer: str = "holistic"             # "holistic" | "null"
    fewshot_dir: str = DEFAULT_FEWSHOT   # rated calibration images for the holistic judge ("" disables)
    pairwise_mode: str = "required"      # "required" | "skip_if_missing" | "off"
    k_opponents: int = 2
    judge_swap: bool = True
    judge_model: str = "qwen/qwen3-vl-32b-instruct"
    judge_retries: int = 2               # extra attempts per judge call; still-failed calls score 0
    w_aesth: float = 0.35
    w_pair: float = 0.20                 # refpool pairwise winrate weight (0.000 all of run 2 -> demoted)
    w_len: float = 0.15
    len_free_tokens: int = 6000
    r_fail: float = -0.3                 # gate: equals the worst possible success score
    r_blank: float = -0.1                # flat/near-uniform render: worse than any attempt, better than a fail
    render_timeout_ms: int = 45000
    tournament: bool = True             # intra-group pairwise tournament
    w_tour: float = 0.30                 # tournament winrate weight (independent of w_pair)
    tournament_k: int = 2                # intra-group opponents per rollout
    n_test: int = 6
    refpool_root: str = DEFAULT_REFPOOL
    system_prompt_path: str = DEFAULT_SYSTEM_PROMPT

    # logging / checkpoints
    eval_every: int = 10
    save_every: int = 10
    wandb_project: str | None = None
    wandb_name: str | None = None
    base_url: str | None = None
    rollout_error_tolerance: bool = False   # False=FailFast; True=RetryOnFailure
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # convenience: flip a batch of defaults for a cheap end-to-end smoke run
    smoke: bool = False


def _apply_smoke(c: CLIConfig) -> CLIConfig:
    """Cheap end-to-end validation: tiny batch, no judge spend, short run."""
    return chz.replace(
        c,
        group_size=4,
        groups_per_batch=2,
        max_tokens=8192,
        max_steps=3,
        save_every=1,
        eval_every=1000,
        scorer="null",
        pairwise_mode="off",
        n_test=0,
    )


async def cli_main(cli_config: CLIConfig):
    _ensure_tinker_key()
    if cli_config.smoke:
        cli_config = _apply_smoke(cli_config)

    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )

    model_slug = cli_config.model_name.replace("/", "-")
    run_name = ("paint-%s-%drank-%dG-%dbatch-seed%d-%s"
                % (model_slug, cli_config.lora_rank, cli_config.group_size,
                   cli_config.groups_per_batch, cli_config.seed,
                   datetime.now().strftime("%Y%m%d-%H%M")))
    log_path = cli_config.log_path or ("runs/%s" % run_name)

    lr = cli_config.learning_rate
    if lr is None:
        lr = hyperparam_utils.get_lr(cli_config.model_name)
        logger.info("using LoRA lr from hyperparam_utils: %g", lr)

    dataset_builder = PaintRLDatasetBuilder(
        model_name=cli_config.model_name,
        log_path=log_path,
        group_size=cli_config.group_size,
        groups_per_batch=cli_config.groups_per_batch,
        renderer_name=renderer_name,
        max_tokens=cli_config.max_tokens,
        n_test=cli_config.n_test,
        seed=cli_config.seed,
        scorer=cli_config.scorer,
        fewshot_dir=cli_config.fewshot_dir,
        pairwise_mode=cli_config.pairwise_mode,
        k_opponents=cli_config.k_opponents,
        judge_swap=cli_config.judge_swap,
        judge_model=cli_config.judge_model,
        judge_retries=cli_config.judge_retries,
        w_aesth=cli_config.w_aesth,
        w_pair=cli_config.w_pair,
        w_len=cli_config.w_len,
        len_free_tokens=cli_config.len_free_tokens,
        r_fail=cli_config.r_fail,
        r_blank=cli_config.r_blank,
        render_timeout_ms=cli_config.render_timeout_ms,
        tournament=cli_config.tournament,
        w_tour=cli_config.w_tour,
        tournament_k=cli_config.tournament_k,
        refpool_root=cli_config.refpool_root,
        system_prompt_path=cli_config.system_prompt_path,
    )

    config = Config(
        learning_rate=lr,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        recipe_name="recipe_paint_rl",
        renderer_name=renderer_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        num_substeps=cli_config.num_substeps,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        remove_constant_reward_groups=True,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        kl_reference_config=(KLReferenceConfig(base_model=cli_config.model_name)
                             if cli_config.kl_penalty_coef > 0 else None),
        rollout_error_tolerance=cli_config.rollout_error_tolerance,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name or run_name,
        max_steps=cli_config.max_steps,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    Path(log_path).mkdir(parents=True, exist_ok=True)
    init_tracker(log_path, resume=True)

    try:
        await main(config)
    finally:
        await close_render_bridge()
        close_orclient()
        try:
            from reward.tracking import get_tracker
            get_tracker().save()
            rep, _, total = get_tracker().report()
            print("\n=== reward pipeline cost/time ===\n%s" % rep)
        except Exception:
            pass


def _reexec_utf8_if_needed() -> None:
    """tinker_cookbook writes HTML/JSON logs with the platform default codec; on
    Windows that is cp1252 and crashes on any non-Latin character in a report.
    Re-exec under UTF-8 mode (equivalent to PYTHONUTF8=1) so every open() in the
    process defaults to utf-8. No-op once already in utf-8 mode."""
    import os

    if sys.platform == "win32" and os.environ.get("PYTHONUTF8") != "1":
        os.environ["PYTHONUTF8"] = "1"
        os.execv(sys.executable, [sys.executable, "-X", "utf8",
                                  os.path.abspath(__file__), *sys.argv[1:]])


if __name__ == "__main__":
    _reexec_utf8_if_needed()
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
