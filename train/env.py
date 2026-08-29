"""train/env.py -- Tinker RL environment for p5.brush watercolour code generation.

Single-turn task: the policy sees the (barebones) system prompt + a "paint this
scene" user turn, emits a full p5.brush JS sketch; we render it to a PNG and score
it. Following the cookbook's `recipes/rubric/env.py` shape:

- PaintEnv.step(): parse the response, gate on stop-sequence + non-empty code +
  successful render, and return the per-step reward (compile gate value on failure,
  else the length penalty). It stashes render state on the env instance.
- PaintEnvGroupBuilder.compute_group_rewards(): after all rollouts, batch the
  aesthetic (holistic VLM) scorer over the compiled PNGs, fan out the pairwise
  judge vs the reference pool, and (when cfg.tournament) judge compiled group
  members against each other; return each rollout's group reward via
  reward.group_reward(). This sums with the per-step reward.

The heavy objects (render bridge, OpenRouter client, scorer) are module-level lazy
singletons fetched inside async code, so the builder stays picklable.
"""

from __future__ import annotations

import asyncio
import base64
import random
import sys
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import chz
import tinker

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinker_cookbook import model_info  # noqa: E402
from tinker_cookbook.completers import StopCondition  # noqa: E402
from tinker_cookbook.renderers import Renderer, get_renderer, get_text_content  # noqa: E402
from tinker_cookbook.rl.types import (  # noqa: E402
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    Metrics,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    Trajectory,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402
from tinker_cookbook.utils import logtree  # noqa: E402

from reward.compat import extract_code, load_examples, split_examples, user_prompt_for  # noqa: E402
from reward.config import RewardConfig  # noqa: E402
from reward.imgcheck import is_blank  # noqa: E402
from reward.judge import pairwise_winrate  # noqa: E402
from reward.refpool import RefPool  # noqa: E402
from reward.render_bridge import get_render_bridge  # noqa: E402
from reward.reward import group_reward, length_penalty  # noqa: E402
from reward.scorers import make_scorer  # noqa: E402
from reward.tracking import get_tracker  # noqa: E402

TRAIN_DIR = ROOT / "train"
DEFAULT_SYSTEM_PROMPT = str(TRAIN_DIR / "system_prompt.txt")
DEFAULT_REFPOOL = str(ROOT / "reference_images")
DEFAULT_FEWSHOT = str(ROOT / "reward" / "vlm_judge_fewshot")


# --------------------------------------------------------------------------- #
# Env
# --------------------------------------------------------------------------- #
class PaintEnv(Env):
    """One rollout: prompt -> sketch -> render -> (gate + length penalty here;
    aesthetic + pairwise added by the group builder)."""

    def __init__(self, renderer: Renderer, combo: dict, system_prompt: str,
                 reward_cfg: RewardConfig, out_dir: str):
        self.renderer = renderer
        self.combo = combo
        self.system_prompt = system_prompt
        self.cfg = reward_cfg
        self.out_dir = Path(out_dir)
        # state populated during step():
        self.compiled: bool = False
        self.blank: bool = False
        self.png_path: str | None = None
        self.n_sampled: int = 0
        self.render_ms: int = 0
        self.render_error: str | None = None
        self.code_len: int = 0

    @property
    def scene_prompt(self) -> str:
        return self.combo["prompt"]

    def _messages(self) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt_for(self.scene_prompt)},
        ]

    async def initial_observation(self) -> tuple[tinker.ModelInput, StopCondition]:
        return (
            self.renderer.build_generation_prompt(self._messages()),
            self.renderer.get_stop_sequences(),
        )

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        self.n_sampled = len(action)
        message, termination = self.renderer.parse_response(action)
        text = get_text_content(message)
        code = extract_code(text)
        self.code_len = len(code)
        format_ok = bool(termination.is_stop_sequence)

        stem = "%s_%s" % (self.combo["id"], uuid.uuid4().hex[:10])
        if code.strip():
            # Always keep the generated source, compiled or not, next to its render:
            # <out_dir>/code/<stem>.js  <->  <out_dir>/renders/<stem>.png
            code_dir = self.out_dir / "code"
            code_dir.mkdir(parents=True, exist_ok=True)
            (code_dir / (stem + ".js")).write_text(code, encoding="utf-8")

        if format_ok and code.strip():
            render_dir = self.out_dir / "renders"
            render_dir.mkdir(parents=True, exist_ok=True)
            out_png = render_dir / (stem + ".png")
            res = await get_render_bridge().render(
                code, out_png, seed=self.cfg.render_seed,
                timeout_ms=self.cfg.render_timeout_ms,
            )
            self.render_ms = res.ms
            get_tracker().add("render", time_s=res.ms / 1000.0, error=not res.ok)
            if res.ok:
                self.compiled = True
                self.png_path = res.out_path
            else:
                self.render_error = res.error
        else:
            self.render_error = "no stop sequence" if not format_ok else "empty code"

        if self.compiled:
            # Deterministic degeneracy check: a flat/near-uniform render gets
            # r_blank, is excluded from all judging, and never scores positive.
            self.blank, _bstats = is_blank(self.png_path, self.cfg.blank_painted_thresh)
            if self.blank:
                step_reward = self.cfg.r_blank
                gate = "blank"
            else:
                step_reward = length_penalty(self.n_sampled, self.cfg)
                gate = "ok"
        else:
            step_reward = self.cfg.r_fail
            gate = "fail"

        self._log(code, gate, step_reward)

        metrics: Metrics = {
            "compiled": float(self.compiled),
            "blank": float(self.blank),
            "n_sampled_tokens": float(self.n_sampled),
            "code_len": float(self.code_len),
            "render_ms": float(self.render_ms),
            "format_ok": float(format_ok),
        }
        logs = {
            "combo_id": self.combo["id"],
            "gate": gate,
            "render_error": self.render_error or "",
        }
        return StepResult(
            reward=step_reward,
            episode_done=True,
            next_observation=tinker.ModelInput.from_ints([]),
            next_stop_condition=[],
            metrics=metrics,
            logs=logs,
        )

    def _log(self, code: str, gate: str, step_reward: float) -> None:
        with logtree.scope_header("Paint rollout: %s" % self.combo["id"]):
            logtree.log_text("scene: %s" % self.scene_prompt)
            logtree.log_text("gate: %s  render_ms: %d  tokens: %d  step_reward: %.3f"
                             % (gate, self.render_ms, self.n_sampled, step_reward))
            if self.render_error:
                logtree.log_text("render_error: %s" % self.render_error)
            logtree.details(code[:2000], summary="sketch code (head)", pre=True)
            if self.compiled and self.png_path:
                try:
                    b64 = base64.b64encode(Path(self.png_path).read_bytes()).decode("ascii")
                    logtree.log_html('<img src="data:image/png;base64,%s" '
                                     'style="max-width:300px;border:1px solid #ccc"/>' % b64)
                except Exception:
                    pass


# --------------------------------------------------------------------------- #
# Group builder (picklable: primitive fields only)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PaintEnvGroupBuilder(EnvGroupBuilder):
    renderer: Renderer
    combo: dict
    system_prompt: str
    reward_cfg: RewardConfig
    group_size: int
    out_dir: str
    refpool_root: str
    seed: int

    async def make_envs(self):
        return [
            PaintEnv(self.renderer, self.combo, self.system_prompt, self.reward_cfg, self.out_dir)
            for _ in range(self.group_size)
        ]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group
    ) -> list[tuple[float, Metrics]]:
        envs = list(env_group)
        cfg = self.reward_cfg
        # Blank renders are excluded here entirely: no judge spend, no tournament
        # slot, group reward 0 — their whole reward is the r_blank step reward.
        compiled_idx = [i for i, e in enumerate(envs)
                        if getattr(e, "compiled", False) and getattr(e, "png_path", None)
                        and not getattr(e, "blank", False)]

        # No gradeable member -> nothing to judge; group reward is 0
        # (step reward already carries r_fail / r_blank).
        if not compiled_idx:
            return [(0.0, {}) for _ in envs]

        pngs = [envs[i].png_path for i in compiled_idx]
        scenes = [envs[i].scene_prompt for i in compiled_idx]

        # Aesthetic (holistic VLM judge or null), batched.
        scorer = make_scorer(cfg)
        aesth_scores = await scorer.score(pngs, scenes)
        aesth_by_idx = dict(zip(compiled_idx, aesth_scores))

        # Pairwise vs reference pool (unless disabled).
        winrate_by_idx: dict[int, float] = {}
        if cfg.pairwise_mode != "off":
            refpool = RefPool(self.refpool_root)
            tasks, idxs = [], []
            for i in compiled_idx:
                rng = random.Random("%d-%s-%d" % (self.seed, self.combo["id"], i))
                refs = refpool.sample(self.combo["id"], self.combo["animal"],
                                      cfg.k_opponents, rng)
                tasks.append(pairwise_winrate(envs[i].png_path, refs, envs[i].scene_prompt, cfg))
                idxs.append(i)
            results = await asyncio.gather(*tasks)
            for i, (wr, _m) in zip(idxs, results):
                winrate_by_idx[i] = wr

        # Intra-group tournament (optional): each compiled rollout vs up to
        # tournament_k of its compiled group-mates. Unlike the refpool, these
        # opponents are always at the policy's own level, so the term carries
        # gradient even while the refpool winrate sits at 0.
        tour_by_idx: dict[int, float] = {}
        if cfg.tournament:
            tasks, idxs = [], []
            for i in compiled_idx:
                others = [j for j in compiled_idx if j != i]
                if not others:
                    tour_by_idx[i] = 0.5  # lone compiled member: uninformative
                    continue
                rng = random.Random("tour-%d-%s-%d" % (self.seed, self.combo["id"], i))
                opp = rng.sample(others, min(cfg.tournament_k, len(others)))
                refs = [Path(envs[j].png_path) for j in opp]
                tasks.append(pairwise_winrate(envs[i].png_path, refs, envs[i].scene_prompt, cfg))
                idxs.append(i)
            if tasks:
                results = await asyncio.gather(*tasks)
                for i, (wr, _m) in zip(idxs, results):
                    tour_by_idx[i] = wr

        out: list[tuple[float, Metrics]] = []
        for i, _e in enumerate(envs):
            if i not in compiled_idx:
                out.append((0.0, {}))
                continue
            aesth = aesth_by_idx.get(i, 0.0)
            wr = winrate_by_idx.get(i, 0.5) if cfg.pairwise_mode != "off" else None
            tw = tour_by_idx.get(i, 0.5) if cfg.tournament else None
            gr = group_reward(aesth, wr, cfg, tournament_winrate=tw)
            m: Metrics = {"aesthetic": aesth, "group_reward": gr}
            if wr is not None:
                m["winrate"] = wr
            if tw is not None:
                m["tournament_winrate"] = tw
            out.append((gr, m))
        return out

    def logging_tags(self) -> list[str]:
        return [self.combo["animal"], self.combo["id"]]


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class PaintRLDataset(RLDataset):
    def __init__(self, combos: list[dict], renderer: Renderer, system_prompt: str,
                 reward_cfg: RewardConfig, group_size: int, groups_per_batch: int,
                 out_dir: str, refpool_root: str, seed: int, eval_mode: bool = False):
        self.combos = combos
        self.renderer = renderer
        self.system_prompt = system_prompt
        self.reward_cfg = reward_cfg
        self.group_size = group_size
        self.groups_per_batch = groups_per_batch
        self.out_dir = out_dir
        self.refpool_root = refpool_root
        self.seed = seed
        self.eval_mode = eval_mode

    def _builder(self, combo: dict, iter_dir: str) -> PaintEnvGroupBuilder:
        return PaintEnvGroupBuilder(
            renderer=self.renderer,
            combo=combo,
            system_prompt=self.system_prompt,
            reward_cfg=self.reward_cfg,
            group_size=self.group_size,
            out_dir=iter_dir,
            refpool_root=self.refpool_root,
            seed=self.seed,
        )

    def _iter_dir(self, index: int) -> str:
        """Per-iteration rollout dir: <out_dir>/iteration_%06d (train) or
        <out_dir>/eval (held-out). The env writes code/ and renders/ inside it."""
        sub = "eval" if self.eval_mode else ("iteration_%06d" % index)
        return str(Path(self.out_dir) / sub)

    def get_batch(self, index: int):
        iter_dir = self._iter_dir(index)
        if self.eval_mode:
            # One group per combo, fixed order (fully consumed by the evaluator).
            return [self._builder(c, iter_dir) for c in self.combos]
        rng = random.Random("%d-%d" % (self.seed, index))
        n = self.groups_per_batch
        if n <= len(self.combos):
            # Unique prompts per batch: sample without replacement.
            chosen = rng.sample(self.combos, n)
        else:
            # More groups requested than unique combos: use each once, then top up.
            chosen = list(self.combos)
            rng.shuffle(chosen)
            chosen += [rng.choice(self.combos) for _ in range(n - len(self.combos))]
        return [self._builder(c, iter_dir) for c in chosen]

    def __len__(self) -> int:
        return 1 if self.eval_mode else 1_000_000


# --------------------------------------------------------------------------- #
# Dataset builder (chz-configurable)
# --------------------------------------------------------------------------- #
@chz.chz
class PaintRLDatasetBuilder(RLDatasetBuilder):
    model_name: str
    log_path: str
    group_size: int = 8
    groups_per_batch: int = 8
    renderer_name: str | None = None
    max_tokens: int = 16384
    n_test: int = 6
    seed: int = 0

    # reward knobs (flow into RewardConfig)
    scorer: str = "holistic"
    fewshot_dir: str = DEFAULT_FEWSHOT
    pairwise_mode: str = "required"
    k_opponents: int = 2
    judge_swap: bool = True
    judge_model: str = "qwen/qwen3-vl-32b-instruct"
    judge_retries: int = 2
    w_aesth: float = 0.35
    w_pair: float = 0.15
    w_len: float = 0.30
    len_free_tokens: int = 6000
    r_fail: float = -0.3
    r_blank: float = -0.1
    render_timeout_ms: int = 45000
    tournament: bool = False
    w_tour: float = 0.30
    tournament_k: int = 2

    refpool_root: str = DEFAULT_REFPOOL
    system_prompt_path: str = DEFAULT_SYSTEM_PROMPT

    def _reward_cfg(self) -> RewardConfig:
        return RewardConfig(
            r_fail=self.r_fail,
            r_blank=self.r_blank,
            w_aesth=self.w_aesth,
            w_pair=self.w_pair,
            w_len=self.w_len,
            len_free_tokens=self.len_free_tokens,
            max_tokens=self.max_tokens,
            k_opponents=self.k_opponents,
            judge_swap=self.judge_swap,
            judge_model=self.judge_model,
            judge_retries=self.judge_retries,
            pairwise_mode=self.pairwise_mode,
            scorer=self.scorer,
            fewshot_dir=self.fewshot_dir,
            render_timeout_ms=self.render_timeout_ms,
            tournament=self.tournament,
            w_tour=self.w_tour,
            tournament_k=self.tournament_k,
        )

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        combos = load_examples()
        train_c, _val, test_c = split_examples(combos, n_val=0, n_test=self.n_test, seed=self.seed)

        cfg = self._reward_cfg()

        # Refpool gate: 'required' mode must not start with missing references.
        if cfg.pairwise_mode == "required":
            rp = RefPool(self.refpool_root)
            missing = rp.missing_combos(train_c)
            if missing:
                report = rp.coverage_report(train_c)
                raise RuntimeError(
                    "pairwise_mode='required' but %d train combo(s) lack references:\n%s\n"
                    "Fill refpool/ (combos/<id>/ or animals/<slug>/), or set "
                    "pairwise_mode='skip_if_missing' or 'off'." % (len(missing), report)
                )

        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(self.model_name)
        renderer = get_renderer(renderer_name, get_tokenizer(self.model_name))
        system_prompt = Path(self.system_prompt_path).read_text(encoding="utf-8")
        # Rollout artifacts land in <log_path>/rollouts/iteration_%06d/{code,renders}
        # (train) and <log_path>/rollouts/eval/{code,renders} (held-out).
        out_dir = str(Path(self.log_path) / "rollouts")
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        train_ds = PaintRLDataset(
            train_c, renderer, system_prompt, cfg,
            self.group_size, self.groups_per_batch, out_dir, self.refpool_root, self.seed,
        )

        test_ds = None
        if test_c:
            # Cheap held-out monitoring: no pairwise (no refpool dependency), small groups.
            test_cfg = replace(cfg, pairwise_mode="off")
            test_ds = PaintRLDataset(
                test_c, renderer, system_prompt, test_cfg,
                group_size=2, groups_per_batch=len(test_c),
                out_dir=out_dir, refpool_root=self.refpool_root, seed=self.seed,
                eval_mode=True,
            )
        return train_ds, test_ds
