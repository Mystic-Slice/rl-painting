"""reward/config.py -- reward shaping knobs for the p5.brush RL loop.

One frozen dataclass carried (as primitive fields) on the picklable
EnvGroupBuilder. Every weight, gate value, and judge setting lives here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    # --- gate --------------------------------------------------------------
    # Kept equal to the worst possible success score (-w_len at the token cap)
    # so failing never beats succeeding, without the gate dwarfing the judge
    # signal the way -1.0 did (which trained a blank-canvas policy).
    r_fail: float = -0.3            # reward for compile/format/render failure

    # Blank-image gate: a render that is a flat shade / near-uniform wash (see
    # reward/imgcheck.py) gets r_blank and is excluded from all judging — worse
    # than any real attempt, better than not compiling, never positive.
    r_blank: float = -0.1
    blank_painted_thresh: float = 0.02  # min fraction of canvas painted over the bg colour

    # --- component weights (success case) ----------------------------------
    # Independent weights, one per term (no implicit coupling between them).
    w_aesth: float = 0.35           # single holistic VLM quality judge (0..1)
    # Refpool winrate was 0.000 for the whole of run 2 (references unreachably
    # strong early on) — kept small until the policy can actually score wins.
    w_pair: float = 0.15            # pairwise winrate vs reference pool (0..1)
    w_len: float = 0.30             # max magnitude of the length penalty (<=0)

    # --- length shaping ----------------------------------------------------
    len_free_tokens: int = 6000     # no penalty at or below this many sampled tokens
    max_tokens: int = 16384         # sampling cap (penalty denominator); set from train Config

    # --- pairwise judge ----------------------------------------------------
    k_opponents: int = 2            # reference opponents sampled per rollout
    judge_swap: bool = True         # judge each pair twice with A/B order swapped
    judge_model: str = "qwen/qwen3-vl-32b-instruct"
    judge_retries: int = 2          # extra attempts per judge call on API error
    # "required": refuse to start if refpool incomplete (MVP);
    # "skip_if_missing": neutral 0.5 winrate when a combo has no refs;
    # "off": drop the pairwise term entirely (smoke).
    pairwise_mode: str = "required"

    # --- intra-group tournament --------------------------------------------
    # When enabled, each compiled rollout is also judged pairwise against
    # tournament_k other compiled rollouts of its own group. Group members are
    # always reachable opponents, so this term carries gradient even while the
    # refpool winrate is pinned near 0.
    tournament: bool = False
    w_tour: float = 0.30            # tournament winrate weight (independent of w_pair)
    tournament_k: int = 2           # intra-group opponents sampled per rollout

    # --- aesthetic scorer --------------------------------------------------
    scorer: str = "holistic"        # "holistic" (single VLM judge) | "null"
    # Directory of rated calibration images for the holistic judge, named
    # "<rating>.png" (rating on the 0-10 scale). Empty string disables few-shot.
    fewshot_dir: str = ""

    # --- render ------------------------------------------------------------
    render_timeout_ms: int = 45000
    render_seed: int | None = None  # p5.brush RNG is not actually seeded (renders vary)
