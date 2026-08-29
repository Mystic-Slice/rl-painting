"""reward/reward.py -- pure, unit-testable reward composition.

The orchestration (render, gate, scorer, pairwise) lives in train/env.py; this
module holds only the arithmetic so it can be tested without node or the network.

Reward shape (per rollout):
    fail (no stop-seq / empty code / render fail):  total = r_fail            (= -1.0)
    success:
        step  = length_penalty(n_sampled)                                    (<= 0)
        group = w_aesth * aesthetic + w_pair * winrate                        (>= 0)
        total = step + group
The trainer sums the per-step reward (returned by Env.step) with the group reward
(returned by EnvGroupBuilder.compute_group_rewards).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from reward.config import RewardConfig


def length_penalty(n_sampled_tokens: int, cfg: RewardConfig) -> float:
    """Zero up to cfg.len_free_tokens, then linear down to -cfg.w_len at max_tokens.

    Penalises sampled tokens (code + any thinking), discouraging runaway output.
    """
    free = cfg.len_free_tokens
    span = max(1, cfg.max_tokens - free)
    over = max(0, n_sampled_tokens - free)
    frac = min(1.0, over / span)
    return -cfg.w_len * frac


def group_reward(
    aesthetic: float,
    winrate: float | None,
    cfg: RewardConfig,
    tournament_winrate: float | None = None,
) -> float:
    """Combine aesthetic score, refpool winrate, and (optional) intra-group
    tournament winrate into the group reward term.

    Each term carries its own independent weight (w_aesth, w_pair, w_tour);
    a None winrate (pairwise_mode='off' / tournament off) drops that term.
    """
    total = cfg.w_aesth * aesthetic
    if winrate is not None:
        total += cfg.w_pair * winrate
    if tournament_winrate is not None:
        total += cfg.w_tour * tournament_winrate
    return total


@dataclass
class RewardBreakdown:
    """Full accounting for one rollout's reward, surfaced into logs/metrics."""
    compiled: bool
    total: float
    step_reward: float = 0.0            # returned from Env.step (gate or length penalty)
    group_reward: float = 0.0           # returned from compute_group_rewards
    length_penalty: float = 0.0
    aesthetic: float | None = None      # 0..1
    winrate: float | None = None        # 0..1
    n_sampled_tokens: int = 0
    render_ms: int = 0
    render_error: str | None = None
    extra: dict = field(default_factory=dict)

    def metrics(self) -> dict:
        """Flat float metrics for ml_log (only present, numeric fields)."""
        m: dict[str, float] = {
            "compiled": float(self.compiled),
            "reward_total": self.total,
            "length_penalty": self.length_penalty,
            "n_sampled_tokens": float(self.n_sampled_tokens),
            "render_ms": float(self.render_ms),
        }
        if self.aesthetic is not None:
            m["aesthetic"] = self.aesthetic
        if self.winrate is not None:
            m["winrate"] = self.winrate
        return m
