"""reward/judge.py -- pairwise VLM judge (candidate vs reference).

The primary visual reward signal, following the blog: instead of scoring a
painting on an absolute scale (which invited reward hacking), ask a VLM which of
two watercolour paintings better depicts the requested scene, and reward the
policy by its win-rate against a pool of reference images.

Each (candidate, reference) pair is judged twice with the A/B order swapped to
cancel position bias. Forced choice: win=1, loss=0, NO ties (run 2's judge tied
essentially every intra-group pair, flattening the tournament signal to a
constant 0.5; for genuinely equal pairs the swapped votes average to ~0.5
anyway). A parse/API failure (after cfg.judge_retries retries inside achat)
scores 0 plus an error metric — a neutral 0.5 fallback proved poisonous,
out-scoring what degenerate images honestly earn. Calls fan out concurrently
and are cost-tracked.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from reward.compat import img_part
from reward.config import RewardConfig
from reward.orclient import achat

PAIRWISE_SYSTEM = (
    "You are a meticulous art critic comparing two WATERCOLOUR paintings that were "
    "produced procedurally by code. You are shown image A and image B and told what "
    "they are meant to depict. Decide which is the better watercolour painting of "
    "that subject, judging: subject recognisability (is it clearly the requested "
    "animal and scene?), watercolour character (soft layered washes, not stiff "
    "outlines or muddy blobs), and composition.\n"
    "RULES:\n"
    "- An image that visibly attempts the subject — figures, shapes, strokes, any "
    "composition — ALWAYS beats a blank canvas, a single flat shade, or a near-empty "
    "canvas with only a stray mark or two, no matter how crude the attempt is.\n"
    "- If both images attempt the subject, prefer the one that depicts it more "
    "recognisably and with more genuine watercolour character.\n"
    "- You MUST pick a winner. There are no ties.\n"
    "Reply with EXACTLY one token: A or B. No other text."
)


def _parse_vote(text: str) -> str | None:
    """Return 'A', 'B', or None (unparseable)."""
    if not text:
        return None
    t = text.strip().upper()
    m = re.search(r"\b(A|B)\b", t)
    if m:
        return m.group(1)
    if t.startswith("A"):
        return "A"
    if t.startswith("B"):
        return "B"
    return None


async def _one_vote(candidate_png: str, reference_png: str, scene_prompt: str,
                    cfg: RewardConfig, candidate_is_a: bool) -> tuple[float, bool]:
    """Judge one ordering. Returns (candidate_score in {0,0.5,1}, error_flag)."""
    first, second = ((candidate_png, reference_png) if candidate_is_a
                     else (reference_png, candidate_png))
    parts = [
        {"type": "text", "text": "Image A:"}, img_part(first),
        {"type": "text", "text": "Image B:"}, img_part(second),
        {"type": "text",
         "text": ("Both are meant to depict: %s.\nWhich is the better watercolour "
                  "painting of that? Reply A or B." % scene_prompt)},
    ]
    messages = [{"role": "system", "content": PAIRWISE_SYSTEM},
                {"role": "user", "content": parts}]
    # The judge may be a reasoning model: the cap must leave room for the verdict
    # after any reasoning (a tight cap yields an empty completion). We still ask
    # for a single-token answer.
    content, metrics = await achat(cfg.judge_model, messages, temperature=0.0,
                                   max_tokens=512, retries=cfg.judge_retries)
    if metrics.get("error"):
        return 0.0, True
    vote = _parse_vote(content)
    if vote is None:
        return 0.0, True
    cand_letter = "A" if candidate_is_a else "B"
    return (1.0 if vote == cand_letter else 0.0), False


async def pairwise_winrate(candidate_png: str, refs: list[Path], scene_prompt: str,
                           cfg: RewardConfig) -> tuple[float, dict]:
    """Win-rate of the candidate against its reference opponents.

    With cfg.judge_swap, each opponent contributes two votes (order swapped).
    Returns (winrate in [0,1], metrics dict).
    """
    if not refs:
        return 0.5, {"pairwise_votes": 0, "pairwise_skipped": 1.0}

    tasks = []
    for ref in refs:
        tasks.append(_one_vote(candidate_png, str(ref), scene_prompt, cfg, candidate_is_a=True))
        if cfg.judge_swap:
            tasks.append(_one_vote(candidate_png, str(ref), scene_prompt, cfg, candidate_is_a=False))

    results = await asyncio.gather(*tasks)
    scores = [s for s, _ in results]
    n_err = sum(1 for _, e in results if e)
    winrate = sum(scores) / len(scores) if scores else 0.5
    return winrate, {
        "pairwise_winrate": winrate,
        "pairwise_votes": float(len(scores)),
        "pairwise_errors": float(n_err),
    }
