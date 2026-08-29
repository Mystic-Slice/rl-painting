"""reward/scorers.py -- aesthetic quality scorer (the old HPS slot).

An AestheticScorer maps rendered PNGs to a quality score in [0,1]. We dropped
HPSv2/HPSv3 (no torch/GPU model); the default scorer is a single holistic VLM
judge call per image that weighs all quality factors and returns one 0-10 score.

Swap the concrete scorer via RewardConfig.scorer:
    "holistic" -> HolisticJudgeScorer   (one VLM call/image)
    "null"     -> NullScorer            (constant 0.5, no judge spend; smoke runs)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol

from reward.compat import img_part, parse_judgement
from reward.config import RewardConfig
from reward.orclient import achat

_FEWSHOT_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _load_fewshot(fewshot_dir: str) -> list[tuple[float, str]]:
    """Load rated calibration images named '<rating>.png' -> [(rating, path)] sorted."""
    if not fewshot_dir:
        return []
    d = Path(fewshot_dir)
    if not d.is_dir():
        return []
    out: list[tuple[float, str]] = []
    for p in d.iterdir():
        if p.is_file() and p.suffix.lower() in _FEWSHOT_EXTS:
            try:
                out.append((float(p.stem), str(p)))
            except ValueError:
                continue
    out.sort(key=lambda x: x[0])
    return out

HOLISTIC_SYSTEM = (
    "You are a meticulous, fair-but-harsh art critic evaluating a WATERCOLOUR "
    "painting produced procedurally by code. You are shown one painting and told "
    "what it is meant to depict. Weigh ALL of these together into a single "
    "judgement: (1) subject recognisability, (2) painterly watercolour looseness "
    "(soft bleeding layered washes, NOT stiff outlines or muddy blobs), "
    "(3) composition and focal clarity, (4) colour harmony, (5) overall aesthetic "
    "appeal.\n\n"
    "Anchor the low end of your scale precisely — differences between weak "
    "attempts matter:\n"
    "- 0: blank canvas, a single flat shade, or a near-empty canvas with only a "
    "stray mark or two. Painting SOMETHING always scores above painting nothing.\n"
    "- 1-2: visible shapes or strokes, but the subject is hard to make out.\n"
    "- 3-4: the subject is guessable; crude or sparse rendering.\n"
    "- 5-6: subject clearly recognisable with some watercolour character.\n"
    "- 7-8: recognisable subject, layered washes, coherent composition.\n"
    "- 9-10: genuinely accomplished; reserve these.\n\n"
    "Respond with ONLY a JSON object, no prose, no code fence:\n"
    '{"score": <number 0-10, one decimal>, "feedback": "<one sentence>"}'
)


class AestheticScorer(Protocol):
    name: str
    async def score(self, png_paths: list[str], scene_prompts: list[str]) -> list[float]:
        ...


class NullScorer:
    name = "null"

    async def score(self, png_paths: list[str], scene_prompts: list[str]) -> list[float]:
        return [0.5 for _ in png_paths]


class HolisticJudgeScorer:
    """One holistic VLM judge call per image -> 0-10 -> normalized to [0,1].

    If cfg.fewshot_dir holds rated calibration images (named '<rating>.png'),
    they are prepended to each call as scored anchors to calibrate the 0-10 scale
    to the user's ratings.
    """
    name = "holistic"

    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg
        self.fewshot = _load_fewshot(cfg.fewshot_dir)

    def _fewshot_parts(self) -> list[dict]:
        if not self.fewshot:
            return []
        parts: list[dict] = [{
            "type": "text",
            "text": ("CALIBRATION EXAMPLES first: watercolour paintings each labelled "
                     "with the score it should receive on your 0-10 scale. Anchor your "
                     "scoring to these before judging the target image."),
        }]
        for rating, path in self.fewshot:
            parts.append({"type": "text", "text": "Example — score %g/10:" % rating})
            parts.append(img_part(path))
        parts.append({"type": "text",
                      "text": "Now score the target painting on that same scale."})
        return parts

    async def _score_one(self, png_path: str, scene_prompt: str) -> float:
        parts = self._fewshot_parts() + [
            {"type": "text",
             "text": "TARGET painting, meant to depict: %s.\nScore it." % scene_prompt},
            img_part(png_path),
        ]
        messages = [{"role": "system", "content": HOLISTIC_SYSTEM},
                    {"role": "user", "content": parts}]
        content, metrics = await achat(self.cfg.judge_model, messages,
                                       temperature=0.2, max_tokens=600,
                                       retries=self.cfg.judge_retries)
        # Score 0 on API/parse failure (post-retries): a neutral 0.5 fallback
        # out-scored what degenerate images honestly earn and fed the collapse.
        if metrics.get("error"):
            return 0.0
        score, _ = parse_judgement(content)
        if score is None:
            return 0.0
        return max(0.0, min(1.0, score / 10.0))

    async def score(self, png_paths: list[str], scene_prompts: list[str]) -> list[float]:
        if not png_paths:
            return []
        return list(await asyncio.gather(
            *[self._score_one(p, s) for p, s in zip(png_paths, scene_prompts)]
        ))


def make_scorer(cfg: RewardConfig) -> AestheticScorer:
    if cfg.scorer == "null":
        return NullScorer()
    if cfg.scorer == "holistic":
        return HolisticJudgeScorer(cfg)
    raise ValueError("unknown scorer: %r (use 'holistic' or 'null')" % cfg.scorer)
