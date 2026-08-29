"""reward/imgcheck.py -- cheap deterministic image degeneracy check.

Detects "blank" renders: a single flat shade or near-uniform wash covering the
whole canvas, possibly with a tiny stray scribble (the mode-collapse attractor
of runs 1 and 2). Pure pixel math, no judge spend, so blanks can be penalised
and excluded from judging before any VLM call is made.

Signal: downsample to 128x128, find the dominant colour (mode over channel-
quantised pixels), and measure the PAINTED FRACTION — the share of pixels whose
colour differs meaningfully from that background. A real attempt covers a large
part of the canvas with non-background paint; a flat wash with a stray squiggle
covers almost none. A full-canvas gradient or layered washes have no dominant
colour bin, score a high painted fraction, and are left for the judge to grade.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

_SIZE = 128
_QUANT = 16          # 256/16 = 16 levels per channel for the mode search
_COLOR_TOL = 24.0    # Chebyshev RGB distance from background to count as painted


def blank_stats(png_path: str) -> dict:
    """Return {'painted_frac': float, 'pixel_std': float} for the image."""
    with Image.open(png_path) as im:
        arr = np.asarray(im.convert("RGB").resize((_SIZE, _SIZE)), dtype=np.float32)
    flat = arr.reshape(-1, 3)
    pixel_std = float(flat.std(axis=0).mean())

    quant = (flat.astype(np.uint8) // _QUANT)
    codes = quant[:, 0].astype(np.int32) * 256 + quant[:, 1] * 16 + quant[:, 2]
    mode_code = np.bincount(codes).argmax()
    bg = flat[codes == mode_code].mean(axis=0)
    painted = np.abs(flat - bg).max(axis=1) > _COLOR_TOL
    return {"painted_frac": float(painted.mean()), "pixel_std": pixel_std}


def is_blank(png_path: str, painted_thresh: float = 0.02) -> tuple[bool, dict]:
    """(blank?, stats). Blank = less than painted_thresh of the canvas carries
    paint that differs from the dominant background colour. Never raises:
    an unreadable image counts as blank."""
    try:
        stats = blank_stats(png_path)
    except Exception as e:
        return True, {"painted_frac": 0.0, "pixel_std": 0.0, "error": str(e)}
    return stats["painted_frac"] < painted_thresh, stats
