"""reward/compat.py -- single source of truth for the datagen plumbing we reuse.

The datagen scripts (`datagen/generate.py`, `datagen/refine.py`,
`datagen/promptopt/optimize_prompt.py`) are standalone scripts, not an installed
package, and they import each other by bare name after mutating `sys.path`. This
module reproduces that path setup once and re-exports the pieces the RL reward and
eval code depend on, so there is exactly one implementation of the OpenRouter
transport, the cost Tracker, the dataset split, and the VLM judges.

Re-exported
-----------
From datagen/generate.py:  load_env, extract_code, user_prompt_for, tag_for
From datagen/refine.py:    chat, img_part, JUDGE_SYSTEM, parse_judgement
From optimize_prompt.py:   Tracker, split_examples, load_examples,
                           judge_image, judge_criterion, CRITERIA_WEIGHTS

Note: importing optimize_prompt pulls in gepa (its reflection engine). That import
cost is paid once and is negligible next to torch/tinker; it keeps Tracker /
split_examples / the 5-criterion eval judge as one shared implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATAGEN = ROOT / "datagen"
PROMPTOPT = DATAGEN / "promptopt"

for _p in (str(DATAGEN), str(PROMPTOPT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# generate.py / refine.py have no heavy deps (just httpx).
from generate import (  # noqa: E402
    load_env,
    extract_code,
    user_prompt_for,
    tag_for,
)
from refine import (  # noqa: E402
    chat,
    img_part,
    JUDGE_SYSTEM,
    parse_judgement,
)

# optimize_prompt.py drags in gepa; import it lazily-but-eagerly here so all reuse
# routes through one module.
from optimize_prompt import (  # noqa: E402
    Tracker,
    split_examples,
    load_examples,
    judge_image,
    judge_criterion,
    CRITERIA_WEIGHTS,
)

__all__ = [
    "load_env",
    "extract_code",
    "user_prompt_for",
    "tag_for",
    "chat",
    "img_part",
    "JUDGE_SYSTEM",
    "parse_judgement",
    "Tracker",
    "split_examples",
    "load_examples",
    "judge_image",
    "judge_criterion",
    "CRITERIA_WEIGHTS",
]
