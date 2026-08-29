"""reward/refpool.py -- reference image pool for the pairwise judge.

Two layouts are supported, both scanned:

1. FLAT (default): images sit directly in the root, named
   `<combo_id>__<model>__r<k>.png` (the datagen output naming). The combo id is
   everything before the first `__`, e.g. `bear_autumn_berries__gpt.png` ->
   `bear_autumn_berries`.
2. SUBDIRS: `combos/<combo_id>/*.png` (scene-specific) and
   `animals/<animal_slug>/*.png` (animal-level fallback).

The pairwise judge samples K opponents per rollout from the requested combo's
references, falling back to the animal-level pool when the combo has none.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REFPOOL = ROOT / "reference_images"
_PROMPTS_JSON = ROOT / "datagen" / "prompts.json"

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def animal_slug(animal: str) -> str:
    return animal.strip().lower().replace(" ", "_")


def _list_images(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in _IMG_EXTS and p.is_file())


class RefPool:
    def __init__(self, root: str | Path = DEFAULT_REFPOOL):
        self.root = Path(root)
        self.combos_dir = self.root / "combos"
        self.animals_dir = self.root / "animals"
        self._flat: dict[str, list[Path]] | None = None
        self._combo_to_animal: dict[str, str] | None = None

    def _flat_index(self) -> dict[str, list[Path]]:
        """Flat root images grouped by combo-id prefix (before the first `__`)."""
        if self._flat is None:
            self._flat = {}
            if self.root.is_dir():
                for p in self.root.iterdir():
                    if p.is_file() and p.suffix.lower() in _IMG_EXTS:
                        cid = p.stem.split("__", 1)[0]
                        self._flat.setdefault(cid, []).append(p)
            for v in self._flat.values():
                v.sort()
        return self._flat

    def _animal_map(self) -> dict[str, str]:
        """combo_id -> animal, from datagen/prompts.json (for flat animal fallback)."""
        if self._combo_to_animal is None:
            self._combo_to_animal = {}
            try:
                data = json.loads(_PROMPTS_JSON.read_text(encoding="utf-8"))
                for c in data["combos"]:
                    self._combo_to_animal[c["id"]] = c["animal"]
            except Exception:
                pass
        return self._combo_to_animal

    def combo_refs(self, combo_id: str) -> list[Path]:
        return _list_images(self.combos_dir / combo_id) + self._flat_index().get(combo_id, [])

    def animal_refs(self, animal: str) -> list[Path]:
        refs = _list_images(self.animals_dir / animal_slug(animal))
        # Flat fallback: every flat image whose combo belongs to this animal.
        amap = self._animal_map()
        flat = self._flat_index()
        target = animal.strip().lower()
        for cid, paths in flat.items():
            if amap.get(cid, "").strip().lower() == target:
                refs = refs + paths
        return refs

    def refs_for(self, combo_id: str, animal: str) -> list[Path]:
        """Combo-specific refs if present, else the animal-level fallback."""
        refs = self.combo_refs(combo_id)
        return refs if refs else self.animal_refs(animal)

    def has_refs(self, combo_id: str, animal: str) -> bool:
        return bool(self.refs_for(combo_id, animal))

    def sample(self, combo_id: str, animal: str, k: int,
               rng: random.Random | None = None) -> list[Path]:
        """Sample up to k reference images for a rollout (without replacement when
        possible; with replacement only if the pool is smaller than k)."""
        pool = self.refs_for(combo_id, animal)
        if not pool:
            return []
        rng = rng or random
        if len(pool) >= k:
            return rng.sample(pool, k)
        return [rng.choice(pool) for _ in range(k)]

    def coverage_report(self, combos: list[dict]) -> str:
        """Human-readable report of which combos have references. `combos` are the
        {id, animal, prompt} dicts from prompts.json."""
        lines = ["RefPool coverage: %s" % self.root]
        n_combo, n_animal, n_missing = 0, 0, 0
        for c in combos:
            cid, animal = c["id"], c["animal"]
            nc = len(self.combo_refs(cid))
            na = len(self.animal_refs(animal))
            if nc:
                tier, n_combo = "combo", n_combo + 1
            elif na:
                tier, n_animal = "animal", n_animal + 1
            else:
                tier, n_missing = "MISSING", n_missing + 1
            lines.append("  %-24s %-12s combo=%d animal=%d [%s]"
                         % (cid, animal, nc, na, tier))
        lines.append("  ---")
        lines.append("  %d combo-covered, %d animal-only, %d MISSING (of %d)"
                     % (n_combo, n_animal, n_missing, len(combos)))
        return "\n".join(lines)

    def missing_combos(self, combos: list[dict]) -> list[str]:
        return [c["id"] for c in combos if not self.has_refs(c["id"], c["animal"])]
