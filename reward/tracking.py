"""reward/tracking.py -- process-wide cost/time Tracker singleton.

Reuses datagen's thread-safe `Tracker` (stages: generation / render / judge /
reflection) so RL cost accounting matches the datagen + GEPA reports. The RL loop
only spends on `render` (local, time-only) and `judge` (OpenRouter $).

The singleton is set once at startup with the run's log dir so it autosaves to
`<log_path>/tracker.json` and resumes from it.
"""

from __future__ import annotations

from pathlib import Path

from reward.compat import Tracker

_TRACKER: Tracker | None = None


def init_tracker(log_path: str | Path, resume: bool = True) -> Tracker:
    """Create (or reuse) the global Tracker, saving under log_path/tracker.json."""
    global _TRACKER
    save_path = Path(log_path) / "tracker.json"
    if _TRACKER is None:
        _TRACKER = Tracker(save_path=save_path)
        if resume:
            _TRACKER.load()
    return _TRACKER


def get_tracker() -> Tracker:
    """Return the global Tracker, or a throwaway in-memory one if uninitialised.

    Reward components call this; in unit tests (no init) they still work, just
    without persistence.
    """
    global _TRACKER
    if _TRACKER is None:
        _TRACKER = Tracker(save_path=None)
    return _TRACKER
