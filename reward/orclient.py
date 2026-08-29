"""reward/orclient.py -- shared async OpenRouter transport for reward judges.

Wraps datagen's sync `chat()` (httpx, never-raises, returns usage cost) in
`asyncio.to_thread` so many judge calls fan out concurrently from the RL loop,
and records every call to the global cost Tracker under the "judge" stage.

One process-wide `httpx.Client` (thread-safe for concurrent requests) is created
lazily. Not stored on any picklable object.
"""

from __future__ import annotations

import asyncio

import httpx

from reward.compat import chat, load_env
from reward.tracking import get_tracker

_CLIENT: httpx.Client | None = None
_KEY: str | None = None
_BASE_URL: str | None = None


def _ensure() -> tuple[httpx.Client, str, str]:
    global _CLIENT, _KEY, _BASE_URL
    if _CLIENT is None:
        _KEY, _BASE_URL = load_env()
        _CLIENT = httpx.Client(timeout=httpx.Timeout(120.0, connect=30.0))
    assert _KEY is not None and _BASE_URL is not None
    return _CLIENT, _BASE_URL, _KEY


async def achat(model: str, messages: list, temperature: float, max_tokens: int,
                stage: str = "judge", retries: int = 2) -> tuple[str, dict]:
    """Async OpenRouter chat call. Returns (content, metrics). Never raises.

    Transport/API errors are retried up to `retries` extra times with a short
    backoff; every attempt (including failed ones) is cost-tracked. The caller
    sees only the final attempt's (content, metrics).
    """
    client, base_url, key = _ensure()
    content, metrics = "", {}
    for attempt in range(retries + 1):
        if attempt:
            await asyncio.sleep(1.5 * attempt)
        content, metrics, _ = await asyncio.to_thread(
            chat, client, base_url, key, model, messages, temperature, max_tokens
        )
        get_tracker().record_metrics(stage, metrics)
        if not metrics.get("error"):
            break
    return content, metrics


def close_orclient() -> None:
    global _CLIENT
    if _CLIENT is not None:
        try:
            _CLIENT.close()
        except Exception:
            pass
        _CLIENT = None
