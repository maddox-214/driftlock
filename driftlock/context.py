"""
Ambient tag propagation via Python's contextvars.

Allows middleware (or any outer scope) to attach labels that are automatically
merged into every DriftlockClient call within that context — without touching
every call site.

Works correctly across sync and async code within the same asyncio task.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Generator

_active_tags: ContextVar[dict] = ContextVar("driftlock_tags", default={})


@contextmanager
def tag(**tags: str) -> Generator[None, None, None]:
    """
    Context manager that attaches key-value tags to every DriftlockClient
    call made within the block.

    Tags merge with (and override) any tags set by an outer scope.
    Per-call ``_dl_labels`` always takes final precedence.

    Example::

        with driftlock.tag(request_id="abc123", user_id="u_42"):
            response = client.chat.completions.create(...)
    """
    merged = {**_active_tags.get(), **tags}
    token = _active_tags.set(merged)
    try:
        yield
    finally:
        _active_tags.reset(token)


def get_active_tags() -> dict:
    """Return the tags currently active in this context (copy)."""
    return dict(_active_tags.get())
