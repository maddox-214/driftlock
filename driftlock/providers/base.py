"""
Provider abstraction layer.

NormalizedUsage is the common telemetry format that every provider adapter
must produce.  The core interception logic (policy, cache, metrics) depends
only on NormalizedUsage — never on provider-specific response shapes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NormalizedUsage:
    """Token counts and model name extracted from any provider response."""

    prompt_tokens: int
    completion_tokens: int
    model: str  # as reported by the provider (may differ from requested model)
