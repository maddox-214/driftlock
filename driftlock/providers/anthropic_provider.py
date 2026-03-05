"""
Anthropic provider adapter.

Normalizes anthropic.types.Message responses into NormalizedUsage.

Anthropic uses input_tokens / output_tokens instead of OpenAI's
prompt_tokens / completion_tokens.
"""

from __future__ import annotations

from typing import Any

from .base import NormalizedUsage


class AnthropicProvider:
    """Adapts Anthropic Messages API responses to NormalizedUsage."""

    def normalize_response(self, response: Any) -> NormalizedUsage:
        usage = response.usage
        return NormalizedUsage(
            prompt_tokens=usage.input_tokens if usage else 0,
            completion_tokens=usage.output_tokens if usage else 0,
            model=getattr(response, "model", None) or "unknown",
        )

    def provider_name(self) -> str:
        return "anthropic"
