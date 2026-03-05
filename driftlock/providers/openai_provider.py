"""
OpenAI provider adapter.

Normalizes openai.types.chat.ChatCompletion responses into NormalizedUsage.
"""

from __future__ import annotations

from typing import Any

from .base import NormalizedUsage


class OpenAIProvider:
    """Adapts OpenAI chat completion responses to NormalizedUsage."""

    def normalize_response(self, response: Any) -> NormalizedUsage:
        usage = response.usage
        return NormalizedUsage(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            model=getattr(response, "model", None) or "unknown",
        )

    def provider_name(self) -> str:
        return "openai"
