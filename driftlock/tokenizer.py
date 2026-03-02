"""
Token counting utilities.

Uses tiktoken when available (installed as a transitive dependency of openai>=1.0).
Falls back to a character-based approximation (~4 chars/token) when not available.
"""

from __future__ import annotations

try:
    import tiktoken as _tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TIKTOKEN_AVAILABLE = False


def _get_encoding(model: str):
    """Return a tiktoken encoding, falling back to cl100k_base for unknown models."""
    try:
        return _tiktoken.encoding_for_model(model)
    except KeyError:
        return _tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in a plain string."""
    if _TIKTOKEN_AVAILABLE:
        return len(_get_encoding(model).encode(text))
    return max(1, len(text) // 4)


def count_messages_tokens(messages: list[dict], model: str = "gpt-4o") -> int:
    """
    Count tokens for a list of OpenAI chat messages.

    Follows OpenAI's token-counting formula:
      - 3 tokens overhead per message (role/name framing)
      - 1 extra token when a "name" field is present
      - 3 tokens for reply priming at the end

    See: https://platform.openai.com/docs/guides/chat/managing-tokens
    """
    if _TIKTOKEN_AVAILABLE:
        enc = _get_encoding(model)
        tokens_per_message = 3
        tokens_per_name = 1
        total = 3  # reply priming

        for message in messages:
            total += tokens_per_message
            for key, value in message.items():
                if isinstance(value, str):
                    total += len(enc.encode(value))
                elif isinstance(value, list):
                    # Multimodal content blocks: only count text parts
                    for block in value:
                        if isinstance(block, dict) and block.get("type") == "text":
                            total += len(enc.encode(block.get("text", "")))
                if key == "name":
                    total += tokens_per_name
        return total

    # Fallback: ~4 chars per token + 4 tokens overhead per message
    total_chars = sum(len(str(v)) for msg in messages for v in msg.values())
    return max(1, total_chars // 4 + len(messages) * 4)
