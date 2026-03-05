"""
Prompt drift tracking.

Detects when the "template" portion of prompts changes over time —
which is the root cause of silent token growth.

Strategy:
  - Hash the system message + the first user message (the template portion).
  - Subsequent user messages are typically dynamic (user input), so they
    are excluded from the hash.
  - When the hash changes for an endpoint, that means a developer changed
    the prompt template — which may have increased token usage.

The hash is stored in the ``prompt_hash`` column of the calls table.
Use ``DriftlockClient.prompt_drift(endpoint)`` or
``storage.prompt_hash_history(endpoint)`` to inspect changes.
"""

from __future__ import annotations

import hashlib
import json


def hash_prompt(
    messages: list[dict],
    system: str | None = None,
) -> str | None:
    """
    Return a short SHA-256 fingerprint of the prompt template.

    The "template" is defined as:
      - The system message (from ``system`` kwarg or first message with role="system")
      - The first non-system message (usually the user instruction template)

    Dynamic values (subsequent turns, user-provided content) are excluded.
    Returns None when there are no messages to hash.
    """
    parts: list[str] = []

    # System content (from explicit kwarg or role="system" in messages)
    if system:
        parts.append(f"system:{system}")
    else:
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = json.dumps(content, sort_keys=True)
                parts.append(f"system:{content}")
                break

    # First non-system message (template instruction)
    for msg in messages:
        if msg.get("role") != "system":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = json.dumps(content, sort_keys=True)
            parts.append(f"{msg.get('role', 'user')}:{content}")
            break

    if not parts:
        return None

    raw = "\n".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def detect_drift(history: list[dict]) -> list[dict]:
    """
    Given a list of prompt_hash_history rows (timestamp, prompt_hash, prompt_tokens),
    return rows where the hash changed from the previous row.

    Each returned dict has:
      - timestamp: when the change was first seen
      - old_hash: previous hash (None for the first entry)
      - new_hash: new hash
      - prompt_tokens: token count at time of change
    """
    changes: list[dict] = []
    prev_hash: str | None = None

    # history is newest-first from the DB; reverse to process oldest-first
    for row in reversed(history):
        h = row.get("prompt_hash")
        if h and h != prev_hash:
            changes.append(
                {
                    "timestamp": row["timestamp"],
                    "old_hash": prev_hash,
                    "new_hash": h,
                    "prompt_tokens": row.get("prompt_tokens", 0),
                }
            )
            prev_hash = h

    return changes
