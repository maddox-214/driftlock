"""
Exact in-memory response cache with LRU eviction and TTL expiry.

"Exact" means the cache key is a deterministic hash of the full request
(model + messages + relevant sampling params). A hit guarantees an identical
response would have been produced — no semantic approximation.

Thread-safe via a single threading.Lock. No external dependencies.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CacheConfig:
    enabled: bool = True
    ttl_seconds: float = 3600.0    # entries expire after 1 hour
    max_entries: int = 1000        # LRU eviction kicks in above this


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    response: Any          # openai ChatCompletion object (stored as-is)
    cached_at: float       # time.time() at write
    prompt_tokens: int
    completion_tokens: int
    hit_count: int = 0


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

# Fields that live in kwargs but should NOT affect the cache key:
#   stream  — we never cache streaming responses
#   user    — OpenAI tracing field, PII, doesn't change the response
#   model / messages — these are passed separately and handled explicitly
_SKIP_IN_KEY: frozenset[str] = frozenset({"stream", "user", "model", "messages"})


def make_cache_key(model: str, messages: list[dict], kwargs: dict) -> str:
    """
    Return a stable SHA-256 hex digest for a chat completion request.

    Call this AFTER the optimization pipeline so the key reflects what
    will actually be sent to the API.
    """
    params = {k: v for k, v in kwargs.items() if k not in _SKIP_IN_KEY}
    payload = {
        "v": 1,            # bump if the key scheme ever changes to bust all entries
        "model": model,
        "messages": messages,
        "params": params,
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# LRU + TTL cache
# ---------------------------------------------------------------------------

class ResponseCache:
    """
    Thread-safe in-memory cache backed by an OrderedDict for O(1) LRU ops.

    Access order:
      - Most-recently used entries sit at the END of the OrderedDict.
      - When the capacity limit is hit, the entry at the FRONT is evicted
        (least recently used).
    """

    def __init__(self, config: CacheConfig) -> None:
        self._cfg = config
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get(self, key: str) -> CacheEntry | None:
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return None

            entry = self._store[key]

            # TTL check
            if time.time() - entry.cached_at > self._cfg.ttl_seconds:
                del self._store[key]
                self._misses += 1
                self._expirations += 1
                return None

            # Promote to most-recently-used position
            self._store.move_to_end(key)
            entry.hit_count += 1
            self._hits += 1
            return entry

    def put(
        self,
        key: str,
        response: Any,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        with self._lock:
            if key in self._store:
                # Refresh TTL on duplicate write; don't double-count
                entry = self._store[key]
                entry.cached_at = time.time()
                self._store.move_to_end(key)
                return

            # Evict LRU entries until we're under capacity
            while len(self._store) >= self._cfg.max_entries:
                self._store.popitem(last=False)   # removes the FRONT (least recently used)
                self._evictions += 1

            self._store[key] = CacheEntry(
                response=response,
                cached_at=time.time(),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._store),
                "max_entries": self._cfg.max_entries,
                "ttl_seconds": self._cfg.ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
                "evictions": self._evictions,
                "expirations": self._expirations,
            }
