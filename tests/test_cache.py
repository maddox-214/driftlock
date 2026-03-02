"""
Tests for the exact response cache.

Covers:
  - ResponseCache: TTL expiry, LRU eviction, hit/miss accounting, stats
  - make_cache_key: stability, sensitivity, kwarg-order invariance
  - Context tags: driftlock.tag() propagation and label precedence
  - Client-level integration: repeated call returns cache hit; OpenAI called once
"""

import pytest
from unittest.mock import MagicMock, patch

import driftlock
from driftlock.cache import CacheConfig, CacheEntry, ResponseCache, make_cache_key
from driftlock.context import tag, get_active_tags
from driftlock import DriftlockClient, DriftlockConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _response(content="Hello!", prompt_tokens=50, completion_tokens=20):
    r = MagicMock()
    r.model = "gpt-4o-mini"
    r.usage.prompt_tokens = prompt_tokens
    r.usage.completion_tokens = completion_tokens
    r.usage.total_tokens = prompt_tokens + completion_tokens
    r.choices[0].message.content = content
    return r


MESSAGES = [{"role": "user", "content": "What is 2+2?"}]


# ---------------------------------------------------------------------------
# ResponseCache — unit tests
# ---------------------------------------------------------------------------

class TestResponseCache:
    def _cache(self, **kwargs) -> ResponseCache:
        return ResponseCache(CacheConfig(**kwargs))

    # --- basic get/put ---

    def test_miss_on_empty(self):
        assert self._cache().get("nonexistent") is None

    def test_hit_after_put(self):
        c = self._cache()
        c.put("k1", _response(), 50, 20)
        entry = c.get("k1")
        assert entry is not None
        assert entry.prompt_tokens == 50
        assert entry.completion_tokens == 20

    def test_hit_count_increments(self):
        c = self._cache()
        c.put("k1", _response(), 50, 20)
        c.get("k1")
        c.get("k1")
        assert c.get("k1").hit_count == 3  # incremented on each get call

    def test_duplicate_put_refreshes_ttl_not_hit_count(self):
        c = self._cache()
        c.put("k1", _response(), 50, 20)
        c.get("k1")  # hit_count = 1
        c.put("k1", _response(), 50, 20)  # duplicate write — refreshes TTL
        entry = c.get("k1")
        assert entry.hit_count == 2  # still counting hits from the original entry

    # --- TTL expiry ---

    def test_expired_entry_returns_none(self):
        c = self._cache(ttl_seconds=60)
        with patch("driftlock.cache.time") as mock_t:
            mock_t.time.return_value = 1000.0
            c.put("k1", _response(), 50, 20)

            mock_t.time.return_value = 1061.0   # 61 seconds later — expired
            assert c.get("k1") is None

    def test_entry_valid_just_before_ttl(self):
        c = self._cache(ttl_seconds=60)
        with patch("driftlock.cache.time") as mock_t:
            mock_t.time.return_value = 1000.0
            c.put("k1", _response(), 50, 20)

            mock_t.time.return_value = 1059.9   # still within TTL
            assert c.get("k1") is not None

    def test_expiration_counted_in_stats(self):
        c = self._cache(ttl_seconds=60)
        with patch("driftlock.cache.time") as mock_t:
            mock_t.time.return_value = 1000.0
            c.put("k1", _response(), 50, 20)

            mock_t.time.return_value = 1061.0
            c.get("k1")   # triggers expiration

        assert c.stats()["expirations"] == 1
        assert c.stats()["size"] == 0

    # --- LRU eviction ---

    def test_lru_evicts_oldest_when_full(self):
        c = self._cache(max_entries=3)
        c.put("a", _response(), 1, 1)
        c.put("b", _response(), 1, 1)
        c.put("c", _response(), 1, 1)
        c.put("d", _response(), 1, 1)  # evicts "a" (oldest)

        assert c.get("a") is None    # evicted
        assert c.get("b") is not None
        assert c.get("c") is not None
        assert c.get("d") is not None
        assert c.stats()["evictions"] == 1

    def test_access_promotes_entry_above_older_entries(self):
        c = self._cache(max_entries=3)
        c.put("a", _response(), 1, 1)
        c.put("b", _response(), 1, 1)
        c.put("c", _response(), 1, 1)

        c.get("a")       # promote "a" — now "b" is the LRU

        c.put("d", _response(), 1, 1)   # should evict "b", not "a"
        assert c.get("b") is None    # evicted
        assert c.get("a") is not None
        assert c.get("c") is not None
        assert c.get("d") is not None

    def test_stats_hit_rate(self):
        c = self._cache()
        c.put("k", _response(), 10, 5)
        c.get("k")      # hit
        c.get("k")      # hit
        c.get("miss1")  # miss
        stats = c.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2 / 3, rel=1e-3)


# ---------------------------------------------------------------------------
# make_cache_key
# ---------------------------------------------------------------------------

class TestCacheKey:
    def test_same_inputs_produce_same_key(self):
        k1 = make_cache_key("gpt-4o-mini", MESSAGES, {"temperature": 0.0})
        k2 = make_cache_key("gpt-4o-mini", MESSAGES, {"temperature": 0.0})
        assert k1 == k2

    def test_different_model_different_key(self):
        k1 = make_cache_key("gpt-4o-mini", MESSAGES, {})
        k2 = make_cache_key("gpt-4o",      MESSAGES, {})
        assert k1 != k2

    def test_different_messages_different_key(self):
        k1 = make_cache_key("gpt-4o-mini", [{"role": "user", "content": "Hi"}], {})
        k2 = make_cache_key("gpt-4o-mini", [{"role": "user", "content": "Bye"}], {})
        assert k1 != k2

    def test_kwarg_order_invariant(self):
        k1 = make_cache_key("gpt-4o-mini", MESSAGES, {"temperature": 0.0, "max_tokens": 512})
        k2 = make_cache_key("gpt-4o-mini", MESSAGES, {"max_tokens": 512, "temperature": 0.0})
        assert k1 == k2

    def test_different_temperature_different_key(self):
        k1 = make_cache_key("gpt-4o-mini", MESSAGES, {"temperature": 0.0})
        k2 = make_cache_key("gpt-4o-mini", MESSAGES, {"temperature": 1.0})
        assert k1 != k2

    def test_stream_kwarg_excluded(self):
        """stream shouldn't affect the key (we never cache streaming responses)."""
        k1 = make_cache_key("gpt-4o-mini", MESSAGES, {"stream": True})
        k2 = make_cache_key("gpt-4o-mini", MESSAGES, {"stream": False})
        k3 = make_cache_key("gpt-4o-mini", MESSAGES, {})
        assert k1 == k2 == k3

    def test_user_kwarg_excluded(self):
        """'user' is PII tracking metadata, not part of the response content."""
        k1 = make_cache_key("gpt-4o-mini", MESSAGES, {"user": "user_a"})
        k2 = make_cache_key("gpt-4o-mini", MESSAGES, {"user": "user_b"})
        assert k1 == k2

    def test_key_is_64_char_hex(self):
        k = make_cache_key("gpt-4o-mini", MESSAGES, {})
        assert len(k) == 64
        assert all(c in "0123456789abcdef" for c in k)


# ---------------------------------------------------------------------------
# Context tags
# ---------------------------------------------------------------------------

class TestContextTags:
    def test_empty_by_default(self):
        assert get_active_tags() == {}

    def test_tags_visible_inside_block(self):
        with tag(user_id="u1", feature="chat"):
            assert get_active_tags() == {"user_id": "u1", "feature": "chat"}

    def test_tags_cleared_after_block(self):
        with tag(user_id="u1"):
            pass
        assert get_active_tags() == {}

    def test_nested_tags_merge(self):
        with tag(a="outer"):
            with tag(b="inner"):
                assert get_active_tags() == {"a": "outer", "b": "inner"}
            assert get_active_tags() == {"a": "outer"}

    def test_inner_tag_overrides_outer(self):
        with tag(key="outer"):
            with tag(key="inner"):
                assert get_active_tags()["key"] == "inner"
            assert get_active_tags()["key"] == "outer"

    def test_tags_are_copied_not_shared(self):
        """Mutations to the returned dict should not affect the context."""
        with tag(a="1"):
            tags = get_active_tags()
            tags["b"] = "injected"
            assert "b" not in get_active_tags()


# ---------------------------------------------------------------------------
# Client integration — cache hit skips OpenAI call
# ---------------------------------------------------------------------------

@pytest.fixture
def cached_client(tmp_path):
    config = DriftlockConfig(db_path=str(tmp_path / "test.db"), log_json=False)
    cache_cfg = CacheConfig(ttl_seconds=3600, max_entries=100)
    with patch("driftlock.client.OpenAI") as MockOpenAI:
        mock_openai = MockOpenAI.return_value
        mock_openai.chat.completions.create.return_value = _response()
        c = DriftlockClient(api_key="sk-test", config=config, cache=cache_cfg)
        yield c, mock_openai


def test_repeated_identical_request_hits_cache(cached_client):
    """Second call with identical inputs must return from cache; OpenAI called once."""
    c, mock_openai = cached_client
    mock_openai.chat.completions.create.return_value = _response(
        prompt_tokens=50, completion_tokens=20
    )

    call_kwargs = dict(model="gpt-4o-mini", messages=MESSAGES, temperature=0.0)

    r1 = c.chat.completions.create(**call_kwargs)
    r2 = c.chat.completions.create(**call_kwargs)

    # OpenAI was only called once
    assert mock_openai.chat.completions.create.call_count == 1
    # Both responses are the same object
    assert r1 is r2


def test_cache_miss_recorded_then_hit_recorded(cached_client):
    """recent_calls reflects one miss then one hit."""
    c, mock_openai = cached_client
    mock_openai.chat.completions.create.return_value = _response(
        prompt_tokens=50, completion_tokens=20
    )

    call_kwargs = dict(model="gpt-4o-mini", messages=MESSAGES, temperature=0.0)
    c.chat.completions.create(**call_kwargs)   # miss
    c.chat.completions.create(**call_kwargs)   # hit

    recent = c.recent_calls(limit=2)
    # Most recent first
    assert recent[0]["cache_hit"] is True
    assert recent[1]["cache_hit"] is False


def test_cache_hit_metrics_show_zero_cost(cached_client):
    c, mock_openai = cached_client
    mock_openai.chat.completions.create.return_value = _response(
        prompt_tokens=100, completion_tokens=40
    )

    call_kwargs = dict(model="gpt-4o-mini", messages=MESSAGES)
    c.chat.completions.create(**call_kwargs)   # miss
    c.chat.completions.create(**call_kwargs)   # hit

    hit = c.recent_calls(limit=1)[0]
    assert hit["cache_hit"] is True
    assert hit["prompt_tokens"] == 0
    assert hit["completion_tokens"] == 0
    assert hit["estimated_cost_usd"] == 0.0
    assert hit["tokens_saved_prompt"] == 100
    assert hit["tokens_saved_completion"] == 40
    assert hit["estimated_savings_usd"] is not None
    assert hit["estimated_savings_usd"] > 0


def test_different_messages_not_cached_together(cached_client):
    """Different messages must produce different cache keys → two OpenAI calls."""
    c, mock_openai = cached_client
    mock_openai.chat.completions.create.return_value = _response()

    c.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "Hello"}]
    )
    c.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "Goodbye"}]
    )

    assert mock_openai.chat.completions.create.call_count == 2


def test_streaming_requests_not_cached(cached_client):
    """stream=True calls should bypass the cache and always hit OpenAI."""
    c, mock_openai = cached_client
    mock_openai.chat.completions.create.return_value = _response()

    c.chat.completions.create(model="gpt-4o-mini", messages=MESSAGES, stream=True)
    c.chat.completions.create(model="gpt-4o-mini", messages=MESSAGES, stream=True)

    assert mock_openai.chat.completions.create.call_count == 2


def test_context_tags_flow_into_cache_hit_metrics(cached_client):
    """Tags set via driftlock.tag() must appear in both the miss and hit records."""
    c, mock_openai = cached_client
    mock_openai.chat.completions.create.return_value = _response()

    with driftlock.tag(user_id="u_42", feature="search"):
        c.chat.completions.create(model="gpt-4o-mini", messages=MESSAGES)  # miss
        c.chat.completions.create(model="gpt-4o-mini", messages=MESSAGES)  # hit

    recent = c.recent_calls(limit=2)
    for record in recent:
        assert record["labels"]["user_id"] == "u_42"
        assert record["labels"]["feature"] == "search"


def test_per_call_labels_override_context_tags(cached_client):
    """_dl_labels wins over driftlock.tag() for the same key."""
    c, mock_openai = cached_client
    mock_openai.chat.completions.create.return_value = _response()

    with driftlock.tag(env="staging"):
        c.chat.completions.create(
            model="gpt-4o-mini",
            messages=MESSAGES,
            _dl_labels={"env": "prod"},
        )

    recent = c.recent_calls(limit=1)
    assert recent[0]["labels"]["env"] == "prod"


def test_cache_stats_reflect_hits_and_misses(cached_client):
    c, mock_openai = cached_client
    mock_openai.chat.completions.create.return_value = _response()

    c.chat.completions.create(model="gpt-4o-mini", messages=MESSAGES)  # miss
    c.chat.completions.create(model="gpt-4o-mini", messages=MESSAGES)  # hit
    c.chat.completions.create(model="gpt-4o-mini", messages=MESSAGES)  # hit

    stats = c.cache_stats()
    assert stats["enabled"] is True
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(2 / 3, rel=1e-3)
