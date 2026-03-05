"""Tests for streaming response interceptor."""

from unittest.mock import MagicMock

import pytest

from driftlock.streaming import StreamingInterceptor
from driftlock.storage import NoopStorage
from driftlock.logger import DriftlockLogger
from driftlock.config import DriftlockConfig


def _make_chunk(content: str = "", usage=None):
    chunk = MagicMock()
    chunk.usage = usage
    choice = MagicMock()
    choice.delta.content = content
    chunk.choices = [choice]
    return chunk


def _make_usage(prompt=50, completion=20):
    u = MagicMock()
    u.prompt_tokens = prompt
    u.completion_tokens = completion
    return u


def _make_interceptor(chunks, pre_tokens=50):
    storage = NoopStorage()
    logger = DriftlockLogger(log_json=False, log_level="WARNING")
    config = DriftlockConfig(storage_backend="none", log_json=False)
    import time
    return StreamingInterceptor(
        stream=iter(chunks),
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        pre_call_prompt_tokens=pre_tokens,
        start_time=time.perf_counter(),
        endpoint="test",
        labels={},
        storage=storage,
        logger=logger,
        config=config,
    )


def test_streaming_yields_all_chunks():
    chunks = [_make_chunk("Hello"), _make_chunk(" world"), _make_chunk("!")]
    interceptor = _make_interceptor(chunks)
    collected = list(interceptor)
    assert len(collected) == 3


def test_streaming_uses_api_usage_when_present():
    final_usage = _make_usage(prompt=60, completion=25)
    chunks = [
        _make_chunk("Hello"),
        _make_chunk("!", usage=final_usage),  # final chunk has usage
    ]
    storage = MagicMock()
    storage.save = MagicMock()
    logger = DriftlockLogger(log_json=False, log_level="WARNING")
    config = DriftlockConfig(storage_backend="none", log_json=False)
    import time
    interceptor = StreamingInterceptor(
        stream=iter(chunks),
        model="gpt-4o-mini",
        messages=[],
        pre_call_prompt_tokens=40,
        start_time=time.perf_counter(),
        endpoint=None,
        labels={},
        storage=storage,
        logger=logger,
        config=config,
    )
    list(interceptor)
    assert storage.save.called
    saved_metrics = storage.save.call_args[0][0]
    assert saved_metrics.prompt_tokens == 60
    assert saved_metrics.completion_tokens == 25


def test_streaming_falls_back_to_char_estimation():
    # No usage on any chunk — falls back to character counting
    chunks = [_make_chunk("Hello world this is a test response")]
    storage = MagicMock()
    storage.save = MagicMock()
    logger = DriftlockLogger(log_json=False, log_level="WARNING")
    config = DriftlockConfig(storage_backend="none", log_json=False)
    import time
    interceptor = StreamingInterceptor(
        stream=iter(chunks),
        model="gpt-4o-mini",
        messages=[],
        pre_call_prompt_tokens=30,
        start_time=time.perf_counter(),
        endpoint=None,
        labels={},
        storage=storage,
        logger=logger,
        config=config,
    )
    list(interceptor)
    saved = storage.save.call_args[0][0]
    assert saved.completion_tokens > 0
    assert saved.prompt_tokens == 30
