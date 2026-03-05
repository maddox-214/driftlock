"""Tests for async acreate() support."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from driftlock import DriftlockClient, DriftlockConfig


def _mock_response(model="gpt-4o-mini", prompt_tokens=50, completion_tokens=20):
    response = MagicMock()
    response.model = model
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    response.usage.total_tokens = prompt_tokens + completion_tokens
    response.choices[0].message.content = "Hello async!"
    return response


@pytest.mark.asyncio
async def test_acreate_returns_response(tmp_path):
    config = DriftlockConfig(db_path=str(tmp_path / "test.db"), log_json=False)
    with (
        patch("driftlock.client.OpenAI"),
        patch("driftlock.client.AsyncOpenAI") as MockAsync,
    ):
        mock_async = MockAsync.return_value
        mock_async.chat.completions.create = AsyncMock(return_value=_mock_response())
        c = DriftlockClient(api_key="sk-test", config=config)

        response = await c.chat.completions.acreate(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert response.choices[0].message.content == "Hello async!"


@pytest.mark.asyncio
async def test_acreate_saves_metrics(tmp_path):
    config = DriftlockConfig(db_path=str(tmp_path / "test.db"), log_json=False)
    with (
        patch("driftlock.client.OpenAI"),
        patch("driftlock.client.AsyncOpenAI") as MockAsync,
    ):
        mock_async = MockAsync.return_value
        mock_async.chat.completions.create = AsyncMock(
            return_value=_mock_response(prompt_tokens=30, completion_tokens=10)
        )
        c = DriftlockClient(api_key="sk-test", config=config)

        await c.chat.completions.acreate(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
            _dl_endpoint="async_fn",
        )

        stats = c.stats(endpoint="async_fn")
        assert stats["calls"] == 1
        assert stats["total_prompt_tokens"] == 30


@pytest.mark.asyncio
async def test_acreate_strips_dl_kwargs(tmp_path):
    config = DriftlockConfig(db_path=str(tmp_path / "test.db"), log_json=False)
    with (
        patch("driftlock.client.OpenAI"),
        patch("driftlock.client.AsyncOpenAI") as MockAsync,
    ):
        mock_async = MockAsync.return_value
        mock_async.chat.completions.create = AsyncMock(return_value=_mock_response())
        c = DriftlockClient(api_key="sk-test", config=config)

        await c.chat.completions.acreate(
            model="gpt-4o-mini",
            messages=[],
            _dl_endpoint="fn",
            _dl_labels={"env": "test"},
        )

        call_kwargs = mock_async.chat.completions.create.call_args.kwargs
        assert "_dl_endpoint" not in call_kwargs
        assert "_dl_labels" not in call_kwargs


@pytest.mark.asyncio
async def test_acreate_kill_switch(tmp_path, monkeypatch):
    monkeypatch.setenv("DRIFTLOCK_ENABLED", "false")
    config = DriftlockConfig(db_path=str(tmp_path / "test.db"), log_json=False)
    with (
        patch("driftlock.client.OpenAI"),
        patch("driftlock.client.AsyncOpenAI") as MockAsync,
    ):
        mock_async = MockAsync.return_value
        mock_async.chat.completions.create = AsyncMock(return_value=_mock_response())
        c = DriftlockClient(api_key="sk-test", config=config)

        await c.chat.completions.acreate(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert c.recent_calls(limit=1) == []
