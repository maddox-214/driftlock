"""
Tests for DriftlockClient using a mocked OpenAI response.
No real API calls are made.
"""

import pytest
from unittest.mock import MagicMock, patch

from driftlock import DriftlockClient, DriftlockConfig
from driftlock.cache import CacheConfig
from driftlock.optimization import OptimizationConfig


def _mock_response(model="gpt-4o-mini", prompt_tokens=50, completion_tokens=20):
    response = MagicMock()
    response.model = model
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    response.usage.total_tokens = prompt_tokens + completion_tokens
    response.choices[0].message.content = "Hello!"
    return response


@pytest.fixture
def client(tmp_path):
    """
    DriftlockClient with a patched OpenAI backend.
    patch() replaces openai.OpenAI before __init__ runs, so self.chat is
    wired to the mock automatically — no manual re-wiring needed.
    """
    config = DriftlockConfig(
        db_path=str(tmp_path / "test.db"),
        log_json=False,
        prompt_token_warning_threshold=100,
    )
    with patch("driftlock.client.OpenAI") as MockOpenAI:
        mock_openai = MockOpenAI.return_value
        mock_openai.chat.completions.create.return_value = _mock_response()
        c = DriftlockClient(api_key="sk-test", config=config)
        yield c, mock_openai


@pytest.fixture
def shadow_client(tmp_path):
    config = DriftlockConfig(db_path=str(tmp_path / "test.db"), log_json=False)
    opt = OptimizationConfig(default_max_output_tokens=123, shadow_mode=True)
    with patch("driftlock.client.OpenAI") as MockOpenAI:
        mock_openai = MockOpenAI.return_value
        mock_openai.chat.completions.create.return_value = _mock_response()
        c = DriftlockClient(api_key="sk-test", config=config, optimization=opt)
        yield c, mock_openai


@pytest.fixture
def sampled_client(tmp_path):
    config = DriftlockConfig(db_path=str(tmp_path / "test.db"), log_json=False)
    opt = OptimizationConfig(default_max_output_tokens=77, sample_rate=0.5)
    with patch("driftlock.client.OpenAI") as MockOpenAI:
        mock_openai = MockOpenAI.return_value
        mock_openai.chat.completions.create.return_value = _mock_response()
        c = DriftlockClient(api_key="sk-test", config=config, optimization=opt)
        yield c, mock_openai


def test_create_returns_response(client):
    c, mock_openai = client
    mock_openai.chat.completions.create.return_value = _mock_response()

    response = c.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert response.choices[0].message.content == "Hello!"


def test_metrics_saved(client):
    c, mock_openai = client
    mock_openai.chat.completions.create.return_value = _mock_response(
        prompt_tokens=50, completion_tokens=20
    )

    c.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        _dl_endpoint="test_fn",
    )

    stats = c.stats(endpoint="test_fn")
    assert stats["calls"] == 1
    assert stats["total_prompt_tokens"] == 50
    assert stats["total_completion_tokens"] == 20


def test_prompt_warning_triggered(client):
    c, mock_openai = client
    mock_openai.chat.completions.create.return_value = _mock_response(
        prompt_tokens=200  # above threshold of 100
    )

    c.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "big prompt"}],
    )

    recent = c.recent_calls(limit=1)
    assert len(recent) == 1
    assert any("large" in w.lower() for w in recent[0]["warnings"])


def test_dl_labels_merged(client):
    c, mock_openai = client
    mock_openai.chat.completions.create.return_value = _mock_response()

    c.chat.completions.create(
        model="gpt-4o-mini",
        messages=[],
        _dl_labels={"env": "test"},
    )

    recent = c.recent_calls(limit=1)
    assert recent[0]["labels"]["env"] == "test"


def test_dl_kwargs_not_forwarded_to_openai(client):
    c, mock_openai = client
    mock_openai.chat.completions.create.return_value = _mock_response()

    c.chat.completions.create(
        model="gpt-4o-mini",
        messages=[],
        _dl_endpoint="my_fn",
        _dl_labels={"k": "v"},
    )

    # _dl_* kwargs must be stripped before reaching the real OpenAI call
    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert "_dl_endpoint" not in call_kwargs
    assert "_dl_labels" not in call_kwargs


def test_stats_aggregate_multiple_calls(client):
    c, mock_openai = client

    for _ in range(3):
        mock_openai.chat.completions.create.return_value = _mock_response(
            prompt_tokens=10, completion_tokens=5
        )
        c.chat.completions.create(
            model="gpt-4o-mini",
            messages=[],
            _dl_endpoint="batch",
        )

    stats = c.stats(endpoint="batch")
    assert stats["calls"] == 3
    assert stats["total_prompt_tokens"] == 30
    assert stats["total_completion_tokens"] == 15


def test_shadow_mode_does_not_modify_outgoing_request(shadow_client):
    c, mock_openai = shadow_client
    call_kwargs = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
    }

    c.chat.completions.create(**call_kwargs)

    sent_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert "max_tokens" not in sent_kwargs
    assert sent_kwargs["messages"] == call_kwargs["messages"]


def test_sampling_is_deterministic_per_key(sampled_client):
    c, mock_openai = sampled_client
    call_kwargs = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "_dl_labels": {"user_id": "u-123"},
    }

    c.chat.completions.create(**call_kwargs)
    c.chat.completions.create(**call_kwargs)

    first = mock_openai.chat.completions.create.call_args_list[0].kwargs
    second = mock_openai.chat.completions.create.call_args_list[1].kwargs
    assert ("max_tokens" in first) == ("max_tokens" in second)


def test_kill_switch_bypasses_optimization_and_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("DRIFTLOCK_ENABLED", "false")
    config = DriftlockConfig(db_path=str(tmp_path / "test.db"), log_json=False)
    opt = OptimizationConfig(default_max_output_tokens=123)
    cache_cfg = CacheConfig(ttl_seconds=3600)

    with patch("driftlock.client.OpenAI") as MockOpenAI:
        mock_openai = MockOpenAI.return_value
        mock_openai.chat.completions.create.return_value = _mock_response()
        c = DriftlockClient(
            api_key="sk-test",
            config=config,
            optimization=opt,
            cache=cache_cfg,
        )

        call_kwargs = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
        }

        c.chat.completions.create(**call_kwargs)
        c.chat.completions.create(**call_kwargs)

        assert mock_openai.chat.completions.create.call_count == 2
        sent_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert "max_tokens" not in sent_kwargs
        assert c.recent_calls(limit=1) == []
