"""Tests for velocity / circuit breaker policy rules."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from driftlock.metrics import CallMetrics
from driftlock.policy import (
    CircuitOpenError,
    CostVelocityRule,
    PolicyEngine,
    PolicyViolationError,
    VelocityLimitRule,
)
from driftlock.storage import SQLiteStorage


def _make_metrics(cost_usd=0.01):
    return CallMetrics(
        model="gpt-4o-mini",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        latency_ms=1.0,
        estimated_cost_usd=cost_usd,
        timestamp=datetime.now(timezone.utc),
    )


def test_velocity_limit_blocks_when_exceeded(tmp_path):
    storage = SQLiteStorage(db_path=str(tmp_path / "test.db"))
    for _ in range(5):
        storage.save(_make_metrics())

    rule = VelocityLimitRule(max_requests=3, window_seconds=3600)
    ctx = {"storage": storage}
    decision = rule.evaluate(ctx)
    assert decision.allow is False
    assert decision.action == "block"
    assert decision.metadata["count"] >= 5


def test_velocity_limit_allows_when_under(tmp_path):
    storage = SQLiteStorage(db_path=str(tmp_path / "test.db"))
    storage.save(_make_metrics())

    rule = VelocityLimitRule(max_requests=10, window_seconds=3600)
    ctx = {"storage": storage}
    decision = rule.evaluate(ctx)
    assert decision.allow is True


def test_velocity_limit_allows_without_storage():
    rule = VelocityLimitRule(max_requests=1, window_seconds=60)
    decision = rule.evaluate({})
    assert decision.allow is True


def test_velocity_limit_user_scope(tmp_path):
    storage = SQLiteStorage(db_path=str(tmp_path / "test.db"))
    for _ in range(6):
        m = _make_metrics()
        m.labels = {"user_id": "u1"}
        storage.save(m)

    rule = VelocityLimitRule(max_requests=5, window_seconds=3600, scope="user")
    ctx = {"storage": storage, "labels": {"user_id": "u1"}}
    decision = rule.evaluate(ctx)
    assert decision.allow is False

    # Different user is unaffected
    ctx2 = {"storage": storage, "labels": {"user_id": "u2"}}
    decision2 = rule.evaluate(ctx2)
    assert decision2.allow is True


def test_cost_velocity_blocks_when_exceeded(tmp_path):
    storage = SQLiteStorage(db_path=str(tmp_path / "test.db"))
    for _ in range(10):
        storage.save(_make_metrics(cost_usd=0.50))  # $5 total

    rule = CostVelocityRule(max_cost_usd=1.0, window_seconds=3600)
    ctx = {"storage": storage}
    decision = rule.evaluate(ctx)
    assert decision.allow is False
    assert decision.action == "block"


def test_cost_velocity_allows_when_under(tmp_path):
    storage = SQLiteStorage(db_path=str(tmp_path / "test.db"))
    storage.save(_make_metrics(cost_usd=0.01))

    rule = CostVelocityRule(max_cost_usd=100.0, window_seconds=3600)
    ctx = {"storage": storage}
    decision = rule.evaluate(ctx)
    assert decision.allow is True


def test_velocity_raises_in_engine(tmp_path):
    from driftlock import DriftlockClient, DriftlockConfig
    from unittest.mock import patch

    storage = SQLiteStorage(db_path=str(tmp_path / "test.db"))
    for _ in range(5):
        storage.save(_make_metrics())

    policy = PolicyEngine([VelocityLimitRule(max_requests=2, window_seconds=3600)])
    config = DriftlockConfig(db_path=str(tmp_path / "test.db"), log_json=False)

    with patch("driftlock.client.OpenAI"):
        c = DriftlockClient(api_key="sk-test", config=config, policy=policy)
        with pytest.raises(PolicyViolationError):
            c.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
            )
