"""Tests for cost forecasting."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from driftlock import DriftlockClient, DriftlockConfig
from driftlock.metrics import CallMetrics
from driftlock.policy import ForecastBudgetRule, PolicyEngine, PolicyViolationError
from driftlock.storage import SQLiteStorage


def _make_metrics(cost_usd=0.10):
    return CallMetrics(
        model="gpt-4o",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        latency_ms=1.0,
        estimated_cost_usd=cost_usd,
        timestamp=datetime.now(timezone.utc),
    )


def test_forecast_empty_returns_zeros(tmp_path):
    config = DriftlockConfig(db_path=str(tmp_path / "test.db"), log_json=False)
    with patch("driftlock.client.OpenAI"), patch("driftlock.client.AsyncOpenAI"):
        c = DriftlockClient(api_key="sk-test", config=config)
    result = c.forecast()
    assert result["projected_monthly_usd"] == 0.0
    assert result["data_points"] == 0


def test_forecast_projects_from_trend(tmp_path):
    storage = SQLiteStorage(db_path=str(tmp_path / "test.db"))
    for _ in range(3):
        storage.save(_make_metrics(cost_usd=1.0))  # $3 total today

    config = DriftlockConfig(db_path=str(tmp_path / "test.db"), log_json=False)
    with patch("driftlock.client.OpenAI"), patch("driftlock.client.AsyncOpenAI"):
        c = DriftlockClient(api_key="sk-test", config=config)
    result = c.forecast(lookback_days=1)
    assert result["daily_avg_usd"] == pytest.approx(3.0, rel=0.01)
    assert result["projected_monthly_usd"] == pytest.approx(90.0, rel=0.01)


def test_forecast_budget_rule_blocks_when_projected_exceeds(tmp_path):
    storage = SQLiteStorage(db_path=str(tmp_path / "test.db"))
    for _ in range(5):
        storage.save(_make_metrics(cost_usd=10.0))  # $50/day projected = $1500/month

    rule = ForecastBudgetRule(monthly_budget_usd=100.0, lookback_days=1)
    ctx = {"storage": storage}
    decision = rule.evaluate(ctx)
    assert decision.allow is False
    assert decision.metadata["projected_monthly_usd"] > 100.0


def test_forecast_budget_rule_allows_when_under(tmp_path):
    storage = SQLiteStorage(db_path=str(tmp_path / "test.db"))
    storage.save(_make_metrics(cost_usd=0.01))

    rule = ForecastBudgetRule(monthly_budget_usd=1000.0, lookback_days=7)
    decision = rule.evaluate({"storage": storage})
    assert decision.allow is True


def test_forecast_budget_rule_allows_without_storage():
    rule = ForecastBudgetRule(monthly_budget_usd=10.0)
    decision = rule.evaluate({})
    assert decision.allow is True
