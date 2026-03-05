"""Tests for per-user budget enforcement."""

from datetime import datetime, timezone

import pytest

from driftlock.metrics import CallMetrics
from driftlock.policy import (
    MonthlyBudgetRule,
    PerUserBudgetRule,
    PolicyViolationError,
)
from driftlock.storage import SQLiteStorage


def _make_metrics(cost_usd=0.05, user_id="u1"):
    m = CallMetrics(
        model="gpt-4o-mini",
        prompt_tokens=50,
        completion_tokens=20,
        total_tokens=70,
        latency_ms=1.0,
        estimated_cost_usd=cost_usd,
        timestamp=datetime.now(timezone.utc),
        labels={"user_id": user_id},
    )
    return m


def test_monthly_budget_user_scope_blocks(tmp_path):
    storage = SQLiteStorage(db_path=str(tmp_path / "test.db"))
    storage.save(_make_metrics(cost_usd=0.10, user_id="u1"))
    storage.save(_make_metrics(cost_usd=0.10, user_id="u1"))

    rule = MonthlyBudgetRule(max_usd=0.15, scope="user")
    ctx = {"storage": storage, "labels": {"user_id": "u1"}, "now": datetime.now(timezone.utc)}
    decision = rule.evaluate(ctx)
    assert decision.allow is False
    assert decision.metadata["scope"] == "user"


def test_monthly_budget_user_scope_does_not_affect_other_user(tmp_path):
    storage = SQLiteStorage(db_path=str(tmp_path / "test.db"))
    storage.save(_make_metrics(cost_usd=0.10, user_id="u1"))
    storage.save(_make_metrics(cost_usd=0.10, user_id="u1"))

    rule = MonthlyBudgetRule(max_usd=0.15, scope="user")
    ctx = {"storage": storage, "labels": {"user_id": "u2"}, "now": datetime.now(timezone.utc)}
    decision = rule.evaluate(ctx)
    assert decision.allow is True


def test_per_user_budget_rule_blocks_specific_user(tmp_path):
    storage = SQLiteStorage(db_path=str(tmp_path / "test.db"))
    storage.save(_make_metrics(cost_usd=0.50, user_id="premium"))
    storage.save(_make_metrics(cost_usd=0.60, user_id="premium"))  # $1.10 total

    rule = PerUserBudgetRule(user_budgets={"premium": 1.00})
    ctx = {"storage": storage, "labels": {"user_id": "premium"}, "now": datetime.now(timezone.utc)}
    decision = rule.evaluate(ctx)
    assert decision.allow is False
    assert decision.metadata["user_id"] == "premium"


def test_per_user_budget_rule_allows_different_tier(tmp_path):
    storage = SQLiteStorage(db_path=str(tmp_path / "test.db"))
    for _ in range(20):
        storage.save(_make_metrics(cost_usd=0.50, user_id="heavy"))

    # "light" user has their own cap that isn't breached
    rule = PerUserBudgetRule(user_budgets={"light": 100.0, "heavy": 100.0})
    ctx = {"storage": storage, "labels": {"user_id": "light"}, "now": datetime.now(timezone.utc)}
    decision = rule.evaluate(ctx)
    assert decision.allow is True


def test_per_user_budget_rule_applies_default_cap(tmp_path):
    storage = SQLiteStorage(db_path=str(tmp_path / "test.db"))
    storage.save(_make_metrics(cost_usd=2.0, user_id="anon"))

    rule = PerUserBudgetRule(user_budgets={}, default_max_usd=1.0)
    ctx = {"storage": storage, "labels": {"user_id": "anon"}, "now": datetime.now(timezone.utc)}
    decision = rule.evaluate(ctx)
    assert decision.allow is False


def test_per_user_budget_rule_skips_when_no_user_id():
    rule = PerUserBudgetRule(user_budgets={"u1": 1.0}, default_max_usd=0.01)
    ctx = {"labels": {}}
    decision = rule.evaluate(ctx)
    assert decision.allow is True


def test_top_users_query(tmp_path):
    storage = SQLiteStorage(db_path=str(tmp_path / "test.db"))
    for _ in range(3):
        storage.save(_make_metrics(cost_usd=0.10, user_id="alice"))
    for _ in range(2):
        storage.save(_make_metrics(cost_usd=0.20, user_id="bob"))

    results = storage.top_users(limit=10)
    assert len(results) == 2
    # bob spent more ($0.40), so should be first
    assert results[0]["user_id"] == "bob"
    assert results[1]["user_id"] == "alice"
