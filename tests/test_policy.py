"""
Tests for the policy engine and client integration.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from driftlock import DriftlockClient, DriftlockConfig
from driftlock.metrics import CallMetrics
from driftlock.policy import (
    MaxCostPerRequestRule,
    MonthlyBudgetRule,
    PolicyEngine,
    PolicyViolationError,
    RestrictModelRule,
    TagBasedModelDowngradeRule,
)
from driftlock.storage import SQLiteStorage


def _mock_response(model="gpt-4o-mini", prompt_tokens=10, completion_tokens=5):
    response = MagicMock()
    response.model = model
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    response.usage.total_tokens = prompt_tokens + completion_tokens
    response.choices[0].message.content = "ok"
    return response


def _make_metrics(cost_usd: float) -> CallMetrics:
    return CallMetrics(
        model="gpt-4o-mini",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        latency_ms=1.0,
        estimated_cost_usd=cost_usd,
        timestamp=datetime.now(timezone.utc),
    )


def test_max_cost_rule_blocks_when_exceeded():
    rule = MaxCostPerRequestRule(max_usd=0.000001)
    ctx = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "kwargs": {"max_tokens": 1000},
    }
    with patch("driftlock.policy.count_messages_tokens", return_value=10000):
        decision = rule.evaluate(ctx)
    assert decision.allow is False
    assert decision.action == "block"


def test_restrict_model_rule_blocks_when_disallowed():
    rule = RestrictModelRule(disallowed_models={"gpt-4o"})
    ctx = {"model": "gpt-4o"}
    decision = rule.evaluate(ctx)
    assert decision.allow is False
    assert decision.action == "block"


def test_tag_based_downgrade_rule_allows_with_action():
    rule = TagBasedModelDowngradeRule(
        condition=lambda c: c.get("labels", {}).get("tier") == "free",
        downgrade_to="gpt-4o-mini",
    )
    ctx = {"labels": {"tier": "free"}}
    decision = rule.evaluate(ctx)
    assert decision.allow is True
    assert decision.action == "downgrade"
    assert decision.metadata["downgrade_to"] == "gpt-4o-mini"


def test_monthly_budget_rule_blocks_when_spent_exceeds(tmp_path):
    storage = SQLiteStorage(db_path=str(tmp_path / "test.db"))
    storage.save(_make_metrics(0.02))
    storage.save(_make_metrics(0.02))

    rule = MonthlyBudgetRule(max_usd=0.01)
    ctx = {"storage": storage, "now": datetime.now(timezone.utc)}
    decision = rule.evaluate(ctx)
    assert decision.allow is False
    assert decision.action == "block"


def test_policy_downgrade_applies_in_client(tmp_path):
    config = DriftlockConfig(db_path=str(tmp_path / "test.db"), log_json=False)
    policy = PolicyEngine(
        [
            TagBasedModelDowngradeRule(
                condition=lambda c: c.get("labels", {}).get("tier") == "free",
                downgrade_to="gpt-4o-mini",
            )
        ]
    )
    with patch("driftlock.client.OpenAI") as MockOpenAI:
        mock_openai = MockOpenAI.return_value
        mock_openai.chat.completions.create.return_value = _mock_response()
        c = DriftlockClient(api_key="sk-test", config=config, policy=policy)

        c.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            _dl_labels={"tier": "free"},
        )

        sent = mock_openai.chat.completions.create.call_args.kwargs
        assert sent["model"] == "gpt-4o-mini"


def test_policy_block_raises_in_client(tmp_path):
    config = DriftlockConfig(db_path=str(tmp_path / "test.db"), log_json=False)
    policy = PolicyEngine([RestrictModelRule(disallowed_models={"gpt-4o"})])
    with patch("driftlock.client.OpenAI") as MockOpenAI:
        mock_openai = MockOpenAI.return_value
        mock_openai.chat.completions.create.return_value = _mock_response()
        c = DriftlockClient(api_key="sk-test", config=config, policy=policy)

        with pytest.raises(PolicyViolationError):
            c.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert mock_openai.chat.completions.create.call_count == 0
