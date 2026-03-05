"""
Policy engine for enforcing spend, model, and velocity controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable

from .pricing import estimate_cost
from .tokenizer import count_messages_tokens


# ------------------------------------------------------------------ #
# Decision + errors
# ------------------------------------------------------------------ #

@dataclass
class RuleDecision:
    allow: bool
    action: str | None = None  # "block" | "fallback" | "downgrade" | None
    metadata: dict = field(default_factory=dict)


class PolicyViolationError(Exception):
    """Raised when a policy rule blocks the request."""

    def __init__(self, rule_name: str, decision: RuleDecision) -> None:
        self.rule_name = rule_name
        self.decision = decision
        super().__init__(f"Policy blocked request: {rule_name}")


class CircuitOpenError(PolicyViolationError):
    """Raised when a velocity circuit breaker trips."""


# ------------------------------------------------------------------ #
# Base rule
# ------------------------------------------------------------------ #

class BaseRule:
    """Base class for policy rules."""

    def evaluate(self, request_context: dict) -> RuleDecision:
        raise NotImplementedError


# ------------------------------------------------------------------ #
# Cost rules
# ------------------------------------------------------------------ #

class MaxCostPerRequestRule(BaseRule):
    def __init__(self, max_usd: float) -> None:
        self._max_usd = max_usd

    def evaluate(self, request_context: dict) -> RuleDecision:
        model = request_context.get("model", "unknown")
        messages = request_context.get("messages", [])
        kwargs = request_context.get("kwargs", {})
        prompt_tokens = count_messages_tokens(messages, model)
        max_output = kwargs.get("max_tokens", 1000)
        estimated = estimate_cost(model, prompt_tokens, max_output)
        if estimated is not None and estimated > self._max_usd:
            return RuleDecision(
                allow=False,
                action="block",
                metadata={"estimated_cost_usd": estimated, "max_usd": self._max_usd},
            )
        return RuleDecision(allow=True)


class MonthlyBudgetRule(BaseRule):
    """
    Block requests once a monthly spend cap is reached.

    scope="workspace"  — aggregate spend across all calls
    scope="user"       — per-user spend, reads user_id from labels
    """

    def __init__(self, max_usd: float, scope: str = "workspace") -> None:
        self._max_usd = max_usd
        self._scope = scope

    def evaluate(self, request_context: dict) -> RuleDecision:
        storage = request_context.get("storage")
        if storage is None:
            return RuleDecision(allow=True)

        now = request_context.get("now") or datetime.now(timezone.utc)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        since = month_start.isoformat()

        if self._scope == "user":
            user_id = request_context.get("labels", {}).get("user_id")
            if not user_id:
                return RuleDecision(allow=True)
            spent = storage.cost_since(since, user_id=user_id)
        else:
            agg = storage.aggregate(since=since)
            spent = agg.get("total_cost_usd", 0.0) or 0.0

        if spent >= self._max_usd:
            return RuleDecision(
                allow=False,
                action="block",
                metadata={"spent_usd": spent, "max_usd": self._max_usd, "scope": self._scope},
            )
        return RuleDecision(allow=True)


class PerUserBudgetRule(BaseRule):
    """
    Enforce individual per-user monthly spend caps.

    user_budgets: mapping of user_id → max USD per month
    default_max_usd: applied to any user not in user_budgets (None = no cap)
    """

    def __init__(
        self,
        user_budgets: dict[str, float],
        default_max_usd: float | None = None,
    ) -> None:
        self._budgets = user_budgets
        self._default = default_max_usd

    def evaluate(self, request_context: dict) -> RuleDecision:
        user_id = request_context.get("labels", {}).get("user_id")
        if not user_id:
            return RuleDecision(allow=True)

        max_usd = self._budgets.get(user_id, self._default)
        if max_usd is None:
            return RuleDecision(allow=True)

        storage = request_context.get("storage")
        if storage is None:
            return RuleDecision(allow=True)

        now = request_context.get("now") or datetime.now(timezone.utc)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        spent = storage.cost_since(month_start.isoformat(), user_id=user_id)

        if spent >= max_usd:
            return RuleDecision(
                allow=False,
                action="block",
                metadata={"user_id": user_id, "spent_usd": spent, "max_usd": max_usd},
            )
        return RuleDecision(allow=True)


class ForecastBudgetRule(BaseRule):
    """
    Block requests when the projected monthly spend will exceed the budget.

    Uses a rolling lookback window to extrapolate current daily spend rate.
    More aggressive than MonthlyBudgetRule — blocks before the budget is hit.
    """

    def __init__(self, monthly_budget_usd: float, lookback_days: int = 7) -> None:
        self._budget = monthly_budget_usd
        self._lookback = lookback_days

    def evaluate(self, request_context: dict) -> RuleDecision:
        storage = request_context.get("storage")
        if storage is None:
            return RuleDecision(allow=True)

        trend = storage.daily_cost_trend(lookback_days=self._lookback)
        if not trend:
            return RuleDecision(allow=True)

        daily_avg = sum(d["cost_usd"] for d in trend) / self._lookback
        projected = daily_avg * 30

        if projected >= self._budget:
            return RuleDecision(
                allow=False,
                action="block",
                metadata={
                    "projected_monthly_usd": round(projected, 4),
                    "monthly_budget_usd": self._budget,
                    "daily_avg_usd": round(daily_avg, 6),
                },
            )
        return RuleDecision(allow=True)


# ------------------------------------------------------------------ #
# Model rules
# ------------------------------------------------------------------ #

class RestrictModelRule(BaseRule):
    def __init__(
        self,
        disallowed_models: set[str],
        condition: Callable[[dict], bool] | None = None,
    ) -> None:
        self._disallowed = disallowed_models
        self._condition = condition or (lambda _ctx: True)

    def evaluate(self, request_context: dict) -> RuleDecision:
        model = request_context.get("model", "unknown")
        if self._condition(request_context) and model in self._disallowed:
            return RuleDecision(
                allow=False,
                action="block",
                metadata={"model": model},
            )
        return RuleDecision(allow=True)


class TagBasedModelDowngradeRule(BaseRule):
    def __init__(
        self,
        condition: Callable[[dict], bool],
        downgrade_to: str,
    ) -> None:
        self._condition = condition
        self._downgrade_to = downgrade_to

    def evaluate(self, request_context: dict) -> RuleDecision:
        if self._condition(request_context):
            return RuleDecision(
                allow=True,
                action="downgrade",
                metadata={"downgrade_to": self._downgrade_to},
            )
        return RuleDecision(allow=True)


# ------------------------------------------------------------------ #
# Velocity / circuit breaker rules
# ------------------------------------------------------------------ #

class VelocityLimitRule(BaseRule):
    """
    Circuit breaker: block requests when call rate exceeds a threshold.

    Prevents runaway loops from burning through budgets.

    scope="workspace"  — counts all calls in the window
    scope="user"       — counts per user_id (reads from labels)

    Example: max 100 requests per 60 seconds workspace-wide.
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: int,
        scope: str = "workspace",
    ) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._scope = scope

    def evaluate(self, request_context: dict) -> RuleDecision:
        storage = request_context.get("storage")
        if storage is None:
            return RuleDecision(allow=True)

        since = (
            datetime.now(timezone.utc) - timedelta(seconds=self._window)
        ).isoformat()

        if self._scope == "user":
            user_id = request_context.get("labels", {}).get("user_id")
            if not user_id:
                return RuleDecision(allow=True)
            count = storage.count_since(since, user_id=user_id)
        else:
            count = storage.count_since(since)

        if count >= self._max:
            return RuleDecision(
                allow=False,
                action="block",
                metadata={
                    "count": count,
                    "max_requests": self._max,
                    "window_seconds": self._window,
                    "scope": self._scope,
                },
            )
        return RuleDecision(allow=True)


class CostVelocityRule(BaseRule):
    """
    Circuit breaker: block requests when spend rate exceeds a threshold.

    More granular than MonthlyBudgetRule — triggers on hourly/daily spikes
    before the monthly cap is exhausted.

    Example: block if more than $5 spent in the last hour.
    """

    def __init__(self, max_cost_usd: float, window_seconds: int) -> None:
        self._max = max_cost_usd
        self._window = window_seconds

    def evaluate(self, request_context: dict) -> RuleDecision:
        storage = request_context.get("storage")
        if storage is None:
            return RuleDecision(allow=True)

        since = (
            datetime.now(timezone.utc) - timedelta(seconds=self._window)
        ).isoformat()
        spent = storage.cost_since(since)

        if spent >= self._max:
            return RuleDecision(
                allow=False,
                action="block",
                metadata={
                    "spent_usd": spent,
                    "max_cost_usd": self._max,
                    "window_seconds": self._window,
                },
            )
        return RuleDecision(allow=True)


# ------------------------------------------------------------------ #
# Engine
# ------------------------------------------------------------------ #

class PolicyEngine:
    def __init__(self, rules: list[BaseRule]) -> None:
        self._rules = rules

    def evaluate(self, request_context: dict) -> list[tuple[str, RuleDecision]]:
        return [
            (rule.__class__.__name__, rule.evaluate(request_context))
            for rule in self._rules
        ]
