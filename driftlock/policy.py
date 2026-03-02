"""
Minimal policy engine for enforcing spend and model controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from .pricing import estimate_cost
from .tokenizer import count_messages_tokens


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


class BaseRule:
    """Base class for policy rules."""

    def evaluate(self, request_context: dict) -> RuleDecision:
        raise NotImplementedError


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


class MonthlyBudgetRule(BaseRule):
    def __init__(self, max_usd: float, scope: str = "workspace") -> None:
        self._max_usd = max_usd
        self._scope = scope

    def evaluate(self, request_context: dict) -> RuleDecision:
        if self._scope != "workspace":
            return RuleDecision(allow=True)
        storage = request_context.get("storage")
        if storage is None:
            return RuleDecision(allow=True)
        now = request_context.get("now") or datetime.now(timezone.utc)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        agg = storage.aggregate(since=month_start.isoformat())
        spent = agg.get("total_cost_usd", 0.0) or 0.0
        if spent >= self._max_usd:
            return RuleDecision(
                allow=False,
                action="block",
                metadata={"spent_usd": spent, "max_usd": self._max_usd},
            )
        return RuleDecision(allow=True)


class PolicyEngine:
    def __init__(self, rules: list[BaseRule]) -> None:
        self._rules = rules

    def evaluate(self, request_context: dict) -> list[tuple[str, RuleDecision]]:
        return [(rule.__class__.__name__, rule.evaluate(request_context)) for rule in self._rules]
