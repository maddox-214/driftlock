"""
OptimizationPipeline — deterministic pre-call optimizations applied before
each OpenAI API request.

Optimizations are applied in this order:
  1. Prompt trimming   — shrink chat history to a token budget
  2. Output cap        — inject max_tokens when the caller omits it
  3. Budget guardrail  — abort or reroute calls that would exceed a cost limit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .pricing import estimate_cost
from .tokenizer import count_messages_tokens


# ---------------------------------------------------------------------------
# Public error
# ---------------------------------------------------------------------------

class BudgetExceededError(Exception):
    """Raised when an API call would exceed the configured per-request budget."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OptimizationConfig:
    # --- Prompt trimming ---
    max_prompt_tokens: int | None = None        # token budget for the prompt
    keep_last_n_messages: int = 10              # always keep the N most recent turns
    always_keep_system: bool = True             # system message is never dropped

    # --- Output cap ---
    default_max_output_tokens: int | None = None  # injected when caller omits max_tokens

    # --- Budget guardrail ---
    max_cost_per_request_usd: float | None = None
    budget_exceeded_action: str = "raise"       # "raise" | "fallback"
    fallback_model: str | None = None           # used when action="fallback"

    # --- Rollout controls ---
    shadow_mode: bool = False
    sample_rate: float = 1.0                    # 0.0 - 1.0
    sample_key: str = "user_id"


# ---------------------------------------------------------------------------
# Optimization report
# ---------------------------------------------------------------------------

@dataclass
class OptimizationReport:
    """
    Describes what was changed before the API call and the estimated savings.

    Costs are prompt-only (completion_tokens=0) — deterministic savings
    we can attribute directly to our changes. Actual output cost depends
    on model generation and is tracked separately in CallMetrics.
    """

    original_prompt_tokens: int
    optimized_prompt_tokens: int
    estimated_prompt_cost_before_usd: float | None   # original model + original tokens
    estimated_prompt_cost_after_usd: float | None    # final model + optimized tokens
    optimizations_applied: list[str]                 # e.g. ["prompt_trim", "output_cap"]
    quality_risk: bool                               # True if content was dropped or model downgraded
    shadow_mode: bool = False
    bypassed_reason: str | None = None

    def tokens_saved(self) -> int:
        return max(0, self.original_prompt_tokens - self.optimized_prompt_tokens)

    def cost_saved_usd(self) -> float | None:
        if (
            self.estimated_prompt_cost_before_usd is None
            or self.estimated_prompt_cost_after_usd is None
        ):
            return None
        return round(
            self.estimated_prompt_cost_before_usd - self.estimated_prompt_cost_after_usd, 8
        )

    def to_dict(self) -> dict:
        return {
            "original_prompt_tokens": self.original_prompt_tokens,
            "optimized_prompt_tokens": self.optimized_prompt_tokens,
            "tokens_saved": self.tokens_saved(),
            "estimated_prompt_cost_before_usd": self.estimated_prompt_cost_before_usd,
            "estimated_prompt_cost_after_usd": self.estimated_prompt_cost_after_usd,
            "cost_saved_usd": self.cost_saved_usd(),
            "optimizations_applied": self.optimizations_applied,
            "quality_risk": self.quality_risk,
            "shadow_mode": self.shadow_mode,
            "bypassed_reason": self.bypassed_reason,
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class OptimizationPipeline:
    def __init__(self, config: OptimizationConfig) -> None:
        self._cfg = config

    def process(
        self,
        model: str,
        messages: list[dict],
        kwargs: dict[str, Any],
        *,
        apply: bool = True,
        shadow_mode: bool = False,
    ) -> tuple[str, list[dict], dict[str, Any], OptimizationReport]:
        """
        Apply all configured optimizations before an API call.

        Returns (model, messages, kwargs, report).
        kwargs is never mutated in-place — a new dict is returned.
        """
        cfg = self._cfg
        original_model = model
        applied: list[str] = []
        quality_risk = False
        original_kwargs = dict(kwargs)  # never mutate caller's dict
        working_kwargs = dict(kwargs)

        original_tokens = count_messages_tokens(messages, model)
        working_messages = list(messages)
        was_trimmed = False

        # ------------------------------------------------------------------ #
        # 1. Prompt trimming
        # ------------------------------------------------------------------ #
        if cfg.max_prompt_tokens is not None:
            working_messages, was_trimmed = _trim_messages(
                messages=working_messages,
                max_tokens=cfg.max_prompt_tokens,
                initial_tokens=original_tokens,
                keep_last_n=cfg.keep_last_n_messages,
                always_keep_system=cfg.always_keep_system,
                model=model,
            )
            if was_trimmed:
                applied.append("prompt_trim")
                quality_risk = True

        # Count optimized tokens (only pays for one extra call when trimming occurred)
        optimized_tokens = (
            count_messages_tokens(working_messages, model) if was_trimmed else original_tokens
        )

        # ------------------------------------------------------------------ #
        # 2. Output cap
        # ------------------------------------------------------------------ #
        if cfg.default_max_output_tokens is not None and "max_tokens" not in working_kwargs:
            working_kwargs["max_tokens"] = cfg.default_max_output_tokens
            applied.append("output_cap")

        effective_max_output: int = working_kwargs.get("max_tokens", 1000)

        # ------------------------------------------------------------------ #
        # 3. Budget guardrail
        # ------------------------------------------------------------------ #
        if cfg.max_cost_per_request_usd is not None:
            estimated_total = estimate_cost(model, optimized_tokens, effective_max_output)
            if estimated_total is not None and estimated_total > cfg.max_cost_per_request_usd:
                if cfg.budget_exceeded_action == "fallback" and cfg.fallback_model:
                    applied.append(f"model_fallback:{model}->{cfg.fallback_model}")
                    model = cfg.fallback_model
                    working_kwargs["model"] = model
                    quality_risk = True
                elif apply:
                    raise BudgetExceededError(
                        f"Estimated request cost ${estimated_total:.6f} exceeds "
                        f"budget ${cfg.max_cost_per_request_usd:.6f} "
                        f"(model={model}, ~{optimized_tokens} prompt tokens, "
                        f"max_output={effective_max_output}). "
                        "Raise max_cost_per_request_usd or set a fallback_model."
                    )

        # ------------------------------------------------------------------ #
        # Build report
        # ------------------------------------------------------------------ #
        report = OptimizationReport(
            original_prompt_tokens=original_tokens,
            optimized_prompt_tokens=optimized_tokens,
            estimated_prompt_cost_before_usd=estimate_cost(original_model, original_tokens, 0),
            estimated_prompt_cost_after_usd=estimate_cost(model, optimized_tokens, 0),
            optimizations_applied=applied,
            quality_risk=quality_risk,
            shadow_mode=shadow_mode,
        )

        if apply:
            return model, working_messages, working_kwargs, report
        return original_model, list(messages), original_kwargs, report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _trim_messages(
    messages: list[dict],
    max_tokens: int,
    initial_tokens: int,
    keep_last_n: int,
    always_keep_system: bool,
    model: str,
) -> tuple[list[dict], bool]:
    """
    Trim a message list to fit within max_tokens.

    Strategy (in order):
      1. Keep system message(s) always (when always_keep_system=True)
      2. Apply keep_last_n to non-system messages
      3. Progressively drop the oldest non-system message until under budget
         (stops when only 1 non-system message remains)

    Returns (trimmed_messages, was_anything_removed).
    """
    if initial_tokens <= max_tokens:
        return messages, False

    if always_keep_system:
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]
    else:
        system_msgs = []
        non_system = list(messages)

    # Step 1: apply keep_last_n
    if len(non_system) > keep_last_n:
        non_system = non_system[-keep_last_n:]

    result = system_msgs + non_system
    was_trimmed = len(result) < len(messages)

    # Step 2: progressive drop from front of non-system messages
    while count_messages_tokens(result, model) > max_tokens and len(non_system) > 1:
        non_system = non_system[1:]
        result = system_msgs + non_system
        was_trimmed = True

    return result, was_trimmed
