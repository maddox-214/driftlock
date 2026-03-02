"""
Tests for OptimizationPipeline.

count_messages_tokens is mocked throughout so tests are independent of
tiktoken availability and run deterministically.
"""

import pytest
from unittest.mock import patch

from driftlock.optimization import (
    BudgetExceededError,
    OptimizationConfig,
    OptimizationPipeline,
    _trim_messages,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Message one"},
    {"role": "assistant", "content": "Response one"},
    {"role": "user", "content": "Message two"},
    {"role": "assistant", "content": "Response two"},
    {"role": "user", "content": "Message three"},
]

_MODULE = "driftlock.optimization.count_messages_tokens"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _pipeline(cfg: OptimizationConfig) -> OptimizationPipeline:
    return OptimizationPipeline(cfg)


# ---------------------------------------------------------------------------
# Prompt trimming — _trim_messages unit tests
# ---------------------------------------------------------------------------

class TestTrimMessages:
    def test_no_trim_when_already_under_budget(self):
        result, trimmed = _trim_messages(
            MESSAGES, max_tokens=10000, initial_tokens=50,
            keep_last_n=10, always_keep_system=True, model="gpt-4o-mini",
        )
        assert result == MESSAGES
        assert trimmed is False

    def test_keep_last_n_drops_older_non_system(self):
        # initial_tokens > max_tokens triggers the trim path;
        # after keep_last_n the while loop sees a value ≤ max_tokens
        with patch(_MODULE, side_effect=[9]):  # while-loop check: under budget
            result, trimmed = _trim_messages(
                MESSAGES, max_tokens=10, initial_tokens=500,
                keep_last_n=2, always_keep_system=True, model="gpt-4o-mini",
            )
        assert trimmed is True
        assert result[0]["role"] == "system"
        non_system = [m for m in result if m["role"] != "system"]
        assert len(non_system) == 2
        assert non_system[-1]["content"] == "Message three"

    def test_system_message_always_preserved(self):
        with patch(_MODULE, side_effect=[9]):
            result, _ = _trim_messages(
                MESSAGES, max_tokens=10, initial_tokens=500,
                keep_last_n=1, always_keep_system=True, model="gpt-4o-mini",
            )
        assert any(m["role"] == "system" for m in result)

    def test_system_not_protected_when_flag_off(self):
        with patch(_MODULE, side_effect=[9]):
            result, trimmed = _trim_messages(
                MESSAGES, max_tokens=10, initial_tokens=500,
                keep_last_n=2, always_keep_system=False, model="gpt-4o-mini",
            )
        assert trimmed is True
        # System is treated as a regular non-system message; only last 2 kept
        assert len(result) == 2

    def test_progressive_drop_reduces_beyond_keep_last_n(self):
        # while loop fires twice before dropping below budget
        with patch(_MODULE, side_effect=[200, 100, 9]):
            result, trimmed = _trim_messages(
                MESSAGES, max_tokens=10, initial_tokens=500,
                keep_last_n=4, always_keep_system=True, model="gpt-4o-mini",
            )
        assert trimmed is True
        non_system = [m for m in result if m["role"] != "system"]
        # keep_last_n=4 gives 4 non-system; 2 progressive drops leaves 2
        assert len(non_system) == 2

    def test_always_keeps_at_least_one_non_system(self):
        # while loop would keep trying to drop, but len(non_system) > 1 guard stops it
        with patch(_MODULE, side_effect=[999, 999, 999]):  # never gets below budget
            result, trimmed = _trim_messages(
                MESSAGES, max_tokens=1, initial_tokens=500,
                keep_last_n=1, always_keep_system=True, model="gpt-4o-mini",
            )
        assert trimmed is True
        non_system = [m for m in result if m["role"] != "system"]
        assert len(non_system) == 1  # minimum: the single last message

    def test_returns_original_list_object_when_no_trim(self):
        result, trimmed = _trim_messages(
            MESSAGES, max_tokens=10000, initial_tokens=50,
            keep_last_n=10, always_keep_system=True, model="gpt-4o-mini",
        )
        assert result is MESSAGES
        assert trimmed is False


# ---------------------------------------------------------------------------
# OptimizationPipeline.process — prompt trimming
# ---------------------------------------------------------------------------

class TestPipelineTrimming:
    def test_no_trim_when_under_budget(self):
        pipeline = _pipeline(OptimizationConfig(max_prompt_tokens=10000))
        with patch(_MODULE, return_value=50):
            _, msgs, _, report = pipeline.process("gpt-4o-mini", MESSAGES, {})
        assert msgs == MESSAGES
        assert "prompt_trim" not in report.optimizations_applied
        assert report.quality_risk is False

    def test_trim_applied_and_flagged(self):
        pipeline = _pipeline(OptimizationConfig(max_prompt_tokens=10, keep_last_n_messages=2))
        # [original_tokens, while-loop check (under budget), optimized_tokens]
        with patch(_MODULE, side_effect=[500, 9, 9]):
            _, msgs, _, report = pipeline.process("gpt-4o-mini", MESSAGES, {})
        assert "prompt_trim" in report.optimizations_applied
        assert report.quality_risk is True
        non_system = [m for m in msgs if m["role"] != "system"]
        assert len(non_system) == 2

    def test_token_counts_in_report(self):
        pipeline = _pipeline(OptimizationConfig(max_prompt_tokens=10, keep_last_n_messages=2))
        with patch(_MODULE, side_effect=[500, 9, 9]):
            _, _, _, report = pipeline.process("gpt-4o-mini", MESSAGES, {})
        assert report.original_prompt_tokens == 500
        assert report.optimized_prompt_tokens == 9
        assert report.tokens_saved() == 491

    def test_no_extra_token_count_when_no_trim(self):
        """When trimming is off, count_messages_tokens is called exactly once."""
        pipeline = _pipeline(OptimizationConfig())
        with patch(_MODULE, return_value=50) as mock_count:
            pipeline.process("gpt-4o-mini", MESSAGES, {})
        assert mock_count.call_count == 1


# ---------------------------------------------------------------------------
# Output cap
# ---------------------------------------------------------------------------

class TestOutputCap:
    def test_injects_max_tokens_when_absent(self):
        pipeline = _pipeline(OptimizationConfig(default_max_output_tokens=512))
        with patch(_MODULE, return_value=100):
            _, _, kwargs, report = pipeline.process("gpt-4o-mini", MESSAGES[:2], {})
        assert kwargs["max_tokens"] == 512
        assert "output_cap" in report.optimizations_applied

    def test_does_not_override_caller_max_tokens(self):
        pipeline = _pipeline(OptimizationConfig(default_max_output_tokens=512))
        with patch(_MODULE, return_value=100):
            _, _, kwargs, report = pipeline.process(
                "gpt-4o-mini", MESSAGES[:2], {"max_tokens": 2048}
            )
        assert kwargs["max_tokens"] == 2048
        assert "output_cap" not in report.optimizations_applied

    def test_no_injection_when_cap_not_configured(self):
        pipeline = _pipeline(OptimizationConfig())
        with patch(_MODULE, return_value=100):
            _, _, kwargs, _ = pipeline.process("gpt-4o-mini", MESSAGES[:2], {})
        assert "max_tokens" not in kwargs


# ---------------------------------------------------------------------------
# Budget guardrail
# ---------------------------------------------------------------------------

class TestBudgetGuardrail:
    def test_raises_when_budget_exceeded(self):
        pipeline = _pipeline(
            OptimizationConfig(max_cost_per_request_usd=0.000001)
        )
        # 10 000 tokens * gpt-4o price will easily exceed $0.000001
        with patch(_MODULE, return_value=10_000):
            with pytest.raises(BudgetExceededError, match="exceeds budget"):
                pipeline.process("gpt-4o", MESSAGES, {})

    def test_no_exception_when_under_budget(self):
        pipeline = _pipeline(OptimizationConfig(max_cost_per_request_usd=1.0))
        with patch(_MODULE, return_value=10):
            model, _, _, report = pipeline.process("gpt-4o-mini", MESSAGES[:2], {})
        assert model == "gpt-4o-mini"
        assert not any("model_fallback" in s for s in report.optimizations_applied)

    def test_fallback_switches_model_and_kwargs(self):
        pipeline = _pipeline(
            OptimizationConfig(
                max_cost_per_request_usd=0.000001,
                budget_exceeded_action="fallback",
                fallback_model="gpt-4o-mini",
            )
        )
        with patch(_MODULE, return_value=10_000):
            model, _, kwargs, report = pipeline.process("gpt-4o", MESSAGES, {})
        assert model == "gpt-4o-mini"
        assert kwargs["model"] == "gpt-4o-mini"
        assert any("model_fallback" in s for s in report.optimizations_applied)
        assert report.quality_risk is True

    def test_fallback_raises_when_no_fallback_model_configured(self):
        pipeline = _pipeline(
            OptimizationConfig(
                max_cost_per_request_usd=0.000001,
                budget_exceeded_action="fallback",
                fallback_model=None,  # fallback requested but not configured
            )
        )
        with patch(_MODULE, return_value=10_000):
            with pytest.raises(BudgetExceededError):
                pipeline.process("gpt-4o", MESSAGES, {})

    def test_guardrail_uses_max_tokens_for_output_estimate(self):
        """Guardrail should use the caller's max_tokens (not 1000 default) in cost estimate."""
        # With a tiny budget and large explicit max_tokens, the guardrail fires
        pipeline = _pipeline(
            OptimizationConfig(max_cost_per_request_usd=0.000001)
        )
        with patch(_MODULE, return_value=10):
            with pytest.raises(BudgetExceededError):
                pipeline.process("gpt-4o", MESSAGES[:2], {"max_tokens": 100_000})


# ---------------------------------------------------------------------------
# OptimizationReport
# ---------------------------------------------------------------------------

class TestOptimizationReport:
    def _make_report(self):
        pipeline = _pipeline(
            OptimizationConfig(max_prompt_tokens=10, keep_last_n_messages=2)
        )
        with patch(_MODULE, side_effect=[500, 9, 9]):
            _, _, _, report = pipeline.process("gpt-4o-mini", MESSAGES, {})
        return report

    def test_report_has_expected_keys(self):
        d = self._make_report().to_dict()
        expected = {
            "original_prompt_tokens",
            "optimized_prompt_tokens",
            "tokens_saved",
            "estimated_prompt_cost_before_usd",
            "estimated_prompt_cost_after_usd",
            "cost_saved_usd",
            "optimizations_applied",
            "quality_risk",
        }
        assert expected.issubset(d.keys())

    def test_tokens_saved_is_difference(self):
        report = self._make_report()
        assert report.tokens_saved() == report.original_prompt_tokens - report.optimized_prompt_tokens

    def test_cost_saved_is_non_negative(self):
        report = self._make_report()
        saved = report.cost_saved_usd()
        if saved is not None:
            assert saved >= 0

    def test_no_optimization_report_fields_when_pipeline_absent(self):
        """Without an optimizer, CallMetrics.to_dict() has no 'optimization' key."""
        from driftlock.metrics import CallMetrics
        m = CallMetrics(
            model="gpt-4o-mini",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            latency_ms=100.0,
            estimated_cost_usd=0.000002,
        )
        assert "optimization" not in m.to_dict()

    def test_optimization_key_present_when_report_attached(self):
        from driftlock.metrics import CallMetrics
        from driftlock.optimization import OptimizationReport
        report = OptimizationReport(
            original_prompt_tokens=100,
            optimized_prompt_tokens=50,
            estimated_prompt_cost_before_usd=0.0005,
            estimated_prompt_cost_after_usd=0.00025,
            optimizations_applied=["prompt_trim"],
            quality_risk=True,
        )
        m = CallMetrics(
            model="gpt-4o-mini",
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
            latency_ms=200.0,
            estimated_cost_usd=0.000008,
            optimization_report=report,
        )
        d = m.to_dict()
        assert "optimization" in d
        assert d["optimization"]["tokens_saved"] == 50
        assert d["optimization"]["quality_risk"] is True


# ---------------------------------------------------------------------------
# kwargs immutability
# ---------------------------------------------------------------------------

class TestKwargsImmutability:
    def test_original_kwargs_not_mutated(self):
        pipeline = _pipeline(OptimizationConfig(default_max_output_tokens=256))
        original_kwargs = {"model": "gpt-4o-mini", "messages": MESSAGES[:2]}
        with patch(_MODULE, return_value=50):
            pipeline.process("gpt-4o-mini", MESSAGES[:2], original_kwargs)
        # original dict should be untouched
        assert "max_tokens" not in original_kwargs
