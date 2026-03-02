import pytest
from driftlock.pricing import estimate_cost, list_supported_models, MODEL_PRICING


def test_known_model_cost():
    # gpt-4o-mini: $0.00015/1K input, $0.0006/1K output
    cost = estimate_cost("gpt-4o-mini", prompt_tokens=1000, completion_tokens=500)
    expected = (1000 / 1000 * 0.00015) + (500 / 1000 * 0.0006)
    assert cost == pytest.approx(expected, rel=1e-6)


def test_gpt4o_cost():
    # gpt-4o: $0.005/1K input, $0.015/1K output
    cost = estimate_cost("gpt-4o", prompt_tokens=2000, completion_tokens=1000)
    expected = (2000 / 1000 * 0.005) + (1000 / 1000 * 0.015)
    assert cost == pytest.approx(expected, rel=1e-6)


def test_unknown_model_returns_none():
    assert estimate_cost("gpt-99-turbo", 100, 100) is None


def test_prefix_match():
    # "gpt-4o-2099-hypothetical" should match the "gpt-4o" entry
    cost = estimate_cost("gpt-4o-2099-hypothetical", 1000, 500)
    assert cost is not None
    expected = (1000 / 1000 * 0.005) + (500 / 1000 * 0.015)
    assert cost == pytest.approx(expected, rel=1e-6)


def test_zero_tokens():
    assert estimate_cost("gpt-4o-mini", 0, 0) == 0.0


def test_list_supported_models():
    models = list_supported_models()
    assert "gpt-4o-mini" in models
    assert "gpt-4o" in models
    assert models == sorted(models)


def test_pricing_table_structure():
    for model, prices in MODEL_PRICING.items():
        assert "input" in prices, f"{model} missing 'input' key"
        assert "output" in prices, f"{model} missing 'output' key"
        assert prices["input"] >= 0
        assert prices["output"] >= prices["input"], f"{model}: output should cost >= input"
