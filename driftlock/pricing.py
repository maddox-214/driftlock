"""
Model pricing table — cost per 1,000 tokens (USD).
Update this table as OpenAI changes prices or releases new models.

Formula: (input_tokens / 1000 * input) + (output_tokens / 1000 * output)
"""

# model_id -> {"input": $/1K tokens, "output": $/1K tokens}
MODEL_PRICING: dict[str, dict[str, float]] = {
    # GPT-4o
    "gpt-4o":                   {"input": 0.005,    "output": 0.015},
    "gpt-4o-2024-11-20":        {"input": 0.005,    "output": 0.015},
    "gpt-4o-2024-08-06":        {"input": 0.005,    "output": 0.015},
    # GPT-4o mini
    "gpt-4o-mini":              {"input": 0.00015,  "output": 0.0006},
    "gpt-4o-mini-2024-07-18":   {"input": 0.00015,  "output": 0.0006},
    # GPT-4 Turbo
    "gpt-4-turbo":              {"input": 0.01,     "output": 0.03},
    "gpt-4-turbo-preview":      {"input": 0.01,     "output": 0.03},
    # GPT-4
    "gpt-4":                    {"input": 0.03,     "output": 0.06},
    # GPT-3.5 Turbo
    "gpt-3.5-turbo":            {"input": 0.0005,   "output": 0.0015},
    "gpt-3.5-turbo-0125":       {"input": 0.0005,   "output": 0.0015},
    # o1 / o3
    "o1":                       {"input": 0.015,    "output": 0.06},
    "o1-mini":                  {"input": 0.003,    "output": 0.012},
    "o3-mini":                  {"input": 0.0011,   "output": 0.0044},
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """
    Return estimated cost in USD for a single API call.
    Returns None if the model is not in the pricing table.
    """
    pricing = MODEL_PRICING.get(model)

    if pricing is None:
        # Prefix-match for dated model variants not explicitly listed
        for key, prices in MODEL_PRICING.items():
            if model.startswith(key):
                pricing = prices
                break

    if pricing is None:
        return None

    cost = (prompt_tokens / 1000 * pricing["input"]) + (completion_tokens / 1000 * pricing["output"])
    return round(cost, 8)


def list_supported_models() -> list[str]:
    return sorted(MODEL_PRICING.keys())
