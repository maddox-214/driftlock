"""
Model pricing table — cost per 1,000 tokens (USD).
Update this table as providers change prices or release new models.

Formula: (input_tokens / 1000 * input) + (output_tokens / 1000 * output)
"""

# model_id -> {"input": $/1K tokens, "output": $/1K tokens}
MODEL_PRICING: dict[str, dict[str, float]] = {
    # ------------------------------------------------------------------ #
    # OpenAI
    # ------------------------------------------------------------------ #
    # GPT-5.2 (flagship, Dec 2025)
    "gpt-5.2":                  {"input": 0.00175,  "output": 0.014},
    # GPT-5.1 (Nov 2025)
    "gpt-5.1":                  {"input": 0.00125,  "output": 0.01},
    # GPT-5 (Aug 2025)
    "gpt-5":                    {"input": 0.00125,  "output": 0.01},
    # GPT-5 mini / nano / pro
    "gpt-5-mini":               {"input": 0.00025,  "output": 0.002},
    "gpt-5-nano":               {"input": 0.00005,  "output": 0.0004},
    "gpt-5-pro":                {"input": 0.015,    "output": 0.12},
    # GPT-4.1 (Apr 2025)
    "gpt-4.1":                  {"input": 0.002,    "output": 0.008},
    "gpt-4.1-2025-04-14":       {"input": 0.002,    "output": 0.008},
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
    # o-series reasoning models
    "o4-mini":                  {"input": 0.0011,   "output": 0.0044},
    "o3":                       {"input": 0.002,    "output": 0.008},
    "o1":                       {"input": 0.015,    "output": 0.06},
    "o1-mini":                  {"input": 0.003,    "output": 0.012},
    "o3-mini":                  {"input": 0.0011,   "output": 0.0044},

    # ------------------------------------------------------------------ #
    # Anthropic Claude
    # ------------------------------------------------------------------ #
    # Claude 4 family
    "claude-opus-4-6":              {"input": 0.015,    "output": 0.075},
    "claude-opus-4":                {"input": 0.015,    "output": 0.075},
    "claude-sonnet-4-6":            {"input": 0.003,    "output": 0.015},
    "claude-sonnet-4":              {"input": 0.003,    "output": 0.015},
    "claude-haiku-4-5":             {"input": 0.0008,   "output": 0.004},
    "claude-haiku-4-5-20251001":    {"input": 0.0008,   "output": 0.004},
    # Claude 3.5 family
    "claude-3-5-sonnet-20241022":   {"input": 0.003,    "output": 0.015},
    "claude-3-5-sonnet-20240620":   {"input": 0.003,    "output": 0.015},
    "claude-3-5-haiku-20241022":    {"input": 0.0008,   "output": 0.004},
    # Claude 3 family
    "claude-3-opus-20240229":       {"input": 0.015,    "output": 0.075},
    "claude-3-sonnet-20240229":     {"input": 0.003,    "output": 0.015},
    "claude-3-haiku-20240307":      {"input": 0.00025,  "output": 0.00125},
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
