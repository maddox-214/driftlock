# Driftlock

**LLM cost governance and control layer for Python.**

Driftlock sits between your application and LLM providers (OpenAI, Anthropic). It enforces cost policies, detects runaway spending, tracks per-call telemetry, and prevents budget overruns — with a drop-in API wrapper and zero changes to existing call sites.

---

## Install

```bash
pip install driftlock
```

With Anthropic support:

```bash
pip install "driftlock[anthropic]"
```

With FastAPI support:

```bash
pip install "driftlock[fastapi]"
```

Requires Python ≥ 3.11.

---

## Try It Now (no API key needed)

Clone the repo and run the full interactive demo:

```bash
git clone https://github.com/your-org/driftlock && cd driftlock
pip install -e .
python examples/demo.py
```

This runs a fully-mocked demo in-process — no API key, no network calls, no cost. It exercises every major feature: tracking, optimization, budget guardrails, cache, and context tags.

---

## 60-Second Quickstart (real API)

```bash
export OPENAI_API_KEY=sk-...       # or ANTHROPIC_API_KEY=sk-ant-...
driftlock demo
```

Driftlock makes one cheap request (`gpt-4o-mini` or `claude-haiku-4-5`) under a default policy and prints a receipt:

```
Driftlock demo  —  provider=openai  model=gpt-4o-mini

  ┌─ Receipt ──────────────────────────────────────────────┐
  │  provider  : openai                                    │
  │  model     : gpt-4o-mini                               │
  │  tokens    : 23 (15 in / 8 out)                        │
  │  cost      : $0.000007                                 │
  │  latency   : 412 ms                                    │
  │  db        : ./driftlock.sqlite                        │
  └────────────────────────────────────────────────────────┘

  Next steps:
    driftlock stats            # aggregate cost + token totals
    driftlock recent           # last 20 calls
    driftlock forecast         # projected monthly spend
```

---

## Basic Integration — OpenAI

```python
from driftlock import DriftlockClient

# Replace openai.OpenAI() with DriftlockClient().
# All other arguments are forwarded to the OpenAI client unchanged.
client = DriftlockClient(api_key="sk-...")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

Every call is logged, costed, and saved to a local SQLite file.

```json
{
  "level": "INFO",
  "logger": "driftlock",
  "message": "model=gpt-4o-mini | tokens=157 | latency=421ms | cost=$0.000033",
  "metrics": {
    "timestamp": "2025-03-01T12:00:00+00:00",
    "model": "gpt-4o-mini",
    "prompt_tokens": 120,
    "completion_tokens": 37,
    "total_tokens": 157,
    "latency_ms": 421.3,
    "estimated_cost_usd": 0.0000330
  }
}
```

### Async

```python
response = await client.chat.completions.acreate(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### Streaming

```python
for chunk in client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me a story."}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
# Metrics are logged and saved when the stream closes.
```

---

## Basic Integration — Anthropic

Requires `pip install -e ".[anthropic]"`.

```python
from driftlock import AnthropicDriftlockClient

client = AnthropicDriftlockClient(api_key="sk-ant-...")

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
```

`max_tokens` is required by Anthropic. The `system` parameter is a top-level kwarg, not a message role — same as the native SDK.

---

## Labelling Calls

Use `_dl_endpoint` and `_dl_labels` to annotate individual calls. These are stripped before the request reaches the provider.

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    _dl_endpoint="summarise_article",        # logical function name
    _dl_labels={"user_id": "u_123", "team": "growth"},
)
```

`user_id` and `team_id` in labels are indexed in SQLite for fast per-user queries.

---

## Ambient Tagging

Attach labels to all calls within a scope without modifying every call site — useful in middleware:

```python
import driftlock

with driftlock.tag(request_id="req_abc", user_id="u_42", feature="chat"):
    response = client.chat.completions.create(...)
```

Tags from nested `driftlock.tag()` blocks merge; inner values override outer ones. Per-call `_dl_labels` always wins.

---

## Configuration

```python
from driftlock import DriftlockClient, DriftlockConfig

config = DriftlockConfig(
    log_json=True,                         # JSON logs (default). False = human-readable.
    log_level="INFO",
    storage_backend="sqlite",             # "sqlite" | "none"
    db_path="driftlock.sqlite",
    prompt_token_warning_threshold=4000,  # Warn if prompt > N tokens.
    cost_warning_threshold=0.10,          # Warn if a single call costs > $X.
    default_labels={"env": "prod"},       # Attached to every tracked call.
)

client = DriftlockClient(api_key="sk-...", config=config)
```

---

## Policy Engine

The policy engine enforces governance rules before every API call. Rules are evaluated in order; the first block raises `PolicyViolationError`.

```python
from driftlock import (
    DriftlockClient,
    PolicyEngine,
    MonthlyBudgetRule,
    MaxCostPerRequestRule,
    VelocityLimitRule,
    CostVelocityRule,
    PerUserBudgetRule,
    ForecastBudgetRule,
    RestrictModelRule,
    TagBasedModelDowngradeRule,
    PolicyViolationError,
)

policy = PolicyEngine(rules=[
    MonthlyBudgetRule(max_usd=100.0),                 # Block at $100/month workspace
    MaxCostPerRequestRule(max_usd=0.10),              # Block single calls > $0.10
    VelocityLimitRule(max_requests=60, window_seconds=60),  # 60 req/min circuit breaker
])

client = DriftlockClient(api_key="sk-...", policy=policy)

try:
    response = client.chat.completions.create(...)
except PolicyViolationError as e:
    print(f"Blocked by {e.rule_name}: {e.decision.metadata}")
```

### Available Rules

| Rule | What it does |
|---|---|
| `MonthlyBudgetRule(max_usd, scope="workspace"\|"user")` | Block once monthly spend cap is reached |
| `MaxCostPerRequestRule(max_usd)` | Block a single call if estimated cost exceeds the limit |
| `PerUserBudgetRule(user_budgets, default_max_usd)` | Per-user monthly caps from a dict |
| `ForecastBudgetRule(monthly_budget_usd, lookback_days=7)` | Block when projected 30-day spend will exceed budget |
| `VelocityLimitRule(max_requests, window_seconds, scope="workspace"\|"user")` | Circuit breaker on request rate |
| `CostVelocityRule(max_cost_usd, window_seconds)` | Circuit breaker on spend rate (e.g. $5/hour) |
| `RestrictModelRule(disallowed_models, condition=None)` | Block calls to specific models |
| `TagBasedModelDowngradeRule(condition, downgrade_to)` | Silently swap model based on labels |

### Per-User Budgets

```python
policy = PolicyEngine(rules=[
    PerUserBudgetRule(
        user_budgets={"power_user": 20.0, "free_tier": 1.0},
        default_max_usd=5.0,   # applied to any user_id not in the dict
    ),
])
# user_id is read from _dl_labels={"user_id": "..."} or ambient tags
```

### Forecast-Based Blocking

```python
policy = PolicyEngine(rules=[
    ForecastBudgetRule(monthly_budget_usd=50.0, lookback_days=7),
])
# Blocks before the budget is actually exhausted — proactive, not reactive
```

### Model Governance

```python
policy = PolicyEngine(rules=[
    # Block GPT-4o on free plan users
    RestrictModelRule(
        disallowed_models={"gpt-4o", "gpt-4"},
        condition=lambda ctx: ctx["labels"].get("plan") == "free",
    ),
    # Auto-downgrade free users to mini
    TagBasedModelDowngradeRule(
        condition=lambda ctx: ctx["labels"].get("plan") == "free",
        downgrade_to="gpt-4o-mini",
    ),
])
```

---

## Alerts

Fire-and-forget notifications when policies trip or cost thresholds are crossed.

```python
from driftlock import DriftlockConfig, WebhookAlertChannel, SlackAlertChannel, LogAlertChannel

config = DriftlockConfig(
    alert_channels=[
        SlackAlertChannel(webhook_url="https://hooks.slack.com/services/..."),
        WebhookAlertChannel(url="https://example.com/hooks/driftlock"),
        LogAlertChannel(),   # logs to Python logging at WARNING level
    ]
)
```

Alert events: `policy_block`, `cost_warning`, `budget_threshold`, `velocity_trip`.

Delivery failures are logged at WARNING level and never propagate to the caller.

---

## Cost Reduction Engine

Enable the optimization pipeline to automatically trim prompts, cap output, and fall back to cheaper models:

```python
from driftlock import DriftlockClient, OptimizationConfig

client = DriftlockClient(
    api_key="sk-...",
    optimization=OptimizationConfig(
        max_prompt_tokens=3000,          # trim history if prompt exceeds this
        keep_last_n_messages=10,         # always keep the N most recent turns
        always_keep_system=True,         # never drop the system message
        default_max_output_tokens=512,   # cap output when caller omits max_tokens
        max_cost_per_request_usd=0.05,   # abort if estimated cost > $0.05
        budget_exceeded_action="fallback",
        fallback_model="gpt-4o-mini",
    ),
)
```

Every call logs an `optimization` block showing tokens and cost saved:

```json
{
  "optimization": {
    "original_prompt_tokens": 3840,
    "optimized_prompt_tokens": 142,
    "tokens_saved": 3698,
    "cost_saved_usd": 0.0005547,
    "optimizations_applied": ["prompt_trim", "output_cap"],
    "quality_risk": true
  }
}
```

---

## Response Cache

Exact in-memory cache (LRU + TTL). Returns stored responses for identical requests without hitting the API:

```python
from driftlock import DriftlockClient, CacheConfig

client = DriftlockClient(
    api_key="sk-...",
    cache=CacheConfig(
        ttl_seconds=600,    # entries expire after 10 minutes
        max_entries=500,    # LRU eviction above this
    ),
)
```

Cache hits report `cost=$0.00` and record tokens and dollars saved. Streaming responses are never cached.

```python
client.cache_stats()
# {"enabled": True, "size": 12, "hits": 48, "misses": 14, "hit_rate": 0.7742}
```

---

## Reading Metrics

```python
# Aggregate stats (all time)
client.stats()
# {'calls': 42, 'total_tokens': 18500, 'total_cost_usd': 0.003245, ...}

# Filter by endpoint, model, or time window
client.stats(endpoint="summarise_article")
client.stats(model="gpt-4o")
client.stats(since="2025-03-01T00:00:00+00:00")

# Recent calls
client.recent_calls(limit=10)

# Projected monthly spend
client.forecast(lookback_days=7)
# {'daily_avg_usd': 0.0004, 'projected_monthly_usd': 0.012, ...}

# Prompt drift detection (detect template changes by endpoint)
client.prompt_drift(endpoint="summarise_article")
```

---

## CLI

Inspect telemetry without writing code:

```bash
driftlock stats                          # aggregate totals
driftlock stats --since 7d              # last 7 days
driftlock stats --endpoint summarise    # filter by endpoint
driftlock recent --limit 20             # last 20 calls
driftlock forecast --lookback 7         # projected monthly spend
driftlock top-endpoints --since 7d      # most expensive endpoints
driftlock top-users --since 30d         # per-user spend
driftlock models                        # spend by model
driftlock drift summarise_article       # prompt change history
driftlock --db /path/to/other.db stats  # point at a different db
```

Set `DRIFTLOCK_DB_PATH` to override the default `driftlock.sqlite` path.

---

## Environment Variables

| Variable | Default | Effect |
|---|---|---|
| `DRIFTLOCK_ENABLED` | `true` | Set to `false` to pass through all calls with zero overhead |
| `DRIFTLOCK_TRACK_ONLY` | `false` | Track metrics but skip optimization and policy enforcement |
| `DRIFTLOCK_DB_PATH` | `driftlock.sqlite` | Override the SQLite file path for CLI commands |

---

## FastAPI Example

See [examples/fastapi_app.py](examples/fastapi_app.py) for a full integration with middleware tagging, optimization, and cache.

```bash
OPENAI_API_KEY=sk-... uvicorn examples.fastapi_app:app --reload
```

---

## Project Structure

```
driftlock/
├── __init__.py          # Public API
├── client.py            # DriftlockClient (OpenAI wrapper, sync + async)
├── anthropic_client.py  # AnthropicDriftlockClient (opt-in)
├── config.py            # DriftlockConfig
├── policy.py            # PolicyEngine + all rules
├── alerts.py            # AlertChannel, Webhook/Slack/Log implementations
├── metrics.py           # CallMetrics dataclass
├── pricing.py           # OpenAI + Anthropic pricing table
├── storage.py           # SQLiteStorage + NoopStorage (auto-migrating)
├── optimization.py      # OptimizationPipeline, OptimizationConfig
├── cache.py             # ResponseCache (LRU+TTL), CacheConfig
├── streaming.py         # StreamingInterceptor (deferred metrics)
├── drift.py             # Prompt hash + drift detection
├── cli.py               # driftlock CLI entry point
├── context.py           # driftlock.tag() context manager
├── logger.py            # Structured JSON logger
├── tokenizer.py         # tiktoken + char fallback
└── providers/           # NormalizedUsage, OpenAIProvider, AnthropicProvider

examples/
├── basic_usage.py
├── fastapi_app.py
└── dashboard_app.py

tests/                   # 131 tests
```

---

## Roadmap

| Feature | Status |
|---|---|
| OpenAI chat wrapper (sync + async) | ✅ |
| Anthropic Messages wrapper (sync + async) | ✅ |
| Token tracking + cost estimation | ✅ |
| Latency measurement | ✅ |
| SQLite storage (auto-migrating) | ✅ |
| Structured JSON logging | ✅ |
| Policy engine (budget, velocity, model) | ✅ |
| Per-user / per-team budget caps | ✅ |
| Forecast-based budget blocking | ✅ |
| Velocity + cost circuit breakers | ✅ |
| Prompt optimization pipeline | ✅ |
| Exact in-memory response cache | ✅ |
| Streaming support | ✅ |
| Prompt drift detection | ✅ |
| Alert channels (Slack, Webhook, Log) | ✅ |
| Ambient tagging context manager | ✅ |
| CLI (stats, forecast, drift, top-users) | ✅ |
| OpenTelemetry export | Planned |
| Redis cache backend | Planned |
| Semantic (embedding-based) cache | Planned |
| Gemini adapter | Planned |
| PyPI release | ✅ |

---

## License

MIT
