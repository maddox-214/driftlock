# Driftlock

**Lightweight LLM cost & token tracking middleware for Python.**

Drop Driftlock into your project in under 5 minutes. It wraps the OpenAI Python SDK transparently, giving you per-call token counts, real-time cost estimates, latency measurement, and persistent local metrics — with zero changes to your existing call sites.

---

## Why Driftlock?

LLM costs compound quietly. A prompt that's 20% too long, a model that's one tier too expensive, a function calling GPT-4o when GPT-4o-mini would do — these are hard to catch without instrumentation.

Driftlock makes the cost of every LLM call visible at the point of code, not three weeks later in a billing alert.

---

## Install

```bash
pip install driftlock
```

With FastAPI support:

```bash
pip install "driftlock[fastapi]"
```

---

## 5-Minute Integration

```python
from driftlock import DriftlockClient

# Swap openai.OpenAI() → DriftlockClient()
# Every other argument is forwarded to the OpenAI client unchanged.
client = DriftlockClient(api_key="sk-...")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

That's it. You get structured logs on every call:

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
    "estimated_cost_usd": 0.0000330,
    "endpoint": null,
    "labels": {},
    "warnings": []
  }
}
```

---

## Configuration

```python
from driftlock import DriftlockClient, DriftlockConfig

config = DriftlockConfig(
    log_json=True,                         # JSON logs (default). False = human-readable.
    log_level="INFO",                      # Standard log level.
    storage_backend="sqlite",             # "sqlite" | "none"
    db_path="driftlock.db",               # SQLite file path.
    prompt_token_warning_threshold=4000,  # Warn if prompt > N tokens.
    cost_warning_threshold=0.10,          # Warn if a single call costs > $X.
    default_labels={"env": "prod"},       # Attached to every tracked call.
)

client = DriftlockClient(api_key="sk-...", config=config)
```

---

## Labelling Calls

Use `_dl_endpoint` and `_dl_labels` to annotate individual calls. These are stripped before the request reaches OpenAI.

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    _dl_endpoint="summarise_article",        # logical function name
    _dl_labels={"user_id": "u_123"},         # arbitrary key/value metadata
)
```

---

## Reading Metrics

```python
# Aggregate stats (all time)
client.stats()
# {'calls': 42, 'total_tokens': 18500, 'total_cost_usd': 0.003245, ...}

# Filter by endpoint or model
client.stats(endpoint="summarise_article")
client.stats(model="gpt-4o")

# Filter since a timestamp (ISO 8601)
client.stats(since="2025-03-01T00:00:00+00:00")

# Recent calls
client.recent_calls(limit=10)
```

---

## FastAPI Example

See [examples/fastapi_app.py](examples/fastapi_app.py) for a working FastAPI integration.

```bash
OPENAI_API_KEY=sk-... uvicorn examples.fastapi_app:app --reload
```

Key endpoints:

| Route | Description |
|---|---|
| `POST /chat` | Chat with GPT, tracked under `endpoint=chat` |
| `POST /summarise` | Summarise text, tracked under `endpoint=summarise` |
| `GET /metrics` | Aggregated stats by endpoint |
| `GET /metrics/recent` | Last N calls |

---

## Cost Reduction Engine v0

Enable the optimization pipeline by passing an `OptimizationConfig` to `DriftlockClient`:

```python
from driftlock import DriftlockClient, OptimizationConfig

client = DriftlockClient(
    api_key="sk-...",
    optimization=OptimizationConfig(
        max_prompt_tokens=3000,          # trim history if prompt exceeds this
        keep_last_n_messages=10,         # always keep the N most recent turns
        always_keep_system=True,         # never drop the system message
        default_max_output_tokens=512,   # cap output when caller omits max_tokens
        max_cost_per_request_usd=0.05,   # guardrail: abort if estimated cost > $0.05
        budget_exceeded_action="fallback",
        fallback_model="gpt-4o-mini",    # switch to this model instead of raising
    ),
)
```

All three features are opt-in and independent — you can enable any combination.

### 1. Prompt Trimming

Trims the chat history before the call to fit within `max_prompt_tokens`.

**Algorithm (deterministic, in order):**
1. Apply `keep_last_n_messages` — drop older non-system turns from the front.
2. If still over budget, drop one non-system message at a time (oldest first) until the prompt fits.
3. The system message is never touched when `always_keep_system=True`.
4. At least one non-system message is always kept regardless of budget.

**Tradeoffs:**
- Dropping messages may degrade response quality (context loss). `quality_risk: true` flags this in the report.
- Token counting uses tiktoken (accurate) or a char/4 fallback if tiktoken is unavailable.

### 2. Hard Output Cap

If the caller does not set `max_tokens`, Driftlock injects `default_max_output_tokens` automatically. This prevents runaway generation from open-ended prompts.

**Tradeoffs:**
- Responses may be cut off. Only enable when the call site tolerates truncated output.
- The cap is silently skipped when the caller already sets `max_tokens`, so it never overrides explicit intent.

### 3. Budget Guardrail

Before every call, Driftlock estimates the total cost:

```
estimated_cost = (prompt_tokens / 1000 * input_price)
               + (effective_max_output_tokens / 1000 * output_price)
```

If this exceeds `max_cost_per_request_usd`:

| `budget_exceeded_action` | Behaviour |
|---|---|
| `"raise"` (default) | Raises `BudgetExceededError` before any API call is made |
| `"fallback"` | Switches to `fallback_model` and retries the cost estimate |

**Tradeoffs:**
- The estimate is **worst-case** (assumes full output generation). Actual cost is often lower.
- Model fallback changes response quality. `quality_risk: true` is set in the report.

### Optimization Report

Every call with an active pipeline appends an `optimization` block to the JSON log and to `CallMetrics`:

```json
{
  "metrics": {
    "model": "gpt-4o-mini",
    "prompt_tokens": 142,
    "optimization": {
      "original_prompt_tokens": 3840,
      "optimized_prompt_tokens": 142,
      "tokens_saved": 3698,
      "estimated_prompt_cost_before_usd": 0.000576,
      "estimated_prompt_cost_after_usd": 0.0000213,
      "cost_saved_usd": 0.0005547,
      "optimizations_applied": ["prompt_trim", "output_cap"],
      "quality_risk": true
    }
  }
}
```

Cost savings reported are **prompt-only** (deterministic). Output cost depends on actual model generation and is tracked in `estimated_cost_usd` after the call completes.

---

## Caching: exact cache (safe) vs semantic cache (future)

### Exact cache (available now)

Driftlock's cache stores the full response for a request and returns it unchanged on repeated identical calls.

```python
client = DriftlockClient(
    api_key="sk-...",
    cache=CacheConfig(
        ttl_seconds=600,    # entries expire after 10 minutes
        max_entries=500,    # LRU eviction above this
    ),
)
```

**How it works:**
- The cache key is a SHA-256 hash of `(model, messages, temperature, max_tokens, …)` — computed *after* the optimization pipeline runs, so you cache the trimmed/capped request, not the original.
- The response object is stored in memory. No serialization, no disk I/O.
- On a cache hit, `prompt_tokens=0`, `completion_tokens=0`, `estimated_cost_usd=0.0`. The savings are reported in `tokens_saved_prompt`, `tokens_saved_completion`, and `estimated_savings_usd`.
- Streaming responses (`stream=True`) are never cached.
- The `user` kwarg (OpenAI tracing, PII) is excluded from the key so different user IDs sharing the same prompt share a cache entry.

**Log output on a cache hit:**
```
[CACHE HIT] | model=gpt-4o-mini | latency=0.1ms | saved=70tok | saved_usd=$0.000011 | endpoint=chat | key=a3f8c21d…
```

**Cache stats** are available in-memory and in the `/metrics` endpoint:
```python
client.cache_stats()
# {"enabled": True, "size": 12, "hits": 48, "misses": 14, "hit_rate": 0.7742, ...}

client.stats()
# {"calls": 62, "cache_hits": 48, "cache_hit_rate": 0.7742,
#  "total_cost_usd": 0.0014, "total_savings_usd": 0.0052, ...}
```

**When is an exact cache appropriate?**

| Use case | Cache? |
|---|---|
| FAQ / help-bot with repeated questions | ✅ Yes — identical prompts, deterministic answers |
| `temperature=0` pipelines (classification, extraction) | ✅ Yes — fully deterministic |
| User-facing chat with creative generation (`temperature>0`) | ⚠️ Cache returns the first response every time — fine for read-heavy FAQ, wrong for conversational UX |
| Unique documents (per-user summaries, personalised content) | ❌ No — cache miss rate near 100%, adds overhead with no benefit |

**Tradeoffs:**
- Memory: each entry holds a full `ChatCompletion` object (~1–5 KB). 1 000 entries ≈ 1–5 MB.
- Stale responses: TTL controls how long entries live. Set lower (`60s`) for frequently-updated knowledge, higher (`3600s`) for stable FAQ content.
- No persistence: the cache is wiped on process restart. For cross-process caching, Redis support is on the roadmap.

### Ambient tagging

Attach labels to all DriftlockClient calls within a scope without modifying every call site — useful in middleware:

```python
import driftlock

# FastAPI middleware example
with driftlock.tag(request_id="req_abc", user_id="u_42", feature="chat"):
    response = client.chat.completions.create(...)
```

Tags from nested `driftlock.tag()` blocks merge; inner values override outer ones. Per-call `_dl_labels` always wins.

### Semantic cache (planned)

A future semantic cache will embed the prompt and return cached responses when cosine similarity exceeds a configurable threshold. This enables cache hits for paraphrased or near-duplicate prompts at the cost of occasional incorrect hits.

Driftlock will implement this as a separate, explicitly opt-in backend with a `similarity_threshold` config and a `quality_risk: true` flag on hits, making the tradeoff transparent.

---

## Warnings

Driftlock emits warnings (elevated to `WARNING` log level) when:

- **Prompt is too large** — prompt token count exceeds `prompt_token_warning_threshold`
- **Call is expensive** — estimated cost exceeds `cost_warning_threshold`

Warnings are also stored in the local SQLite record for later inspection.

---

## Project Structure

```
driftlock/
├── __init__.py          # Public API: DriftlockClient, DriftlockConfig, CacheConfig, tag, …
├── client.py            # DriftlockClient + chat interceptor
├── config.py            # DriftlockConfig dataclass
├── metrics.py           # CallMetrics dataclass (includes cache + optimization fields)
├── pricing.py           # Model pricing table + cost estimator
├── storage.py           # SQLiteStorage + NoopStorage (auto-migrating schema)
├── logger.py            # Structured JSON logger
├── optimization.py      # OptimizationPipeline, OptimizationConfig, OptimizationReport
├── tokenizer.py         # Token counting (tiktoken + char fallback)
├── cache.py             # ResponseCache (LRU+TTL), CacheConfig, make_cache_key
└── context.py           # driftlock.tag() context manager (ContextVar)

examples/
├── basic_usage.py       # Minimal script example
└── fastapi_app.py       # FastAPI with middleware tagging + optimization + cache

tests/
├── test_pricing.py
├── test_client.py
├── test_storage.py
├── test_optimization.py
└── test_cache.py        # LRU, TTL, key stability, context tags, client integration
```

---

## Roadmap

| Feature | Status |
|---|---|
| OpenAI chat wrapper | ✅ |
| Token tracking | ✅ |
| Cost estimation | ✅ |
| Latency measurement | ✅ |
| SQLite storage | ✅ |
| Structured JSON logging | ✅ |
| Prompt-length warnings | ✅ |
| Cost-per-call warnings | ✅ |
| Async (`acreate`) support | Planned |
| Anthropic / Gemini adapters | Planned |
| Smart model routing | Planned |
| Prompt compression hooks | Planned |
| Per-user / per-team budget caps | Planned |
| OpenTelemetry export | Planned |

---

## License

MIT
