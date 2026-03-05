# DRIFTLOCK PROJECT CONTEXT

## What Driftlock Is

Driftlock is an infrastructure layer that helps companies **control and limit the cost of LLM usage** in production systems.

As companies integrate LLM APIs like OpenAI, Anthropic, and Azure OpenAI, they quickly run into a major problem:

LLM costs can spiral out of control.

Developers often ship features without realizing:

- prompts are getting longer
- models are more expensive than necessary
- background jobs are burning tokens
- a bug can trigger thousands of expensive calls

Driftlock solves this by acting as a **cost control and policy enforcement layer for LLM requests**.

Instead of simply calling an LLM provider directly, applications call through Driftlock which:

- calculates cost before execution
- enforces spending policies
- automatically downgrades models when appropriate
- tracks usage for debugging
- prevents runaway LLM spend

Driftlock is essentially a **financial safety system for LLM infrastructure**.

Think of it as:

Stripe for payments  
Datadog for observability  
but specifically for **LLM cost governance**.

---

# Core Problem Driftlock Solves

LLMs are expensive and unpredictable.

Problems teams encounter:

1. **Runaway Costs**

A bug or loop can generate thousands of LLM calls.

Example:

A background job accidentally calls GPT-4 inside a loop.

Result:
$20 → $2000 overnight.

2. **Model Overuse**

Developers default to expensive models when cheaper models would work.

Example:

Using GPT-4 when GPT-4o-mini would suffice.

3. **Hidden Token Growth**

Prompts evolve over time.

Small prompt changes slowly increase token usage.

Costs silently creep up.

4. **Lack of Cost Awareness**

Developers rarely see the **dollar cost** of each request.

They only see tokens.

Driftlock exposes **real dollar impact of every call**.

---

# Design Philosophy

Driftlock follows five core principles.

## 1. Cost Control First

The primary function of Driftlock is **preventing unnecessary LLM spending**.

Every request should be evaluated for cost impact.

Key capabilities:

- cost estimation before execution
- budget enforcement
- automatic model downgrades
- cost alerts

Observability exists to support cost control.

Cost governance is the priority.

---

## 2. Minimal Integration

Developers should integrate Driftlock in minutes.

Example:
from driftlock import track_llm

response = track_llm(
client.chat.completions.create(
model="gpt-4o",
messages=[...]
)
)


or decorator style:


@driftlock.track()
def generate_summary():


Integration should require **minimal changes to existing code**.

---

## 3. Vendor Agnostic

Driftlock should support all major LLM providers.

Providers include:

OpenAI  
Anthropic  
Azure OpenAI  
Groq  
local models  
future providers

Responses should be normalized into a **single telemetry format**.

---

## 4. Lightweight Runtime

Driftlock should add minimal overhead.

The SDK should remain:

- fast
- dependency light
- easy to embed

No heavy frameworks inside the SDK.

---

## 5. Policy Driven

LLM usage should be governed by policies.

Examples:

- Maximum cost per request
- Monthly spending limits
- Model restrictions
- Model downgrades for background tasks

Policies must be configurable.

---

# High Level Architecture

Driftlock sits between the application and the LLM provider.

Application  
↓  
Driftlock SDK  
↓  
LLM Provider API

The SDK captures telemetry and applies policies before execution.

Architecture flow:

Application Code  
↓  
Driftlock SDK  
↓  
Policy Engine  
↓  
LLM Provider  
↓  
Telemetry Recorder

---

# Key Components

## 1. SDK Instrumentation

The SDK wraps LLM API calls.

Responsibilities:

- intercept requests
- estimate cost
- evaluate policies
- log telemetry
- execute provider request

Example wrapper:
track_llm(openai.chat.completions.create(...))


---

## 2. Cost Estimation

Before a request is executed Driftlock estimates cost.

Cost formula:


cost =
prompt_tokens * prompt_price +
completion_tokens * completion_price


Model pricing is maintained in a pricing table.

Example:


MODEL_PRICING = {
"gpt-4o": {
"prompt": 5.0 / 1_000_000,
"completion": 15.0 / 1_000_000
}
}


This allows Driftlock to predict cost before execution.

---

## 3. Policy Engine

The policy engine decides whether a request should proceed.

Policies evaluate request context.

Example context:


model
estimated_cost
tags
user_id
environment
request_type


Policies return a decision.


class RuleDecision:
allow: bool
action: str | None
metadata: dict


Possible actions:

allow  
block  
downgrade  
fallback  
log_only

---

# Example Policies

## MaxCostPerRequestRule

Prevents expensive calls.

Example:

Block requests exceeding $0.10.

---

## ModelRestrictionRule

Prevent certain models from being used.

Example:

Disallow GPT-4 for free tier users.

---

## Automatic Model Downgrade

Downgrade expensive models automatically.

Example:

GPT-4 → GPT-4o-mini

When:

- background jobs
- non-critical requests
- high traffic workloads

---

## MonthlyBudgetRule

Track total spend.

If monthly budget exceeded:

Block requests or downgrade models.

---

# Telemetry Model

Every LLM call generates a record.


class LLMRecord:

model: str
provider: str

prompt_tokens: int
completion_tokens: int
total_tokens: int

estimated_cost_usd: float

latency_ms: int

timestamp: datetime

request_id: str
user_id: Optional[str]

tags: dict

Telemetry enables debugging and cost analytics.

---

# Dashboard (Secondary Priority)

A dashboard helps developers understand usage patterns.

Built using:

FastAPI backend  
React or Next.js frontend

Metrics include:

- total cost
- requests per minute
- token usage
- model usage distribution
- most expensive prompts

However the dashboard is **secondary to the SDK and policy engine**.

---

# Project Structure


driftlock/

core/
tracker.py
telemetry.py
pricing.py

policy/
engine.py
rules.py

providers/
openai.py
anthropic.py
base_provider.py

storage/
recorder.py

dashboard/
fastapi_app.py


---

# Storage

Telemetry records may be stored in:

Local mode:

SQLite  
JSON logs

Cloud mode:

Postgres  
ClickHouse  
BigQuery

---

# Future Features

## Budget Forecasting

Predict monthly LLM spending.

## Token Usage Anomaly Detection

Detect spikes in token usage.

## Prompt Diff Tracking

Detect prompt growth over time.

## Model Routing

Automatically route requests to cheaper models when possible.

Example:

Simple queries → GPT-4o-mini  
Complex queries → GPT-4o

---

# Target Users

Driftlock is designed for:

AI startups  
SaaS companies integrating LLMs  
ML platform teams  
AI infrastructure engineers

Anyone operating LLMs in production.

---

# Long Term Vision

Driftlock aims to become the **control plane for LLM usage**.

Companies should rely on Driftlock to:

control cost  
enforce policies  
route models  
monitor usage  
prevent runaway spending

The goal is to make **LLM infrastructure financially predictable**.

---

# Instructions for AI Coding Assistants

When generating code for this repository:

Prefer simple modular architecture.

Avoid heavy dependencies.

Prioritize cost tracking and policy enforcement.

Provider implementations should be abstracted behind a common interface.

Telemetry structures should remain consistent across modules.

The primary goal of Driftlock is **LLM cost governance**.