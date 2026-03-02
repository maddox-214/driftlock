"""
Example FastAPI application showing how to integrate DriftlockClient
with optimization, caching, and automatic per-request tag injection.

Run with:
    uvicorn examples.fastapi_app:app --reload

Set OPENAI_API_KEY in your environment before starting.
"""

import os
import uuid

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

import driftlock
from driftlock import CacheConfig, DriftlockClient, DriftlockConfig, OptimizationConfig

# ---------------------------------------------------------------------------
# Initialise the client once at startup.
# ---------------------------------------------------------------------------
client = DriftlockClient(
    api_key=os.environ["OPENAI_API_KEY"],
    config=DriftlockConfig(
        log_json=True,
        prompt_token_warning_threshold=3000,
        cost_warning_threshold=0.05,
    ),
    optimization=OptimizationConfig(
        max_prompt_tokens=3000,
        keep_last_n_messages=10,
        default_max_output_tokens=512,
        max_cost_per_request_usd=0.10,
        budget_exceeded_action="fallback",
        fallback_model="gpt-4o-mini",
    ),
    cache=CacheConfig(
        ttl_seconds=600,    # cache responses for 10 minutes
        max_entries=500,
    ),
)

app = FastAPI(title="Driftlock Example", version="0.1.0")


# ---------------------------------------------------------------------------
# Middleware: inject request_id and route into every DriftlockClient call
# made within this request, with zero changes to the route handlers.
# ---------------------------------------------------------------------------
class DriftlockTagMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        with driftlock.tag(
            request_id=request_id,
            route=request.url.path,
            method=request.method,
        ):
            response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(DriftlockTagMiddleware)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    model: str = "gpt-4o-mini"


class ChatResponse(BaseModel):
    reply: str
    model: str
    cache_hit: bool = False


class SummaryRequest(BaseModel):
    text: str
    model: str = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Simple chat endpoint.
    Tags endpoint='chat' per-call; request_id/route come from middleware.
    """
    try:
        response = client.chat.completions.create(
            model=req.model,
            messages=[{"role": "user", "content": req.message}],
            temperature=0.0,           # deterministic → cache-friendly
            _dl_endpoint="chat",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    recent = client.recent_calls(limit=1)
    was_cached = recent[0]["cache_hit"] if recent else False

    return ChatResponse(
        reply=response.choices[0].message.content,
        model=response.model,
        cache_hit=was_cached,
    )


@app.post("/summarise", response_model=ChatResponse)
async def summarise(req: SummaryRequest):
    """Summarise arbitrary text. Long inputs are trimmed by the optimizer."""
    try:
        response = client.chat.completions.create(
            model=req.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise summariser. Return a 2-3 sentence summary.",
                },
                {"role": "user", "content": req.text},
            ],
            temperature=0.0,
            _dl_endpoint="summarise",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    return ChatResponse(
        reply=response.choices[0].message.content,
        model=response.model,
    )


@app.get("/metrics")
async def metrics():
    """Aggregated usage + savings stats from local SQLite."""
    return {
        "all_time": client.stats(),
        "by_endpoint": {
            "chat": client.stats(endpoint="chat"),
            "summarise": client.stats(endpoint="summarise"),
        },
        "cache": client.cache_stats(),
    }


@app.get("/metrics/recent")
async def recent_calls(limit: int = 10):
    """Return the N most recent tracked calls (includes cache hits)."""
    return client.recent_calls(limit=limit)
