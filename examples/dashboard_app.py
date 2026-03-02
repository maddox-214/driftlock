"""
Minimal internal dashboard for Driftlock.

Run with:
    uvicorn examples.dashboard_app:app --reload

Optionally set DRIFTLOCK_DB_PATH to point at a specific driftlock.db file.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

DB_PATH = os.environ.get("DRIFTLOCK_DB_PATH", "driftlock.db")
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

app = FastAPI(title="Driftlock Dashboard", version="0.1.0")


def _db_exists() -> bool:
    return Path(DB_PATH).exists()


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return bool(row)


def _ensure_indexes(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE INDEX IF NOT EXISTS idx_calls_timestamp ON calls(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_calls_endpoint ON calls(endpoint)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_calls_model ON calls(model)")
    conn.commit()


def _empty_summary() -> dict[str, Any]:
    return {
        "total_requests": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_cost_usd": 0.0,
        "total_tokens_saved_prompt": 0,
        "total_tokens_saved_completion": 0,
        "total_savings_usd": 0.0,
        "cache_hit_rate": 0.0,
        "optimization_applied_count": 0,
        "savings_percentage": 0.0,
    }


def get_summary(start_time: str) -> dict[str, Any]:
    if not _db_exists():
        return _empty_summary()

    with _get_conn() as conn:
        if not _table_exists(conn, "calls"):
            return _empty_summary()

        _ensure_indexes(conn)
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_requests,
                COALESCE(SUM(prompt_tokens), 0) AS total_prompt_tokens,
                COALESCE(SUM(completion_tokens), 0) AS total_completion_tokens,
                COALESCE(SUM(estimated_cost_usd), 0) AS total_cost_usd,
                COALESCE(SUM(tokens_saved_prompt), 0) AS total_tokens_saved_prompt,
                COALESCE(SUM(tokens_saved_completion), 0) AS total_tokens_saved_completion,
                COALESCE(SUM(estimated_savings_usd), 0) AS total_savings_usd,
                COALESCE(SUM(cache_hit), 0) AS cache_hits,
                COALESCE(SUM(optimization_enabled), 0) AS optimization_applied_count
            FROM calls
            WHERE timestamp >= ?
            """,
            (start_time,),
        ).fetchone()

    total_requests = int(row["total_requests"] or 0)
    cache_hits = int(row["cache_hits"] or 0)
    total_cost_usd = round(float(row["total_cost_usd"] or 0.0), 6)
    total_savings_usd = round(float(row["total_savings_usd"] or 0.0), 6)
    denom = total_cost_usd + total_savings_usd
    savings_percentage = round((total_savings_usd / denom), 4) if denom > 0 else 0.0

    return {
        "total_requests": total_requests,
        "total_prompt_tokens": int(row["total_prompt_tokens"] or 0),
        "total_completion_tokens": int(row["total_completion_tokens"] or 0),
        "total_cost_usd": total_cost_usd,
        "total_tokens_saved_prompt": int(row["total_tokens_saved_prompt"] or 0),
        "total_tokens_saved_completion": int(row["total_tokens_saved_completion"] or 0),
        "total_savings_usd": total_savings_usd,
        "cache_hit_rate": round(cache_hits / total_requests, 4) if total_requests else 0.0,
        "optimization_applied_count": int(row["optimization_applied_count"] or 0),
        "savings_percentage": savings_percentage,
    }


def get_top_endpoints(limit: int) -> list[dict[str, Any]]:
    if not _db_exists():
        return []

    with _get_conn() as conn:
        if not _table_exists(conn, "calls"):
            return []

        _ensure_indexes(conn)
        rows = conn.execute(
            """
            SELECT
                COALESCE(endpoint, 'unknown') AS endpoint,
                COUNT(*) AS total_requests,
                COALESCE(SUM(estimated_cost_usd), 0) AS total_cost_usd
            FROM calls
            GROUP BY endpoint
            ORDER BY total_cost_usd DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [
        {
            "endpoint": r["endpoint"],
            "total_requests": int(r["total_requests"] or 0),
            "total_cost_usd": round(float(r["total_cost_usd"] or 0.0), 6),
        }
        for r in rows
    ]


def get_recent_calls(limit: int) -> list[dict[str, Any]]:
    if not _db_exists():
        return []

    with _get_conn() as conn:
        if not _table_exists(conn, "calls"):
            return []

        _ensure_indexes(conn)
        rows = conn.execute(
            """
            SELECT
                timestamp,
                model,
                endpoint,
                estimated_cost_usd,
                cache_hit,
                tokens_saved_prompt,
                tokens_saved_completion,
                optimization_enabled
            FROM calls
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    result = []
    for r in rows:
        result.append(
            {
                "timestamp": r["timestamp"],
                "model": r["model"],
                "endpoint": r["endpoint"] or "(none)",
                "cost_usd": round(float(r["estimated_cost_usd"] or 0.0), 6),
                "cache_hit": bool(r["cache_hit"] or 0),
                "tokens_saved_prompt": int(r["tokens_saved_prompt"] or 0),
                "tokens_saved_completion": int(r["tokens_saved_completion"] or 0),
                "optimization_applied": bool(r["optimization_enabled"] or 0),
            }
        )
    return result


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    now = datetime.now(timezone.utc)
    start_24h = (now - timedelta(hours=24)).isoformat()
    start_7d = (now - timedelta(days=7)).isoformat()

    summary_24h = get_summary(start_24h)
    summary_7d = get_summary(start_7d)

    return TEMPLATES.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "summary_24h": summary_24h,
            "summary_7d": summary_7d,
            "top_endpoints": get_top_endpoints(5),
            "recent_calls": get_recent_calls(20),
            "generated_at": now.isoformat(timespec="seconds"),
            "db_path": DB_PATH,
        },
    )


@app.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse(url="/dashboard")


@app.on_event("startup")
def _startup() -> None:
    if not _db_exists():
        return
    with _get_conn() as conn:
        if _table_exists(conn, "calls"):
            _ensure_indexes(conn)
