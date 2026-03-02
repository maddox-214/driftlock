"""
Local SQLite storage for metrics.
Schema is intentionally minimal — one row per API call.

Migration: _migrate() adds new columns to existing databases so users don't
need to delete and recreate driftlock.db when upgrading.
"""

import json
import sqlite3
import threading
from pathlib import Path

from .metrics import CallMetrics


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS calls (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp               TEXT NOT NULL,
    model                   TEXT NOT NULL,
    endpoint                TEXT,
    prompt_tokens           INTEGER NOT NULL,
    completion_tokens       INTEGER NOT NULL,
    total_tokens            INTEGER NOT NULL,
    latency_ms              REAL NOT NULL,
    estimated_cost_usd      REAL,
    labels                  TEXT,
    warnings                TEXT,
    request_id              TEXT,
    -- cache fields (NULL / 0 for non-cached calls)
    cache_hit               INTEGER DEFAULT 0,
    cache_key               TEXT,
    tokens_saved_prompt     INTEGER DEFAULT 0,
    tokens_saved_completion INTEGER DEFAULT 0,
    estimated_savings_usd   REAL,
    -- optimization rollout status
    optimization_enabled    INTEGER DEFAULT 0,
    optimization_shadow     INTEGER DEFAULT 0,
    sampled_out             INTEGER DEFAULT 0,
    quality_regression      INTEGER DEFAULT 0
)
"""

# Columns added after the initial schema — applied via ALTER TABLE on existing DBs
_MIGRATION_COLUMNS: dict[str, str] = {
    "cache_hit":               "INTEGER DEFAULT 0",
    "cache_key":               "TEXT",
    "tokens_saved_prompt":     "INTEGER DEFAULT 0",
    "tokens_saved_completion": "INTEGER DEFAULT 0",
    "estimated_savings_usd":   "REAL",
    "optimization_enabled":    "INTEGER DEFAULT 0",
    "optimization_shadow":     "INTEGER DEFAULT 0",
    "sampled_out":             "INTEGER DEFAULT 0",
    "quality_regression":      "INTEGER DEFAULT 0",
}


class SQLiteStorage:
    def __init__(self, db_path: str = "driftlock.db") -> None:
        self._path = Path(db_path)
        self._local = threading.local()
        conn = self._conn()
        conn.execute(_CREATE_TABLE)
        self._migrate(conn)
        conn.commit()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(str(self._path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add any missing columns to an existing database."""
        existing = {row[1] for row in conn.execute("PRAGMA table_info(calls)")}
        for col, definition in _MIGRATION_COLUMNS.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE calls ADD COLUMN {col} {definition}")

    def save(self, m: CallMetrics) -> None:
        conn = self._conn()
        conn.execute(
            """
            INSERT INTO calls (
                timestamp, model, endpoint,
                prompt_tokens, completion_tokens, total_tokens,
                latency_ms, estimated_cost_usd,
                labels, warnings, request_id,
                cache_hit, cache_key,
                tokens_saved_prompt, tokens_saved_completion, estimated_savings_usd,
                optimization_enabled, optimization_shadow, sampled_out, quality_regression
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                m.timestamp.isoformat(),
                m.model,
                m.endpoint,
                m.prompt_tokens,
                m.completion_tokens,
                m.total_tokens,
                m.latency_ms,
                m.estimated_cost_usd,
                json.dumps(m.labels),
                json.dumps(m.warnings),
                m.request_id,
                int(m.cache_hit),
                m.cache_key,
                m.tokens_saved_prompt,
                m.tokens_saved_completion,
                m.estimated_savings_usd,
                int(m.optimization_enabled),
                int(m.optimization_shadow),
                int(m.sampled_out),
                int(m.quality_regression),
            ),
        )
        conn.commit()

    def aggregate(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        since: str | None = None,
    ) -> dict:
        """
        Return aggregated stats filtered by endpoint / model / since (ISO timestamp).
        Includes cache savings so callers can see total ROI.
        """
        where_clauses: list[str] = []
        params: list = []

        if endpoint:
            where_clauses.append("endpoint = ?")
            params.append(endpoint)
        if model:
            where_clauses.append("model = ?")
            params.append(model)
        if since:
            where_clauses.append("timestamp >= ?")
            params.append(since)

        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        row = self._conn().execute(
            f"""
            SELECT
                COUNT(*)                            AS calls,
                SUM(cache_hit)                      AS cache_hits,
                SUM(optimization_enabled)           AS optimization_enabled_count,
                SUM(optimization_shadow)            AS optimization_shadow_count,
                SUM(sampled_out)                    AS sampled_out_count,
                SUM(quality_regression)             AS quality_regression_count,
                SUM(prompt_tokens)                  AS total_prompt_tokens,
                SUM(completion_tokens)              AS total_completion_tokens,
                SUM(total_tokens)                   AS total_tokens,
                SUM(estimated_cost_usd)             AS total_cost_usd,
                SUM(tokens_saved_prompt)            AS total_tokens_saved_prompt,
                SUM(tokens_saved_completion)        AS total_tokens_saved_completion,
                SUM(estimated_savings_usd)          AS total_savings_usd,
                AVG(latency_ms)                     AS avg_latency_ms,
                MAX(latency_ms)                     AS max_latency_ms
            FROM calls {where}
            """,
            params,
        ).fetchone()

        calls = row["calls"] or 0
        cache_hits = row["cache_hits"] or 0
        return {
            "calls": calls,
            "cache_hits": cache_hits,
            "cache_hit_rate": round(cache_hits / calls, 4) if calls else 0.0,
            "optimization_enabled_count": row["optimization_enabled_count"] or 0,
            "optimization_shadow_count": row["optimization_shadow_count"] or 0,
            "sampled_out_count": row["sampled_out_count"] or 0,
            "quality_regression_count": row["quality_regression_count"] or 0,
            "total_prompt_tokens": row["total_prompt_tokens"] or 0,
            "total_completion_tokens": row["total_completion_tokens"] or 0,
            "total_tokens": row["total_tokens"] or 0,
            "total_cost_usd": round(row["total_cost_usd"] or 0, 6),
            "total_tokens_saved_prompt": row["total_tokens_saved_prompt"] or 0,
            "total_tokens_saved_completion": row["total_tokens_saved_completion"] or 0,
            "total_savings_usd": round(row["total_savings_usd"] or 0, 6),
            "avg_latency_ms": round(row["avg_latency_ms"] or 0, 2),
            "max_latency_ms": round(row["max_latency_ms"] or 0, 2),
        }

    def recent(self, limit: int = 20) -> list[dict]:
        rows = self._conn().execute(
            "SELECT * FROM calls ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["labels"] = json.loads(d["labels"] or "{}")
            d["warnings"] = json.loads(d["warnings"] or "[]")
            d["cache_hit"] = bool(d.get("cache_hit", 0))
            d["optimization_enabled"] = bool(d.get("optimization_enabled", 0))
            d["optimization_shadow"] = bool(d.get("optimization_shadow", 0))
            d["sampled_out"] = bool(d.get("sampled_out", 0))
            d["quality_regression"] = bool(d.get("quality_regression", 0))
            result.append(d)
        return result


class NoopStorage:
    """Discards all data (useful in tests or when storage is disabled)."""

    def save(self, m: CallMetrics) -> None:
        pass

    def aggregate(self, **kwargs) -> dict:
        return {}

    def recent(self, limit: int = 20) -> list[dict]:
        return []
