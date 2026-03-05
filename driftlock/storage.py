"""
Local SQLite storage for metrics.
Schema is intentionally minimal — one row per API call.

Migration: _migrate() adds new columns to existing databases so users don't
need to delete and recreate driftlock.db when upgrading.
"""

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .metrics import CallMetrics


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS calls (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp               TEXT NOT NULL,
    provider                TEXT NOT NULL DEFAULT 'openai',
    model                   TEXT NOT NULL,
    endpoint                TEXT,
    user_id                 TEXT,
    team_id                 TEXT,
    prompt_tokens           INTEGER NOT NULL,
    completion_tokens       INTEGER NOT NULL,
    total_tokens            INTEGER NOT NULL,
    latency_ms              REAL NOT NULL,
    estimated_cost_usd      REAL,
    labels                  TEXT,
    warnings                TEXT,
    request_id              TEXT,
    prompt_hash             TEXT,
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

_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_calls_timestamp ON calls(timestamp);
CREATE INDEX IF NOT EXISTS idx_calls_endpoint  ON calls(endpoint);
CREATE INDEX IF NOT EXISTS idx_calls_model     ON calls(model);
CREATE INDEX IF NOT EXISTS idx_calls_user_id   ON calls(user_id);
CREATE INDEX IF NOT EXISTS idx_calls_team_id   ON calls(team_id);
CREATE INDEX IF NOT EXISTS idx_calls_provider  ON calls(provider);
"""

# Columns added after the initial schema — applied via ALTER TABLE on existing DBs
_MIGRATION_COLUMNS: dict[str, str] = {
    "provider":                "TEXT NOT NULL DEFAULT 'openai'",
    "user_id":                 "TEXT",
    "team_id":                 "TEXT",
    "prompt_hash":             "TEXT",
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
        for stmt in _CREATE_INDEXES.strip().splitlines():
            stmt = stmt.strip().rstrip(";")
            if stmt:
                conn.execute(stmt)
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
        user_id = m.labels.get("user_id") if m.labels else None
        team_id = m.labels.get("team_id") if m.labels else None
        conn = self._conn()
        conn.execute(
            """
            INSERT INTO calls (
                timestamp, provider, model, endpoint, user_id, team_id,
                prompt_tokens, completion_tokens, total_tokens,
                latency_ms, estimated_cost_usd,
                labels, warnings, request_id, prompt_hash,
                cache_hit, cache_key,
                tokens_saved_prompt, tokens_saved_completion, estimated_savings_usd,
                optimization_enabled, optimization_shadow, sampled_out, quality_regression
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                m.timestamp.isoformat(),
                m.provider,
                m.model,
                m.endpoint,
                user_id,
                team_id,
                m.prompt_tokens,
                m.completion_tokens,
                m.total_tokens,
                m.latency_ms,
                m.estimated_cost_usd,
                json.dumps(m.labels),
                json.dumps(m.warnings),
                m.request_id,
                m.prompt_hash,
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
        provider: str | None = None,
        user_id: str | None = None,
        team_id: str | None = None,
        since: str | None = None,
    ) -> dict:
        """
        Return aggregated stats filtered by any combination of dimensions.
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
        if provider:
            where_clauses.append("provider = ?")
            params.append(provider)
        if user_id:
            where_clauses.append("user_id = ?")
            params.append(user_id)
        if team_id:
            where_clauses.append("team_id = ?")
            params.append(team_id)
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

    # ------------------------------------------------------------------ #
    # Velocity helpers (used by circuit breaker rules)
    # ------------------------------------------------------------------ #

    def count_since(self, since: str, user_id: str | None = None) -> int:
        """Return number of calls recorded since an ISO timestamp."""
        if user_id:
            row = self._conn().execute(
                "SELECT COUNT(*) FROM calls WHERE timestamp >= ? AND user_id = ?",
                (since, user_id),
            ).fetchone()
        else:
            row = self._conn().execute(
                "SELECT COUNT(*) FROM calls WHERE timestamp >= ?", (since,)
            ).fetchone()
        return row[0] or 0

    def cost_since(self, since: str, user_id: str | None = None) -> float:
        """Return total estimated cost (USD) since an ISO timestamp."""
        if user_id:
            row = self._conn().execute(
                "SELECT COALESCE(SUM(estimated_cost_usd), 0.0) FROM calls "
                "WHERE timestamp >= ? AND user_id = ?",
                (since, user_id),
            ).fetchone()
        else:
            row = self._conn().execute(
                "SELECT COALESCE(SUM(estimated_cost_usd), 0.0) FROM calls WHERE timestamp >= ?",
                (since,),
            ).fetchone()
        return float(row[0] or 0.0)

    # ------------------------------------------------------------------ #
    # Analytics queries (used by dashboard / CLI)
    # ------------------------------------------------------------------ #

    def top_users(self, since: str | None = None, limit: int = 20) -> list[dict]:
        """Return per-user cost breakdown, highest spend first."""
        where = "WHERE user_id IS NOT NULL"
        params: list = []
        if since:
            where += " AND timestamp >= ?"
            params.append(since)
        rows = self._conn().execute(
            f"""
            SELECT user_id,
                   COUNT(*)                          AS calls,
                   COALESCE(SUM(estimated_cost_usd), 0) AS total_cost_usd
            FROM calls {where}
            GROUP BY user_id
            ORDER BY total_cost_usd DESC
            LIMIT ?
            """,
            params + [limit],
        ).fetchall()
        return [
            {
                "user_id": r["user_id"],
                "calls": r["calls"],
                "total_cost_usd": round(float(r["total_cost_usd"]), 6),
            }
            for r in rows
        ]

    def model_distribution(self, since: str | None = None) -> list[dict]:
        """Return per-model call count and cost share."""
        where = ""
        params: list = []
        if since:
            where = "WHERE timestamp >= ?"
            params.append(since)
        rows = self._conn().execute(
            f"""
            SELECT model,
                   COUNT(*)                          AS calls,
                   COALESCE(SUM(estimated_cost_usd), 0) AS total_cost_usd
            FROM calls {where}
            GROUP BY model
            ORDER BY total_cost_usd DESC
            """,
            params,
        ).fetchall()
        return [
            {
                "model": r["model"],
                "calls": r["calls"],
                "total_cost_usd": round(float(r["total_cost_usd"]), 6),
            }
            for r in rows
        ]

    # ------------------------------------------------------------------ #
    # Prompt drift queries (used by drift tracker)
    # ------------------------------------------------------------------ #

    def prompt_hash_history(
        self, endpoint: str, limit: int = 30
    ) -> list[dict]:
        """Return recent (timestamp, prompt_hash, prompt_tokens) rows for an endpoint."""
        rows = self._conn().execute(
            """
            SELECT timestamp, prompt_hash, prompt_tokens
            FROM calls
            WHERE endpoint = ? AND prompt_hash IS NOT NULL
            ORDER BY id DESC LIMIT ?
            """,
            (endpoint, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Forecasting helpers
    # ------------------------------------------------------------------ #

    def daily_cost_trend(self, lookback_days: int = 7) -> list[dict]:
        """
        Return per-day cost totals for the last N days (oldest first).
        Days with no calls are omitted.
        """
        since = (
            datetime.now(timezone.utc) - timedelta(days=lookback_days)
        ).isoformat()
        rows = self._conn().execute(
            """
            SELECT SUBSTR(timestamp, 1, 10)          AS day,
                   COUNT(*)                          AS calls,
                   COALESCE(SUM(estimated_cost_usd), 0) AS cost_usd
            FROM calls
            WHERE timestamp >= ?
            GROUP BY day
            ORDER BY day ASC
            """,
            (since,),
        ).fetchall()
        return [
            {
                "day": r["day"],
                "calls": r["calls"],
                "cost_usd": round(float(r["cost_usd"]), 6),
            }
            for r in rows
        ]


class NoopStorage:
    """Discards all data (useful in tests or when storage is disabled)."""

    def save(self, m: CallMetrics) -> None:
        pass

    def aggregate(self, **kwargs) -> dict:
        return {}

    def recent(self, limit: int = 20) -> list[dict]:
        return []

    def count_since(self, since: str, user_id: str | None = None) -> int:
        return 0

    def cost_since(self, since: str, user_id: str | None = None) -> float:
        return 0.0

    def top_users(self, since: str | None = None, limit: int = 20) -> list[dict]:
        return []

    def model_distribution(self, since: str | None = None) -> list[dict]:
        return []

    def prompt_hash_history(self, endpoint: str, limit: int = 30) -> list[dict]:
        return []

    def daily_cost_trend(self, lookback_days: int = 7) -> list[dict]:
        return []
