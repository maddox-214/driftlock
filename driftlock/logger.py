"""
Structured JSON logging for Driftlock.
Uses Python's stdlib logging so it slots naturally into existing log pipelines.
"""

import json
import logging
import sys
from typing import Any

from .metrics import CallMetrics

_LOGGER_NAME = "driftlock"


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "metrics"):
            payload["metrics"] = record.metrics  # type: ignore[attr-defined]
        return json.dumps(payload, default=str)


def _build_logger(log_json: bool, log_level: str) -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)

    if log_json:
        handler.setFormatter(_JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] driftlock: %(message)s")
        )

    logger.addHandler(handler)
    logger.propagate = False
    return logger


class DriftlockLogger:
    def __init__(self, log_json: bool = True, log_level: str = "INFO") -> None:
        self._logger = _build_logger(log_json, log_level)

    def log_call(self, metrics: CallMetrics) -> None:
        extra = {"metrics": metrics.to_dict()}

        if metrics.cache_hit:
            parts = [
                "[CACHE HIT]",
                f"model={metrics.model}",
                f"latency={metrics.latency_ms:.1f}ms",
                f"saved={metrics.tokens_saved_prompt + metrics.tokens_saved_completion}tok",
            ]
            if metrics.estimated_savings_usd is not None:
                parts.append(f"saved_usd=${metrics.estimated_savings_usd:.6f}")
            if metrics.endpoint:
                parts.append(f"endpoint={metrics.endpoint}")
            if metrics.cache_key:
                parts.append(f"key={metrics.cache_key}…")
            self._logger.info(" | ".join(parts), extra=extra)
            return

        parts = [
            f"model={metrics.model}",
            f"tokens={metrics.total_tokens}",
            f"latency={metrics.latency_ms:.0f}ms",
        ]
        if metrics.estimated_cost_usd is not None:
            parts.append(f"cost=${metrics.estimated_cost_usd:.6f}")
        if metrics.endpoint:
            parts.append(f"endpoint={metrics.endpoint}")

        # Optimization savings
        if metrics.optimization_report:
            report = metrics.optimization_report
            if report.shadow_mode:
                parts.append("shadow=true")
                if report.tokens_saved() > 0:
                    parts.append(f"projected_saved={report.tokens_saved()}tok")
                if report.cost_saved_usd() is not None:
                    parts.append(f"projected_saved_usd=${report.cost_saved_usd():.6f}")
            else:
                if report.tokens_saved() > 0:
                    parts.append(f"saved={report.tokens_saved()}tok")
            if report.bypassed_reason:
                parts.append(f"bypass={report.bypassed_reason}")
            if report.optimizations_applied:
                parts.append(f"opts=[{','.join(report.optimizations_applied)}]")

        msg = " | ".join(parts)

        if metrics.warnings:
            self._logger.warning(msg, extra=extra)
            for w in metrics.warnings:
                self._logger.warning("  -> %s", w)
        else:
            self._logger.info(msg, extra=extra)
