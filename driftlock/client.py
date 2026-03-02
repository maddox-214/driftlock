"""
DriftlockClient — a transparent wrapper around the OpenAI Python SDK.

Usage::

    from driftlock import DriftlockClient, DriftlockConfig, OptimizationConfig, CacheConfig

    client = DriftlockClient(
        api_key="sk-...",
        optimization=OptimizationConfig(max_prompt_tokens=3000),
        cache=CacheConfig(ttl_seconds=600),
    )

    # Optionally set ambient tags for a whole block (e.g. from middleware):
    with driftlock.tag(request_id="req_123", user_id="u_42"):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello!"}],
            _dl_endpoint="my_function",   # per-call label
            _dl_labels={"env": "prod"},   # per-call labels
        )
"""

import hashlib
import os
import time
import uuid
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletion

from .cache import CacheConfig, ResponseCache, make_cache_key
from .config import DriftlockConfig
from .context import get_active_tags
from .logger import DriftlockLogger
from .metrics import CallMetrics
from .optimization import OptimizationConfig, OptimizationPipeline
from .policy import PolicyEngine, PolicyViolationError
from .pricing import estimate_cost
from .storage import NoopStorage, SQLiteStorage


def _env_flag(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() not in {"0", "false", "no", "off"}


def _sample_key_value(sample_key: str, labels: dict, kwargs: dict) -> str:
    if sample_key in labels:
        return str(labels[sample_key])
    if sample_key in kwargs:
        return str(kwargs[sample_key])
    return "unknown"


def _is_sampled_in(sample_value: str, rate: float) -> bool:
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    digest = hashlib.sha256(sample_value.encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return bucket < rate


class _ChatCompletionsWrapper:
    """Intercepts chat.completions.create calls."""

    def __init__(self, openai_completions, client: "DriftlockClient") -> None:
        self._completions = openai_completions
        self._dl = client

    def create(self, *args, **kwargs) -> ChatCompletion:
        # ------------------------------------------------------------------ #
        # 1. Strip Driftlock-only kwargs
        # ------------------------------------------------------------------ #
        endpoint: str | None = kwargs.pop("_dl_endpoint", None)
        labels: dict = kwargs.pop("_dl_labels", {})

        # Merge precedence: default_labels < context tags < per-call labels
        merged_labels: dict = {
            **self._dl._config.default_labels,
            **get_active_tags(),
            **labels,
        }

        enabled = _env_flag("DRIFTLOCK_ENABLED", True)
        track_only = _env_flag("DRIFTLOCK_TRACK_ONLY", False)

        if not enabled and not track_only:
            return self._completions.create(*args, **kwargs)

        # ------------------------------------------------------------------ #
        # 2. Optimization pipeline (trimming → output cap → budget guardrail)
        # ------------------------------------------------------------------ #
        opt_report = None
        policy_decisions: list[dict] = []
        optimization_enabled = False
        optimization_shadow = False
        sampled_out = False

        optimizer = self._dl._optimizer if enabled else None
        cache = self._dl._cache if enabled else None

        if optimizer is not None:
            cfg = self._dl._optimization_config
            optimization_shadow = bool(cfg.shadow_mode)
            sample_rate = max(0.0, min(1.0, cfg.sample_rate))
            sample_value = _sample_key_value(cfg.sample_key, merged_labels, kwargs)
            sampled_in = _is_sampled_in(sample_value, sample_rate)
            sampled_out = not sampled_in
            apply = sampled_in and not optimization_shadow

            model, messages, kwargs, opt_report = optimizer.process(
                model=kwargs.get("model", "unknown"),
                messages=kwargs.get("messages", []),
                kwargs=kwargs,
                apply=apply,
                shadow_mode=optimization_shadow,
            )
            if opt_report and sampled_out:
                opt_report.bypassed_reason = "sampled_out"

            kwargs["model"] = model
            kwargs["messages"] = messages
            optimization_enabled = not sampled_out

        # ------------------------------------------------------------------ #
        # 3. Policy evaluation (after optimization, before cache)
        # ------------------------------------------------------------------ #
        if self._dl._policy is not None:
            ctx = {
                "model": kwargs.get("model", "unknown"),
                "messages": kwargs.get("messages", []),
                "kwargs": kwargs,
                "labels": merged_labels,
                "optimization_report": opt_report,
                "storage": self._dl._storage,
            }
            for rule_name, decision in self._dl._policy.evaluate(ctx):
                policy_decisions.append(
                    {
                        "rule": rule_name,
                        "allow": decision.allow,
                        "action": decision.action,
                        "metadata": decision.metadata,
                    }
                )
                if not decision.allow or decision.action == "block":
                    raise PolicyViolationError(rule_name, decision)
                if decision.action in {"downgrade", "fallback"}:
                    target = (
                        decision.metadata.get("downgrade_to")
                        or decision.metadata.get("fallback_to")
                        or decision.metadata.get("fallback_model")
                    )
                    if target:
                        kwargs["model"] = target
                        ctx["model"] = target

        # ------------------------------------------------------------------ #
        # 4. Cache lookup (key computed AFTER optimization)
        # ------------------------------------------------------------------ #
        cache_key: str | None = None
        if cache is not None and not kwargs.get("stream", False):
            cache_key = make_cache_key(
                model=kwargs.get("model", "unknown"),
                messages=kwargs.get("messages", []),
                kwargs=kwargs,
            )
            t0 = time.perf_counter()
            entry = cache.get(cache_key)
            latency_ms = (time.perf_counter() - t0) * 1000

            if entry is not None:
                model_name = kwargs.get("model", "unknown")
                savings_usd = estimate_cost(
                    model_name, entry.prompt_tokens, entry.completion_tokens
                )
                metrics = CallMetrics(
                    model=model_name,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    latency_ms=latency_ms,
                    estimated_cost_usd=0.0,
                    endpoint=endpoint,
                    labels=merged_labels,
                    request_id=str(uuid.uuid4()),
                    optimization_report=opt_report,
                    policy_decisions=policy_decisions,
                    cache_hit=True,
                    cache_key=cache_key[:8],
                    tokens_saved_prompt=entry.prompt_tokens,
                    tokens_saved_completion=entry.completion_tokens,
                    estimated_savings_usd=savings_usd,
                    optimization_enabled=optimization_enabled,
                    optimization_shadow=optimization_shadow,
                    sampled_out=sampled_out,
                )
                self._dl._logger.log_call(metrics)
                self._dl._storage.save(metrics)
                return entry.response

        # ------------------------------------------------------------------ #
        # 4. Cache miss — call the real API
        # ------------------------------------------------------------------ #
        start = time.perf_counter()
        response: ChatCompletion = self._completions.create(*args, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0
        model_name = response.model or kwargs.get("model", "unknown")

        # Store in cache before computing metrics so the entry is available
        # if another thread is making an identical concurrent request.
        if cache is not None and cache_key is not None:
            cache.put(cache_key, response, prompt_tokens, completion_tokens)

        # ------------------------------------------------------------------ #
        # 5. Metrics, warnings, logging, storage
        # ------------------------------------------------------------------ #
        cost = estimate_cost(model_name, prompt_tokens, completion_tokens)
        cfg = self._dl._config
        warnings: list[str] = []

        if prompt_tokens > cfg.prompt_token_warning_threshold:
            warnings.append(
                f"Prompt is large: {prompt_tokens} tokens "
                f"(threshold: {cfg.prompt_token_warning_threshold})"
            )
        if cfg.cost_warning_threshold and cost and cost > cfg.cost_warning_threshold:
            warnings.append(
                f"Call cost ${cost:.6f} exceeds warning threshold "
                f"${cfg.cost_warning_threshold:.6f}"
            )

        metrics = CallMetrics(
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=cost,
            endpoint=endpoint,
            labels=merged_labels,
            request_id=str(uuid.uuid4()),
            warnings=warnings,
            optimization_report=opt_report,
            policy_decisions=policy_decisions,
            optimization_enabled=optimization_enabled,
            optimization_shadow=optimization_shadow,
            sampled_out=sampled_out,
        )

        self._dl._logger.log_call(metrics)
        self._dl._storage.save(metrics)
        return response

    async def acreate(self, *args, **kwargs):
        raise NotImplementedError(
            "DriftlockClient does not yet wrap async calls. "
            "Use the synchronous client or contribute async support."
        )


class _ChatWrapper:
    def __init__(self, openai_chat, client: "DriftlockClient") -> None:
        self.completions = _ChatCompletionsWrapper(openai_chat.completions, client)


class DriftlockClient:
    """
    Drop-in wrapper around openai.OpenAI.

    Adds token tracking, cost estimation, latency measurement, structured
    logging, an optional pre-call optimization pipeline, and an optional
    exact in-memory response cache.

    All kwargs not listed below are forwarded to openai.OpenAI.
    """

    def __init__(
        self,
        *,
        config: DriftlockConfig | None = None,
        optimization: OptimizationConfig | None = None,
        cache: CacheConfig | None = None,
        policy: PolicyEngine | None = None,
        **openai_kwargs: Any,
    ) -> None:
        self._config = config or DriftlockConfig()
        self._openai = OpenAI(**openai_kwargs)
        self._logger = DriftlockLogger(
            log_json=self._config.log_json,
            log_level=self._config.log_level,
        )
        self._optimization_config = optimization or OptimizationConfig()
        self._optimizer = (
            OptimizationPipeline(self._optimization_config) if optimization else None
        )
        self._cache = ResponseCache(cache) if (cache and cache.enabled) else None
        self._policy = policy

        if self._config.storage_backend == "sqlite":
            self._storage = SQLiteStorage(self._config.db_path)
        else:
            self._storage = NoopStorage()

        self.chat = _ChatWrapper(self._openai.chat, self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._openai, name)

    def stats(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        since: str | None = None,
    ) -> dict:
        """Return aggregated metrics from local storage (includes cache savings)."""
        return self._storage.aggregate(endpoint=endpoint, model=model, since=since)

    def recent_calls(self, limit: int = 20) -> list[dict]:
        """Return the N most recent tracked calls."""
        return self._storage.recent(limit=limit)

    def cache_stats(self) -> dict:
        """Return live cache hit/miss stats (in-memory only, not persisted)."""
        if self._cache is None:
            return {"enabled": False}
        return {"enabled": True, **self._cache.stats()}
