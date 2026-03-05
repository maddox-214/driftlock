from .alerts import (
    AlertChannel,
    LogAlertChannel,
    SlackAlertChannel,
    WebhookAlertChannel,
)
from .cache import CacheConfig
from .client import DriftlockClient
from .config import DriftlockConfig
from .context import tag
from .drift import detect_drift, hash_prompt
from .optimization import BudgetExceededError, OptimizationConfig
from .policy import (
    BaseRule,
    CircuitOpenError,
    CostVelocityRule,
    ForecastBudgetRule,
    MaxCostPerRequestRule,
    MonthlyBudgetRule,
    PerUserBudgetRule,
    PolicyEngine,
    PolicyViolationError,
    RestrictModelRule,
    RuleDecision,
    TagBasedModelDowngradeRule,
    VelocityLimitRule,
)
from .providers import AnthropicProvider, NormalizedUsage, OpenAIProvider

__version__ = "0.2.0"
__all__ = [
    # Core clients
    "DriftlockClient",
    "DriftlockConfig",
    "OptimizationConfig",
    "BudgetExceededError",
    "CacheConfig",
    "tag",
    # Policy engine
    "PolicyEngine",
    "PolicyViolationError",
    "CircuitOpenError",
    "RuleDecision",
    "BaseRule",
    "MaxCostPerRequestRule",
    "RestrictModelRule",
    "TagBasedModelDowngradeRule",
    "MonthlyBudgetRule",
    "PerUserBudgetRule",
    "VelocityLimitRule",
    "CostVelocityRule",
    "ForecastBudgetRule",
    # Alerts
    "AlertChannel",
    "WebhookAlertChannel",
    "SlackAlertChannel",
    "LogAlertChannel",
    # Providers
    "NormalizedUsage",
    "OpenAIProvider",
    "AnthropicProvider",
    # Drift
    "hash_prompt",
    "detect_drift",
]

# Anthropic client is opt-in (requires `pip install driftlock[anthropic]`)
try:
    from .anthropic_client import AnthropicDriftlockClient
    __all__.append("AnthropicDriftlockClient")
except ImportError:
    pass
