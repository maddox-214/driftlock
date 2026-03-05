from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .alerts import AlertChannel


@dataclass
class DriftlockConfig:
    """Configuration for DriftlockClient / AnthropicDriftlockClient."""

    # Logging
    log_json: bool = True
    log_level: str = "INFO"

    # Storage
    storage_backend: str = "sqlite"  # "sqlite" | "none"
    db_path: str = "driftlock.sqlite"

    # Warnings
    prompt_token_warning_threshold: int = 4000  # warn if prompt exceeds this
    cost_warning_threshold: float | None = None  # warn if single call exceeds $X

    # Labels applied to all tracked calls
    default_labels: dict = field(default_factory=dict)

    # Alert channels — fired on policy violations and cost warnings
    alert_channels: list["AlertChannel"] = field(default_factory=list)
