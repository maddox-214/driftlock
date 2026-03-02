from dataclasses import dataclass, field


@dataclass
class DriftlockConfig:
    """Configuration for DriftlockClient."""

    # Logging
    log_json: bool = True
    log_level: str = "INFO"

    # Storage
    storage_backend: str = "sqlite"  # "sqlite" | "none"
    db_path: str = "driftlock.db"

    # Warnings
    prompt_token_warning_threshold: int = 4000  # warn if prompt exceeds this
    cost_warning_threshold: float | None = None  # warn if single call exceeds $X

    # Labels applied to all tracked calls
    default_labels: dict = field(default_factory=dict)
