"""Configuration for the standalone audit extension."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class LeakageAuditConfig:
    """Explicit assumptions that cannot safely be inferred from a data table."""

    mode: Literal["warn", "drop_high_confidence", "fail"] = "warn"
    id_columns: list[str] = field(default_factory=list)
    forbidden_features: list[str] = field(default_factory=list)
    group_column: str | None = None
    time_column: str | None = None
    identifier_unique_ratio: float = 0.98
    target_proxy_threshold: float = 0.98

    def __post_init__(self) -> None:
        if not 0 <= self.identifier_unique_ratio <= 1:
            raise ValueError("identifier_unique_ratio 必须在 [0, 1] 内")
        if not 0 <= self.target_proxy_threshold <= 1:
            raise ValueError("target_proxy_threshold 必须在 [0, 1] 内")
