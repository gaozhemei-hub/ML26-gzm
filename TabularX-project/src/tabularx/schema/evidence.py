"""Schema types for evidence-delivery contracts and audit snapshots."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

EvidenceStatus = Literal[
    "pending", "generated", "failed", "missing", "not_enabled", "not_applicable"
]


@dataclass
class EvidenceAttempt:
    """One execution attempt for a declared evidence producer."""

    started_at: str
    ended_at: str
    status: EvidenceStatus
    reason: str
    paths: list[str] = field(default_factory=list)


@dataclass
class EvidenceRecord:
    """Declared requirement and its latest observed delivery state."""

    required: bool
    applicable: bool
    condition: str
    enabled: bool
    status: EvidenceStatus
    reason: str
    paths: list[str] = field(default_factory=list)
    attempts: list[EvidenceAttempt] = field(default_factory=list)
    recoverable: bool = True


@dataclass
class EvidenceManifest:
    """Versioned audit snapshot for one TabularX run."""

    run_id: str
    task: str
    complete: bool
    required_count: int
    generated_required_count: int
    completeness: float | None
    boundary: str
    records: dict[str, EvidenceRecord]
    schema_version: str = "1.0"

    @classmethod
    def from_dict(cls, payload: dict) -> "EvidenceManifest":
        """Validate and parse a JSON-compatible manifest payload."""
        version = payload.get("schema_version")
        if version != "1.0":
            raise ValueError(f"Unsupported evidence schema_version: {version!r}")
        valid_statuses = {"pending", "generated", "failed", "missing", "not_enabled", "not_applicable"}
        records = {}
        for key, raw in payload["records"].items():
            if raw.get("status") not in valid_statuses:
                raise ValueError(f"Invalid evidence status for {key!r}: {raw.get('status')!r}")
            attempts = []
            for attempt in raw.get("attempts", []):
                if attempt.get("status") not in valid_statuses:
                    raise ValueError(f"Invalid evidence attempt status for {key!r}")
                attempts.append(EvidenceAttempt(**attempt))
            records[key] = EvidenceRecord(
                required=raw["required"], applicable=raw["applicable"],
                condition=raw["condition"], enabled=raw["enabled"],
                status=raw["status"], reason=raw["reason"],
                paths=list(raw.get("paths", [])), attempts=attempts,
                recoverable=raw.get("recoverable", True))
        return cls(run_id=payload["run_id"], task=payload["task"],
                   complete=payload["complete"], required_count=payload["required_count"],
                   generated_required_count=payload["generated_required_count"],
                   completeness=payload["completeness"], boundary=payload["boundary"],
                   records=records, schema_version=version)

    def to_dict(self) -> dict:
        """Return the stable JSON representation of this contract."""
        return asdict(self)


__all__ = ["EvidenceStatus", "EvidenceAttempt", "EvidenceRecord", "EvidenceManifest"]
