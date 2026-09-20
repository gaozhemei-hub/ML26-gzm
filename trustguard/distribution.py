"""Feature-distribution drift and sensitive-group coverage audit."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DistributionAuditConfig:
    psi_warning: float = 0.10
    psi_high: float = 0.25
    ks_warning: float = 0.10
    ks_high: float = 0.20
    categorical_tvd_warning: float = 0.10
    categorical_tvd_high: float = 0.25
    min_group_size: int = 20
    bins: int = 10


@dataclass
class FeatureDriftRecord:
    feature: str
    feature_type: str
    comparison: str
    psi: float | None
    ks_statistic: float | None
    total_variation: float | None
    unseen_rate: float | None
    severity: str
    recommendation: str


@dataclass
class GroupCoverageRecord:
    sensitive_feature: str
    group: str
    split: str
    count: int
    proportion: float
    severity: str


@dataclass
class DistributionAuditResult:
    records: list[FeatureDriftRecord] = field(default_factory=list)
    group_coverage: list[GroupCoverageRecord] = field(default_factory=list)

    @property
    def risk_level(self) -> str:
        order = {"info": 0, "low": 1, "medium": 2, "high": 3}
        levels = [r.severity for r in self.records] + [r.severity for r in self.group_coverage]
        return max(levels, key=lambda level: order[level], default="info")

    def to_dict(self) -> dict:
        return {"risk_level": self.risk_level, "feature_records": [asdict(r) for r in self.records], "group_coverage": [asdict(r) for r in self.group_coverage]}


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    values = np.sort(np.unique(np.concatenate([a, b])))
    if len(values) == 0:
        return 0.0
    return float(np.max(np.abs(np.searchsorted(np.sort(a), values, side="right") / len(a) - np.searchsorted(np.sort(b), values, side="right") / len(b))))


def _psi(reference: np.ndarray, other: np.ndarray, bins: int) -> float:
    if len(reference) == 0 or len(other) == 0 or np.nanmin(reference) == np.nanmax(reference):
        return 0.0
    # Define bins using training data only: test-distribution information must not shape the baseline.
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    # A small epsilon below keeps zero-frequency bins numerically stable.
    p = np.histogram(reference, bins=edges)[0] / len(reference)
    q = np.histogram(other, bins=edges)[0] / len(other)
    eps = 1e-6
    return float(np.sum((p - q) * np.log((p + eps) / (q + eps))))


def _severity(value: float, warning: float, high: float) -> str:
    if value >= high:
        return "high"
    if value >= warning:
        return "medium"
    return "info"


def _numeric_record(feature: str, reference: pd.Series, other: pd.Series, comparison: str, cfg: DistributionAuditConfig) -> FeatureDriftRecord:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    candidate = pd.to_numeric(other, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(ref) or not len(candidate):
        return FeatureDriftRecord(feature, "numeric", comparison, None, None, None, None, "low", "样本为空或无法解析；检查缺失值处理。")
    psi, ks = _psi(ref, candidate, cfg.bins), _ks(ref, candidate)
    severity = max((_severity(psi, cfg.psi_warning, cfg.psi_high), _severity(ks, cfg.ks_warning, cfg.ks_high)), key=("info", "low", "medium", "high").index)
    return FeatureDriftRecord(feature, "numeric", comparison, round(psi, 6), round(ks, 6), None, None, severity, "若风险为中/高，检查切分策略、时间漂移或训练数据覆盖。")


def _categorical_record(feature: str, reference: pd.Series, other: pd.Series, comparison: str, cfg: DistributionAuditConfig) -> FeatureDriftRecord:
    ref = reference.astype("string").fillna("<NA>")
    candidate = other.astype("string").fillna("<NA>")
    categories = sorted(set(ref) | set(candidate))
    p = ref.value_counts(normalize=True)
    q = candidate.value_counts(normalize=True)
    tvd = 0.5 * sum(abs(float(p.get(c, 0.0)) - float(q.get(c, 0.0))) for c in categories)
    unseen = float((~candidate.isin(set(ref))).mean())
    severity = _severity(max(tvd, unseen), cfg.categorical_tvd_warning, cfg.categorical_tvd_high)
    return FeatureDriftRecord(feature, "categorical", comparison, None, None, round(tvd, 6), round(unseen, 6), severity, "若风险为中/高，检查新类别、编码策略及训练数据代表性。")


def audit_distribution(
    X: pd.DataFrame,
    train_indices: Iterable[int],
    valid_indices: Iterable[int],
    test_indices: Iterable[int],
    *,
    sensitive_features: Iterable[str] = (),
    config: DistributionAuditConfig | None = None,
) -> DistributionAuditResult:
    """Compare each validation/test feature distribution to the training set."""
    cfg = config or DistributionAuditConfig()
    train, valid, test = X.iloc[list(train_indices)], X.iloc[list(valid_indices)], X.iloc[list(test_indices)]
    result = DistributionAuditResult()
    for feature in X.columns:
        maker = _numeric_record if pd.api.types.is_numeric_dtype(X[feature]) else _categorical_record
        result.records.append(maker(feature, train[feature], valid[feature], "train_vs_validation", cfg))
        result.records.append(maker(feature, train[feature], test[feature], "train_vs_test", cfg))
    for feature in sensitive_features:
        if feature not in X.columns:
            continue
        # Use the union of groups from the complete input so that a group that
        # is entirely absent from validation/test is recorded with count=0.
        # Iterating only over value_counts() would silently hide the most
        # serious coverage failure.
        all_groups = sorted(
            X[feature].astype("string").fillna("<NA>").unique().tolist()
        )
        for split_name, frame in (("train", train), ("validation", valid), ("test", test)):
            counts = frame[feature].astype("string").fillna("<NA>").value_counts(dropna=False)
            total = max(len(frame), 1)
            for group in all_groups:
                count = int(counts.get(group, 0))
                severity = "high" if count < cfg.min_group_size else "info"
                result.group_coverage.append(GroupCoverageRecord(feature, str(group), split_name, count, round(count / total, 6), severity))
    return result


def write_distribution_artifacts(result: DistributionAuditResult, output_dir: str | Path) -> dict[str, Path]:
    destination = Path(output_dir) / "distribution_audit"
    destination.mkdir(parents=True, exist_ok=True)
    summary = destination / "summary.json"
    summary.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    records = destination / "feature_drift.csv"
    pd.DataFrame([asdict(r) for r in result.records]).to_csv(records, index=False, encoding="utf-8-sig")
    coverage = destination / "group_coverage.csv"
    pd.DataFrame([asdict(r) for r in result.group_coverage]).to_csv(coverage, index=False, encoding="utf-8-sig")
    return {"summary": summary, "feature_drift": records, "group_coverage": coverage}
