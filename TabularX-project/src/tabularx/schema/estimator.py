"""AutoEstimator 结果的数据类型定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tabularx.schema.profile import DataProfile
from tabularx.schema.split import SplitResult
from tabularx.schema.preprocess import PreprocessedData
from tabularx.schema.learning import (
    ModelSearchResult,
    ShapResult,
    FairnessResult,
)
from tabularx.schema.evidence import EvidenceManifest


@dataclass
class AutoResult:
    """AutoEstimator.fit() 的完整结果。

    持有所有中间输出，供用户查看和 save() 使用。
    """

    profile: DataProfile
    split: SplitResult
    preprocessed: PreprocessedData
    searched: ModelSearchResult
    shap: ShapResult | None = None
    fairness: FairnessResult | None = None
    evidence: EvidenceManifest | None = None
    output_dir: Path | None = None
    report_paths: dict[str, str] = field(default_factory=dict)
    """报告文件路径字典，键为报告类型，如 "profile" / "model_search" 等。"""


__all__ = [
    "AutoResult",
]