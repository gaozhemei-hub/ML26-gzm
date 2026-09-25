"""可信审计的公共数据类型定义。

本模块只定义跨阶段共享的审计契约，不包含任何计算逻辑：
- ``AuditFinding`` / ``AuditCheckResult``：单项风险与固定检查项判定。
- ``AuditReport``：审计结果基类，统一风险等级、高危计数与序列化。
- ``ProfileAuditResult``：画像阶段的静态泄漏预检结果。
- ``SplitAuditResult``：划分阶段的完整性审计结果。

各阶段的具体审计算法分别放在 ``core/profile.py`` 与 ``core/split.py``。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal

AuditSeverity = Literal["info", "low", "medium", "high", "critical"]
AuditCheckStatus = Literal["通过", "发现风险", "未执行"]

SEVERITY_ORDER: dict[str, int] = {
    "info": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

HIGH_RISK_SEVERITIES = frozenset({"high", "critical"})


@dataclass
class AuditFinding:
    """单项风险/合规发现项。"""

    code: str
    severity: AuditSeverity
    title: str
    evidence: str
    recommendation: str
    columns: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


@dataclass
class AuditCheckResult:
    """固定审计检查项目的判定记录（含通过、风险、未执行）。"""

    code: str
    check: str
    status: AuditCheckStatus
    result: str


@dataclass
class AuditReport:
    """审计结果的公共基类。

    子类可追加自己的结构化字段；``to_dict()`` 会自动带上子类字段，
    因此新增字段无需重写序列化逻辑。
    """

    findings: list[AuditFinding] = field(default_factory=list)
    check_results: list[AuditCheckResult] = field(default_factory=list)
    model_info_questions: list[str] = field(default_factory=list)

    # ── 派生指标 ────────────────────────────────────────────

    @property
    def risk_level(self) -> AuditSeverity:
        """最高风险等级；无发现项时为 ``info``。"""
        return max(
            (f.severity for f in self.findings),
            key=lambda s: SEVERITY_ORDER.get(s, 0),
            default="info",
        )

    @property
    def high_risk_count(self) -> int:
        """高危与严重级别的发现项数量。"""
        return sum(f.severity in HIGH_RISK_SEVERITIES for f in self.findings)

    @property
    def passed(self) -> bool:
        """是否没有任何发现项。"""
        return not self.findings

    # ── 序列化 ──────────────────────────────────────────────

    def to_dict(self) -> dict:
        """转成可直接写入 JSON 的字典（含子类字段）。"""
        payload: dict = {
            "risk_level": self.risk_level,
            "high_risk_count": self.high_risk_count,
            "findings": [dataclasses.asdict(f) for f in self.findings],
            "check_results": [dataclasses.asdict(c) for c in self.check_results],
            "model_info_questions": self.model_info_questions,
        }
        for f in dataclasses.fields(self):
            if f.name in payload or f.name.startswith("_"):
                continue
            payload[f.name] = getattr(self, f.name)
        return payload


@dataclass
class ProfileAuditResult(AuditReport):
    """画像阶段的静态数据泄漏与合规预检结果。"""

    suggested_drop_columns: list[str] = field(default_factory=list)
    """建议从模型特征中排除的列（仅供预处理阶段参考，本阶段不修改数据）。"""


@dataclass
class SplitAuditResult(AuditReport):
    """划分阶段的完整性审计结果。"""

    suggested_actions: list[str] = field(default_factory=list)
    """针对发现项的建议动作，如「改用 group 切分」「切分前去重」。"""
    duplicate_count: int = 0
    """跨 train/valid/test 重复的原始特征行数量。"""
    group_overlap_count: int = 0
    """跨集合出现的实体数量（仅在指定 group_column 时有意义）。"""
    temporal_ordered: bool | None = None
    """三集时间顺序是否合规；未指定 time_column 时为 None。"""


__all__ = [
    "AuditSeverity",
    "AuditCheckStatus",
    "SEVERITY_ORDER",
    "HIGH_RISK_SEVERITIES",
    "AuditFinding",
    "AuditCheckResult",
    "AuditReport",
    "ProfileAuditResult",
    "SplitAuditResult",
]
