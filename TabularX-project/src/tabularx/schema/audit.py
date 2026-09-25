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
from typing import Any, Literal

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
        """转成可直接写入 JSON 的字典（含子类字段）。

        子类若追加了 dataclass 列表（如漂移记录、群体覆盖记录），会被递归
        转成普通字典；否则 ``json.dumps`` 只能把它们序列化成无用的字符串。
        """
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
            payload[f.name] = _serialize_value(getattr(self, f.name))
        return payload


def _serialize_value(value: Any) -> Any:
    """把字段值转成 JSON 可序列化的形式（递归处理 dataclass 与容器）。"""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    return value


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


@dataclass
class FeatureDriftRecord:
    """单个特征在某个方向上的分布漂移记录。"""

    feature: str
    """特征名（原始列名，不是预处理后的特征名）。"""
    feature_type: str
    """特征类型：numeric / categorical。"""
    comparison: str
    """对比方向：train_vs_validation / train_vs_test。"""
    psi: float | None = None
    """群体稳定性指标（仅数值特征）。"""
    ks_statistic: float | None = None
    """Kolmogorov-Smirnov 统计量（仅数值特征）。"""
    total_variation: float | None = None
    """总变差距离（仅类别特征）。"""
    unseen_rate: float | None = None
    """验证/测试集中未在训练集出现过的类别占比（仅类别特征）。"""
    severity: AuditSeverity = "info"
    recommendation: str = ""


@dataclass
class GroupCoverageRecord:
    """敏感群体在某个子集中的覆盖情况。"""

    sensitive_feature: str
    """敏感特征列名。"""
    group: str
    """该特征下的一个取值（群体）。"""
    split: str
    """所属子集：train / validation / test。"""
    count: int
    """该群体在此子集中的样本数（可能为 0）。"""
    proportion: float
    """该群体在此子集中的占比。"""
    severity: AuditSeverity = "info"


@dataclass
class PreprocessAuditResult(AuditReport):
    """预处理阶段的特征分布与群体覆盖审计结果。"""

    drift_records: list[FeatureDriftRecord] = field(default_factory=list)
    """逐特征、逐对比方向的漂移记录。"""
    group_coverage: list[GroupCoverageRecord] = field(default_factory=list)
    """敏感群体在各子集中的覆盖记录（未提供敏感特征时为空）。"""
    excluded_by_audit: list[str] = field(default_factory=list)
    """因审计建议或分组声明而被排除出模型输入的列。"""
    skipped_columns: list[str] = field(default_factory=list)
    """被跳过漂移审计的列（已被排除出模型输入，其分布差异不构成建模风险）。"""
    max_psi: float | None = None
    """所有数值特征中的最大 PSI（便于快速判断漂移程度）。"""
    audited_features: int = 0
    """实际参与漂移审计的特征数。"""
    drift_truncated: bool = False
    """是否因特征数超过上限而只审计了部分特征。"""

    @property
    def high_drift_features(self) -> list[str]:
        """被判定为高风险漂移的特征名（去重）。"""
        return list(dict.fromkeys(
            r.feature for r in self.drift_records if r.severity in HIGH_RISK_SEVERITIES
        ))

    @property
    def uncovered_groups(self) -> list[str]:
        """样本数为 0 的群体标识（形如 ``特征=群体@子集``）。"""
        return [
            f"{r.sensitive_feature}={r.group}@{r.split}"
            for r in self.group_coverage
            if r.count == 0
        ]

    # ── 覆盖基类指标：漂移与覆盖记录也参与风险判定 ──────────

    @property
    def risk_level(self) -> AuditSeverity:
        """最高风险等级，同时考虑 findings、漂移记录与群体覆盖记录。

        本阶段的结构化记录（漂移、覆盖）本身就带 severity，若只看 findings，
        调用方漏建 finding 时 risk_level 会错误地显示为 info。
        """
        candidates = (
            [f.severity for f in self.findings]
            + [r.severity for r in self.drift_records]
            + [r.severity for r in self.group_coverage]
        )
        return max(
            candidates,
            key=lambda s: SEVERITY_ORDER.get(s, 0),
            default="info",
        )

    @property
    def high_risk_count(self) -> int:
        """高危与严重级别的记录数（含漂移与覆盖记录）。"""
        return (
            sum(f.severity in HIGH_RISK_SEVERITIES for f in self.findings)
            + sum(r.severity in HIGH_RISK_SEVERITIES for r in self.drift_records)
            + sum(r.severity in HIGH_RISK_SEVERITIES for r in self.group_coverage)
        )

    @property
    def passed(self) -> bool:
        """是否没有任何风险记录。"""
        return self.risk_level == "info"


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
    "FeatureDriftRecord",
    "GroupCoverageRecord",
    "PreprocessAuditResult",
]
