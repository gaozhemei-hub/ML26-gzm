"""Evidence-based data leakage checks, independent of TabularX internals."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Literal

import pandas as pd

from .config import LeakageAuditConfig

Severity = Literal["info", "low", "medium", "high", "critical"]
_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
_ID = re.compile(r"(^|[_-])(id|uuid|guid|record|patient|user|account|device|code)([_-]|$)", re.I)
_OUTCOME = re.compile(r"(target|label|outcome|diagnos|post[_-]|result|ground[_-]?truth)", re.I)


@dataclass
class AuditFinding:
    code: str
    severity: Severity
    title: str
    evidence: str
    recommendation: str
    columns: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


@dataclass
class AuditCheckResult:
    """One fixed audit check, including successful and skipped checks."""

    code: str
    check: str
    status: Literal["通过", "发现风险", "未执行"]
    result: str


@dataclass
class AuditResult:
    findings: list[AuditFinding] = field(default_factory=list)
    excluded_columns: list[str] = field(default_factory=list)
    checked_stages: list[str] = field(default_factory=list)
    model_info_questions: list[str] = field(default_factory=list)
    check_results: list[AuditCheckResult] = field(default_factory=list)

    @property
    def risk_level(self) -> Severity:
        return max((f.severity for f in self.findings), key=lambda s: _ORDER[s], default="info")

    @property
    def high_risk_count(self) -> int:
        return sum(f.severity in {"high", "critical"} for f in self.findings)

    def to_dict(self) -> dict:
        return {"risk_level": self.risk_level, "high_risk_count": self.high_risk_count,
                "excluded_columns": self.excluded_columns, "checked_stages": self.checked_stages,
                "findings": [asdict(f) for f in self.findings],
                "check_results": [asdict(item) for item in self.check_results],
                "model_info_questions": self.model_info_questions}


class LeakageAuditor:
    """Run before-split and after-split checks without mutating source data."""

    def __init__(self, config: LeakageAuditConfig | None = None):
        self.config = config or LeakageAuditConfig()

    def preflight(self, X: pd.DataFrame, y: pd.Series, task_type: str) -> AuditResult:
        result = AuditResult(checked_stages=["raw_data"])
        for col in X.columns:
            series = X[col]
            ratio = series.nunique(dropna=True) / max(1, len(series))
            id_name = bool(_ID.search(str(col)))
            if id_name or ratio >= self.config.identifier_unique_ratio:
                result.findings.append(AuditFinding(
                    "identifier_like_column", "high" if id_name and ratio >= self.config.identifier_unique_ratio else "medium",
                    "疑似标识符特征", f"{col!r} 唯一值比例为 {ratio:.1%}" + ("，列名疑似 ID。" if id_name else "。"),
                    "确认预测时是否可得；通常应作为分组列或从模型输入排除。", [col], {"unique_ratio": ratio}))
                if self.config.mode == "drop_high_confidence" and (id_name or col in self.config.id_columns):
                    result.excluded_columns.append(col)

            same = series.astype("string").fillna("<NA>").equals(y.astype("string").fillna("<NA>"))
            score = self._univariate_score(series, y, task_type)
            if same:
                result.findings.append(AuditFinding("target_copy", "critical", "特征与目标完全相同", f"列 {col!r} 与目标逐行一致。", "立即删除该列并追溯数据导出流程。", [col]))
                result.excluded_columns.append(col)
            elif _OUTCOME.search(str(col)) or (score is not None and score >= self.config.target_proxy_threshold):
                evidence = f"列名 {col!r} 命中后验/结果字段规则。" if _OUTCOME.search(str(col)) else f"单特征分数为 {score:.3f}。"
                result.findings.append(AuditFinding("possible_target_proxy", "high", "疑似目标代理特征", evidence, "在模型信息表中说明其在预测时为何可得。", [col], {"score": score}))

        for col in self.config.forbidden_features:
            if col in X.columns:
                result.findings.append(AuditFinding("forbidden_feature", "critical", "用户声明的非法特征", f"{col!r} 已声明为预测时不可得。", "从模型特征中删除。", [col]))
                result.excluded_columns.append(col)
        result.excluded_columns = list(dict.fromkeys(result.excluded_columns))
        result.model_info_questions = [
            "每个保留特征是否在真实预测时已经可获得，而不是结果发生后才产生？",
            "测试集是否代表要泛化到的人群、地点与时间段？",
            "是否存在实体、时间、空间或层级依赖，需要相应的分组切分？",
        ]
        self._set_check_from_findings(result, "target_copy", "目标复制", "未发现特征与目标逐行相同。")
        self._set_check_from_findings(result, "possible_target_proxy", "目标代理", "未发现超过阈值的疑似目标代理特征。")
        self._set_check_from_findings(result, "identifier_like_column", "ID/高唯一性字段", "未发现疑似 ID 或高唯一性字段。")
        self._set_check(result, "cross_split_duplicates", "跨集合重复", "未执行", "尚未执行数据划分完整性检查。")
        self._set_check(
            result, "cross_split_groups", "实体分组泄露", "未执行",
            "未指定 group_column，无法判断同一实体是否跨集合出现。" if not self.config.group_column
            else "尚未执行数据划分完整性检查。",
        )
        self._set_check(
            result, "temporal_overlap", "时间穿越", "未执行",
            "未指定 time_column，无法检查训练、验证与测试数据的时间顺序。" if not self.config.time_column
            else "尚未执行数据划分完整性检查。",
        )
        self._raise_if_strict(result)
        return result

    def check_split(self, X: pd.DataFrame, train_idx, valid_idx, test_idx, result: AuditResult | None = None) -> AuditResult:
        result = result or AuditResult()
        result.checked_stages.append("split_integrity")
        parts = [X.iloc[idx] for idx in (train_idx, valid_idx, test_idx)]
        hashes = [set(pd.util.hash_pandas_object(p.astype("string").fillna("<NA>"), index=False).tolist()) for p in parts]
        duplicate_count = sum(len(hashes[i] & hashes[j]) for i in range(3) for j in range(i + 1, 3))
        if duplicate_count:
            result.findings.append(AuditFinding("cross_split_duplicates", "critical", "跨集合重复样本", f"发现 {duplicate_count} 个原始特征行跨 train/valid/test 重复。", "在切分前去重，或让重复实体只进入一个集合。", details={"count": duplicate_count}))
            self._set_check(result, "cross_split_duplicates", "跨集合重复", "发现风险", f"发现 {duplicate_count} 个原始特征行跨训练集、验证集或测试集重复。")
        else:
            self._set_check(result, "cross_split_duplicates", "跨集合重复", "通过", "重复数为 0。")
        self._check_group(X, (train_idx, valid_idx, test_idx), result)
        self._check_time(X, (train_idx, valid_idx, test_idx), result)
        self._raise_if_strict(result)
        return result

    def _check_group(self, X, indices, result):
        col = self.config.group_column
        if not col:
            self._set_check(result, "cross_split_groups", "实体分组泄露", "未执行", "未指定 group_column，无法判断同一实体是否跨集合出现。")
            return
        if col not in X:
            result.findings.append(AuditFinding("missing_group_column", "high", "分组列不存在", f"{col!r} 不在数据中。", "检查 group_column 配置。", [col]))
            self._set_check(result, "cross_split_groups", "实体分组泄露", "发现风险", f"配置的 group_column={col!r} 不在数据中。")
            return
        groups = [set(X.iloc[idx][col].dropna().astype("string")) for idx in indices]
        overlap = sum(len(groups[i] & groups[j]) for i in range(3) for j in range(i + 1, 3))
        if overlap:
            result.findings.append(AuditFinding("cross_split_groups", "critical", "同一实体跨集合出现", f"{col!r} 有 {overlap} 个实体跨集合。", "使用 GroupShuffleSplit/GroupKFold，并从模型输入排除该列。", [col], {"count": overlap}))
            self._set_check(result, "cross_split_groups", "实体分组泄露", "发现风险", f"字段 {col!r} 有 {overlap} 个实体跨集合出现。")
        else:
            self._set_check(result, "cross_split_groups", "实体分组泄露", "通过", f"字段 {col!r} 未发现实体跨集合重叠。")

    def _check_time(self, X, indices, result):
        col = self.config.time_column
        if not col:
            self._set_check(result, "temporal_overlap", "时间穿越", "未执行", "未指定 time_column，无法检查训练、验证与测试数据的时间顺序。")
            return
        if col not in X:
            result.findings.append(AuditFinding("missing_time_column", "high", "时间列不存在", f"{col!r} 不在数据中。", "检查 time_column 配置。", [col]))
            self._set_check(result, "temporal_overlap", "时间穿越", "发现风险", f"配置的 time_column={col!r} 不在数据中。")
            return
        time = pd.to_datetime(X[col], errors="coerce")
        ranges = [(time.iloc[idx].min(), time.iloc[idx].max()) for idx in indices]
        if any(pd.isna(lo) or pd.isna(hi) for lo, hi in ranges):
            result.findings.append(AuditFinding("invalid_time_values", "medium", "时间列不可解析", f"{col!r} 有空值或不可解析时间。", "清洗时间后再进行 temporal 切分。", [col]))
            self._set_check(result, "temporal_overlap", "时间穿越", "发现风险", f"字段 {col!r} 存在空值或不可解析时间。")
            return
        # Every earlier split must finish before the next split starts.
        # Comparing valid_min with test_min is insufficient because the
        # validation range itself may extend into the test period.
        if not (ranges[0][1] <= ranges[1][0] and ranges[1][1] <= ranges[2][0]):
            result.findings.append(AuditFinding("temporal_overlap", "critical", "时间顺序泄露", f"train/valid/test 范围为 {ranges}。", "确保训练数据严格早于验证和测试数据。", [col]))
            self._set_check(result, "temporal_overlap", "时间穿越", "发现风险", f"字段 {col!r} 的训练、验证和测试时间范围发生重叠。")
        else:
            self._set_check(result, "temporal_overlap", "时间穿越", "通过", f"字段 {col!r} 的时间顺序符合训练集 → 验证集 → 测试集。")

    @staticmethod
    def _set_check(result: AuditResult, code: str, check: str, status: str, message: str) -> None:
        item = AuditCheckResult(code=code, check=check, status=status, result=message)
        for index, existing in enumerate(result.check_results):
            if existing.code == code:
                result.check_results[index] = item
                return
        result.check_results.append(item)

    def _set_check_from_findings(self, result: AuditResult, code: str, check: str, pass_message: str) -> None:
        matched = [item for item in result.findings if item.code == code]
        if matched:
            columns = list(dict.fromkeys(col for item in matched for col in item.columns))
            detail = "、".join(map(str, columns)) if columns else f"{len(matched)} 项"
            self._set_check(result, code, check, "发现风险", f"发现 {len(matched)} 项：{detail}。")
        else:
            self._set_check(result, code, check, "通过", pass_message)

    @staticmethod
    def _univariate_score(series, y, task_type):
        if not pd.api.types.is_numeric_dtype(series): return None
        mask = series.notna() & y.notna()
        if mask.sum() < 10: return None
        try:
            if task_type == "binary" and y[mask].nunique() == 2:
                # Mann-Whitney AUC, implemented locally to keep the extension lightweight.
                labels = pd.factorize(y[mask])[0]
                score = series[mask].astype(float)
                ranks = score.rank(method="average").to_numpy()
                n_pos = int((labels == 1).sum()); n_neg = int((labels == 0).sum())
                auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
                return float(max(auc, 1 - auc))
            if task_type == "regression":
                actual = y[mask].astype(float); predicted = series[mask].astype(float)
                denom = float(((actual - actual.mean()) ** 2).sum())
                return max(0.0, 1 - float(((actual - predicted) ** 2).sum()) / denom) if denom else None
        except ValueError: pass
        return None

    def _raise_if_strict(self, result):
        if self.config.mode == "fail" and result.high_risk_count:
            raise ValueError("LeakageAudit 严格模式阻止训练：" + "; ".join(f.title for f in result.findings if f.severity in {"high", "critical"}))
