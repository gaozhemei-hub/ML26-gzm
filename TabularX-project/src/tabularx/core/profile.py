"""数据概况模块：观测数据事实，不做任何决策。

角色等价于 MLJAR 的 DataInfo.compute，但输出强类型 dataclass。
"""

from __future__ import annotations

import hashlib
import re
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target

from tabularx.core.params import ProfileConfig
from tabularx.exceptions import (
    EmptyDataError,
    EmptyTargetError,
    InsufficientDataError,
    InvalidDataError,
    MismatchedLengthError,
    TaskTypeError,
)
from tabularx.schema.audit import (
    AuditCheckResult,
    AuditFinding,
    ProfileAuditResult,
)
from tabularx.schema.profile import (
    ColumnFact,
    DataProfile,
    Evidence,
    TargetFact,
)

# 合法的任务类型取值
TASK_TYPES = ("binary", "multiclass", "regression")

# 正则：识别疑似 ID 字段与结果/目标代理字段
_ID_PATTERN = re.compile(
    r"(^|[_-])(id|uuid|guid|record|patient|user|account|device|code)([_-]|$)",
    re.I,
)
_OUTCOME_PATTERN = re.compile(
    r"(target|label|outcome|diagnos|post[_-]|result|ground[_-]?truth)",
    re.I,
)


class ProfileUtils:
    """列类型推断、统计量、scale 判据等工具方法集。"""

    CATEGORICAL = "categorical"
    NUMERIC = "numeric"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"

    # ── 列类型推断 ──────────────────────────────────────────

    @staticmethod
    def infer_column_type(
        col: pd.Series,
        config: ProfileConfig | None = None,
    ) -> str:
        """判断列的类型（MLJAR 式逐列分析）。

        规则（参考 MLJAR PreprocessingUtils.get_type）：
        1. 全空列 → "categorical"（保守标记，build_column_fact 会加 empty_column 标记）
        2. datetime → "datetime"
        3. bool     → "boolean"
        4. object / str / category → 看唯一值数量：
           - >{text_unique_threshold} 且 >{text_unique_ratio} 行唯一 → "text"
           - 否则 → "categorical"
        5. int / uint → "categorical"（如果唯一值 ≤ {int_categorical_threshold}） 否则 → "numeric"
        6. float → "numeric"

        Args:
            col: 列数据。
            config: 可选参数配置，默认使用 ProfileConfig()。
        """
        cfg = config if config is not None else ProfileConfig()
        col_clean = col.dropna()

        # 全空列：无法判断类型，保守标记为 categorical
        if col_clean.empty:
            return ProfileUtils.CATEGORICAL

        if pd.api.types.is_datetime64_any_dtype(col):
            return ProfileUtils.DATETIME

        if pd.api.types.is_bool_dtype(col):
            return ProfileUtils.BOOLEAN

        dtype = col.dtype

        # 类别型 / 文本型
        is_string_like = (
            pd.api.types.is_object_dtype(col)
            or pd.api.types.is_string_dtype(col)
            or isinstance(dtype, pd.CategoricalDtype)
        )
        if is_string_like:
            unique_cnt = col_clean.nunique()
            if (
                unique_cnt > cfg.text_unique_threshold
                and unique_cnt > int(cfg.text_unique_ratio * len(col_clean))
            ):
                return ProfileUtils.TEXT
            return ProfileUtils.CATEGORICAL

        # 整数型
        if pd.api.types.is_integer_dtype(col):
            unique_cnt = col_clean.nunique()
            if unique_cnt <= cfg.int_categorical_threshold:
                return ProfileUtils.CATEGORICAL
            return ProfileUtils.NUMERIC

        # 浮点型
        return ProfileUtils.NUMERIC

    # ── scale 判据 ──────────────────────────────────────────

    @staticmethod
    def is_scale_needed(
        col: pd.Series,
        config: ProfileConfig | None = None,
    ) -> bool:
        """判断数值列是否需要标准化（|mean|>{scale_mean_threshold} or std>{scale_std_threshold}）。

        Args:
            col: 列数据。
            config: 可选参数配置，默认使用 ProfileConfig()。
        """
        cfg = config if config is not None else ProfileConfig()
        col_clean = col.dropna()
        if col_clean.empty:
            return False
        abs_mean = abs(np.mean(col_clean))
        std = np.std(col_clean)
        return bool(abs_mean > cfg.scale_mean_threshold or std > cfg.scale_std_threshold)

    # ── 任务类型推断 ─────────────────────────────────────────

    @staticmethod
    def infer_task_type(y: pd.Series) -> Literal["binary", "multiclass", "regression"]:
        """推断任务类型，显式支持 'binary'/'multiclass'/'regression'。

        用 sklearn 的 type_of_target 做基础判断，对边界情况做兜底。
        """
        y_clean = y.dropna()

        if y_clean.empty:
            raise EmptyTargetError("y 为空，无法推断任务类型")

        # ── 第 1 层：type_of_target 直接能判定的 ──────────────
        raw = type_of_target(y_clean)

        # 明确是回归的（continuous），直接返回
        if raw in ("continuous", "continuous-multioutput"):
            return "regression"

        # 明确是二分类的，直接返回
        if raw == "binary":
            return "binary"

        # ── 第 2 层：type_of_target 判为 multiclass 但实际可能是回归 ──
        if raw == "multiclass" and pd.api.types.is_float_dtype(y_clean):
            unique_cnt = y_clean.nunique()
            if unique_cnt > 20:
                return "regression"
            return "multiclass"

        # ── 第 3 层：type_of_target 判为 multiclass 且是整数 ──
        if raw == "multiclass":
            if pd.api.types.is_integer_dtype(y_clean):
                unique_cnt = y_clean.nunique()
                if unique_cnt <= 20:
                    return "binary" if unique_cnt == 2 else "multiclass"
            return "multiclass"

        # ── 第 4 层：兜底 ────────────────────────────────────
        raise TaskTypeError(
            f"无法推断任务类型，type_of_target 返回: {raw}，"
            f"dtype={y_clean.dtype}，unique_cnt={y_clean.nunique()}。"
            "建议：使用 build_profile(X, y, task_type=...) 手动指定。"
        )

    # ── 统计量 ──────────────────────────────────────────────

    @staticmethod
    def compute_basic_stats(col: pd.Series) -> dict:
        """计算数值列的基本统计量。"""
        col_clean = col.dropna()
        if col_clean.empty:
            return {"mean": None, "std": None, "min": None, "max": None}
        return {
            "mean": float(np.mean(col_clean)),
            "std": float(np.std(col_clean)),
            "min": float(np.min(col_clean)),
            "max": float(np.max(col_clean)),
        }

    @staticmethod
    def compute_skewness(col: pd.Series) -> float | None:
        """计算数值列的偏度（Fisher-Pearson 系数）。"""
        col_clean = col.dropna()
        if len(col_clean) < 3:
            return None
        return float(col_clean.skew())

    # ── 数据指纹 ─────────────────────────────────────────────

    @staticmethod
    def compute_fingerprint(X: pd.DataFrame, y: pd.Series) -> str:
        """对 X+y 计算 sha256 指纹，用于复现校验。"""
        hasher = hashlib.sha256()
        x_hash = pd.util.hash_pandas_object(X).values
        y_hash = pd.util.hash_pandas_object(y).values
        hasher.update(x_hash.tobytes())
        hasher.update(y_hash.tobytes())
        return hasher.hexdigest()[:16]


# ── 顶层函数 ─────────────────────────────────────────────────


def build_column_fact(
    col: pd.Series,
    config: ProfileConfig | None = None,
) -> ColumnFact:
    """构建单列的事实记录。

    Args:
        col: 列数据。
        config: 可选参数配置，默认使用 ProfileConfig()。
    """
    cfg = config if config is not None else ProfileConfig()
    col_clean = col.dropna()
    n_missing = int(col.isna().sum())
    missing_rate = n_missing / len(col) if len(col) > 0 else 0.0
    n_unique = int(col_clean.nunique())
    evidences: list[Evidence] = []

    # 全空列：保守标记 categorical，并记录证据
    if col_clean.empty:
        evidences.append(
            Evidence("n_non_missing", 0, "非缺失值数量为 0，判定为全空列")
        )
        return ColumnFact(
            col_name=col.name if col.name is not None else "",
            dtype=str(col.dtype),
            n_missing=n_missing,
            missing_rate=missing_rate,
            n_unique=0,
            tags=["categorical", "empty_column"],
            evidences=evidences,
        )

    # 常数列：所有值相同，对下游无信息量，preprocess 阶段应丢弃
    if n_unique == 1:
        evidences.append(Evidence("n_unique", 1, "仅 1 个唯一值，判定为常数列"))
        return ColumnFact(
            col_name=col.name if col.name is not None else "",
            dtype=str(col.dtype),
            n_missing=n_missing,
            missing_rate=missing_rate,
            n_unique=n_unique,
            tags=["constant_column"],
            evidences=evidences,
        )

    # 正常列：推断列类型
    col_type = ProfileUtils.infer_column_type(col, config=cfg)
    evidences.append(
        Evidence("col_type", col_type, f"列类型判定为 {col_type}")
    )

    # 确定标签 tags
    tags = [col_type]

    # ID/高唯一性标识初步标记
    ratio = n_unique / max(1, len(col_clean))
    is_id_name = bool(_ID_PATTERN.search(str(col.name)))
    if is_id_name or ratio >= cfg.identifier_unique_ratio:
        tags.append("identifier_like")
        evidences.append(
            Evidence(
                "identifier_like",
                f"{ratio:.2%}",
                f"唯一值占比为 {ratio:.2%}" + ("，列名命中 ID 模式" if is_id_name else "，唯一率较高"),
            )
        )

    # 高基数类别检测
    if col_type == ProfileUtils.CATEGORICAL and n_unique > cfg.high_cardinality_threshold:
        tags.append("high_cardinality")
        evidences.append(
            Evidence(
                "n_unique",
                n_unique,
                f"唯一值数量为 {n_unique}，基数较高，建议使用目标编码或频率编码",
            )
        )

    if col_type == ProfileUtils.NUMERIC:
        scale_needed = ProfileUtils.is_scale_needed(col, config=cfg)
        rule_str = f"|mean|>{cfg.scale_mean_threshold} or std>{cfg.scale_std_threshold}"
        if scale_needed:
            tags.append("scale_needed")
            evidences.append(
                Evidence(
                    "scale_rule",
                    rule_str,
                    f"|mean|={abs(float(np.mean(col_clean))):.4f}, "
                    f"std={float(np.std(col_clean)):.4f}，满足判据 → scale_needed",
                )
            )
        else:
            evidences.append(
                Evidence(
                    "scale_rule",
                    rule_str,
                    f"|mean|={abs(float(np.mean(col_clean))):.4f}, "
                    f"std={float(np.std(col_clean)):.4f}，不满足判据 → 无需 scale",
                )
            )
        # 偏度检测
        skewness = ProfileUtils.compute_skewness(col)
        if skewness is not None and abs(skewness) > cfg.skew_threshold:
            tags.append("skewed")
            evidences.append(
                Evidence(
                    "skewness",
                    round(skewness, 4),
                    f"偏度 = {skewness:.4f}，|skew| > {cfg.skew_threshold}，"
                    f"分布偏斜明显，建议评估 log 变换或缩尾处理",
                )
            )
    if n_missing > 0:
        tags.append("has_missing")
        evidences.append(
            Evidence(
                "n_missing",
                n_missing,
                f"存在 {n_missing} 个缺失值（{missing_rate:.2%}）",
            )
        )

    # 数值列额外统计
    stats = ProfileUtils.compute_basic_stats(col) if col_type == ProfileUtils.NUMERIC else {}
    if col_type == ProfileUtils.NUMERIC:
        stats["skewness"] = ProfileUtils.compute_skewness(col)

    return ColumnFact(
        col_name=col.name if col.name is not None else "",
        dtype=str(col.dtype),
        n_missing=n_missing,
        missing_rate=missing_rate,
        n_unique=n_unique,
        tags=tags,
        evidences=evidences,
        **stats,
    )


def build_target_fact(
    y: pd.Series,
    task_type: str = "auto",
    config: ProfileConfig | None = None,
) -> TargetFact:
    """构建目标变量的事实记录。

    Args:
        y: 目标 Series。
        task_type: "auto"（自动推断）| "classification" | "regression"。
        config: 可选参数配置，默认使用 ProfileConfig()。
    """
    cfg = config if config is not None else ProfileConfig()

    if task_type not in ("auto", "classification", "regression"):
        raise TaskTypeError(
            f"task_type 必须是 'auto'/'classification'/'regression'，收到 {task_type!r}。"
            "建议：使用 'auto'（自动推断）或 'classification'/'regression'（手动指定）。"
        )

    y_clean = y.dropna()
    missing_rate = (len(y) - len(y_clean)) / len(y) if len(y) > 0 else 0.0
    n_unique = int(y_clean.nunique())
    evidences: list[Evidence] = []

    # 确定任务类型
    if task_type == "regression":
        inferred = "regression"
        evidences.append(Evidence("task_type_source", "user", "用户手动指定为回归任务"))
    elif task_type == "classification":
        inferred = "binary" if n_unique == 2 else "multiclass"
        evidences.append(
            Evidence(
                "task_type_source",
                "user+auto",
                f"用户手动指定分类任务；按唯一值数 {n_unique} 细分为 {inferred}",
            )
        )
    else:  # auto
        inferred = ProfileUtils.infer_task_type(y_clean)
        evidences.append(
            Evidence(
                "task_type_source",
                "auto",
                f"自动推断：唯一值 {n_unique} 个，dtype={y_clean.dtype}，"
                f"type_of_target 判定为 {inferred}",
            )
        )

    if inferred in ("binary", "multiclass"):
        value_counts = y_clean.value_counts(normalize=True)
        class_balance = {str(k): float(v) for k, v in value_counts.items()}

        # 类别不平衡检测
        min_ratio = min(class_balance.values())
        if min_ratio < cfg.imbalance_ratio:
            target_tags = ["imbalanced"]
            evidences.append(
                Evidence(
                    "imbalanced",
                    f"{min_ratio:.2%}",
                    f"最小类别占比 {min_ratio:.2%} < {cfg.imbalance_ratio}，"
                    f"类别不平衡。建议使用 F1/ROC-AUC 评估，并考虑 class_weight 参数。",
                )
            )
        else:
            target_tags = []

        return TargetFact(
            task_type=inferred,  # type: ignore[arg-type]
            n_classes=n_unique,
            class_balance=class_balance,
            mean=None,
            std=None,
            missing_rate=missing_rate,
            tags=target_tags,
            evidences=evidences,
        )
    else:
        # 回归
        return TargetFact(
            task_type="regression",
            n_classes=None,
            class_balance=None,
            mean=float(np.mean(y_clean)) if not y_clean.empty else None,
            std=float(np.std(y_clean)) if not y_clean.empty else None,
            missing_rate=missing_rate,
            tags=[],
            evidences=evidences,
        )


def _compute_univariate_score(
    series: pd.Series,
    y: pd.Series,
    task_type: str,
) -> float | None:
    """计算单个数值特征对目标的预测力，用于检测强相关目标代理特征。"""
    if not pd.api.types.is_numeric_dtype(series):
        return None
    mask = series.notna() & y.notna()
    if mask.sum() < 10:
        return None
    try:
        if task_type == "binary" and y[mask].nunique() == 2:
            labels = pd.factorize(y[mask])[0]
            score = series[mask].astype(float)
            ranks = score.rank(method="average").to_numpy()
            n_pos = int((labels == 1).sum())
            n_neg = int((labels == 0).sum())
            if n_pos == 0 or n_neg == 0:
                return None
            auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
            return float(max(auc, 1 - auc))
        if task_type == "regression":
            actual = y[mask].astype(float)
            predicted = series[mask].astype(float)
            denom = float(((actual - actual.mean()) ** 2).sum())
            if denom == 0.0:
                return None
            r2 = 1.0 - float(((actual - predicted) ** 2).sum()) / denom
            return max(0.0, float(r2))
    except Exception:
        pass
    return None


def _set_check(
    result: ProfileAuditResult,
    code: str,
    check: str,
    status: Literal["通过", "发现风险", "未执行"],
    message: str,
) -> None:
    item = AuditCheckResult(code=code, check=check, status=status, result=message)
    for index, existing in enumerate(result.check_results):
        if existing.code == code:
            result.check_results[index] = item
            return
    result.check_results.append(item)


def _set_check_from_findings(
    result: ProfileAuditResult,
    code: str,
    check: str,
    pass_message: str,
) -> None:
    matched = [item for item in result.findings if item.code == code]
    if matched:
        columns = list(dict.fromkeys(col for item in matched for col in item.columns))
        detail = "、".join(map(str, columns)) if columns else f"{len(matched)} 项"
        _set_check(result, code, check, "发现风险", f"发现 {len(matched)} 项：{detail}。")
    else:
        _set_check(result, code, check, "通过", pass_message)


def audit_profile_preflight(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    config: ProfileConfig | None = None,
) -> ProfileAuditResult:
    """在划分前进行静态数据风险与泄漏预检（纯观测，零副作用）。"""
    cfg = config if config is not None else ProfileConfig()
    result = ProfileAuditResult()

    if not cfg.enable_leakage_audit:
        return result

    suggested_drops: list[str] = []

    for col in X.columns:
        series = X[col]
        clean_s = series.dropna()
        ratio = clean_s.nunique() / max(1, len(clean_s))
        id_name = bool(_ID_PATTERN.search(str(col)))
        is_user_id = col in cfg.id_columns

        # 1. 标识符特征 / 高唯一性特征
        if is_user_id or id_name or ratio >= cfg.identifier_unique_ratio:
            sev: Literal["high", "medium"] = (
                "high" if (is_user_id or (id_name and ratio >= cfg.identifier_unique_ratio)) else "medium"
            )
            result.findings.append(
                AuditFinding(
                    code="identifier_like_column",
                    severity=sev,
                    title="疑似标识符特征",
                    evidence=f"{col!r} 唯一值比例为 {ratio:.1%}"
                    + ("，列名命中标识符模式" if id_name else "。"),
                    recommendation="确认预测时是否可得；通常应作为分组列或从模型特征中排除。",
                    columns=[str(col)],
                    details={"unique_ratio": ratio},
                )
            )
            if sev == "high" or is_user_id:
                suggested_drops.append(str(col))

        # 2. 目标直接复制
        same = series.astype("string").fillna("<NA>").equals(y.astype("string").fillna("<NA>"))
        if same:
            result.findings.append(
                AuditFinding(
                    code="target_copy",
                    severity="critical",
                    title="特征与目标完全相同",
                    evidence=f"列 {col!r} 与目标逐行完全一致。",
                    recommendation="立即从特征集中排除该列，并追溯数据导出流程。",
                    columns=[str(col)],
                )
            )
            suggested_drops.append(str(col))
        else:
            # 3. 疑似目标代理特征 (Outcome Proxy / Leakage)
            score = _compute_univariate_score(series, y, task_type)
            name_hit = bool(_OUTCOME_PATTERN.search(str(col)))
            high_score = score is not None and score >= cfg.target_proxy_threshold
            if name_hit or high_score:
                ev_text = (
                    f"列名 {col!r} 命中后验/结果字段规则。"
                    if name_hit
                    else f"单特征判别力分数为 {score:.3f}（阈值 {cfg.target_proxy_threshold}）。"
                )
                result.findings.append(
                    AuditFinding(
                        code="possible_target_proxy",
                        severity="high",
                        title="疑似目标代理特征",
                        evidence=ev_text,
                        recommendation="检查该特征是否在真实预测时已产生，避免时序事后泄漏。",
                        columns=[str(col)],
                        details={"score": score},
                    )
                )

    # 4. 用户显式指定的不可用特征
    for col in cfg.forbidden_features:
        if col in X.columns:
            result.findings.append(
                AuditFinding(
                    code="forbidden_feature",
                    severity="critical",
                    title="声明的非法/禁用特征",
                    evidence=f"{col!r} 已被用户声明为预测时不可获得。",
                    recommendation="从建模特征集中排除。",
                    columns=[str(col)],
                )
            )
            suggested_drops.append(str(col))

    result.suggested_drop_columns = list(dict.fromkeys(suggested_drops))
    result.model_info_questions = [
        "每个保留特征是否在真实预测时已经可获得，而不是结果发生后才产生？",
        "测试集是否代表要泛化到的人群、地点与时间段？",
        "是否存在实体、时间、空间或层级依赖，需要相应的分组切分？",
    ]

    _set_check_from_findings(result, "target_copy", "目标复制检测", "未发现特征与目标逐行一致。")
    _set_check_from_findings(result, "possible_target_proxy", "目标代理特征检测", "未发现超过阈值的疑似目标代理特征。")
    _set_check_from_findings(result, "identifier_like_column", "ID/高唯一性字段检测", "未发现疑似 ID 或超高唯一性字段。")
    _set_check_from_findings(result, "forbidden_feature", "用户禁用特征校验", "未发现包含声明的非法特征。")

    return result


def build_profile(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str = "auto",
    config: ProfileConfig | None = None,
) -> DataProfile:
    """纯观测，零决策：构建数据集的完整事实记录。

    Args:
        X: 特征 DataFrame，必须有列名。
        y: 目标 Series。
        task_type: "auto"（默认，自动推断）| "classification" | "regression"。
        config: 可选参数配置，默认使用 ProfileConfig()。

    Returns:
        DataProfile：包含所有列和目标的观测事实。

    Raises:
        InvalidDataError: X 不是 DataFrame 或 y 不是 Series。
        EmptyDataError: X 或 y 为空。
        MismatchedLengthError: X 与 y 行数不一致。
        InsufficientDataError: 样本数不足以进行后续划分。
    """
    cfg = config if config is not None else ProfileConfig()

    if not isinstance(X, pd.DataFrame):
        raise InvalidDataError(
            f"X 必须是 pd.DataFrame，收到 {type(X)}。"
            "建议：使用 pd.DataFrame(X, columns=[...]) 转换。"
        )
    if not isinstance(y, pd.Series):
        raise InvalidDataError(
            f"y 必须是 pd.Series，收到 {type(y)}。建议：使用 pd.Series(y) 转换。"
        )

    # 数据非空检查
    if len(X) == 0 or len(X.columns) == 0:
        raise EmptyDataError(
            f"X 为空（shape={X.shape}）。请提供至少 1 行 1 列的特征数据。"
        )
    if len(y) == 0:
        raise EmptyDataError("y 为空。请提供至少 1 个样本的目标值。")

    # X 与 y 长度对齐检查
    if len(X) != len(y):
        raise MismatchedLengthError(
            f"X 行数（{len(X)}）与 y 长度（{len(y)}）不一致。"
            "请确保 X 与 y 对应同一批样本。"
        )

    columns = [build_column_fact(X[col], config=cfg) for col in X.columns]
    target = build_target_fact(y, task_type=task_type, config=cfg)
    fingerprint = ProfileUtils.compute_fingerprint(X, y)

    # ── 数据量下限检查 ──────────────────────────────────────
    if target.task_type == "regression":
        if len(X) < cfg.min_samples_regression:
            raise InsufficientDataError(
                f"回归任务至少需要 {cfg.min_samples_regression} 个样本，"
                f"当前仅 {len(X)} 个。可通过增加数据量或调小 "
                f"ProfileConfig.min_samples_regression 解决。"
            )
    else:
        # 分类任务：计算每类最少样本数
        min_class_size = min(
            (y == cls).sum() for cls in y.unique()
        )
        if min_class_size < cfg.min_samples_per_class:
            raise InsufficientDataError(
                f"分类任务每类至少需要 {cfg.min_samples_per_class} 个样本，"
                f"当前最小类别仅 {min_class_size} 个。可通过增加数据量或调小 "
                f"ProfileConfig.min_samples_per_class 解决。"
            )

    # ── 静态泄漏与合规预检（纯观测） ──────────────────────
    audit = audit_profile_preflight(X, y, target.task_type, config=cfg)

    return DataProfile(
        n_rows=len(X),
        n_cols=len(X.columns),
        columns=columns,
        target=target,
        fingerprint=fingerprint,
        audit=audit,
    )