"""数据预处理模块：根据 Profile 的观测事实做数据变换。

设计哲学：
- 所有变换**在训练集上拟合**，在训练/验证/测试集上应用，避免数据泄露。
- 输出包含变换后的 DataFrames、变换记录及训练集拟合状态，支持原始输入重载推断。
- 每列的变换动作由 profile.columns[i].tags 驱动。
- 阶段 3 新增：执行画像/划分的审计建议（排除高危列与分组列），并在切分后
  对原始特征做分布漂移与敏感群体覆盖审计（纯观测，不修改数据）。
"""

from __future__ import annotations

import warnings
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer,
)

from tabularx.core.params import PreprocessConfig
from tabularx.core.inference import FittedPreprocessor
from tabularx.exceptions.preprocess import (
    PreprocessError,
    TextColumnNotSupportedError,
    TransformError,
)
from tabularx.schema.audit import (
    HIGH_RISK_SEVERITIES,
    AuditCheckResult,
    AuditFinding,
    FeatureDriftRecord,
    GroupCoverageRecord,
    PreprocessAuditResult,
)
from tabularx.schema.preprocess import PreprocessedData, TransformRecord, ColumnMapping
from tabularx.schema.profile import DataProfile
from tabularx.schema.split import SplitResult


# ── 内置小工具 ─────────────────────────────────────────────────


def _make_imputer(method: str, constant_value: float) -> SimpleImputer:
    """根据配置创建数值列/类别列的填补器。"""
    if method == "median":
        return SimpleImputer(strategy="median")
    elif method == "mean":
        return SimpleImputer(strategy="mean")
    elif method == "most_frequent":
        return SimpleImputer(strategy="most_frequent")
    elif method == "constant":
        return SimpleImputer(strategy="constant", fill_value=constant_value)
    else:
        raise TransformError(f"不支持的填补方法 {method!r}")


def _make_scaler(method: str) -> StandardScaler | MinMaxScaler | RobustScaler:
    """根据配置创建标准化器。"""
    if method == "standard":
        return StandardScaler()
    elif method == "minmax":
        return MinMaxScaler()
    elif method == "robust":
        return RobustScaler()
    else:
        raise TransformError(f"不支持的标准化方法 {method!r}")


def _resolve_scale_needed(profile: DataProfile, col_name: str) -> bool:
    """判断某列是否需要标准化（查 profile tag）。"""
    for col in profile.columns:
        if col.col_name == col_name:
            return "scale_needed" in col.tags
    return False


def _resolve_skewed(profile: DataProfile, col_name: str) -> bool:
    """判断某列是否偏斜（查 profile tag）。"""
    for col in profile.columns:
        if col.col_name == col_name:
            return "skewed" in col.tags
    return False


def _resolve_col_type(profile: DataProfile, col_name: str) -> str | None:
    """获取列的基础类型（numeric / categorical / text / datetime / boolean）。"""
    for col in profile.columns:
        if col.col_name == col_name:
            for t in col.tags:
                if t in ("numeric", "categorical", "text", "datetime", "boolean"):
                    return t
    return None


def _resolve_n_unique(profile: DataProfile, col_name: str) -> int:
    """获取列的原始唯一值数。"""
    for col in profile.columns:
        if col.col_name == col_name:
            return col.n_unique
    return 0


def _has_tag(profile: DataProfile, col_name: str, tag: str) -> bool:
    """判断列是否包含某 tag。"""
    for col in profile.columns:
        if col.col_name == col_name and tag in col.tags:
            return True
    return False


# ── 列级变换函数 ──────────────────────────────────────────────


def _transform_numeric(
    col_train: pd.Series,
    col_valid: pd.Series,
    col_test: pd.Series,
    col_name: str,
    profile: DataProfile,
    config: PreprocessConfig,
    state: dict | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series, list[str], str | None]:
    """处理数值列：填补 → （可选）标准化或偏度校正。

    Args:
        col_train/col_valid/col_test: 三组数据。
        col_name: 列名。
        profile/profile: 用于查 tag。
        config: 参数配置。

    Returns:
        (train, valid, test, actions, note)
    """
    actions: list[str] = []
    note: str | None = None

    # ── 步骤 1：填补 ─────────────────────────────────────────
    n_missing_train = int(col_train.isna().sum())
    if n_missing_train > 0:
        imputer = _make_imputer(config.numeric_impute, config.impute_constant_value)
        imputer.fit(col_train.values.reshape(-1, 1))
        if state is not None:
            state["steps"].append(("transform", imputer))
        col_train = pd.Series(
            imputer.transform(col_train.values.reshape(-1, 1)).ravel(),
            index=col_train.index,
            name=col_name,
        )
        col_valid = pd.Series(
            imputer.transform(col_valid.values.reshape(-1, 1)).ravel(),
            index=col_valid.index,
            name=col_name,
        )
        col_test = pd.Series(
            imputer.transform(col_test.values.reshape(-1, 1)).ravel(),
            index=col_test.index,
            name=col_name,
        )
        actions.append(f"{config.numeric_impute}_impute({n_missing_train})")

    # ── 步骤 2：scale / skew 处理 ────────────────────────────
    # 规则：skewed + scale_needed 同时存在时只做 YeoJohnson（它自带标准化效果）
    is_skewed = _resolve_skewed(profile, col_name)
    is_scale = _resolve_scale_needed(profile, col_name)

    if is_skewed:
        if config.skew_method == "yeo-johnson":
            pt = PowerTransformer(method="yeo-johnson", standardize=True)
            pt.fit(col_train.values.reshape(-1, 1))
            if state is not None:
                state["steps"].append(("transform", pt))
            col_train = pd.Series(
                pt.transform(col_train.values.reshape(-1, 1)).ravel(),
                index=col_train.index,
                name=col_name,
            )
            col_valid = pd.Series(
                pt.transform(col_valid.values.reshape(-1, 1)).ravel(),
                index=col_valid.index,
                name=col_name,
            )
            col_test = pd.Series(
                pt.transform(col_test.values.reshape(-1, 1)).ravel(),
                index=col_test.index,
                name=col_name,
            )
            action_name = "yeo-johnson"
            if is_scale:
                action_name += "+scale"
            actions.append(action_name)
        elif config.skew_method == "log1p":
            # log1p: 需要确保数据非负
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fn = FunctionTransformer(np.log1p, validate=True)
                fn.fit(col_train.values.reshape(-1, 1))
                if state is not None:
                    state["steps"].append(("transform", fn))
                col_train = pd.Series(
                    fn.transform(col_train.values.reshape(-1, 1)).ravel(),
                    index=col_train.index,
                    name=col_name,
                )
                col_valid = pd.Series(
                    fn.transform(col_valid.values.reshape(-1, 1)).ravel(),
                    index=col_valid.index,
                    name=col_name,
                )
                col_test = pd.Series(
                    fn.transform(col_test.values.reshape(-1, 1)).ravel(),
                    index=col_test.index,
                    name=col_name,
                )
            action_name = "log1p"
            if is_scale:
                action_name += "+scale"
            actions.append(action_name)
            note = "log1p 要求输入 > -1，负值较多的列请换用 yeo-johnson"
    elif is_scale:
        scaler = _make_scaler(config.scale_method)
        scaler.fit(col_train.values.reshape(-1, 1))
        if state is not None:
            state["steps"].append(("transform", scaler))
        col_train = pd.Series(
            scaler.transform(col_train.values.reshape(-1, 1)).ravel(),
            index=col_train.index,
            name=col_name,
        )
        col_valid = pd.Series(
            scaler.transform(col_valid.values.reshape(-1, 1)).ravel(),
            index=col_valid.index,
            name=col_name,
        )
        col_test = pd.Series(
            scaler.transform(col_test.values.reshape(-1, 1)).ravel(),
            index=col_test.index,
            name=col_name,
        )
        actions.append(f"{config.scale_method}_scale")

    return col_train, col_valid, col_test, actions, note


def _transform_boolean(
    col_train: pd.Series,
    col_valid: pd.Series,
    col_test: pd.Series,
    col_name: str,
) -> tuple[pd.Series, pd.Series, pd.Series, list[str]]:
    """布尔列：填补为 False → 转 int(0/1)。"""
    actions: list[str] = []

    n_missing_train = int(col_train.isna().sum())
    if n_missing_train > 0:
        col_train = col_train.fillna(False)
        col_valid = col_valid.fillna(False)
        col_test = col_test.fillna(False)
        actions.append(f"fillna_False({n_missing_train})")

    col_train = col_train.astype(int)
    col_valid = col_valid.astype(int)
    col_test = col_test.astype(int)
    actions.append("to_int")

    return col_train, col_valid, col_test, actions


def _transform_categorical(
    col_train: pd.Series,
    col_valid: pd.Series,
    col_test: pd.Series,
    y_train: pd.Series,
    y_valid: pd.Series,
    y_test: pd.Series,
    col_name: str,
    profile: DataProfile,
    config: PreprocessConfig,
    state: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str], str | None]:
    """处理类别列：填补 → 编码。

    Args:
        三组 X 列和三组 y（目标编码需要 y）。

    Returns:
        (train_df, valid_df, test_df, actions, output_features, note)
        返回 DataFrame 是因为 OneHot 可能输出多列。
    """
    actions: list[str] = []
    note: str | None = None

    n_unique = _resolve_n_unique(profile, col_name)

    # ── 步骤 1：填补 ─────────────────────────────────────────
    n_missing_train = int(col_train.isna().sum())
    if n_missing_train > 0:
        imputer = _make_imputer(config.categorical_impute, config.impute_constant_value)
        imputer.fit(col_train.values.reshape(-1, 1))
        if state is not None:
            state["steps"].append(("transform", imputer))
        col_train = pd.Series(
            imputer.transform(col_train.values.reshape(-1, 1)).ravel(),
            index=col_train.index,
            name=col_name,
        )
        col_valid = pd.Series(
            imputer.transform(col_valid.values.reshape(-1, 1)).ravel(),
            index=col_valid.index,
            name=col_name,
        )
        col_test = pd.Series(
            imputer.transform(col_test.values.reshape(-1, 1)).ravel(),
            index=col_test.index,
            name=col_name,
        )
        actions.append(f"{config.categorical_impute}_impute({n_missing_train})")

    # ── 步骤 2：编码 ─────────────────────────────────────────
    # 分辨率：低基数用 low_cardinality_encoding，高基数用 high_cardinality_encoding
    use_high_card = n_unique > config.max_onehot_categories

    if use_high_card:
        # 高基数编码
        if config.high_cardinality_encoding == "target":
            from sklearn.preprocessing import TargetEncoder

            te = TargetEncoder(random_state=42)
            te.fit(col_train.values.reshape(-1, 1), y_train)
            if state is not None:
                state["steps"].append(("transform", te))
            col_train_arr = te.transform(col_train.values.reshape(-1, 1))
            col_valid_arr = te.transform(col_valid.values.reshape(-1, 1))
            col_test_arr = te.transform(col_test.values.reshape(-1, 1))
            actions.append("target_encode")
            note = f"高基数（{n_unique} 类），使用目标编码"
            # target encode 输出 1 列
            out_df_train = pd.DataFrame(
                col_train_arr, index=col_train.index, columns=[col_name]
            )
            out_df_valid = pd.DataFrame(
                col_valid_arr, index=col_valid.index, columns=[col_name]
            )
            out_df_test = pd.DataFrame(
                col_test_arr, index=col_test.index, columns=[col_name]
            )
            output_features = [col_name]

        else:  # frequency
            freq = col_train.value_counts(normalize=True)
            if state is not None:
                state["steps"].append(("frequency", freq))
            col_train = col_train.map(freq)
            col_valid = col_valid.map(freq)
            col_test = col_test.map(freq)
            # 测试集出现未见过的类别 → 用全局频率 0
            col_valid = col_valid.fillna(0.0)
            col_test = col_test.fillna(0.0)
            actions.append("frequency_encode")
            note = f"高基数（{n_unique} 类），使用频率编码"
            out_df_train = col_train.to_frame()
            out_df_valid = col_valid.to_frame()
            out_df_test = col_test.to_frame()
            output_features = [col_name]

    else:
        # 低基数编码
        if config.low_cardinality_encoding == "onehot":
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            ohe.fit(col_train.values.reshape(-1, 1))
            if state is not None:
                state["steps"].append(("transform", ohe))
            col_train_arr = ohe.transform(col_train.values.reshape(-1, 1))
            col_valid_arr = ohe.transform(col_valid.values.reshape(-1, 1))
            col_test_arr = ohe.transform(col_test.values.reshape(-1, 1))

            # 构造特征名
            cats = ohe.categories_[0]
            output_features = [f"{col_name}_{c}" for c in cats]
            actions.append(f"onehot({len(cats)})")

            out_df_train = pd.DataFrame(
                col_train_arr, index=col_train.index, columns=output_features
            )
            out_df_valid = pd.DataFrame(
                col_valid_arr, index=col_valid.index, columns=output_features
            )
            out_df_test = pd.DataFrame(
                col_test_arr, index=col_test.index, columns=output_features
            )
        else:  # label
            oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            oe.fit(col_train.values.reshape(-1, 1))
            if state is not None:
                state["steps"].append(("transform", oe))
            col_train_arr = oe.transform(col_train.values.reshape(-1, 1)).ravel()
            col_valid_arr = oe.transform(col_valid.values.reshape(-1, 1)).ravel()
            col_test_arr = oe.transform(col_test.values.reshape(-1, 1)).ravel()

            actions.append("label_encode")
            output_features = [col_name]
            out_df_train = pd.DataFrame(
                col_train_arr, index=col_train.index, columns=output_features
            )
            out_df_valid = pd.DataFrame(
                col_valid_arr, index=col_valid.index, columns=output_features
            )
            out_df_test = pd.DataFrame(
                col_test_arr, index=col_test.index, columns=output_features
            )

    return out_df_train, out_df_valid, out_df_test, actions, output_features, note


def _transform_datetime(
    col_train: pd.Series,
    col_valid: pd.Series,
    col_test: pd.Series,
    col_name: str,
    config: PreprocessConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """日期列：抽成年/月/日等数值特征。"""
    actions: list[str] = []
    all_sets = []
    for col in [col_train, col_valid, col_test]:
        d = pd.to_datetime(col)
        parts = {"year": d.dt.year, "month": d.dt.month, "day": d.dt.day}
        if config.datetime_features == "full":
            parts["dayofweek"] = d.dt.dayofweek
            parts["quarter"] = d.dt.quarter
        all_sets.append(pd.DataFrame(parts, index=col.index))
        # 列名加前缀以区分来源
        for k in parts:
            all_sets[-1].rename(columns={k: f"{col_name}_{k}"}, inplace=True)

    out_features = list(all_sets[0].columns)
    actions.append(f"datetime_extract({config.datetime_features})")

    return all_sets[0], all_sets[1], all_sets[2], actions, out_features


def _ks_critical(n_reference: int, n_other: int, coefficient: float = 1.36) -> float:
    """两样本 KS 检验在 alpha=0.05 下的临界值。

    KS 统计量的零分布只取决于样本量。样本越少，抽样噪声造成的 KS 越大，
    因此不能拿固定阈值判定漂移，必须先过这个显著性门槛。
    """
    if n_reference <= 0 or n_other <= 0:
        return 1.0
    return float(coefficient * np.sqrt((n_reference + n_other) / (n_reference * n_other)))


def _ks_statistic(reference: np.ndarray, other: np.ndarray) -> float:
    """两个样本的经验分布函数最大间距（Kolmogorov-Smirnov 统计量）。"""
    if len(reference) == 0 or len(other) == 0:
        return 0.0
    values = np.sort(np.unique(np.concatenate([reference, other])))
    ref_sorted, other_sorted = np.sort(reference), np.sort(other)
    cdf_ref = np.searchsorted(ref_sorted, values, side="right") / len(reference)
    cdf_other = np.searchsorted(other_sorted, values, side="right") / len(other)
    return float(np.max(np.abs(cdf_ref - cdf_other)))


def _psi(
    reference: np.ndarray,
    other: np.ndarray,
    bins: int,
    min_bin_count: int,
) -> float | None:
    """群体稳定性指标；样本不足以支撑分箱时返回 None。

    分箱边界只用训练集分位数确定——否则测试集的分布信息会反过来塑造基线，
    使漂移被系统性低估。箱外边界设为 ±inf，保证任何取值都落在某个箱内。

    分箱数按样本量自适应下调：每个箱的期望样本数不低于 ``min_bin_count``，
    否则 PSI 会被抽样噪声主导（小数据集上动辄给出 >1 的假性漂移）。
    连 2 个箱都支撑不起时返回 None，交由调用方按「样本不足」处理。
    """
    if len(reference) == 0 or len(other) == 0:
        return None
    if np.nanmin(reference) == np.nanmax(reference):
        # 参考分布为常量：无法定义有意义的箱，视为无漂移
        return 0.0
    effective_bins = min(
        bins,
        max(1, len(other) // max(1, min_bin_count)),
        max(1, len(reference) // max(1, min_bin_count)),
    )
    if effective_bins < 2:
        return None
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, effective_bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(reference, bins=edges)[0] / len(reference)
    q = np.histogram(other, bins=edges)[0] / len(other)
    eps = 1e-6  # 避免零频箱取对数发散
    return float(np.sum((p - q) * np.log((p + eps) / (q + eps))))


def _severity(value: float, warning: float, high: float) -> str:
    if value >= high:
        return "high"
    if value >= warning:
        return "medium"
    return "info"


def _drift_numeric(
    feature: str,
    reference: pd.Series,
    other: pd.Series,
    comparison: str,
    cfg: PreprocessConfig,
) -> FeatureDriftRecord:
    """数值特征的 PSI + KS 漂移记录。

    判定分两步：先看 KS 统计量是否显著大于同样本量下的临界值（否则差异
    只是抽样噪声），再按 PSI 分级。小样本上 PSI 会被噪声主导，因此统计
    不显著时不给出漂移结论。
    """
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    candidate = pd.to_numeric(other, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(ref) or not len(candidate):
        return FeatureDriftRecord(
            feature, "numeric", comparison, severity="low",
            recommendation="样本为空或无法解析；检查缺失值处理与子集规模。",
        )
    psi = _psi(ref, candidate, cfg.drift_bins, cfg.psi_min_bin_count)
    ks = _ks_statistic(ref, candidate)
    critical = _ks_critical(len(ref), len(candidate))

    if ks <= critical:
        # 差异未超过同样本量下的抽样噪声水平，不下漂移结论
        return FeatureDriftRecord(
            feature, "numeric", comparison,
            psi=round(psi, 6) if psi is not None else None,
            ks_statistic=round(ks, 6), severity="info",
            recommendation=f"KS={ks:.4f} 未超过同样本量临界值 {critical:.4f}，差异在抽样噪声范围内。",
        )

    severity = _severity(psi, cfg.psi_warning, cfg.psi_high) if psi is not None else "medium"
    return FeatureDriftRecord(
        feature, "numeric", comparison,
        psi=round(psi, 6) if psi is not None else None,
        ks_statistic=round(ks, 6), severity=severity,
        recommendation=(
            "漂移偏高：检查切分策略、时间漂移或训练集覆盖范围。"
            if severity in HIGH_RISK_SEVERITIES
            else "KS 显著但 PSI 未超阈值，建议关注该特征。"
        ),
    )


def _tvd_critical(
    counts_ref: pd.Series,
    counts_other: pd.Series,
    n_ref: int,
    n_other: int,
    z: float = 1.96,
) -> float:
    """两样本总变差距离在近似 95% 置信水平下的临界值。

    零假设下 ``p_i - q_i`` 的方差约为 ``p_i(1-p_i)(1/n1 + 1/n2)``，总变差是
    各分量绝对差的一半之和，用 delta 方法近似得到下述临界值。样本越小，
    抽样噪声导致的 TVD 就越大，因此必须按样本量给出门槛。
    """
    if n_ref <= 0 or n_other <= 0:
        return 1.0
    pooled = (
        counts_ref.reindex(counts_ref.index.union(counts_other.index)).fillna(0)
        + counts_other.reindex(counts_ref.index.union(counts_other.index)).fillna(0)
    )
    total = float(pooled.sum())
    if total <= 0:
        return 1.0
    probs = pooled / total
    variance = float((probs * (1 - probs)).sum())
    return float(z * 0.5 * np.sqrt(variance) * np.sqrt(1 / n_ref + 1 / n_other))


def _drift_categorical(
    feature: str,
    reference: pd.Series,
    other: pd.Series,
    comparison: str,
    cfg: PreprocessConfig,
) -> FeatureDriftRecord:
    """类别特征的总变差距离与未见类别比例记录。

    与数值特征同理：先按样本量判断 TVD 是否显著，再分级，避免小样本噪声
    被当作真实漂移。未见类别比例是确定性事实（测试集出现了训练集没有的
    取值），不受样本量影响，因此单独作为信号保留。
    """
    ref = reference.astype("string").fillna("<NA>")
    candidate = other.astype("string").fillna("<NA>")
    if ref.empty or candidate.empty:
        return FeatureDriftRecord(
            feature, "categorical", comparison, severity="low",
            recommendation="样本为空；检查子集规模。",
        )
    counts_ref = ref.value_counts()
    counts_other = candidate.value_counts()
    categories = sorted(set(ref) | set(candidate))
    p = ref.value_counts(normalize=True)
    q = candidate.value_counts(normalize=True)
    tvd = 0.5 * sum(abs(float(p.get(c, 0.0)) - float(q.get(c, 0.0))) for c in categories)
    unseen = float((~candidate.isin(set(ref))).mean())
    critical = _tvd_critical(counts_ref, counts_other, len(ref), len(candidate))

    if tvd <= critical:
        return FeatureDriftRecord(
            feature, "categorical", comparison,
            total_variation=round(tvd, 6), unseen_rate=round(unseen, 6), severity="info",
            recommendation=f"TVD={tvd:.4f} 未超过同样本量临界值 {critical:.4f}，差异在抽样噪声范围内。",
        )

    severity = _severity(max(tvd, unseen), cfg.tvd_warning, cfg.tvd_high)
    return FeatureDriftRecord(
        feature, "categorical", comparison,
        total_variation=round(tvd, 6), unseen_rate=round(unseen, 6), severity=severity,
        recommendation=(
            "类别分布变化较大：检查新类别、编码策略及训练数据代表性。"
            if severity in HIGH_RISK_SEVERITIES
            else "类别分布有变化但幅度有限，建议关注。"
        ),
    )


def audit_drift(
    X: pd.DataFrame,
    train_indices: Iterable[int],
    valid_indices: Iterable[int],
    test_indices: Iterable[int],
    *,
    sensitive_features: Iterable[str] | None = None,
    config: PreprocessConfig | None = None,
    exclude_columns: Iterable[str] | None = None,
) -> PreprocessAuditResult:
    """对切分后的原始特征做分布漂移与敏感群体覆盖审计（纯观测）。

    每个特征都比较「训练集 vs 验证集」和「训练集 vs 测试集」两个方向；
    群体覆盖取全体数据中出现的群体并集，因此某群体在某个子集中完全缺失时
    会被记为 count=0，而不会因为 value_counts 不返回该键而被静默漏掉。

    Args:
        X: 原始特征 DataFrame（切分前）。
        train_indices / valid_indices / test_indices: 三集在 X 中的行索引。
        sensitive_features: 需要做群体覆盖审计的列名；为 None 时跳过该部分。
        config: 预处理配置，提供各项阈值。
        exclude_columns: 不参与漂移审计的列（如已被排除出模型输入的列）。
            被排除的列不影响模型，其分布差异不构成建模风险；分组列尤其如此——
            分组切分会让各集合的实体天然不同，"未见率 100%" 正是预期行为。

    Returns:
        PreprocessAuditResult：逐特征漂移记录与群体覆盖记录。
    """
    cfg = config if config is not None else PreprocessConfig()
    result = PreprocessAuditResult()

    if not cfg.enable_drift_audit or not isinstance(X, pd.DataFrame):
        return result

    skip = set(exclude_columns or [])
    train_idx = np.asarray(list(train_indices))
    valid_idx = np.asarray(list(valid_indices))
    test_idx = np.asarray(list(test_indices))
    train = X.iloc[train_idx]
    valid = X.iloc[valid_idx]
    test = X.iloc[test_idx]

    features = [c for c in X.columns if c not in skip]
    if cfg.drift_max_features > 0 and len(features) > cfg.drift_max_features:
        features = features[: cfg.drift_max_features]
        result.drift_truncated = True
    result.audited_features = len(features)
    result.skipped_columns = sorted(skip & set(X.columns))

    for feature in features:
        maker = _drift_numeric if pd.api.types.is_numeric_dtype(X[feature]) else _drift_categorical
        result.drift_records.append(
            maker(feature, train[feature], valid[feature], "train_vs_validation", cfg)
        )
        result.drift_records.append(
            maker(feature, train[feature], test[feature], "train_vs_test", cfg)
        )

    # 只统计通过显著性门槛的记录：未过门槛的 PSI 由抽样噪声主导，
    # 把它当作「最大漂移」展示会误导阅读报告的人。
    significant_psi = [
        r.psi for r in result.drift_records
        if r.psi is not None and r.severity != "info"
    ]
    result.max_psi = round(max(significant_psi), 6) if significant_psi else None

    # ── 敏感群体覆盖 ────────────────────────────────────────
    for feature in list(sensitive_features or []):
        if feature not in X.columns:
            result.findings.append(
                AuditFinding(
                    code="missing_sensitive_feature",
                    severity="medium",
                    title="声明的敏感特征不存在",
                    evidence=f"敏感特征 {feature!r} 不在数据列中，无法审计群体覆盖。",
                    recommendation="检查列名拼写，或从 fit(sensitive_features=...) 中移除。",
                    columns=[feature],
                )
            )
            continue
        # 用全体数据的群体并集，保证「某群体在某子集完全缺失」被记录为 count=0
        all_groups = sorted(X[feature].astype("string").fillna("<NA>").unique().tolist())
        for split_name, frame in (("train", train), ("validation", valid), ("test", test)):
            counts = frame[feature].astype("string").fillna("<NA>").value_counts(dropna=False)
            total = max(len(frame), 1)
            for group in all_groups:
                count = int(counts.get(group, 0))
                severity: str = "high" if count < cfg.min_group_size else "info"
                result.group_coverage.append(
                    GroupCoverageRecord(
                        str(feature), str(group), split_name, count,
                        round(count / total, 6), severity,
                    )
                )

    # ── 汇总为 findings 与固定检查项 ────────────────────────
    high_drift = result.high_drift_features
    if high_drift:
        result.findings.append(
            AuditFinding(
                code="feature_drift",
                severity="high",
                title="特征分布漂移偏大",
                evidence=f"{len(high_drift)} 个特征在验证集或测试集相对训练集漂移偏高："
                f"{high_drift[:5]}{' 等' if len(high_drift) > 5 else ''}。"
                f"最大 PSI = {result.max_psi}。",
                recommendation="检查切分方式是否引入时间或实体偏差；必要时补充训练数据覆盖。",
                columns=high_drift,
                details={"max_psi": result.max_psi},
            )
        )

    uncovered = result.uncovered_groups
    if uncovered:
        result.findings.append(
            AuditFinding(
                code="group_coverage_gap",
                severity="high",
                title="敏感群体覆盖缺失",
                evidence=f"{len(uncovered)} 个「群体 × 子集」组合样本数为 0："
                f"{uncovered[:5]}{' 等' if len(uncovered) > 5 else ''}。",
                recommendation="该群体在对应子集中无样本，相关公平性结论不可用；"
                "考虑调整切分或补充数据。",
                details={"uncovered": uncovered},
            )
        )
    else:
        small_groups = [
            f"{r.sensitive_feature}={r.group}@{r.split}"
            for r in result.group_coverage
            if r.severity == "high"
        ]
        if small_groups:
            result.findings.append(
                AuditFinding(
                    code="small_group_coverage",
                    severity="medium",
                    title="部分群体样本量偏小",
                    evidence=f"{len(small_groups)} 个「群体 × 子集」组合样本数低于 "
                    f"{cfg.min_group_size}：{small_groups[:5]}"
                    f"{' 等' if len(small_groups) > 5 else ''}。",
                    recommendation="样本量过小会让该群体的指标估计不稳定，解读时需谨慎。",
                )
            )

    result.model_info_questions = [
        "验证集与测试集是否与训练集来自同一分布与同一时间范围？",
        "敏感群体在三集中是否都有足够样本量支撑公平性判断？",
    ]
    _set_audit_check(
        result, "feature_drift", "特征分布漂移",
        "发现风险" if high_drift else "通过",
        f"{len(high_drift)} 个特征漂移偏高。" if high_drift
        else f"{result.audited_features} 个特征在两两方向上漂移均未超过高危阈值。",
    )
    if not sensitive_features:
        _set_audit_check(
            result, "group_coverage", "敏感群体覆盖", "未执行",
            "未提供敏感特征，无法审计群体覆盖；如需该检查请传入 sensitive_features。",
        )
    else:
        small_count = sum(r.severity == "high" for r in result.group_coverage)
        _set_audit_check(
            result, "group_coverage", "敏感群体覆盖",
            "发现风险" if small_count else "通过",
            f"{len(uncovered)} 个「群体 × 子集」组合样本数为 0，"
            f"{small_count} 个组合低于最小样本量 {cfg.min_group_size}。"
            if small_count
            else f"{len(result.group_coverage)} 个「群体 × 子集」组合均有足够样本。",
        )
    return result


def _set_audit_check(
    result: PreprocessAuditResult,
    code: str,
    check: str,
    status: str,
    message: str,
) -> None:
    """登记/覆盖一个固定审计检查项。"""
    item = AuditCheckResult(code=code, check=check, status=status, result=message)  # type: ignore[arg-type]
    for index, existing in enumerate(result.check_results):
        if existing.code == code:
            result.check_results[index] = item
            return
    result.check_results.append(item)


def _resolve_audit_exclusions(
    profile: DataProfile,
    split_result: SplitResult,
    cfg: PreprocessConfig,
) -> dict[str, str]:
    """计算因审计建议或分组声明需要排除的列，返回 {列名: 排除原因}。

    与 ``dropped_columns`` 分开记录，便于追溯「这一列为什么被丢掉」：
    常数列/空列属于兜底清理，这里则是明确的合规或泄漏处置。
    """
    exclusions: dict[str, str] = {}

    # 1. 分组列：声明即语义，与是否启用划分审计无关
    group_column = split_result.group_column
    if group_column and cfg.drop_group_column:
        exclusions[group_column] = (
            f"声明的实体分组列（split_strategy={split_result.strategy}），"
            "作为标识列不参与建模，避免模型记忆实体"
        )

    # 2. 画像阶段的建议排除列，按配置的级别筛选
    if cfg.apply_audit_drops != "off" and profile.audit is not None:
        profile_columns = {c.col_name for c in profile.columns}
        for finding in profile.audit.findings:
            if finding.code == "potential_grouping_need":
                continue  # 建议性提示，不构成排除依据
            if cfg.apply_audit_drops == "high" and finding.severity not in HIGH_RISK_SEVERITIES:
                continue
            for col in finding.columns:
                if col in profile_columns and col not in exclusions:
                    exclusions[col] = f"画像审计（{finding.code}/{finding.severity}）：{finding.title}"

    return exclusions


# ── 顶层函数 ─────────────────────────────────────────────────


def preprocess(
    X: pd.DataFrame,
    y: pd.Series,
    profile: DataProfile,
    split_result: SplitResult,
    config: PreprocessConfig | None = None,
    sensitive_features: list[str] | None = None,
) -> PreprocessedData:
    """执行数据预处理：填补 → 编码 → 缩放，全部拟合于训练集。

    函数签名沿用已有模式：
        result = stage_function(X, y, input_contract, config?)

    内部步骤：
        1. 根据 split_result 的索引切出 train/valid/test 三集。
        2. 逐列检查 profile.tags，决定变换方案。
        3. 在训练集上拟合变换，应用到三集。
        4. 收集 TransformRecord，返回 PreprocessedData。

    Args:
        X: 原始特征 DataFrame。
        y: 原始目标 Series。
        profile: build_profile 输出的 DataProfile，含每列的 tags。
        split_result: split_data 输出的 SplitResult，含三集索引。
        config: 可选预处理参数，默认使用 PreprocessConfig()。
        sensitive_features: 需要做群体覆盖审计的原始列名；为 None 时跳过该检查。

    Returns:
        PreprocessedData：变换后的三集数据 + 变换记录 + 审计结果。

    Raises:
        TextColumnNotSupportedError: handle_text="keep" 但发现文本列。
        TransformError: 变换执行失败。
    """
    cfg = config if config is not None else PreprocessConfig()

    # ── 步骤 1：切分数据 ────────────────────────────────────
    train_idx = split_result.train_indices
    valid_idx = split_result.valid_indices
    test_idx = split_result.test_indices

    X_train = X.iloc[train_idx].copy()
    X_valid = X.iloc[valid_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_valid = y.iloc[valid_idx].copy()
    y_test = y.iloc[test_idx].copy()

    all_columns = list(X.columns)
    fitted_states = {name: dict(kind="passthrough", steps=[], outputs=[name]) for name in all_columns}
    n_features_in = len(all_columns)
    transforms: list[TransformRecord] = []
    column_mappings: list[ColumnMapping] = []
    dropped_columns: list[str] = []
    excluded_by_audit: list[str] = []
    feature_names_out: list[str] = []

    # ── 步骤 1.5：审计排除（分组列 + 画像建议列）───────────
    audit_exclusions = _resolve_audit_exclusions(profile, split_result, cfg)

    # ── 步骤 2：逐列处理 ────────────────────────────────────
    for col_name in all_columns:
        col_type = _resolve_col_type(profile, col_name)

        # ── 审计排除列 ───────────────────────────────────────
        if col_name in audit_exclusions:
            reason = audit_exclusions[col_name]
            saved_dtype = str(X_train[col_name].dtype)
            for df in [X_train, X_valid, X_test]:
                df.drop(columns=[col_name], inplace=True)
            excluded_by_audit.append(col_name)
            transforms.append(
                TransformRecord(
                    col_name=col_name,
                    original_dtype=saved_dtype,
                    original_n_unique=_resolve_n_unique(profile, col_name),
                    action_sequence=["dropped_by_audit"],
                    dropped=True,
                    n_features_out=0,
                    note=reason,
                )
            )
            column_mappings.append(
                ColumnMapping(
                    original_name=col_name,
                    output_names=[],
                    is_dropped=True,
                    n_features_out=0,
                )
            )
            continue

        # ── 丢弃型列 ─────────────────────────────────────────
        if _has_tag(profile, col_name, "constant_column") or _has_tag(
            profile, col_name, "empty_column"
        ):
            saved_dtype = str(X_train[col_name].dtype)
            for df in [X_train, X_valid, X_test]:
                df.drop(columns=[col_name], inplace=True)
            dropped_columns.append(col_name)
            transforms.append(
                TransformRecord(
                    col_name=col_name,
                    original_dtype=saved_dtype,
                    original_n_unique=_resolve_n_unique(profile, col_name),
                    action_sequence=["dropped"],
                    dropped=True,
                    n_features_out=0,
                    note="常数列或全空列，对下游无信息量",
                )
            )
            column_mappings.append(
                ColumnMapping(
                    original_name=col_name,
                    output_names=[],
                    is_dropped=True,
                    n_features_out=0,
                )
            )
            continue

        # ── 文本列 ──────────────────────────────────────────
        if col_type == "text":
            if cfg.handle_text == "drop":
                saved_dtype = str(X_train[col_name].dtype)
                for df in [X_train, X_valid, X_test]:
                    df.drop(columns=[col_name], inplace=True)
                dropped_columns.append(col_name)
                transforms.append(
                    TransformRecord(
                        col_name=col_name,
                        original_dtype=saved_dtype,
                        original_n_unique=_resolve_n_unique(profile, col_name),
                        action_sequence=["dropped"],
                        dropped=True,
                        n_features_out=0,
                        note="文本列，按配置丢弃",
                    )
                )
                column_mappings.append(
                    ColumnMapping(
                        original_name=col_name,
                        output_names=[],
                        is_dropped=True,
                        n_features_out=0,
                    )
                )
                continue
            else:
                raise TextColumnNotSupportedError(
                    f"发现文本列 {col_name!r}（唯一值="
                    f"{_resolve_n_unique(profile, col_name)}），"
                    f"handle_text='keep' 当前不支持 NLP 特征提取。"
                    f"请设置 handle_text='drop' 丢弃文本列，"
                    f"或等待后续版本实现。"
                )

        # ── 日期列 ──────────────────────────────────────────
        if col_type == "datetime":
            if cfg.handle_datetime == "drop":
                saved_dtype = str(X_train[col_name].dtype)
                for df in [X_train, X_valid, X_test]:
                    df.drop(columns=[col_name], inplace=True)
                dropped_columns.append(col_name)
                transforms.append(
                    TransformRecord(
                        col_name=col_name,
                        original_dtype=saved_dtype,
                        original_n_unique=_resolve_n_unique(profile, col_name),
                        action_sequence=["dropped"],
                        dropped=True,
                        n_features_out=0,
                        note="日期列，按配置丢弃",
                    )
                )
                column_mappings.append(
                    ColumnMapping(
                        original_name=col_name,
                        output_names=[],
                        is_dropped=True,
                        n_features_out=0,
                    )
                )
                continue
            elif cfg.handle_datetime == "keep":
                feature_names_out.append(col_name)
                transforms.append(
                    TransformRecord(
                        col_name=col_name,
                        original_dtype=str(X_train[col_name].dtype),
                        original_n_unique=_resolve_n_unique(profile, col_name),
                        action_sequence=["kept"],
                        n_features_out=1,
                        note="日期列按原样保留",
                    )
                )
                column_mappings.append(
                    ColumnMapping(
                        original_name=col_name,
                        output_names=[col_name],
                        n_features_out=1,
                    )
                )
                continue
            else:  # extract
                col_train_s = X_train[col_name]
                col_valid_s = X_valid[col_name]
                col_test_s = X_test[col_name]

                (
                    out_train,
                    out_valid,
                    out_test,
                    actions,
                    out_features,
                ) = _transform_datetime(
                    col_train_s, col_valid_s, col_test_s, col_name, cfg
                )

                # 替换原列
                X_train = pd.concat(
                    [X_train.drop(columns=[col_name]), out_train], axis=1
                )
                X_valid = pd.concat(
                    [X_valid.drop(columns=[col_name]), out_valid], axis=1
                )
                X_test = pd.concat(
                    [X_test.drop(columns=[col_name]), out_test], axis=1
                )
                feature_names_out.extend(out_features)
                transforms.append(
                    TransformRecord(
                        col_name=col_name,
                        original_dtype=str(X_train[col_name].dtype)
                        if col_name in X_train.columns
                        else "datetime",
                        original_n_unique=_resolve_n_unique(profile, col_name),
                        action_sequence=actions,
                        n_features_out=len(out_features),
                    )
                )
                column_mappings.append(
                    ColumnMapping(
                        original_name=col_name,
                        output_names=list(out_features),
                        n_features_out=len(out_features),
                    )
                )
                continue

        # ── 数值列 ──────────────────────────────────────────
        if col_type == "numeric":
            col_train_s = X_train[col_name]
            col_valid_s = X_valid[col_name]
            col_test_s = X_test[col_name]

            out_train, out_valid, out_test, actions, note = _transform_numeric(
                col_train_s, col_valid_s, col_test_s, col_name, profile, cfg, fitted_states[col_name]
            )

            X_train[col_name] = out_train
            X_valid[col_name] = out_valid
            X_test[col_name] = out_test
            feature_names_out.append(col_name)
            transforms.append(
                TransformRecord(
                    col_name=col_name,
                    original_dtype=str(col_train_s.dtype),
                    original_n_unique=_resolve_n_unique(profile, col_name),
                    action_sequence=actions,
                    n_features_out=1,
                    note=note,
                )
            )
            column_mappings.append(
                ColumnMapping(
                    original_name=col_name,
                    output_names=[col_name],
                    n_features_out=1,
                )
            )
            continue

        # ── 布尔列 ──────────────────────────────────────────
        if col_type == "boolean":
            col_train_s = X_train[col_name]
            col_valid_s = X_valid[col_name]
            col_test_s = X_test[col_name]

            out_train, out_valid, out_test, actions = _transform_boolean(
                col_train_s, col_valid_s, col_test_s, col_name
            )

            X_train[col_name] = out_train
            X_valid[col_name] = out_valid
            X_test[col_name] = out_test
            feature_names_out.append(col_name)
            transforms.append(
                TransformRecord(
                    col_name=col_name,
                    original_dtype=str(col_train_s.dtype),
                    original_n_unique=2,
                    action_sequence=actions,
                    n_features_out=1,
                )
            )
            column_mappings.append(
                ColumnMapping(
                    original_name=col_name,
                    output_names=[col_name],
                    n_features_out=1,
                )
            )
            continue

        # ── 类别列 ──────────────────────────────────────────
        if col_type == "categorical":
            col_train_s = X_train[col_name]
            col_valid_s = X_valid[col_name]
            col_test_s = X_test[col_name]

            (
                out_train_df,
                out_valid_df,
                out_test_df,
                actions,
                out_features,
                note,
            ) = _transform_categorical(
                col_train_s,
                col_valid_s,
                col_test_s,
                y_train,
                y_valid,
                y_test,
                col_name,
                profile,
                cfg,
                fitted_states[col_name],
            )

            # 替换原列（可能多列）
            X_train = pd.concat(
                [X_train.drop(columns=[col_name]), out_train_df], axis=1
            )
            X_valid = pd.concat(
                [X_valid.drop(columns=[col_name]), out_valid_df], axis=1
            )
            X_test = pd.concat(
                [X_test.drop(columns=[col_name]), out_test_df], axis=1
            )
            feature_names_out.extend(out_features)
            transforms.append(
                TransformRecord(
                    col_name=col_name,
                    original_dtype=str(col_train_s.dtype),
                    original_n_unique=_resolve_n_unique(profile, col_name),
                    action_sequence=actions,
                    n_features_out=len(out_features),
                    note=note,
                )
            )
            column_mappings.append(
                ColumnMapping(
                    original_name=col_name,
                    output_names=list(out_features),
                    n_features_out=len(out_features),
                )
            )
            continue

        # ── 兜底：未知类型，原样保留 ─────────────────────────
        feature_names_out.append(col_name)
        transforms.append(
            TransformRecord(
                col_name=col_name,
                original_dtype=str(X_train[col_name].dtype),
                original_n_unique=_resolve_n_unique(profile, col_name),
                action_sequence=["kept"],
                n_features_out=1,
                note="未知类型，原样保留",
            )
        )
        column_mappings.append(
            ColumnMapping(
                original_name=col_name,
                output_names=[col_name],
                n_features_out=1,
            )
        )

    # Match model input order, including columns appended by one-hot/date expansion.
    feature_names_out = list(X_train.columns)
    if not X_train.columns.is_unique:
        raise TransformError("预处理后特征名重复，无法可靠保存推断流水线")
    for mapping in column_mappings:
        state = fitted_states[mapping.original_name]
        state["outputs"] = list(mapping.output_names)
        col_type = _resolve_col_type(profile, mapping.original_name)
        if mapping.is_dropped:
            state["kind"] = "drop"
        elif col_type == "boolean":
            state["kind"] = "boolean"
        elif col_type == "datetime" and cfg.handle_datetime == "extract":
            state["kind"] = "datetime"
            state["date_parts"] = ["year", "month", "day"] + (["dayofweek", "quarter"] if cfg.datetime_features == "full" else [])
    fitted_preprocessor = FittedPreprocessor(fitted_states, feature_names_out)
    n_features_out = len(feature_names_out)

    # ── 步骤 3：分布漂移与群体覆盖审计（在原始特征上，纯观测）──
    # 已被排除出模型输入的列不参与漂移审计：它们不影响建模，
    # 且分组列在分组切分下"未见率 100%"是预期行为而非风险。
    audit = audit_drift(
        X, train_idx, valid_idx, test_idx,
        sensitive_features=sensitive_features, config=cfg,
        exclude_columns=excluded_by_audit,
    )
    audit.excluded_by_audit = list(excluded_by_audit)
    if excluded_by_audit:
        # 记录动作本身而非风险：severity 保持 info，避免把「已按预期排除」
        # 误报成整体风险升高。
        audit.findings.append(
            AuditFinding(
                code="columns_excluded",
                severity="info",
                title="已排除的列",
                evidence=f"{len(excluded_by_audit)} 列未进入模型输入：{excluded_by_audit}。",
                recommendation="如需保留其中某列作为特征，请调整 apply_audit_drops 或 drop_group_column。",
                columns=list(excluded_by_audit),
            )
        )

    return PreprocessedData(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        n_features_in=n_features_in,
        n_features_out=n_features_out,
        dropped_columns=dropped_columns,
        feature_names_out=feature_names_out,
        transforms=transforms,
        column_mappings=column_mappings,
        fitted_preprocessor=fitted_preprocessor,
        audit=audit,
        excluded_by_audit=excluded_by_audit,
    )


__all__ = [
    "preprocess",
    "audit_drift",
]