"""数据划分模块：按比例划分训练/验证/测试集，并审计划分完整性。

设计哲学：
- Split 是实验设计（组织实验），不做数据变换。
- 输出只存索引和元信息，不存实际数据，避免内存重复。
- Preprocess 阶段通过 train_indices / valid_indices / test_indices
  从原始 X, y 中切片获取各子集数据。
- 阶段 2 新增：实体（group）与时序（temporal）感知切分，以及划分后的
  完整性审计（跨集合重复、实体泄漏、时间穿越）。审计为纯观测，不修改数据。

关于 ``group_column`` / ``time_column`` 的语义：它们只**声明**数据结构并触发
相应审计；实际怎么切由 ``split_strategy`` 决定。这样保留「用随机切分 + 审计
暴露泄漏」的诊断用法（把策略显式设为 random/stratified 即可）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from tabularx.core.params import SplitConfig
from tabularx.exceptions.split import (
    InvalidSplitError,
    SplitColumnMissingError,
    StratifyError,
    TimeParseError,
)
from tabularx.schema.audit import (
    AuditCheckResult,
    AuditFinding,
    SplitAuditResult,
)
from tabularx.schema.profile import DataProfile, Evidence
from tabularx.schema.split import SplitResult

# 合法的切分策略
SPLIT_STRATEGIES = ("auto", "stratified", "random", "group", "temporal")


def _compute_class_distribution(
    y: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    """计算指定索引子集的类别分布。"""
    subset = y[indices]
    unique, counts = np.unique(subset, return_counts=True)
    total = counts.sum()
    return {str(k): float(v / total) for k, v in zip(unique, counts)}


def _max_distribution_shift(
    dists: list[dict[str, float]],
) -> float:
    """计算多个分布之间的最大类别比例差异。"""
    all_classes = set()
    for d in dists:
        all_classes.update(d.keys())
    max_diff = 0.0
    for cls in all_classes:
        ratios = [d.get(cls, 0.0) for d in dists]
        diff = max(ratios) - min(ratios)
        max_diff = max(max_diff, diff)
    return max_diff


# ── 策略解析 ─────────────────────────────────────────────────


def resolve_split_strategy(
    profile: DataProfile,
    config: SplitConfig | None = None,
) -> str:
    """把 ``auto`` 解析成实际策略（group / temporal / stratified / random）。"""
    cfg = config if config is not None else SplitConfig()
    strategy = cfg.split_strategy
    if strategy != "auto":
        return strategy
    if cfg.group_column:
        return "group"
    if cfg.time_column:
        return "temporal"
    return "stratified" if profile.target.task_type in ("binary", "multiclass") else "random"


def _require_dataframe(X: object, column: str, role: str) -> pd.DataFrame:
    """分组/时序切分需要按列取值，因此要求 X 是 DataFrame。"""
    if not isinstance(X, pd.DataFrame):
        raise SplitColumnMissingError(
            f"声明了 {role}（{column!r}）时必须传入 pandas DataFrame，"
            f"以便按列取值；当前收到 {type(X)}。"
            "建议：保持 DataFrame 输入，或改用 split_strategy='random'/'stratified'。"
        )
    if column not in X.columns:
        raise SplitColumnMissingError(
            f"声明的 {role} {column!r} 不在 X 的列中。"
            f"  X 现有列：{list(X.columns)[:10]}{' ...' if X.shape[1] > 10 else ''}。"
            "建议：检查列名拼写，或从 SplitConfig 中移除该声明。"
        )
    return X


def _parse_time_column(X: pd.DataFrame, column: str) -> pd.Series:
    """解析时间列；存在空值或不可解析值时直接报错（时序切分不能容忍）。"""
    parsed = pd.to_datetime(X[column], errors="coerce")
    invalid = int(parsed.isna().sum())
    if invalid:
        raise TimeParseError(
            f"时间列 {column!r} 有 {invalid} 个空值或不可解析的时间，无法进行时序切分。"
            "建议：清洗时间列后重试，或改用随机切分并由审计报告暴露时间风险。"
        )
    return parsed


# ── 切分实现 ─────────────────────────────────────────────────


def _split_random_or_stratified(
    X: object,
    y: object,
    profile: DataProfile,
    cfg: SplitConfig,
    stratify: bool,
    evidences: list[Evidence],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """原有两阶段划分：先分测试集，再从剩余部分分验证集。"""
    n_total = len(X)  # type: ignore[arg-type]
    stratify_y = y if stratify else None
    if stratify:
        evidences.append(
            Evidence(
                "stratify_reason",
                str(profile.target.task_type),
                f"目标任务类型为 {profile.target.task_type}，使用分层采样保证各集类别分布一致。",
            )
        )
    else:
        reason = (
            "目标任务类型为回归，使用纯随机采样。"
            if profile.target.task_type == "regression"
            else f"策略为 {cfg.split_strategy}，使用纯随机采样（未分层）。"
        )
        evidences.append(Evidence("stratify_reason", "random", reason))

    evidences.append(
        Evidence(
            "random_seed",
            cfg.random_state,
            f"随机种子 = {cfg.random_state}，保证划分可复现。",
        )
    )

    indices = np.arange(n_total)
    try:
        train_rest_idx, test_idx = train_test_split(
            indices,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=stratify_y,
        )
    except ValueError as e:
        raise StratifyError(
            f"分层采样失败（阶段 1：分出测试集）。"
            f"原始样本数 {n_total}，test_size={cfg.test_size}，"
            f"任务类型={profile.target.task_type}。"
            f"原始错误：{e}。"
            "建议：增大数据集，或调小 test_size/validation_size。"
        ) from e

    if stratify:
        stratify_rest = y.iloc[train_rest_idx] if hasattr(y, "iloc") else y[train_rest_idx]  # type: ignore[index]
    else:
        stratify_rest = None

    try:
        train_idx, valid_idx = train_test_split(
            train_rest_idx,
            test_size=cfg.validation_size,
            random_state=cfg.random_state,
            stratify=stratify_rest,
        )
    except ValueError as e:
        raise StratifyError(
            f"分层采样失败（阶段 2：分出验证集）。"
            f"剩余样本数 {len(train_rest_idx)}，validation_size={cfg.validation_size}，"
            f"任务类型={profile.target.task_type}。"
            f"原始错误：{e}。"
            "建议：增大数据集，或调小 validation_size。"
        ) from e

    return np.sort(train_idx), np.sort(valid_idx), np.sort(test_idx)


def _split_group(
    X: pd.DataFrame,
    y: object,
    cfg: SplitConfig,
    evidences: list[Evidence],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按实体分组切分：同一实体只进入一个集合。"""
    column = cfg.group_column
    assert column is not None  # 由 SplitConfig 校验保证
    _require_dataframe(X, column, "group_column")

    groups_all = X[column].astype("string").fillna("<NA>").to_numpy()
    n_total = len(X)
    n_groups = len(pd.unique(groups_all))

    evidences.append(
        Evidence(
            "split_strategy",
            "group",
            f"按实体列 {column!r} 分组切分，共 {n_groups} 个实体，"
            "同一实体只会进入训练集、验证集或测试集中的某一个。",
        )
    )
    evidences.append(
        Evidence(
            "random_seed",
            cfg.random_state,
            f"随机种子 = {cfg.random_state}，保证划分可复现。",
        )
    )

    indices = np.arange(n_total)
    splitter_test = GroupShuffleSplit(
        n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state
    )
    try:
        train_rest_idx, test_idx = next(
            splitter_test.split(indices, groups=groups_all)
        )
    except ValueError as e:
        raise InvalidSplitError(
            f"分组切分失败（阶段 1：分出测试集）。实体数 {n_groups}，"
            f"test_size={cfg.test_size}。原始错误：{e}。"
            "建议：增加实体数，或调小 test_size。"
        ) from e

    splitter_valid = GroupShuffleSplit(
        n_splits=1, test_size=cfg.validation_size, random_state=cfg.random_state
    )
    try:
        train_idx, valid_idx = next(
            splitter_valid.split(train_rest_idx, groups=groups_all[train_rest_idx])
        )
    except ValueError as e:
        raise InvalidSplitError(
            f"分组切分失败（阶段 2：分出验证集）。剩余实体数 "
            f"{len(pd.unique(groups_all[train_rest_idx]))}，"
            f"validation_size={cfg.validation_size}。原始错误：{e}。"
            "建议：增加实体数，或调小 validation_size。"
        ) from e

    # 索引是相对 train_rest_idx 的位置，需映射回原始行号
    train_idx = train_rest_idx[train_idx]
    valid_idx = train_rest_idx[valid_idx]

    return np.sort(train_idx), np.sort(valid_idx), np.sort(test_idx)


def _split_temporal(
    X: pd.DataFrame,
    y: object,
    cfg: SplitConfig,
    evidences: list[Evidence],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按时间顺序切分：训练集最早、验证集居中、测试集最晚。"""
    column = cfg.time_column
    assert column is not None  # 由 SplitConfig 校验保证
    _require_dataframe(X, column, "time_column")
    parsed = _parse_time_column(X, column)

    n_total = len(X)
    order = np.argsort(parsed.to_numpy(), kind="stable")

    n_test = int(np.ceil(n_total * cfg.test_size))
    n_remain = n_total - n_test
    n_valid = int(np.ceil(n_remain * cfg.validation_size))
    n_train = n_remain - n_valid

    evidences.append(
        Evidence(
            "split_strategy",
            "temporal",
            f"按时间列 {column!r} 顺序切分：训练集最早、验证集居中、测试集最晚，"
            "避免用未来数据预测过去样本。",
        )
    )
    evidences.append(
        Evidence(
            "temporal_boundaries",
            f"train<valid<test",
            f"三集时间边界：训练集 {parsed.iloc[order[:n_train]].min()} ~ "
            f"{parsed.iloc[order[:n_train]].max()}；"
            f"验证集 {parsed.iloc[order[n_train:n_train + n_valid]].min()} ~ "
            f"{parsed.iloc[order[n_train:n_train + n_valid]].max()}；"
            f"测试集 {parsed.iloc[order[n_train + n_valid:]].min()} ~ "
            f"{parsed.iloc[order[n_train + n_valid:]].max()}。",
        )
    )

    train_idx = order[:n_train]
    valid_idx = order[n_train:n_train + n_valid]
    test_idx = order[n_train + n_valid:]

    return np.sort(train_idx), np.sort(valid_idx), np.sort(test_idx)


# ── 划分完整性审计 ───────────────────────────────────────────


def _row_hashes(df: pd.DataFrame, index: np.ndarray) -> set[int]:
    """对指定行子集计算逐行哈希集合，用于跨集合重复检测。"""
    subset = df.iloc[index].astype("string").fillna("<NA>")
    return set(pd.util.hash_pandas_object(subset, index=False).tolist())


def _set_check(
    result: SplitAuditResult,
    code: str,
    check: str,
    status: str,
    message: str,
) -> None:
    item = AuditCheckResult(code=code, check=check, status=status, result=message)  # type: ignore[arg-type]
    for index, existing in enumerate(result.check_results):
        if existing.code == code:
            result.check_results[index] = item
            return
    result.check_results.append(item)


def _set_check_from_findings(
    result: SplitAuditResult,
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


def audit_split(
    X: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    test_idx: np.ndarray,
    profile: DataProfile | None = None,
    config: SplitConfig | None = None,
) -> SplitAuditResult:
    """对既有切分做完整性审计（纯观测，不修改数据）。

    检查三项固定内容：跨集合重复样本、实体分组泄漏、时间顺序穿越。
    未声明的检查项记录为「未执行」，而不是谎报通过。
    """
    cfg = config if config is not None else SplitConfig()
    result = SplitAuditResult()
    indices = (train_idx, valid_idx, test_idx)
    set_names = ("训练集", "验证集", "测试集")

    if not cfg.enable_split_audit:
        return result

    is_dataframe = isinstance(X, pd.DataFrame)

    # ── 检查 1：跨集合重复样本 ──────────────────────────────
    if is_dataframe and len(X) > 0:
        n_rows = len(X)
        sampled = n_rows > cfg.duplicate_audit_max_rows
        if sampled:
            rng = np.random.default_rng(cfg.random_state)
            pick = rng.choice(
                n_rows, size=cfg.duplicate_audit_max_rows, replace=False
            )
            pick_set = set(pick.tolist())
            work_indices = tuple(
                np.array([i for i in idx if i in pick_set]) for idx in indices
            )
        else:
            work_indices = tuple(np.asarray(idx) for idx in indices)

        hash_sets = [_row_hashes(X, idx) for idx in work_indices]
        duplicate_count = 0
        for i in range(3):
            for j in range(i + 1, 3):
                duplicate_count += len(hash_sets[i] & hash_sets[j])

        result.duplicate_count = duplicate_count
        sample_note = (
            f"（数据超过 {cfg.duplicate_audit_max_rows} 行，本次抽样 "
            f"{cfg.duplicate_audit_max_rows} 行检测）"
            if sampled
            else ""
        )
        if duplicate_count:
            result.findings.append(
                AuditFinding(
                    code="cross_split_duplicates",
                    severity="critical",
                    title="跨集合重复样本",
                    evidence=f"发现 {duplicate_count} 个原始特征行同时出现在多个集合中{sample_note}。",
                    recommendation="在切分前去重，或让重复实体只进入一个集合，否则测试分数会被高估。",
                    details={"count": duplicate_count, "sampled": sampled},
                )
            )
            result.suggested_actions.append("对原始数据去重后重新切分")
            _set_check(
                result, "cross_split_duplicates", "跨集合重复样本", "发现风险",
                f"发现 {duplicate_count} 个原始特征行跨训练集、验证集或测试集重复{sample_note}。",
            )
        else:
            _set_check(
                result, "cross_split_duplicates", "跨集合重复样本", "通过",
                f"三个集合之间没有重复的原始特征行{sample_note}。",
            )
    else:
        _set_check(
            result, "cross_split_duplicates", "跨集合重复样本", "未执行",
            "X 不是 pandas DataFrame，无法按行比对原始特征是否跨集合重复。",
        )

    # ── 检查 2：实体分组泄漏 ────────────────────────────────
    group_column = cfg.group_column
    if not group_column:
        _set_check(
            result, "cross_split_groups", "实体分组泄漏", "未执行",
            "未声明 group_column，无法判断同一实体是否跨集合出现。"
            "若数据存在患者、用户或设备等实体依赖，建议声明该列。",
        )
    elif not is_dataframe or group_column not in X.columns:
        result.findings.append(
            AuditFinding(
                code="missing_group_column",
                severity="high",
                title="分组列不存在",
                evidence=f"声明的 group_column={group_column!r} 不在数据列中。",
                recommendation="检查列名拼写；或移除该声明以关闭分组泄漏检查。",
                columns=[group_column],
            )
        )
        _set_check(
            result, "cross_split_groups", "实体分组泄漏", "发现风险",
            f"声明的 group_column={group_column!r} 不在数据中，无法执行检查。",
        )
    else:
        groups = [
            set(X.iloc[idx][group_column].dropna().astype("string")) for idx in indices
        ]
        overlap_pairs = 0
        overlap_examples: list[str] = []
        for i in range(3):
            for j in range(i + 1, 3):
                shared = groups[i] & groups[j]
                overlap_pairs += len(shared)
                if shared and len(overlap_examples) < 5:
                    overlap_examples.extend(sorted(shared)[: 5 - len(overlap_examples)])
        result.group_overlap_count = overlap_pairs
        if overlap_pairs:
            result.findings.append(
                AuditFinding(
                    code="cross_split_groups",
                    severity="critical",
                    title="同一实体跨集合出现",
                    evidence=f"{group_column!r} 有 {overlap_pairs} 个实体跨集合出现，"
                    f"例如 {overlap_examples}。",
                    recommendation="改用 group 切分（split_strategy='group'），"
                    "并从模型输入中排除该实体列。",
                    columns=[group_column],
                    details={"count": overlap_pairs, "examples": overlap_examples},
                )
            )
            result.suggested_actions.append(
                f"将 split_strategy 设为 'group' 并保留 group_column={group_column!r}"
            )
            _set_check(
                result, "cross_split_groups", "实体分组泄漏", "发现风险",
                f"字段 {group_column!r} 有 {overlap_pairs} 个实体跨集合出现。",
            )
        else:
            _set_check(
                result, "cross_split_groups", "实体分组泄漏", "通过",
                f"字段 {group_column!r} 的实体未跨越集合边界。",
            )

    # ── 检查 3：时间顺序穿越 ────────────────────────────────
    time_column = cfg.time_column
    if not time_column:
        _set_check(
            result, "temporal_overlap", "时间穿越", "未执行",
            "未声明 time_column，无法判断训练、验证与测试的时间顺序是否合规。"
            "若数据存在时间依赖，建议声明该列。",
        )
    elif not is_dataframe or time_column not in X.columns:
        result.findings.append(
            AuditFinding(
                code="missing_time_column",
                severity="high",
                title="时间列不存在",
                evidence=f"声明的 time_column={time_column!r} 不在数据列中。",
                recommendation="检查列名拼写；或移除该声明以关闭时间穿越检查。",
                columns=[time_column],
            )
        )
        _set_check(
            result, "temporal_overlap", "时间穿越", "发现风险",
            f"声明的 time_column={time_column!r} 不在数据中，无法执行检查。",
        )
    else:
        time = pd.to_datetime(X[time_column], errors="coerce")
        n_invalid = int(time.isna().sum())
        if n_invalid:
            result.findings.append(
                AuditFinding(
                    code="invalid_time_values",
                    severity="medium",
                    title="时间列不可解析",
                    evidence=f"时间列 {time_column!r} 有 {n_invalid} 个空值或不可解析的时间。",
                    recommendation="先清洗时间列，再评估时间顺序是否合规。",
                    columns=[time_column],
                    details={"invalid_count": n_invalid},
                )
            )
            _set_check(
                result, "temporal_overlap", "时间穿越", "发现风险",
                f"字段 {time_column!r} 存在空值或不可解析的时间，无法完成时间顺序检查。",
            )
        else:
            ranges = [(time.iloc[idx].min(), time.iloc[idx].max()) for idx in indices]
            # 每个较早的集合必须先于下一个集合结束（用 valid 起点比 test 起点不够，
            # 因为验证集自身可能延伸到测试期）。
            ordered = bool(
                ranges[0][1] <= ranges[1][0] and ranges[1][1] <= ranges[2][0]
            )
            result.temporal_ordered = ordered
            readable = "；".join(
                f"{set_names[i]} {ranges[i][0]} ~ {ranges[i][1]}" for i in range(3)
            )
            if not ordered:
                result.findings.append(
                    AuditFinding(
                        code="temporal_overlap",
                        severity="critical",
                        title="时间顺序穿越",
                        evidence=f"三集时间区间为：{readable}，存在重叠或倒置。",
                        recommendation="把 split_strategy 设为 'temporal'，"
                        "确保训练数据严格早于验证和测试数据。",
                        columns=[time_column],
                    )
                )
                result.suggested_actions.append(
                    f"将 split_strategy 设为 'temporal' 并保留 time_column={time_column!r}"
                )
                _set_check(
                    result, "temporal_overlap", "时间穿越", "发现风险",
                    f"字段 {time_column!r} 的训练、验证和测试时间区间发生重叠或倒置。",
                )
            else:
                _set_check(
                    result, "temporal_overlap", "时间穿越", "通过",
                    f"字段 {time_column!r} 的时间顺序符合训练集 → 验证集 → 测试集。",
                )

    # ── 与画像联动的建议性检查 ──────────────────────────────
    if profile is not None and profile.audit is not None and not group_column:
        id_like = [
            f
            for f in profile.audit.findings
            if f.code == "identifier_like_column" and f.severity in {"high", "critical"}
        ]
        if id_like:
            columns = list(dict.fromkeys(c for f in id_like for c in f.columns))
            result.findings.append(
                AuditFinding(
                    code="potential_grouping_need",
                    severity="low",
                    title="疑似实体列未声明分组",
                    evidence=f"画像已把 {columns} 判定为高唯一性标识符列，"
                    "但本次未声明 group_column，无法校验实体是否跨集合。",
                    recommendation="若这些列代表患者、用户或设备等实体，"
                    "请声明 group_column 并改用 group 切分。",
                    columns=columns,
                )
            )

    result.model_info_questions = [
        "测试集是否代表要泛化到的人群、地点与时间段？",
        "是否存在实体、时间、空间或层级依赖，需要相应的分组切分？",
        "原始数据是否已去重，避免同一样本同时出现在训练与评估集合？",
    ]

    # 仅在用户确实声明了对应列时，才登记「列可用性」检查项
    if group_column:
        _set_check_from_findings(
            result, "missing_group_column", "分组列可用性", "声明的分组列存在且可用。"
        )
    if time_column:
        _set_check_from_findings(
            result, "missing_time_column", "时间列可用性", "声明的时间列存在且可用。"
        )

    return result


# ── 顶层入口 ─────────────────────────────────────────────────


def split_data(
    X: np.ndarray | object,
    y: np.ndarray | object,
    profile: DataProfile,
    config: SplitConfig | None = None,
) -> SplitResult:
    """按 test_size → validation_size 两阶段划分数据，并审计划分完整性。

    划分顺序：
      1. 先从原始数据中分出 test_size 比例的测试集。
      2. 再从剩余部分分出 validation_size 比例的验证集。
      3. 剩余部分为训练集。

    分类任务（binary / multiclass）默认使用分层采样（stratify）；
    回归任务（regression）默认使用纯随机采样。
    ``split_strategy`` 为 ``group`` 或 ``temporal`` 时改用实体感知或时序切分，
    此时 ``auto`` 之外的策略由 ``config.split_strategy`` 显式决定。

    Args:
        X: 特征数据（DataFrame 或 ndarray，长度与 y 一致）。
        y: 目标数据（Series 或 ndarray，长度与 X 一致）。
        profile: build_profile 输出的 DataProfile，需包含 target.task_type。
        config: 可选划分参数，默认使用 SplitConfig()。

    Returns:
        SplitResult：包含划分索引、切分策略与完整性审计结果。

    Raises:
        InvalidSplitError: 子集样本数不足 min_samples_split。
        StratifyError: 分层采样失败（如类别样本数不足）。
        SplitColumnMissingError: 声明了分组列/时间列但数据中不存在。
        TimeParseError: 时间列存在空值或不可解析的值。
    """
    cfg = config if config is not None else SplitConfig()

    n_total = len(X)  # type: ignore[arg-type]
    evidences: list[Evidence] = []

    # ── 解析实际策略 ─────────────────────────────────────────
    strategy = resolve_split_strategy(profile, cfg)
    is_classification = profile.target.task_type in ("binary", "multiclass")
    # 分层只在 stratified 策略下启用；random 策略即使分类任务也不分层
    stratify = is_classification and strategy == "stratified"
    # group / temporal 切分无法同时满足分层约束，此时以结构约束优先并说明
    if is_classification and strategy in ("group", "temporal"):
        evidences.append(
            Evidence(
                "stratify_tradeoff",
                strategy,
                f"策略为 {strategy}，为保证"
                + ("实体不跨集合" if strategy == "group" else "时间顺序不乱")
                + "，本次未使用分层采样。",
            )
        )

    if strategy == "group":
        train_idx, valid_idx, test_idx = _split_group(X, y, cfg, evidences)  # type: ignore[arg-type]
    elif strategy == "temporal":
        train_idx, valid_idx, test_idx = _split_temporal(X, y, cfg, evidences)  # type: ignore[arg-type]
    else:
        train_idx, valid_idx, test_idx = _split_random_or_stratified(
            X, y, profile, cfg, stratify=stratify, evidences=evidences
        )

    n_train = len(train_idx)
    n_valid = len(valid_idx)
    n_test = len(test_idx)

    # ── 子集样本数检查 ───────────────────────────────────────
    min_ok = True
    for name, n in [("训练集", n_train), ("验证集", n_valid), ("测试集", n_test)]:
        if n < cfg.min_samples_split:
            min_ok = False
            evidences.append(
                Evidence(
                    "min_samples_check",
                    f"{name}: {n} < {cfg.min_samples_split}",
                    f"{name}仅 {n} 个样本，低于最低要求 {cfg.min_samples_split} 个。"
                    f"可通过调小 min_samples_split 或增大数据集解决。",
                )
            )

    if not min_ok:
        raise InvalidSplitError(
            f"划分后子集样本数不足（要求每集 ≥ {cfg.min_samples_split} 个）。"
            f"训练集 {n_train} / 验证集 {n_valid} / 测试集 {n_test}。"
            "建议：增大数据集，调小 test_size/validation_size，"
            "或调小 SplitConfig.min_samples_split。"
        )

    # ── 记录比例证据 ─────────────────────────────────────────
    actual_test_ratio = n_test / n_total
    actual_valid_ratio = n_valid / n_total
    actual_train_ratio = n_train / n_total

    evidences.append(
        Evidence(
            "train_ratio",
            round(actual_train_ratio, 4),
            f"训练集 {n_train} 样本，占 {actual_train_ratio:.2%}",
        )
    )
    evidences.append(
        Evidence(
            "valid_ratio",
            round(actual_valid_ratio, 4),
            f"验证集 {n_valid} 样本，占 {actual_valid_ratio:.2%}",
        )
    )
    evidences.append(
        Evidence(
            "test_ratio",
            round(actual_test_ratio, 4),
            f"测试集 {n_test} 样本，占 {actual_test_ratio:.2%}",
        )
    )

    # ── 分类任务：类别分布对比（仅记录，不阻断） ────────────
    if is_classification:
        y_arr = y.values if hasattr(y, "values") else np.asarray(y)  # type: ignore[union-attr]
        dist_train = _compute_class_distribution(y_arr, train_idx)
        dist_valid = _compute_class_distribution(y_arr, valid_idx)
        dist_test = _compute_class_distribution(y_arr, test_idx)

        max_shift = _max_distribution_shift([dist_train, dist_valid, dist_test])
        shift_note = (
            f"三集间最大类别比例差异 = {max_shift:.4f}"
            + ("（差异较小，分布一致 ✅）" if max_shift < 0.05 else "（差异较大，建议检查 ⚠️）")
        )
        evidences.append(
            Evidence(
                "stratify_quality",
                round(max_shift, 4),
                shift_note,
            )
        )

    # ── 划分完整性审计（纯观测） ─────────────────────────────
    audit = audit_split(X, train_idx, valid_idx, test_idx, profile=profile, config=cfg)

    # ── 构建结果 ─────────────────────────────────────────────
    return SplitResult(
        train_indices=train_idx,
        valid_indices=valid_idx,
        test_indices=test_idx,
        test_size=cfg.test_size,
        validation_size=cfg.validation_size,
        random_state=cfg.random_state,
        stratify=stratify,
        n_train=n_train,
        n_valid=n_valid,
        n_test=n_test,
        train_ratio=actual_train_ratio,
        valid_ratio=actual_valid_ratio,
        test_ratio=actual_test_ratio,
        evidences=evidences,
        strategy=strategy,
        group_column=cfg.group_column,
        time_column=cfg.time_column,
        audit=audit,
    )


__all__ = [
    "split_data",
    "audit_split",
    "resolve_split_strategy",
    "SPLIT_STRATEGIES",
]
