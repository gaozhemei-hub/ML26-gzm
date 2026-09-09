"""数据划分模块：按比例划分训练/验证/测试集。

设计哲学：
- Split 是实验设计（组织实验），不做数据变换。
- 输出只存索引和元信息，不存实际数据，避免内存重复。
- Preprocess 阶段通过 train_indices / valid_indices / test_indices
  从原始 X, y 中切片获取各子集数据。
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from tabularx.core.params import SplitConfig
from tabularx.exceptions.split import InvalidSplitError, StratifyError
from tabularx.schema.profile import DataProfile, Evidence
from tabularx.schema.split import SplitResult


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


def split_data(
    X: np.ndarray | object,
    y: np.ndarray | object,
    profile: DataProfile,
    config: SplitConfig | None = None,
) -> SplitResult:
    """按 test_size → validation_size 两阶段划分数据。

    划分顺序：
      1. 先从原始数据中分出 test_size 比例的测试集。
      2. 再从剩余部分分出 validation_size 比例的验证集。
      3. 剩余部分为训练集。

    分类任务（binary / multiclass）使用分层采样（stratify），
    回归任务（regression）使用纯随机采样。

    Args:
        X: 特征数据（DataFrame 或 ndarray，长度与 y 一致）。
        y: 目标数据（Series 或 ndarray，长度与 X 一致）。
        profile: build_profile 输出的 DataProfile，需包含 target.task_type。
        config: 可选划分参数，默认使用 SplitConfig()。

    Returns:
        SplitResult：包含划分索引和元信息。

    Raises:
        InvalidSplitError: test_size + validation_size 超过可接受范围，
                           或子集样本数不足 min_samples_split。
        StratifyError: 分层采样失败（如类别样本数不足）。
    """
    cfg = config if config is not None else SplitConfig()

    # ── 参数校验 ──────────────────────────────────────────────
    if cfg.test_size <= 0 or cfg.test_size >= 1:
        raise InvalidSplitError(
            f"test_size 必须在 (0, 1) 之间，收到 {cfg.test_size}。"
        )
    if cfg.validation_size <= 0 or cfg.validation_size >= 1:
        raise InvalidSplitError(
            f"validation_size 必须在 (0, 1) 之间，收到 {cfg.validation_size}。"
        )

    # 计算各子集实际占比
    test_ratio = cfg.test_size
    valid_ratio = cfg.validation_size * (1 - cfg.test_size)
    train_ratio = 1 - test_ratio - valid_ratio

    if train_ratio <= 0:
        raise InvalidSplitError(
            f"test_size ({cfg.test_size}) + "
            f"validation_size ({cfg.validation_size}) 占比过大，"
            f"导致训练集占比为 {train_ratio:.4f}（应为正数）。"
            "请减小 test_size 或 validation_size。"
        )

    n_total = len(X)
    evidences: list[Evidence] = []

    # ── 是否为分类任务（需要 stratify）──
    is_classification = profile.target.task_type in ("binary", "multiclass")
    stratify_y = y if is_classification else None

    if is_classification:
        evidences.append(
            Evidence(
                "stratify_reason",
                str(profile.target.task_type),
                f"目标任务类型为 {profile.target.task_type}，使用分层采样保证各集类别分布一致。",
            )
        )
    else:
        evidences.append(
            Evidence(
                "stratify_reason",
                "regression",
                "目标任务类型为回归，使用纯随机采样。",
            )
        )

    evidences.append(
        Evidence(
            "random_seed",
            cfg.random_state,
            f"随机种子 = {cfg.random_state}，保证划分可复现。",
        )
    )

    # ── 阶段 1：分测试集 ──────────────────────────────────────
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

    # ── 阶段 2：从剩余部分分验证集 ──────────────────────────
    if is_classification:
        stratify_rest = y.iloc[train_rest_idx] if hasattr(y, "iloc") else y[train_rest_idx]
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

    # ── 排序（保持原始顺序） ──────────────────────────────────
    train_idx = np.sort(train_idx)
    valid_idx = np.sort(valid_idx)
    test_idx = np.sort(test_idx)

    n_train = len(train_idx)
    n_valid = len(valid_idx)
    n_test = len(test_idx)

    # ── 阶段 3：子集样本数检查 ────────────────────────────────
    min_ok = True
    for name, n in [("训练集", n_train), ("验证集", n_valid), ("测试集", n_test)]:
        if n < cfg.min_samples_split:
            min_ok = False
            evidences.append(
                Evidence(
                    f"min_samples_check",
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
        y_arr = y.values if hasattr(y, "values") else np.asarray(y)
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

    # ── 构建结果 ─────────────────────────────────────────────
    return SplitResult(
        train_indices=train_idx,
        valid_indices=valid_idx,
        test_indices=test_idx,
        test_size=cfg.test_size,
        validation_size=cfg.validation_size,
        random_state=cfg.random_state,
        stratify=is_classification,
        n_train=n_train,
        n_valid=n_valid,
        n_test=n_test,
        train_ratio=actual_train_ratio,
        valid_ratio=actual_valid_ratio,
        test_ratio=actual_test_ratio,
        evidences=evidences,
    )


__all__ = [
    "split_data",
]