"""数据预处理模块：根据 Profile 的观测事实做数据变换。

设计哲学：
- 所有变换**在训练集上拟合**，在训练/验证/测试集上应用，避免数据泄露。
- 输出不持有任何 sklearn 对象，而是变换后的 DataFrames + 变换记录，
  确保可序列化。
- 每列的变换动作由 profile.columns[i].tags 驱动。
"""

from __future__ import annotations

import warnings
from typing import Any

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
from tabularx.exceptions.preprocess import (
    PreprocessError,
    TextColumnNotSupportedError,
    TransformError,
)
from tabularx.schema.preprocess import PreprocessedData, TransformRecord
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


# ── 顶层函数 ─────────────────────────────────────────────────


def preprocess(
    X: pd.DataFrame,
    y: pd.Series,
    profile: DataProfile,
    split_result: SplitResult,
    config: PreprocessConfig | None = None,
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

    Returns:
        PreprocessedData：变换后的三集数据 + 变换记录。

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
    n_features_in = len(all_columns)
    transforms: list[TransformRecord] = []
    dropped_columns: list[str] = []
    feature_names_out: list[str] = []

    # ── 步骤 2：逐列处理 ────────────────────────────────────
    for col_name in all_columns:
        col_type = _resolve_col_type(profile, col_name)

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
                continue

        # ── 数值列 ──────────────────────────────────────────
        if col_type == "numeric":
            col_train_s = X_train[col_name]
            col_valid_s = X_valid[col_name]
            col_test_s = X_test[col_name]

            out_train, out_valid, out_test, actions, note = _transform_numeric(
                col_train_s, col_valid_s, col_test_s, col_name, profile, cfg
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

    n_features_out = len(feature_names_out)

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
    )


__all__ = [
    "preprocess",
]