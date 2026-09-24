"""Optuna 超参数调优模块。

对单个模型执行 Optuna 调参：
- 每次 trial 在训练集上训练，在验证集上评分
- 支持声明式 ParamDef 搜索空间 + 配置层注入的 fixed_params / overrides
- 返回 OptunaResult（含最佳参数、全部 trial 记录）
- 支持设置随机种子保证可复现
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import optuna
import pandas as pd

from tabularx.core.learning.metrics import (
    compute_all_metrics,
    resolve_primary_metric,
)
from tabularx.exceptions.learning import OptunaError
from tabularx.schema.learning import ModelSpec, OptunaResult, TrialRecord


def _objective(
    trial: optuna.Trial,
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    metric_name: str,
    task_type: str,
    fixed_params: dict[str, Any] | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
    device: str = "cpu",
) -> float:
    """Optuna 目标函数：一次 trial 的训练+评估。

    基于声明式 ParamDef 遍历搜索空间定义：
    - 若参数在 fixed_params 中 → 直接赋值，不调用 trial.suggest_*()
    - 否则按 ParamDef 类型调用对应的 trial.suggest_*()
    - overrides 中的字段优先于 ParamDef 的默认值。

    Args:
        trial: Optuna trial 对象。
        spec: 模型规约（含 search_space_defs）。
        X_train/y_train: 训练集。
        X_valid/y_valid: 验证集。
        metric_name: 优化目标指标名。
        task_type: "regression" / "binary" / "multiclass"。
        fixed_params: 固定参数字典（跳过搜索）。
        overrides: 搜索范围覆盖字典，key 为参数名，value 为覆盖字段。
        device: 计算设备，"cpu"（默认）| "gpu" | "auto"。

    Returns:
        验证集上的指标值。
    """
    # 1. 遍历 ParamDef 生成参数
    fixed = fixed_params or {}
    override_map = overrides or {}
    params: dict[str, Any] = {}

    for pdef in spec.search_space_defs:
        # 固定参数跳过搜索
        if pdef.name in fixed:
            params[pdef.name] = fixed[pdef.name]
            continue

        # 应用覆盖
        o = override_map.get(pdef.name, {})

        if pdef.type == "int":
            lo = o.get("low", pdef.low)
            hi = o.get("high", pdef.high)
            step = o.get("step", pdef.step)
            if step is not None:
                params[pdef.name] = trial.suggest_int(
                    pdef.name, lo, hi, step=step
                )
            else:
                params[pdef.name] = trial.suggest_int(pdef.name, lo, hi)
        elif pdef.type == "float":
            lo = o.get("low", pdef.low)
            hi = o.get("high", pdef.high)
            log = o.get("log", pdef.log)
            params[pdef.name] = trial.suggest_float(
                pdef.name, lo, hi, log=log
            )
        elif pdef.type == "categorical":
            choices = o.get("choices", pdef.choices)
            params[pdef.name] = trial.suggest_categorical(
                pdef.name, choices
            )

    # 2. 注入默认参数（固定参数已由 fixed_params 处理，这里只补未设置的）
    if spec.default_params:
        for k, v in spec.default_params.items():
            if k not in params:
                params[k] = v

    # 3. 根据 device 注入硬件参数（覆盖搜索空间和默认参数）
    if device == "cpu":
        if spec.name == "xgboost":
            params["device"] = "cpu"
            params["tree_method"] = "hist"
    elif device == "gpu":
        if spec.name == "xgboost":
            params["device"] = "cuda"
            params["tree_method"] = "hist"
        elif spec.name == "lightgbm":
            params["device"] = "gpu"
        elif spec.name == "catboost":
            params["task_type"] = "GPU"
    # "auto" — 不做任何注入，让各库自行决定

    # 4. 创建并训练模型
    model = spec.factory(**params)
    model.fit(X_train, y_train)

    # 5. 验证集预测
    y_pred = model.predict(X_valid)
    y_prob = None
    if task_type != "regression" and spec.supports_predict_proba:
        try:
            y_prob = model.predict_proba(X_valid)
        except Exception:
            y_prob = None

    # 6. 计算指标
    metrics = compute_all_metrics(
        task_type=task_type,
        y_true=y_valid,
        y_pred=y_pred,
        y_prob=y_prob,
    )

    return float(metrics.get(metric_name, 0.0))


def tune_model(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    task_type: str,
    metric_name: str,
    direction: str,
    n_trials: int = 50,
    random_state: int = 42,
    study_name: str | None = None,
    fixed_params: dict[str, Any] | None = None,
    search_space_overrides: dict[str, dict[str, Any]] | None = None,
    device: str = "cpu",
) -> OptunaResult:
    """对单个模型执行 Optuna 调参。

    支持通过 fixed_params 固定某些参数（不参与搜索），
    通过 search_space_overrides 覆盖搜索边界。

    Args:
        spec: 模型规约。
        X_train/y_train: 训练集。
        X_valid/y_valid: 验证集。
        task_type: 任务类型。
        metric_name: 优化指标名。
        direction: "maximize" 或 "minimize"。
        n_trials: trial 数量。
        random_state: 随机种子。
        study_name: 可选 study 名称。
        fixed_params: 固定参数（不参与搜索）。
        search_space_overrides: 搜索范围覆盖。
        device: 计算设备，"cpu"（默认）| "gpu" | "auto"。

    Returns:
        OptunaResult 包含最佳参数和 trial 记录。

    Raises:
        OptunaError: 调参过程失败。
    """
    start_time = time.time()

    try:
        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )

        # 静默日志，避免 trial 过程打印过多
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study.optimize(
            lambda trial: _objective(
                trial=trial,
                spec=spec,
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                metric_name=metric_name,
                task_type=task_type,
                fixed_params=fixed_params,
                overrides=search_space_overrides,
                device=device,
            ),
            n_trials=n_trials,
            show_progress_bar=False,
        )

    except Exception as e:
        raise OptunaError(
            f"{spec.name} 调参失败：{e}"
        ) from e

    elapsed = time.time() - start_time

    # 将固定参数合并到 best_params 中（固定参数不经过 study.best_params）
    best_params = dict(study.best_params)
    if fixed_params:
        for k, v in fixed_params.items():
            best_params.setdefault(k, v)

    # 构造 trial 记录
    trials: list[TrialRecord] = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            trials.append(
                TrialRecord(
                    trial_number=t.number,
                    params=t.params,
                    value=t.value if t.value is not None else 0.0,
                    state=str(t.state.name),
                    duration_seconds=t.duration.total_seconds()
                    if t.duration
                    else None,
                )
            )

    return OptunaResult(
        model_name=spec.name,
        task_type=task_type,
        best_params=best_params,
        best_value=float(study.best_value),
        n_trials=n_trials,
        trials=trials,
        study_name=study_name,
        duration_seconds=round(elapsed, 2),
    )


__all__ = [
    "tune_model",
]