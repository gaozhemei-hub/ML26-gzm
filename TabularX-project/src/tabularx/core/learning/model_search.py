"""模型搜索编排——主入口模块。

职责：
1. 根据配置和任务类型确定要运行的模型列表
2. 多进程并行启动各模型的 Optuna 调参
3. 收集结果，按验证集指标选出最佳模型
4. 选择 Top-K 模型可用于后续 Voting/Stacking
5. 输出 ModelSearchResult
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd

from tabularx.core.learning.metrics import (
    compute_all_metrics,
    resolve_primary_metric,
)
from tabularx.core.learning.model_registry import (
    ModelRegistry,
    get_default_registry,
)
from tabularx.core.learning.optuna_tuner import tune_model
from tabularx.core.params import ModelSearchConfig
from tabularx.exceptions.learning import ModelSearchError, ModelNotAvailableError
from tabularx.schema.learning import (
    ModelRunResult,
    ModelSearchResult,
    ModelSpec,
    OptunaResult,
)
from tabularx.schema.preprocess import PreprocessedData
from tabularx.schema.profile import DataProfile


# ═══════════════════════════════════════════════════════════════
# 内部：单个模型在子进程中运行的入口
# ═══════════════════════════════════════════════════════════════


def _run_model_tuning(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    task_type: str,
    metric_name: str,
    direction: str,
    n_trials: int,
    random_state: int,
    device: str = "cpu",
    fixed_params: dict[str, Any] | None = None,
    search_space_overrides: dict[str, dict[str, Any]] | None = None,
) -> tuple[str, OptunaResult | None, dict[str, float] | None,
           dict[str, float] | None, dict[str, float] | None, str | None]:
    """在子进程中运行单个模型的调参与评估。

    步骤:
      1) Optuna 在 train 上调参，在 valid 上评估
      2) 用最佳参数在 train 上训练，计算 train + valid 指标
      3) 返回验证指标；不接收或访问测试数据

    此函数为模块顶层函数，确保可被 ProcessPoolExecutor pickle。

    Args:
        device: 计算设备，"cpu"（默认）| "gpu" | "auto"。

    Returns:
        (model_name, optuna_result, train_metrics, valid_metrics, test_metrics, error)
    """
    try:
        optuna_result = tune_model(
            spec=spec,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            task_type=task_type,
            metric_name=metric_name,
            direction=direction,
            n_trials=n_trials,
            random_state=random_state,
            study_name=f"{spec.name}_{task_type}",
            fixed_params=fixed_params,
            search_space_overrides=search_space_overrides,
        )

        # 构造完整参数
        params = dict(optuna_result.best_params)
        if spec.default_params:
            for k, v in spec.default_params.items():
                params.setdefault(k, v)

        # ── 根据 device 注入硬件参数（覆盖搜索空间和默认参数）──
        if device == "cpu":
            if spec.name == "xgboost":
                params["device"] = "cpu"
                params["tree_method"] = "hist"
            # LightGBM 和 CatBoost 默认就是 CPU，无需额外设置
        elif device == "gpu":
            if spec.name == "xgboost":
                params["device"] = "cuda"
                params["tree_method"] = "hist"
            elif spec.name == "lightgbm":
                params["device"] = "gpu"
            elif spec.name == "catboost":
                params["task_type"] = "GPU"
        # "auto" — 不做任何注入，让各库自行决定

        # ── 阶段 A：在 train 上训练，评估 train + valid ──────
        model = spec.factory(**params)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_train_prob = None
        if task_type != "regression" and spec.supports_predict_proba:
            try:
                y_train_prob = model.predict_proba(X_train)
            except Exception:
                y_train_prob = None
        train_metrics = compute_all_metrics(
            task_type=task_type,
            y_true=y_train,
            y_pred=y_train_pred,
            y_prob=y_train_prob,
        )

        y_valid_pred = model.predict(X_valid)
        y_valid_prob = None
        if task_type != "regression" and spec.supports_predict_proba:
            try:
                y_valid_prob = model.predict_proba(X_valid)
            except Exception:
                y_valid_prob = None
        valid_metrics = compute_all_metrics(
            task_type=task_type,
            y_true=y_valid,
            y_pred=y_valid_pred,
            y_prob=y_valid_prob,
        )

        # Final held-out evaluation belongs to the locked selected model only.
        return spec.name, optuna_result, train_metrics, valid_metrics, None, None

    except Exception as e:
        err_msg = str(e)
        # 检测 CUDA 驱动不兼容错误
        if spec.name == "xgboost" and (
            "CUDA" in err_msg
            or "cudaError" in err_msg
            or "cuda_dr_utils" in err_msg
            or "cudaErrorInsufficientDriver" in err_msg
        ):
            return spec.name, None, None, None, None, (
                "XGBoost 跳过：当前环境的 NVIDIA 驱动与 XGBoost 不兼容。\n"
                "  - 如果有 NVIDIA 显卡，请安装正确的 CUDA 驱动\n"
                "  - 如果没有显卡，请安装纯 CPU 版：uv pip install --no-binary xgboost xgboost"
            )
        return spec.name, None, None, None, None, str(e)


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════


def search_models(
    preprocessed_data: PreprocessedData,
    profile: DataProfile,
    config: ModelSearchConfig | None = None,
    registry: ModelRegistry | None = None,
) -> ModelSearchResult:
    """执行模型搜索的主入口。

    Args:
        preprocessed_data: 预处理后的数据（含 train/valid/test）。
        profile: Profile 阶段输出，用于获取 task_type 和 tags。
        config: 可选搜索配置，默认使用 ModelSearchConfig()。
        registry: 可选模型注册表，默认使用全局单例。

    Returns:
        ModelSearchResult 包含所有模型搜索结果。

    Raises:
        ModelNotAvailableError: 配置指定的模型不可用。
        ModelSearchError: 搜索过程整体失败（如无可用模型）。
    """
    # ── 1. 解析配置 ────────────────────────────────────────
    cfg = config if config is not None else ModelSearchConfig()
    reg = registry if registry is not None else get_default_registry()

    task_type = profile.target.task_type  # "binary" / "multiclass" / "regression"
    registry_task = "classification" if task_type in ("binary", "multiclass") else "regression"
    primary_metric, direction = resolve_primary_metric(
        task_type=task_type,
        primary_metric=cfg.primary_metric,
        profile=profile,
    )

    # ── 2. 确定模型列表 ────────────────────────────────────
    if cfg.models == ["auto"]:
        model_names = reg.list_models(registry_task)
    else:
        model_names = cfg.models
        for name in model_names:
            try:
                reg.get(name, registry_task)
            except ModelNotAvailableError as e:
                raise ModelNotAvailableError(
                    f"配置中指定的模型 {name!r} 在 {task_type} 任务中不可用。  {e}"
                ) from e

    if not model_names:
        raise ModelSearchError(f"在 {task_type} 任务中没有可用的模型。")

    specs = [reg.get(name, registry_task) for name in model_names]

    # ── 3. 并行调参 ────────────────────────────────────────
    model_results: list[ModelRunResult] = []
    start_time = time.time()

    n_jobs = min(len(specs), cfg.n_jobs) if cfg.n_jobs > 0 else len(specs)

    if n_jobs <= 1:
        _results = []
        for spec in specs:
            per_fixed = cfg.fixed_params.get(spec.name, {})
            per_overrides = cfg.search_space_overrides.get(spec.name, {})
            name, opt, train_m, valid_m, test_m, err = _run_model_tuning(
                spec=spec,
                X_train=preprocessed_data.X_train,
                y_train=preprocessed_data.y_train,
                X_valid=preprocessed_data.X_valid,
                y_valid=preprocessed_data.y_valid,
                task_type=task_type,
                metric_name=primary_metric,
                direction=direction,
                n_trials=cfg.n_trials,
                random_state=cfg.random_state,
                device=cfg.device,
                fixed_params=per_fixed,
                search_space_overrides=per_overrides,
            )
            _results.append((name, opt, train_m, valid_m, test_m, err))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {}
            for spec in specs:
                per_fixed = cfg.fixed_params.get(spec.name, {})
                per_overrides = cfg.search_space_overrides.get(spec.name, {})
                fut = executor.submit(
                    _run_model_tuning,
                    spec=spec,
                    X_train=preprocessed_data.X_train,
                    y_train=preprocessed_data.y_train,
                    X_valid=preprocessed_data.X_valid,
                    y_valid=preprocessed_data.y_valid,
                    task_type=task_type,
                    metric_name=primary_metric,
                    direction=direction,
                    n_trials=cfg.n_trials,
                    random_state=cfg.random_state,
                    device=cfg.device,
                    fixed_params=per_fixed,
                    search_space_overrides=per_overrides,
                )
                futures[fut] = spec.name
            _results = []
            for future in as_completed(futures):
                try:
                    _results.append(future.result())
                except Exception as e:
                    name = futures[future]
                    _results.append((name, None, None, None, None, str(e)))

    # ── 4. 组装 ModelRunResult ─────────────────────────────
    for name, opt, train_m, valid_m, test_m, err in _results:
        if opt is not None and valid_m is not None:
            model_results.append(
                ModelRunResult(
                    model_name=name,
                    task_type=task_type,
                    status="success",
                    best_params=opt.best_params,
                    n_trials=opt.n_trials,
                    optuna_duration_seconds=opt.duration_seconds,
                    optuna_trials=opt.trials,
                    train_metrics=train_m or {},
                    validation_metrics=valid_m,
                    test_metrics=test_m,
                )
            )
        else:
            model_results.append(
                ModelRunResult(
                    model_name=name,
                    task_type=task_type,
                    status="failed",
                    best_params={},
                    n_trials=cfg.n_trials,
                    validation_metrics={},
                    error=err or "未知错误",
                )
            )

    # ── 5. 验证集选模；测试分数不得参与排序 ────────────────
    succeeded = [r for r in model_results if r.status == "success"
                 and primary_metric in r.validation_metrics]
    reverse = direction == "maximize"
    succeeded.sort(
        key=lambda r: r.validation_metrics[primary_metric],
        reverse=reverse,
    )

    best_model_name: str | None = None
    best_model_params: dict[str, Any] | None = None
    best_test_metrics: dict[str, float] | None = None

    if succeeded:
        best_run = succeeded[0]
        best_model_name = best_run.model_name
        best_model_params = best_run.best_params
        best_test_metrics = best_run.test_metrics

    elapsed = time.time() - start_time

    n_success = len([r for r in model_results if r.status == "success"])
    n_failed = len([r for r in model_results if r.status == "failed"])

    return ModelSearchResult(
        task_type=task_type,
        primary_metric=primary_metric,
        optimization_direction=direction,
        model_results=model_results,
        best_model_name=best_model_name,
        best_model_params=best_model_params,
        best_train_metrics=best_run.train_metrics if succeeded else {},
        best_validation_metrics=best_run.validation_metrics if succeeded else {},
        best_test_metrics=best_test_metrics,
        n_models_total=len(specs),
        n_models_succeeded=n_success,
        n_models_failed=n_failed,
    )


__all__ = [
    "search_models",
]
