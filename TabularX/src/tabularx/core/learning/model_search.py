"""模型搜索编排——主入口模块。

职责：
1. 根据配置和任务类型确定要运行的模型列表
2. 多进程并行启动各模型的 Optuna 调参
3. 收集结果，按验证集指标选出最佳模型
4. 选择 Top-K 模型可用于后续 Voting/Stacking
5. 重训最佳模型（train+valid）并评估测试集
6. 输出 ModelSearchResult
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
) -> tuple[str, OptunaResult | None, dict[str, float] | None, str | None]:
    """在子进程中运行单个模型的调参与评估。

    此函数为模块顶层函数，确保可被 ProcessPoolExecutor pickle。

    Returns:
        (model_name, optuna_result, validation_metrics, error_message)
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
        )

        # 用最佳参数重新在训练集上训练，计算验证集完整指标
        params = dict(optuna_result.best_params)
        if spec.default_params:
            for k, v in spec.default_params.items():
                if k not in params:
                    params[k] = v

        model = spec.factory(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        y_prob = None
        if task_type != "regression" and spec.supports_predict_proba:
            try:
                y_prob = model.predict_proba(X_valid)
            except Exception:
                y_prob = None

        valid_metrics = compute_all_metrics(
            task_type=task_type,
            y_true=y_valid,
            y_pred=y_pred,
            y_prob=y_prob,
        )

        return spec.name, optuna_result, valid_metrics, None

    except Exception as e:
        return spec.name, None, None, str(e)


# ═══════════════════════════════════════════════════════════════
# 内部：最终评估（在 train+valid 上训练，在 test 上评估）
# ═══════════════════════════════════════════════════════════════


def _evaluate_on_test(
    spec: ModelSpec,
    best_params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str,
) -> dict[str, float] | None:
    """用最佳参数在 train+valid 上重训，在 test 上评估。

    Returns:
        测试集指标字典，失败返回 None。
    """
    try:
        # 合并训练集和验证集
        X_full = pd.concat([X_train, X_valid], axis=0)
        y_full = pd.concat([y_train, y_valid], axis=0)

        params = dict(best_params)
        if spec.default_params:
            for k, v in spec.default_params.items():
                if k not in params:
                    params[k] = v

        model = spec.factory(**params)
        model.fit(X_full, y_full)
        y_pred = model.predict(X_test)
        y_prob = None
        if task_type != "regression" and spec.supports_predict_proba:
            try:
                y_prob = model.predict_proba(X_test)
            except Exception:
                y_prob = None

        return compute_all_metrics(
            task_type=task_type,
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
        )
    except Exception:
        return None


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
    primary_metric, direction = resolve_primary_metric(
        task_type=task_type,
        primary_metric=cfg.primary_metric,
        profile=profile,
    )

    # ── 2. 确定模型列表 ────────────────────────────────────
    if cfg.models == ["auto"]:
        model_names = reg.list_models(task_type)
    else:
        model_names = cfg.models
        # 验证所有模型在指定任务类型下可用
        for name in model_names:
            try:
                reg.get(name, task_type)
            except ModelNotAvailableError as e:
                raise ModelNotAvailableError(
                    f"配置中指定的模型 {name!r} 在 {task_type} 任务中不可用。"
                    f"  {e}"
                ) from e

    if not model_names:
        raise ModelSearchError(
            f"在 {task_type} 任务中没有可用的模型。"
            f"  请确认已安装可选依赖。"
        )

    specs = [reg.get(name, task_type) for name in model_names]

    # ── 3. 并行调参 ────────────────────────────────────────
    model_results: list[ModelRunResult] = []
    start_time = time.time()

    # 单进程或小数据集时直接用主进程
    n_jobs = min(len(specs), cfg.n_jobs) if cfg.n_jobs > 0 else len(specs)

    if n_jobs <= 1:
        _results = []
        for spec in specs:
            name, opt_result, valid_metrics, error = _run_model_tuning(
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
            )
            _results.append((name, opt_result, valid_metrics, error))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(
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
                ): spec.name
                for spec in specs
            }
            _results = []
            for future in as_completed(futures):
                try:
                    _results.append(future.result())
                except Exception as e:
                    name = futures[future]
                    _results.append((name, None, None, str(e)))

    # ── 4. 组装 ModelRunResult ─────────────────────────────
    for name, opt_result, valid_metrics, error in _results:
        if opt_result is not None and valid_metrics is not None:
            model_results.append(
                ModelRunResult(
                    model_name=name,
                    task_type=task_type,
                    status="success",
                    best_params=opt_result.best_params,
                    n_trials=opt_result.n_trials,
                    optuna_duration_seconds=opt_result.duration_seconds,
                    validation_metrics=valid_metrics,
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
                    error=error or "未知错误",
                )
            )

    # ── 5. 按验证集指标排序 ────────────────────────────────
    succeeded = [r for r in model_results if r.status == "success"]
    reverse = direction == "maximize"
    succeeded.sort(
        key=lambda r: r.validation_metrics.get(primary_metric, 0.0),
        reverse=reverse,
    )

    # ── 6. 最佳模型测试集评估 ──────────────────────────────
    best_model_name: str | None = None
    best_model_params: dict[str, Any] | None = None
    best_test_metrics: dict[str, float] | None = None

    if succeeded:
        best_run = succeeded[0]
        best_model_name = best_run.model_name
        best_model_params = best_run.best_params
        best_run.validation_metrics  # 已记录

        if cfg.retrain_on_train_valid:
            best_spec = reg.get(best_model_name, task_type)
            best_test_metrics = _evaluate_on_test(
                spec=best_spec,
                best_params=best_run.best_params,
                X_train=preprocessed_data.X_train,
                y_train=preprocessed_data.y_train,
                X_valid=preprocessed_data.X_valid,
                y_valid=preprocessed_data.y_valid,
                X_test=preprocessed_data.X_test,
                y_test=preprocessed_data.y_test,
                task_type=task_type,
            )
            # 将测试指标写回到该模型的 result 中
            best_run.test_metrics = best_test_metrics

    elapsed = time.time() - start_time

    # ── 7. 构造最终结果 ────────────────────────────────────
    n_success = len([r for r in model_results if r.status == "success"])
    n_failed = len([r for r in model_results if r.status == "failed"])

    top_k = succeeded[: cfg.top_k_models] if cfg.top_k_models > 0 else succeeded

    return ModelSearchResult(
        task_type=task_type,
        primary_metric=primary_metric,
        optimization_direction=direction,
        model_results=model_results,
        best_model_name=best_model_name,
        best_model_params=best_model_params,
        best_validation_metrics=best_run.validation_metrics if succeeded else {},
        best_test_metrics=best_test_metrics,
        n_models_total=len(specs),
        n_models_succeeded=n_success,
        n_models_failed=n_failed,
    )


__all__ = [
    "search_models",
]