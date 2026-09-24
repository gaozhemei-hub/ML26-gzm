"""模型组合模块（Voting / Stacking）。

从 ModelSearchResult 中取出验证集 Top-K 模型，
使用 scikit-learn 的 VotingClassifier / VotingRegressor 构造组合模型。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    VotingClassifier,
    VotingRegressor,
)

from tabularx.core.learning.metrics import compute_all_metrics
from tabularx.core.learning.model_registry import ModelRegistry, get_default_registry
from tabularx.exceptions.learning import ModelSearchError
from tabularx.schema.learning import ModelRunResult, ModelSearchResult
from tabularx.schema.preprocess import PreprocessedData


def build_voting(
    search_result: ModelSearchResult,
    preprocessed_data: PreprocessedData,
    task_type: str,
    registry: ModelRegistry | None = None,
) -> ModelRunResult | None:
    """构建 Voting 验证集比较模型，不访问测试集，也不替换已选择的单模型。

    Args:
        search_result: 模型搜索结果（含 top_k 模型）。
        preprocessed_data: 预处理后的数据。
        task_type: 任务类型。
        registry: 可选模型注册表。

    Returns:
        ModelRunResult（若成功），None（若无足够模型）。
    """
    reg = registry if registry is not None else get_default_registry()

    # 映射任务类型到注册表使用的分类
    registry_task = "classification" if task_type in ("binary", "multiclass") else "regression"

    ranking = search_result.ranking
    if len(ranking) < 2:
        return None  # 模型不足，跳过组合

    try:
        # 为每个候选模型创建实例
        estimators: list[tuple[str, Any]] = []
        for run in ranking:
            spec = reg.get(run.model_name, registry_task)
            params = dict(run.best_params)
            if spec.default_params:
                for k, v in spec.default_params.items():
                    if k not in params:
                        params[k] = v
            model = spec.factory(**params)
            estimators.append((run.model_name, model))

        # 创建 Voting 模型
        if task_type == "regression":
            voting = VotingRegressor(estimators)
        else:
            voting = VotingClassifier(estimators, voting="soft")

        # 在训练集上训练
        voting.fit(preprocessed_data.X_train, preprocessed_data.y_train)

        # 验证集评估
        y_pred = voting.predict(preprocessed_data.X_valid)
        y_prob = None
        if task_type != "regression" and hasattr(voting, "predict_proba"):
            try:
                y_prob = voting.predict_proba(preprocessed_data.X_valid)
            except Exception:
                y_prob = None

        valid_metrics = compute_all_metrics(
            task_type=task_type,
            y_true=preprocessed_data.y_valid,
            y_pred=y_pred,
            y_prob=y_prob,
        )

        return ModelRunResult(
            model_name="voting",
            task_type=task_type,
            status="success",
            best_params={"estimators": [e[0] for e in estimators]},
            n_trials=0,
            validation_metrics=valid_metrics,
            test_metrics=None,
            note=f"Voting 组合：{', '.join(e[0] for e in estimators)}",
        )

    except Exception as e:
        return ModelRunResult(
            model_name="voting",
            task_type=task_type,
            status="failed",
            best_params={},
            n_trials=0,
            error=f"Voting 组合失败：{e}",
        )


__all__ = [
    "build_voting",
]