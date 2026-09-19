"""A standalone, test-set-safe model-search alternative for TabularX.

Unlike the current core search implementation, this module never passes test
data into the tuning or ranking loop.  It is intentionally kept outside
``src/tabularx`` so the team can review it as an independent change.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from tabularx.core.learning.metrics import compute_all_metrics, resolve_primary_metric
from tabularx.core.learning.model_registry import ModelRegistry, get_default_registry
from tabularx.core.learning.optuna_tuner import tune_model


@dataclass
class SafeModelRun:
    """One candidate evaluated only on train and validation partitions."""
    model_name: str
    best_params: dict[str, Any]
    train_metrics: dict[str, float]
    validation_metrics: dict[str, float]
    n_trials: int
    error: str | None = None


@dataclass
class SafeSearchResult:
    """Selection result with at most one final test-set evaluation."""
    primary_metric: str
    direction: str
    candidates: list[SafeModelRun] = field(default_factory=list)
    selected_model_name: str | None = None
    selected_params: dict[str, Any] = field(default_factory=dict)
    final_test_metrics: dict[str, float] | None = None
    final_model: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_metric": self.primary_metric,
            "direction": self.direction,
            "selected_model_name": self.selected_model_name,
            "selected_params": self.selected_params,
            "final_test_metrics": self.final_test_metrics,
            "candidates": [asdict(c) for c in self.candidates],
            "test_access_policy": "Only the validation-selected candidate is evaluated on test.",
        }


def _metrics(model, X: pd.DataFrame, y: pd.Series, task_type: str, supports_proba: bool) -> dict[str, float]:
    pred = model.predict(X)
    prob = None
    if task_type != "regression" and supports_proba:
        try:
            prob = model.predict_proba(X)
        except (AttributeError, ValueError):
            pass
    return compute_all_metrics(task_type=task_type, y_true=y, y_pred=pred, y_prob=prob)


def safe_search_models(
    preprocessed_data,
    profile,
    *,
    model_names: list[str] | None = None,
    n_trials: int = 10,
    random_state: int = 42,
    registry: ModelRegistry | None = None,
) -> SafeSearchResult:
    """Tune candidates on train/valid, select on valid, evaluate one final model.

    ``preprocessed_data`` is the existing TabularX ``PreprocessedData`` object.
    It is accepted by duck typing so this extension does not change its schema.
    """
    task_type = profile.target.task_type
    registry_task = "classification" if task_type in ("binary", "multiclass") else "regression"
    reg = registry or get_default_registry()
    metric, direction = resolve_primary_metric(task_type, "auto", profile)
    # Optional dependencies are isolated per candidate: unavailable libraries
    # become recorded candidate failures rather than aborting the experiment.
    names = model_names or ["random_forest", "xgboost", "lightgbm", "catboost"]
    result = SafeSearchResult(primary_metric=metric, direction=direction)
    specs: dict[str, Any] = {}

    for name in names:
        try:
            spec = reg.get(name, registry_task)
            tuned = tune_model(
                spec, preprocessed_data.X_train, preprocessed_data.y_train,
                preprocessed_data.X_valid, preprocessed_data.y_valid,
                task_type, metric, direction, n_trials=n_trials, random_state=random_state,
            )
            params = dict(spec.default_params or {})
            params.update(tuned.best_params)
            model = spec.factory(**params)
            model.fit(preprocessed_data.X_train, preprocessed_data.y_train)
            result.candidates.append(SafeModelRun(
                name, params,
                _metrics(model, preprocessed_data.X_train, preprocessed_data.y_train, task_type, spec.supports_predict_proba),
                _metrics(model, preprocessed_data.X_valid, preprocessed_data.y_valid, task_type, spec.supports_predict_proba),
                tuned.n_trials,
            ))
            specs[name] = spec
        except Exception as exc:  # Candidates fail independently, as in AutoML.
            result.candidates.append(SafeModelRun(name, {}, {}, {}, n_trials, error=str(exc)))

    succeeded = [c for c in result.candidates if c.error is None]
    if not succeeded:
        raise RuntimeError("没有候选模型成功完成安全搜索。")
    reverse = direction == "maximize"
    selected = sorted(succeeded, key=lambda c: c.validation_metrics[metric], reverse=reverse)[0]
    result.selected_model_name, result.selected_params = selected.model_name, selected.best_params

    # This is the sole test-set access in the module.
    spec = specs[selected.model_name]
    final_model = spec.factory(**selected.best_params)
    X_full = pd.concat([preprocessed_data.X_train, preprocessed_data.X_valid])
    y_full = pd.concat([preprocessed_data.y_train, preprocessed_data.y_valid])
    final_model.fit(X_full, y_full)
    result.final_model = final_model
    result.final_test_metrics = _metrics(final_model, preprocessed_data.X_test, preprocessed_data.y_test, task_type, spec.supports_predict_proba)
    return result


__all__ = ["SafeModelRun", "SafeSearchResult", "safe_search_models"]
