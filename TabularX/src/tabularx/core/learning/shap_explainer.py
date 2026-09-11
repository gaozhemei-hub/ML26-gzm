"""SHAP 可解释性分析模块。

对训练好的模型执行 SHAP 解释，输出：
- beeswarm 图：展示每个特征对预测的影响分布
- 特征重要性排序柱状图
- 单个样本的 waterfall 解释

设计原则：
- shap 是可选依赖，惰性导入
- 优先使用 TreeExplainer（RF/XGBoost/LightGBM/CatBoost 均兼容）
- 失败时 fallback 到 KernelExplainer（模型无关）
- 所有绘图使用 Agg 后端，不依赖显示器
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from tabularx.exceptions.learning import ShapError
from tabularx.schema.learning import ModelSpec, ShapResult

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════
# 内部：重建模型（合并参数）
# ═══════════════════════════════════════════════════════════════


def _rebuild_model(
    spec: ModelSpec,
    best_params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Any:
    """用最佳参数重建模型并在训练集上训练。

    Args:
        spec: 模型规约（含工厂函数和默认参数）。
        best_params: 最佳参数（来自 ModelRunResult.best_params）。
        X_train: 训练集。
        y_train: 训练标签。

    Returns:
        训练好的模型实例。
    """
    params = dict(best_params)
    if spec.default_params:
        for k, v in spec.default_params.items():
            params.setdefault(k, v)
    model = spec.factory(**params)
    model.fit(X_train, y_train)
    return model


# ═══════════════════════════════════════════════════════════════
# 内部：计算 SHAP 值
# ═══════════════════════════════════════════════════════════════


def _compute_shap_values(
    model: Any,
    X_explain: pd.DataFrame,
    X_background: pd.DataFrame | None = None,
    max_explain_samples: int = 200,
    n_background: int = 100,
) -> tuple[Any, str, str]:
    """计算 SHAP 值。

    Args:
        model: 训练好的模型。
        X_explain: 要解释的数据。
        X_background: 背景数据（KernelExplainer 需要）。
        max_explain_samples: 解释样本数上限。
        n_background: KernelExplainer 背景采样数。

    Returns:
        (shap_values, explainer_type, error_or_empty)

        shap_values: 若成功则为 Explanation 对象或数组，否则为 None。
        explainer_type: "tree" / "kernel" / ""。
        error: 若失败则为错误信息字符串，否则为 ""。
    """
    try:
        import shap
    except ImportError as e:
        raise ShapError("SHAP 未安装。请执行：uv pip install shap") from e

    # 截断解释样本数
    if len(X_explain) > max_explain_samples:
        X_explain = X_explain.iloc[:max_explain_samples]

    # 尝试 TreeExplainer（树模型专用，精确且快）
    explainer_type = ""
    shap_values = None
    last_error: str | None = None

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_explain)
        explainer_type = "tree"
        return shap_values, explainer_type, ""
    except Exception as e:
        last_error = str(e)

    # Fallback: KernelExplainer（模型无关，较慢）
    if X_background is not None and len(X_background) > n_background:
        X_bg = X_background.sample(n=n_background, random_state=42)
    elif X_background is not None:
        X_bg = X_background
    else:
        X_bg = X_explain

    try:
        # 如果是分类，用 predict_proba；否则用 predict
        if hasattr(model, "predict_proba"):
            predict_fn = model.predict_proba
        else:
            predict_fn = model.predict

        explainer = shap.KernelExplainer(predict_fn, X_bg)
        shap_values = explainer(X_explain)
        explainer_type = "kernel"
        return shap_values, explainer_type, ""
    except Exception as e:
        last_error = f"TreeExplainer / KernelExplainer 均失败：{e} ({last_error})"

    return None, "", last_error or "SHAP 计算失败"


# ═══════════════════════════════════════════════════════════════
# 内部：提取特征重要性
# ═══════════════════════════════════════════════════════════════


def _extract_feature_importance(
    shap_values: Any,
    feature_names: list[str],
) -> dict[str, float]:
    """从 SHAP 值中提取特征重要性（平均 |SHAP|）。

    Args:
        shap_values: shap.Explanation 对象或数组。
        feature_names: 特征名列表。

    Returns:
        特征名 -> 平均 |SHAP| 值（按降序排列的字典）。
    """
    values = shap_values.values if hasattr(shap_values, "values") else shap_values

    # 多分类：values 形状为 (n_samples, n_features, n_classes)
    # 取所有类的 |SHAP| 均值
    if values.ndim == 3:
        # Shape: (samples, features, classes)
        abs_vals = np.abs(values).mean(axis=(0, 2))  # (features,)
    else:
        # Shape: (samples, features)
        abs_vals = np.abs(values).mean(axis=0)  # (features,)

    # 确保长度匹配
    n_features = min(len(feature_names), len(abs_vals))
    pairs = [(feature_names[i], float(abs_vals[i])) for i in range(n_features)]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return dict(pairs)


# ═══════════════════════════════════════════════════════════════
# 内部：生成图表
# ═══════════════════════════════════════════════════════════════


def _generate_plots(
    shap_values: Any,
    X_explain: pd.DataFrame,
    feature_names: list[str],
    output_dir: Path,
    model_name: str,
) -> dict[str, str]:
    """生成 SHAP 解释图。

    Args:
        shap_values: shap.Explanation 对象或数组。
        X_explain: 解释用的数据。
        feature_names: 特征名列表。
        output_dir: 图片保存目录。
        model_name: 模型名（用于文件名）。

    Returns:
        图片路径字典，如 {"shap_summary": "assets/model_shap_summary.png"}。
    """
    import shap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确保 matplotlib 使用 Agg 后端
    plt.switch_backend("Agg")

    # 准备特征名（截断到实际特征数）
    if hasattr(shap_values, "values"):
        n_actual = shap_values.values.shape[1]
    else:
        n_actual = shap_values.shape[1]
    feat_names = list(feature_names[:n_actual])

    artifact_paths: dict[str, str] = {}

    # ── 图 1：Summary（beeswarm）──
    try:
        # 多分类时 beeswarm 可能需要选取特定的类
        if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
            # 多分类：取平均 |SHAP| 最高的类画 summary
            abs_means = np.abs(shap_values.values).mean(axis=(0, 1))  # per class
            top_class = int(np.argmax(abs_means))
            shap_1d = shap_values[..., top_class]
        else:
            shap_1d = shap_values

        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_1d,
            X_explain,
            feature_names=feat_names,
            show=False,
        )
        plt.tight_layout()
        path = output_dir / f"{model_name}_shap_summary.png"
        fig.savefig(str(path), dpi=120, bbox_inches="tight")
        plt.close(fig)
        artifact_paths["shap_summary"] = f"assets/{path.name}"
    except Exception as e:
        plt.close("all")

    # ── 图 2：特征重要性柱状图 ──
    try:
        if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
            shap_1d = shap_values[..., top_class]
        else:
            shap_1d = shap_values

        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_1d,
            X_explain,
            feature_names=feat_names,
            plot_type="bar",
            show=False,
        )
        plt.tight_layout()
        path = output_dir / f"{model_name}_shap_importance.png"
        fig.savefig(str(path), dpi=120, bbox_inches="tight")
        plt.close(fig)
        artifact_paths["shap_importance"] = f"assets/{path.name}"
    except Exception as e:
        plt.close("all")

    # ── 图 3：Waterfall（单样本） ──
    try:
        # 取预测值最极端的样本（分类取最确定/最不确定，回归取残差最大）
        if hasattr(shap_values, "values"):
            raw_values = shap_values.values
        else:
            raw_values = shap_values

        if raw_values.ndim == 3:
            # 多分类：选 top_class 中 |SHAP| 总和最大的样本
            cls_vals = raw_values[:, :, top_class]
            score = np.abs(cls_vals).sum(axis=1)
            sample_idx = int(np.argmax(score))
            exp_single = shap_values[sample_idx, :, top_class]
        elif raw_values.ndim == 2:
            score = np.abs(raw_values).sum(axis=1)
            sample_idx = int(np.argmax(score))
            exp_single = shap_values[sample_idx]
        else:
            exp_single = shap_values[0]

        fig = plt.figure(figsize=(10, 5))
        shap.plots.waterfall(exp_single, max_display=10, show=False)
        plt.tight_layout()
        path = output_dir / f"{model_name}_shap_waterfall.png"
        fig.savefig(str(path), dpi=120, bbox_inches="tight")
        plt.close(fig)
        artifact_paths["shap_waterfall"] = f"assets/{path.name}"
    except Exception as e:
        plt.close("all")

    return artifact_paths


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════


def explain_model(
    spec: ModelSpec,
    best_params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_explain: pd.DataFrame,
    feature_names: list[str],
    task_type: str,
    output_dir: str | Path,
    model_name: str = "best_model",
    max_explain_samples: int = 200,
    n_background: int = 100,
) -> ShapResult:
    """对最优模型执行 SHAP 可解释性分析。

    步骤：
      1. 重建模型：合并 best_params + default_params → 工厂函数创建 → fit
      2. 惰性导入 shap，优先 TreeExplainer，失败则 fallback KernelExplainer
      3. 计算 SHAP 值（测试集上）
      4. 生成 3 张图：summary（beeswarm）、特征重要性柱状图、waterfall
      5. 提取特征重要性排序

    Args:
        spec: 模型规约（含工厂函数和默认参数）。
        best_params: 最佳参数（来自 ModelRunResult.best_params）。
        X_train: 训练数据（通常为 X_train + X_valid 合并）。
        y_train: 训练标签（通常为 y_train + y_valid 合并）。
        X_explain: 要解释的数据（通常为测试集）。
        feature_names: 特征名列表。
        task_type: 任务类型："binary" / "multiclass" / "regression"。
        output_dir: 图片保存目录。
        model_name: 模型名称（用于文件名）。
        max_explain_samples: 解释样本数上限（默认 200）。
        n_background: KernelExplainer 背景采样数（默认 100）。

    Returns:
        ShapResult 对象。

    Raises:
        ShapError: shap 未安装或所有解释器均失败。
    """
    output_path = Path(output_dir)

    # ── 1. 重建模型 ─────────────────────────────────────────
    try:
        model = _rebuild_model(spec, best_params, X_train, y_train)
    except Exception as e:
        return ShapResult(
            model_name=model_name,
            error=f"模型重建失败：{e}",
            prediction_type=task_type,
        )

    # ── 2. 计算 SHAP 值 ──────────────────────────────────────
    shap_values, explainer_type, error_msg = _compute_shap_values(
        model=model,
        X_explain=X_explain,
        X_background=X_train,
        max_explain_samples=max_explain_samples,
        n_background=n_background,
    )

    if shap_values is None or error_msg:
        return ShapResult(
            model_name=model_name,
            error=error_msg or "SHAP 计算失败",
            prediction_type=task_type,
        )

    # ── 3. 提取特征重要性 ────────────────────────────────────
    feature_importance = _extract_feature_importance(shap_values, feature_names)
    top_k = list(feature_importance.keys())

    # ── 4. 生成图表 ──────────────────────────────────────────
    artifact_paths = _generate_plots(
        shap_values=shap_values,
        X_explain=X_explain.iloc[: max(min(len(X_explain), max_explain_samples), 50)],
        feature_names=feature_names,
        output_dir=output_path,
        model_name=model_name,
    )

    return ShapResult(
        model_name=model_name,
        feature_importance=feature_importance,
        top_k_features=top_k,
        prediction_type=task_type,
        n_features=len(feature_names),
        artifact_paths=artifact_paths,
    )


__all__ = [
    "explain_model",
]