"""Run the shared MLJAR–TabularX benchmark protocol for one dataset.

The script keeps one external 20% holdout, uses it only after both tools have
finished model selection, and writes machine-readable metrics plus an artifact
inventory.  Dataset-specific choices are declared in ``DATASETS`` so that the
same protocol is reused across all runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from fnmatch import fnmatch
from html import escape
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[2]
TABULARX_SRC = ROOT / "TabularX-project" / "src"
if str(TABULARX_SRC) not in sys.path:
    sys.path.insert(0, str(TABULARX_SRC))

import tabularx as tbx  # noqa: E402
from supervised.automl import AutoML  # noqa: E402


SEED = 42
ALGORITHMS_TBX = ["random_forest", "xgboost", "lightgbm", "catboost"]
ALGORITHMS_MLJAR = ["Random Forest", "Xgboost", "LightGBM", "CatBoost"]

DATASETS = {
    "adult": {
        "title": "Adult Income",
        "task": "classification",
        "task_cn": "二分类",
        "source": "OpenML dataset 1590（adult）",
        "description": "Adult Income 数据集包含 48,842 条人口统计记录和 14 个输入特征，目标是预测年收入是否超过 50,000 美元。",
        "target": "class",
        "sensitive": "sex",
        "privileged": "Male",
        "underprivileged": "Female",
        "drop": ["sex"],
        "openml_id": 1590,
    },
    "breast-cancer": {
        "title": "Breast Cancer Wisconsin",
        "task": "classification",
        "task_cn": "二分类",
        "source": "UCI Breast Cancer Wisconsin（仓库固定副本）",
        "description": "Breast Cancer Wisconsin 数据集包含 569 个肿瘤样本及 30 项测量特征，目标是区分恶性与良性肿瘤。",
        "target": "diagnosis",
        "drop": ["id"],
        "csv": "breast_cancer_wisconsin/data.csv",
    },
    "titanic": {
        "title": "Titanic",
        "task": "classification",
        "task_cn": "二分类",
        "source": "Kaggle Titanic（仓库固定副本）",
        "description": "Titanic 数据集包含 891 名乘客的舱位、年龄和家庭关系等信息，目标是预测乘客是否生还。",
        "target": "Survived",
        "sensitive": "Sex",
        "privileged": "male",
        "underprivileged": "female",
        "drop": ["PassengerId", "Name", "Sex", "Ticket", "Cabin"],
        "csv": "Titanic/train.csv",
    },
    "iris": {
        "title": "Iris",
        "task": "classification",
        "task_cn": "三分类",
        "source": "UCI Iris（仓库固定副本）",
        "description": "Iris 数据集包含 150 个鸢尾花样本及 4 项形态测量特征，目标是识别三个鸢尾花类别。",
        "target": "class",
        "drop": [],
        "csv": "iris/data.csv",
    },
    "california-housing": {
        "title": "California Housing",
        "task": "regression",
        "task_cn": "回归",
        "source": "California Housing（仓库固定副本）",
        "description": "California Housing 数据集包含 20,640 个街区的住房与人口统计特征，目标是预测街区房价中位数。",
        "target": "MedHouseVal",
        "drop": [],
        "csv": "housing_california/california_housing.csv",
    },
}


def datasets_root() -> Path:
    configured = os.environ.get("TABULARX_DATASETS_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    default = ROOT.parent / "datasets-for-start"
    if not default.exists():
        raise FileNotFoundError("未找到数据仓库；请设置 TABULARX_DATASETS_ROOT")
    return default


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        series = result[column]
        if isinstance(series.dtype, pd.CategoricalDtype):
            result[column] = series.astype(object)
        if result[column].dtype == object:
            result[column] = result[column].map(
                lambda value: value.decode().strip() if isinstance(value, bytes) else value
            )
            result[column] = result[column].replace({"?": np.nan, " ?": np.nan})
    return result


def load_dataset(config: dict) -> tuple[pd.DataFrame, pd.Series, dict]:
    if "openml_id" in config:
        bundle = fetch_openml(
            data_id=config["openml_id"],
            as_frame=True,
            data_home=str(Path.home() / "scikit_learn_data"),
            parser="auto",
        )
        frame = normalize_frame(bundle.frame)
        source_id = {"openml_dataset_id": config["openml_id"]}
    else:
        path = datasets_root() / config["csv"]
        frame = normalize_frame(pd.read_csv(path))
        source_id = {
            "repository_commit": "17bab1c32cdf57faf16ca949f37c7f54b1de7423",
            "repository_path": config["csv"],
        }
    target = config["target"]
    if target not in frame:
        raise KeyError(f"目标列不存在: {target}")
    y_raw = pd.Series(frame.pop(target), name=target).reset_index(drop=True)
    return frame.reset_index(drop=True), y_raw, source_id


def encode_target(y_raw: pd.Series, task: str) -> tuple[pd.Series, dict | None]:
    if task == "regression":
        return pd.to_numeric(y_raw, errors="raise").astype(float).rename("target"), None
    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(y_raw.astype("string")), name="target")
    mapping = {str(label): int(index) for index, label in enumerate(encoder.classes_)}
    return y, mapping


def fingerprint(X: pd.DataFrame, y_raw: pd.Series) -> str:
    frame = pd.concat(
        [X.astype("string"), y_raw.astype("string").rename("target_raw")], axis=1
    )
    values = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(values).hexdigest()


def matching(root: Path, patterns: list[str]) -> list[str]:
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and any(fnmatch(path.name.lower(), pattern.lower()) for pattern in patterns)
    )


def classification_metrics(
    name: str,
    y_test: pd.Series,
    y_all: pd.Series,
    prediction: np.ndarray,
    probability: np.ndarray,
) -> dict:
    row = {
        "工具": name,
        "Accuracy": accuracy_score(y_test, prediction),
        "Balanced Accuracy": balanced_accuracy_score(y_test, prediction),
        "Macro-F1": f1_score(y_test, prediction, average="macro"),
        "混淆矩阵": confusion_matrix(y_test, prediction).tolist(),
    }
    try:
        if y_all.nunique() == 2:
            score = probability[:, 1] if probability.ndim == 2 else probability
            row["ROC-AUC"] = roc_auc_score(y_test, score)
        else:
            row["ROC-AUC"] = roc_auc_score(
                y_test, probability, multi_class="ovr", average="macro"
            )
    except Exception:
        row["ROC-AUC"] = np.nan
    return row


def regression_metrics(name: str, y_test: pd.Series, prediction: np.ndarray) -> dict:
    return {
        "工具": name,
        "MAE": mean_absolute_error(y_test, prediction),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, prediction))),
        "R2": r2_score(y_test, prediction),
    }


def build_artifact_matrix(
    tbx_run_dir: Path,
    mljar_dir: Path,
    task: str,
    sensitive: str | None,
) -> pd.DataFrame:
    checks = [
        ("最佳模型及可重载文件", ["pipeline.joblib"], ["*.lightgbm", "*.catboost", "*.random_forest", "*.xgboost.json", "ensemble.json"]),
        ("候选模型与验证分数", ["model_scores.json"], ["leaderboard.csv", "params.json"]),
        ("预处理记录", ["preprocess.html"], ["README.md"]),
        ("特征重要性", ["*shap_importance*"], ["*importance*"]),
        ("SHAP", ["*shap_summary*"], ["*shap_summary*"]),
        ("ALE", ["ale*.png", "*_ale_*.png"], ["ale*.png", "*_ale_*.png"]),
        ("学习曲线或调参轨迹", ["*trial_trajectory*"], ["*learning_curve*"]),
        ("混淆矩阵", ["*confusion_matrix*"], ["*confusion_matrix*"]),
        ("公平性", ["fairness.json"], ["*fairness*"]),
        ("泄露、划分与评估协议审计", ["audit.json", "model_search_audit.json"], ["*leakage*", "*audit*"]),
        ("漂移与群体覆盖", ["audit.json"], ["*drift*", "*coverage*"]),
        ("重载一致性", ["reload_check.json"], ["*reload*"]),
        ("交付状态与失败原因", ["delivery.json"], ["*delivery*"]),
        ("原生 HTML 报告", ["report.html"], ["*.html"]),
    ]
    rows = []
    for label, tbx_patterns, mljar_patterns in checks:
        tbx_files = matching(tbx_run_dir, tbx_patterns)
        mljar_files = matching(mljar_dir, mljar_patterns)
        not_applicable = (label == "公平性" and not sensitive) or (
            label == "混淆矩阵" and task == "regression"
        )
        rows.append(
            {
                "工件": label,
                "TabularX": "生成并核验" if tbx_files else ("不适用" if not_applicable else "未观测"),
                "MLJAR": "生成并核验" if mljar_files else ("不适用" if not_applicable else "未观测"),
                "TabularX 证据": "；".join(tbx_files[:3]),
                "MLJAR 证据": "；".join(mljar_files[:3]),
            }
        )
    return pd.DataFrame(rows)


def write_html(
    output: Path,
    config: dict,
    metrics: pd.DataFrame,
    artifacts: pd.DataFrame,
    tbx_best: str,
    mljar_best: str,
    tbx_seconds: float,
    mljar_seconds: float,
) -> Path:
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><style>
body{{font-family:"Microsoft YaHei",sans-serif;max-width:1120px;margin:32px auto;line-height:1.65;color:#243042}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #ccd5e0;padding:7px;vertical-align:top}}th{{background:#edf3f8}}
.note{{padding:12px;background:#fff8dc;border-left:4px solid #d9a400}}</style>
<h1>MLJAR 与 TabularX 完整工具包对比</h1>
<p><strong>数据集：{escape(config['title'])}（{escape(config['source'])}，{escape(config['task_cn'])}）</strong>。{escape(config['description'])}</p>
<p class="note">双方使用相同开发集和外部测试集；性能只描述本次固定配置，核心比较对象是实际生成的工件。</p>
<h2>1. 外部测试集结果</h2>{metrics.to_html(index=False, float_format=lambda value: f"{value:.4f}")}
<h2>2. 工件盘点</h2>{artifacts.to_html(index=False, escape=True)}
<h2>3. 对比结果</h2><ul>
<li><strong>TabularX：</strong>将测试隔离、可信审计、模型重载核验和交付状态纳入同一次运行。</li>
<li><strong>MLJAR：</strong>Explain 模式提供逐模型说明、学习曲线、混淆矩阵和特征重要性。</li>
<li>TabularX 最佳模型：{escape(tbx_best)}；MLJAR 最佳模型：{escape(mljar_best)}。</li>
</ul><h2>4. 运行信息</h2><p>TabularX：{tbx_seconds:.1f} 秒；MLJAR：{mljar_seconds:.1f} 秒；随机种子 {SEED}。</p>
<p>“未观测”只表示本次固定配置的运行目录中没有找到相应工件，不外推为软件在任何条件下均不支持。</p></html>'''
    path = output / "comparison_report.html"
    path.write_text(html, encoding="utf-8")
    return path


def run(dataset_key: str) -> Path:
    if dataset_key not in DATASETS:
        raise KeyError(f"未知数据集: {dataset_key}")
    config = DATASETS[dataset_key]
    output = ROOT / "benchmark-output" / dataset_key
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib-cache"))
    os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(output / "joblib-temp"))

    X, y_raw, source_id = load_dataset(config)
    y, label_mapping = encode_target(y_raw, config["task"])
    sensitive = config.get("sensitive")
    sensitive_columns = [sensitive] if sensitive else []
    sensitive_data = X[[sensitive]].copy() if sensitive else None
    X_tbx = X.copy()
    X_model = X.drop(columns=config.get("drop", []), errors="ignore")
    forbidden = list(config.get("drop", []))

    indices = np.arange(len(X))
    stratify = y if config["task"] == "classification" else None
    dev_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=SEED, stratify=stratify
    )
    X_dev_tbx = X_tbx.iloc[dev_idx].reset_index(drop=True)
    X_test_tbx = X_tbx.iloc[test_idx].reset_index(drop=True)
    X_dev = X_model.iloc[dev_idx].reset_index(drop=True)
    X_test = X_model.iloc[test_idx].reset_index(drop=True)
    y_dev = y.iloc[dev_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    S_dev = sensitive_data.iloc[dev_idx].reset_index(drop=True) if sensitive else None

    shared = output / "shared-evidence"
    shared.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(shared / "split_indices.npz", dev=dev_idx, test=test_idx)
    manifest = {
        **source_id,
        "dataset": config["title"],
        "task": config["task_cn"],
        "rows": len(X),
        "original_features": X.shape[1],
        "model_input_features": X_model.shape[1],
        "dev_rows": len(dev_idx),
        "test_rows": len(test_idx),
        "seed": SEED,
        "dataset_sha256": fingerprint(X, y_raw),
        "label_mapping": label_mapping,
        "forbidden_features": forbidden,
        "sensitive_columns": sensitive_columns,
        "external_test_policy": "locked_until_final_models_selected",
    }
    (shared / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    ae = tbx.AutoEstimator(
        task=config["task"], output_dir=output / "tabularx", pdf_report=False
    )
    ae.param.model_search.set_param("models", ALGORITHMS_TBX)
    ae.param.model_search.set_param("n_trials", 3)
    ae.param.model_search.set_param("n_jobs", 1)
    ae.param.model_search.set_param("top_k_models", 2)
    ae.param.model_search.set_param("random_state", SEED)
    ae.param.split.set_param("random_state", SEED)
    ae.param.profile.forbidden_features = forbidden
    ae.param.preprocess.apply_audit_drops = "high"
    if config["task"] == "regression":
        ae.param.preprocess.high_cardinality_encoding = "frequency"
        ae.param.preprocess.low_cardinality_encoding = "label"

    started = time.time()
    ae.fit(X_dev_tbx, y_dev, sensitive_features=sensitive_columns)
    tbx_seconds = time.time() - started
    tbx_run_dir = Path(ae.output_dir)
    tbx_pipeline = joblib.load(tbx_run_dir / "pipeline.joblib")
    tbx_prediction = np.asarray(tbx_pipeline.predict(X_test_tbx))
    tbx_probability = (
        np.asarray(tbx_pipeline.predict_proba(X_test_tbx))
        if config["task"] == "classification"
        else None
    )

    mljar_dir = output / "mljar"
    mljar_kwargs = {
        "mode": "Explain",
        "results_path": str(mljar_dir),
        "algorithms": ALGORITHMS_MLJAR,
        "total_time_limit": 600,
        "train_ensemble": True,
        "random_state": SEED,
        "explain_level": 2,
        "eval_metric": (
            "auc"
            if config["task"] == "classification" and y.nunique() == 2
            else ("logloss" if config["task"] == "classification" else "rmse")
        ),
    }
    if sensitive and y.nunique() == 2:
        mljar_kwargs.update(
            fairness_metric="demographic_parity_ratio",
            fairness_threshold=0.8,
            privileged_groups=[{sensitive: config["privileged"]}],
            underprivileged_groups=[{sensitive: config["underprivileged"]}],
        )
    automl = AutoML(**mljar_kwargs)
    started = time.time()
    if sensitive and y.nunique() == 2:
        automl.fit(X_dev, y_dev, sensitive_features=S_dev)
    else:
        automl.fit(X_dev, y_dev)
    mljar_seconds = time.time() - started
    (mljar_dir / "benchmark_runtime.json").write_text(
        json.dumps({"first_full_run_seconds": mljar_seconds}, indent=2), encoding="utf-8"
    )
    mljar_prediction = np.asarray(automl.predict(X_test))
    mljar_probability = (
        np.asarray(automl.predict_proba(X_test))
        if config["task"] == "classification"
        else None
    )
    try:
        automl.report_structured(format="json")
    except Exception as error:
        (mljar_dir / "structured_report_error.txt").write_text(
            f"{type(error).__name__}: {error}", encoding="utf-8"
        )

    if config["task"] == "classification":
        metrics = pd.DataFrame(
            [
                classification_metrics("TabularX", y_test, y, tbx_prediction, tbx_probability),
                classification_metrics("MLJAR", y_test, y, mljar_prediction, mljar_probability),
            ]
        )
    else:
        metrics = pd.DataFrame(
            [
                regression_metrics("TabularX", y_test, tbx_prediction),
                regression_metrics("MLJAR", y_test, mljar_prediction),
            ]
        )
    metrics.to_json(
        output / "external_test_metrics.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )

    artifacts = build_artifact_matrix(
        tbx_run_dir, mljar_dir, config["task"], sensitive
    )
    artifacts.to_csv(output / "artifact_matrix.csv", index=False, encoding="utf-8-sig")
    mljar_params = json.loads((mljar_dir / "params.json").read_text(encoding="utf-8"))
    mljar_best = str(mljar_params.get("best_model", "未记录"))
    write_html(
        output,
        config,
        metrics,
        artifacts,
        str(ae.result.searched.best_model_name),
        mljar_best,
        tbx_seconds,
        mljar_seconds,
    )
    run_summary = {
        "dataset": config["title"],
        "tabularx_best_model": ae.result.searched.best_model_name,
        "mljar_best_model": mljar_best,
        "tabularx_seconds": tbx_seconds,
        "mljar_seconds": mljar_seconds,
        "tabularx_run_dir": str(tbx_run_dir.relative_to(ROOT)),
        "mljar_run_dir": str(mljar_dir.relative_to(ROOT)),
    }
    (output / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(DATASETS))
    args = parser.parse_args()
    run(args.dataset)


if __name__ == "__main__":
    main()
