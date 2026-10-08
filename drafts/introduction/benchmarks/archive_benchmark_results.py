"""Archive benchmark evidence and build the ten-dataset synthesis report."""

from __future__ import annotations

import json
import shutil
from html import escape
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "benchmark-output"
ARCHIVE_ROOT = ROOT / "docs" / "validation" / "mljar-tabularx-comparison"

NEW_DATASETS = {
    "adult": "Adult Income",
    "breast-cancer": "Breast Cancer Wisconsin",
    "titanic": "Titanic",
    "iris": "Iris",
    "california-housing": "California Housing",
}

ALL_DATASETS = {
    "bank-marketing": ("Bank Marketing", "二分类"),
    "credit-scoring": ("Credit Scoring", "二分类"),
    "employee-attrition": ("Employee Attrition", "二分类"),
    "wine": ("Wine", "三分类"),
    "house-prices": ("House Prices", "回归"),
    "adult": ("Adult Income", "二分类"),
    "breast-cancer": ("Breast Cancer Wisconsin", "二分类"),
    "titanic": ("Titanic", "二分类"),
    "iris": ("Iris", "三分类"),
    "california-housing": ("California Housing", "回归"),
}


def archive_new_results() -> None:
    for key, title in NEW_DATASETS.items():
        source = OUTPUT_ROOT / key
        target = ARCHIVE_ROOT / key
        target.mkdir(parents=True, exist_ok=True)
        files = {
            "external_test_metrics.json": "external_test_metrics.json",
            "artifact_matrix.csv": "artifact_matrix.csv",
            "comparison_report.html": "comparison_report.html",
            "comparison_report.pdf": f"对比报告-{title}.pdf",
            f"{title}.tex": f"{title}.tex",
            "run_summary.json": "run_summary.json",
        }
        for source_name, target_name in files.items():
            path = source / source_name
            if not path.exists():
                raise FileNotFoundError(path)
            shutil.copy2(path, target / target_name)


def normalize_artifact_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    if "TabularX 状态" in frame:
        rename["TabularX 状态"] = "TabularX"
    if "MLJAR 状态" in frame:
        rename["MLJAR 状态"] = "MLJAR"
    normalized = frame.rename(columns=rename)
    for column in ["TabularX", "MLJAR"]:
        normalized[column] = normalized[column].replace(
            {"已生成并验证": "生成并核验", "未观察到": "未观测"}
        )
    normalized["工件"] = normalized["工件"].replace(
        {
            "最佳模型及可重载流水线": "最佳模型及可重载文件",
            "学习曲线": "学习曲线或调参轨迹",
            "泄露/划分/评估协议审计": "泄露、划分与评估协议审计",
            "HTML 报告": "原生 HTML 报告",
        }
    )
    return normalized


def build_summary_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    artifact_rows = []
    for key, (title, task) in ALL_DATASETS.items():
        folder = ARCHIVE_ROOT / key
        metrics = pd.read_json(folder / "external_test_metrics.json")
        for _, row in metrics.iterrows():
            record = {"数据集": title, "任务": task, "工具": row["工具"]}
            for column in ["Accuracy", "Balanced Accuracy", "Macro-F1", "ROC-AUC", "MAE", "RMSE", "R2"]:
                if column in row and pd.notna(row[column]):
                    record[column] = row[column]
            metric_rows.append(record)

        artifacts = normalize_artifact_columns(pd.read_csv(folder / "artifact_matrix.csv"))
        for _, row in artifacts.iterrows():
            artifact_rows.append(
                {
                    "数据集": title,
                    "任务": task,
                    "工件": row["工件"],
                    "TabularX": row["TabularX"],
                    "MLJAR": row["MLJAR"],
                }
            )

    metrics = pd.DataFrame(metric_rows)
    artifacts = pd.DataFrame(artifact_rows)
    status_rows = []
    for tool in ["TabularX", "MLJAR"]:
        for status in ["生成并核验", "已生成并验证", "未观测", "未观察到", "不适用"]:
            count = int((artifacts[tool] == status).sum())
            if count:
                status_rows.append({"工具": tool, "状态": status, "数量": count})
    status = pd.DataFrame(status_rows)
    return metrics, artifacts, status


def build_html(metrics: pd.DataFrame, artifacts: pd.DataFrame, status: pd.DataFrame) -> str:
    classification = metrics.loc[
        metrics["任务"] != "回归",
        ["数据集", "任务", "工具", "Accuracy", "Balanced Accuracy", "Macro-F1", "ROC-AUC"],
    ].copy()
    regression = metrics.loc[
        metrics["任务"] == "回归", ["数据集", "任务", "工具", "MAE", "RMSE", "R2"]
    ].copy()
    artifact_generated = artifacts.copy()
    for tool in ["TabularX", "MLJAR"]:
        artifact_generated[tool] = artifact_generated[tool].isin(["生成并核验", "已生成并验证"])
    cross = (
        artifact_generated.groupby("工件")[["TabularX", "MLJAR"]]
        .sum()
        .astype(int)
        .reset_index()
        .rename(columns={"TabularX": "TabularX 观察次数", "MLJAR": "MLJAR 观察次数"})
    )
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><style>
@page{{size:A4;margin:18mm}}body{{font-family:"Microsoft YaHei",sans-serif;max-width:1120px;margin:32px auto;line-height:1.7;color:#243042}}
h1,h2{{color:#123f4a}}table{{border-collapse:collapse;width:100%;font-size:12px;margin:12px 0 26px}}th,td{{border:1px solid #cbd5df;padding:7px;vertical-align:top}}th{{background:#e8f1f3}}.status-table{{break-inside:avoid;page-break-inside:avoid}}
.note{{padding:12px 16px;background:#fff7d8;border-left:4px solid #c99000}}.conclusion{{padding:16px;background:#e8f4f3;border-left:5px solid #087f8c}}</style>
<h1>MLJAR 与 TabularX 十数据集完整工具包对比报告</h1>
<p class="note">本报告汇总十个公开表格数据集上的固定配置实验。性能指标用于描述本次运行，不构成严格等预算的普遍性能排名；“未观测”仅指本次运行目录中未找到对应工件。</p>
<h2>1. 实验设计</h2><p>实验覆盖八个分类任务和两个回归任务。每个数据集按随机种子 42 划分为 80% 开发集与 20% 外部测试集；外部测试集仅在两种工具完成训练与模型选择后评估一次。候选算法统一为随机森林、XGBoost、LightGBM 与 CatBoost。</p>
<h2>2. 分类任务外部测试结果</h2>{classification.to_html(index=False, float_format=lambda value: f"{value:.4f}", na_rep="—")}
<h2>3. 回归任务外部测试结果</h2>{regression.to_html(index=False, float_format=lambda value: f"{value:.4f}", na_rep="—")}
<h2>4. 跨数据集工件观察次数</h2>{cross.to_html(index=False)}
<h2>5. 工件状态汇总</h2>{status.to_html(index=False, classes="status-table")}
<h2>6. 结果边界</h2><ul>
<li>两种工具采用相同外部测试集与算法集合，但搜索机制和公平性缓解流程并非严格等价，因此不将性能差异解释为工具的普遍优劣。</li>
<li>Adult 与 Titanic 使用具有明确语义的敏感字段进行分组审计；Bank Marketing 与 Credit Scoring 的年龄分组属于预先声明阈值下的探索性审计。</li>
<li>公平性差异用于描述群体结果，不单独证明模型公平，也不替代场景规范与因果分析。</li>
</ul>
<div class="conclusion"><strong>综合结论：</strong>两种工具均能够完成主流模型训练并生成常规解释工件。跨数据集结果显示，TabularX 的主要增量集中在测试隔离、运行审计、模型重载核验以及工件状态与失败原因的统一交付；该结论指向证据交付流程的工程差异，不等同于预测性能优势。</div>
</html>'''


def build_tex(metrics: pd.DataFrame, artifacts: pd.DataFrame) -> str:
    generated = artifacts.copy()
    for tool in ["TabularX", "MLJAR"]:
        generated[tool] = generated[tool].isin(["生成并核验", "已生成并验证"]).astype(int)
    cross = generated.groupby("工件")[["TabularX", "MLJAR"]].sum().reset_index()
    rows = []
    for _, row in cross.iterrows():
        label = str(row["工件"]).replace("_", r"\_").replace("&", r"\&")
        rows.append(f"{label} & {int(row['TabularX'])} & {int(row['MLJAR'])} \\\\")
    table = "\n".join(rows)
    return rf'''% !TeX encoding = UTF-8
% !TeX program = xelatex
\documentclass[degree=bachelor,language=chinese,fontset=fandol]{{thuthesis}}
\usepackage{{booktabs,longtable,array,xcolor,hyperref,enumitem}}
\hypersetup{{colorlinks=true,linkcolor=black,urlcolor=blue}}
\setlist[itemize]{{leftmargin=2em,itemsep=0.25em}}
\begin{{document}}
\begin{{titlepage}}\centering\vspace*{{2.3cm}}
{{\zihao{{1}}\bfseries MLJAR 与 TabularX 完整工具包对比\par}}
\vspace{{1.2cm}}{{\zihao{{3}}十数据集综合实验报告\par}}
\vspace{{2.2cm}}{{\zihao{{4}}TabularX 课程项目\par}}
\vfill{{\zihao{{-4}}实验日期：2026 年 9 月 30 日\par}}\vspace{{1.2cm}}
\end{{titlepage}}
\frontmatter\tableofcontents\mainmatter
\chapter{{实验设计}}
本实验覆盖八个分类任务和两个回归任务。每个数据集按随机种子 42 划分为 80\% 开发集与 20\% 外部测试集；外部测试集仅在模型锁定后评估一次。两种工具使用随机森林、XGBoost、LightGBM 与 CatBoost。

性能指标描述本次固定配置，不构成严格等预算的性能排名。主要比较对象是运行结束后实际生成并能够核对的解释性、审计性与复现性工件。

\chapter{{跨数据集工件结果}}
\begin{{longtable}}{{p{{0.55\textwidth}}cc}}
\toprule 工件 & TabularX & MLJAR \\\\ \midrule\endfirsthead
\toprule 工件 & TabularX & MLJAR \\\\ \midrule\endhead
{table}
\bottomrule\end{{longtable}}

\chapter{{结果边界}}
\begin{{itemize}}
\item 两种工具的搜索机制与公平性缓解流程并非严格等价，因此不将性能差异解释为工具的普遍优劣。
\item Adult 与 Titanic 使用具有明确语义的敏感字段；Bank Marketing 与 Credit Scoring 的年龄分组属于预先声明阈值下的探索性审计。
\item 公平性差异用于描述群体结果，不单独证明模型公平，也不替代场景规范与因果分析。
\end{{itemize}}

\chapter{{结论}}
两种工具均完成主流模型训练并生成常规解释工件。跨数据集结果显示，TabularX 的主要增量集中在测试隔离、运行审计、模型重载核验以及工件状态与失败原因的统一交付。该结论描述证据交付流程的工程差异，不等同于预测性能优势。
\end{{document}}
'''


def build_notebooks() -> None:
    for key, title in NEW_DATASETS.items():
        folder = ROOT / "introduction" / "benchmarks" / key
        folder.mkdir(parents=True, exist_ok=True)
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"# MLJAR 与 TabularX 完整工具包对比：{title}\n",
                        "\n",
                        "本 Notebook 对已完成的固定配置运行进行复核，并提供使用同一协议重新运行的入口。\n",
                    ],
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "from pathlib import Path\n",
                        "import json\n",
                        "import pandas as pd\n",
                        "ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'TabularX-project').exists())\n",
                        f"OUTPUT = ROOT / 'benchmark-output' / '{key}'\n",
                        "pd.DataFrame(json.loads((OUTPUT / 'external_test_metrics.json').read_text(encoding='utf-8')))\n",
                    ],
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "pd.read_csv(OUTPUT / 'artifact_matrix.csv')\n",
                    ],
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## 重新运行\n",
                        "\n",
                        f"在项目根目录运行 `python drafts/introduction/benchmarks/run_toolkit_benchmark.py {key}`。完整运行会重新训练两种工具并覆盖对应的 `benchmark-output/{key}` 目录。\n",
                    ],
                },
            ],
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.12"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        path = folder / f"{key.replace('-', '_')}_mljar_vs_tabularx.ipynb"
        if not path.exists():
            path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    archive_new_results()
    metrics, artifacts, status = build_summary_tables()
    metrics.to_csv(ARCHIVE_ROOT / "ten_dataset_metrics.csv", index=False, encoding="utf-8-sig")
    artifacts.to_csv(ARCHIVE_ROOT / "ten_dataset_artifacts.csv", index=False, encoding="utf-8-sig")
    status.to_csv(ARCHIVE_ROOT / "ten_dataset_status.csv", index=False, encoding="utf-8-sig")
    (ARCHIVE_ROOT / "十数据集综合报告.html").write_text(
        build_html(metrics, artifacts, status), encoding="utf-8"
    )
    (ARCHIVE_ROOT / "十数据集综合报告.tex").write_text(
        build_tex(metrics, artifacts), encoding="utf-8"
    )
    build_notebooks()
    print(ARCHIVE_ROOT)


if __name__ == "__main__":
    main()
