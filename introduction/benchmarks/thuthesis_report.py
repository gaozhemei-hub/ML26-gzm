from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "benchmark-output"
NOTEBOOK_ROOT = ROOT / "introduction" / "benchmarks"

DATASETS = {
    "credit-scoring": {
        "title": "Credit Scoring",
        "source": "OpenML dataset 31（credit-g）",
        "task": "二分类",
        "description": "Credit Scoring 数据集包含 1,000 名信贷申请人的 20 个金融与人口统计特征，目标是预测申请人的信用风险为 good 或 bad。",
        "test_size": 200,
        "tbx_model": "CatBoost",
        "mljar_model": "Ensemble",
        "fairness": "按 age\_25\_plus 年龄分组生成公平性与群体覆盖结果；该审计分组不作为模型输入。",
    },
    "employee-attrition": {
        "title": "Employee Attrition",
        "source": "OpenML dataset 43893（ibm-employee-attrition）",
        "task": "二分类",
        "description": "Employee Attrition 数据集包含 1,470 名员工的 34 个个人与工作特征，目标是预测员工是否离职。",
        "test_size": 294,
        "tbx_model": "LightGBM",
        "mljar_model": "Ensemble",
        "fairness": "按 Gender 生成公平性与群体覆盖结果；Gender、EmployeeNumber 等声明列不进入模型。",
    },
    "wine": {
        "title": "Wine",
        "source": "scikit-learn load\_wine（源自 UCI Wine）",
        "task": "三分类",
        "description": "Wine 数据集包含 178 个葡萄酒样本的 13 项化学测量特征，目标是判断样本属于三个葡萄酒品种中的哪一类。",
        "test_size": 36,
        "tbx_model": "Random Forest",
        "mljar_model": "ModelFramework",
        "fairness": "数据集没有在本实验中声明敏感属性，因此公平性工件标记为不适用。",
    },
    "house-prices": {
        "title": "House Prices",
        "source": "OpenML dataset 42165（house\_prices）",
        "task": "回归",
        "description": "House Prices 数据集包含 1,460 套住宅的 80 个原始描述特征，目标是预测住宅销售价格 SalePrice；标识列 Id 不进入模型。",
        "test_size": 292,
        "tbx_model": "LightGBM",
        "mljar_model": "Ensemble",
        "fairness": "数据集没有在本实验中声明敏感属性，因此公平性工件标记为不适用。",
    },
}

DESCRIPTIONS = {
    "最佳模型及可重载文件": "两者均保存最终模型；TabularX 另行核验模型接收原始表格后的预测一致性。",
    "候选模型与验证分数": "TabularX 保存 model\_scores.json；MLJAR 保存 leaderboard.csv 或参数记录。",
    "预处理记录": "两者均提供本次运行可查阅的预处理记录。",
    "特征重要性": "两者均生成特征重要性文件或图。",
    "SHAP": "两者本次均观察到 SHAP 工件；覆盖范围以实际模型目录为准。",
    "ALE": "本次固定配置下，两者均未观察到 ALE 工件。",
    "学习曲线或调参轨迹": "TabularX 输出调参轨迹；MLJAR 输出学习曲线。",
    "混淆矩阵": "分类任务用于核对各类别预测错误；回归任务不适用。",
    "公平性": "@@FAIRNESS@@",
    "泄露、划分与评估协议审计": "TabularX 将数据风险、划分完整性和测试集访问顺序写入审计证据。",
    "漂移与群体覆盖": "TabularX 检查开发集与外部测试集的分布差异及已声明群体覆盖。",
    "重载一致性": "TabularX 保存 reload\_check.json，核对持久化前后预测是否一致。",
    "交付状态与失败原因": "TabularX 保存 delivery.json，区分已生成、失败、缺失、未启用与不适用。",
    "原生 HTML 报告": "TabularX 生成汇总 HTML；MLJAR Explain 本次主要输出逐模型 Markdown 与图片。",
}

ROW_END = " " + chr(92) * 2


def esc(value: object) -> str:
    text = str(value)
    for old, new in [
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
    ]:
        text = text.replace(old, new)
    return text


def status(value: str) -> str:
    return {"生成并核验": r"\yes", "未观测": r"\no", "不适用": r"\na"}.get(value, esc(value))


def metric_table(metrics: pd.DataFrame, cfg: dict) -> str:
    rows = []
    for index, row in metrics.iterrows():
        model = cfg["tbx_model"] if index == 0 else cfg["mljar_model"]
        if cfg["task"] == "回归":
            values = [row["工具"], model, f'{row["MAE"]:.2f}', f'{row["RMSE"]:.2f}', f'{row["R2"]:.4f}']
        else:
            values = [row["工具"], model, f'{row["Accuracy"]:.4f}', f'{row["Balanced Accuracy"]:.4f}',
                      f'{row["Macro-F1"]:.4f}', f'{row["ROC-AUC"]:.4f}']
        rows.append("    " + " & ".join(map(esc, values)) + ROW_END)
    if cfg["task"] == "回归":
        spec, header = "llccc", "工具 & 最终模型 & MAE & RMSE & $R^2$"
    else:
        spec, header = "llcccc", "工具 & 最终模型 & Accuracy & Balanced Accuracy & Macro-F1 & ROC-AUC"
    return "\n".join([
        r"\begin{table}[htbp]", r"  \centering", r"  \small",
        r"  \caption{共同外部测试集上的模型表现}", f"  \\begin{{tabular}}{{{spec}}}",
        r"    \toprule", f"    {header}" + ROW_END, r"    \midrule", *rows,
        r"    \bottomrule", r"  \end{tabular}", r"\end{table}",
    ])


def interpretation(metrics: pd.DataFrame, cfg: dict) -> str:
    a, b = metrics.iloc[0], metrics.iloc[1]
    if cfg["task"] == "回归":
        return (f"本次固定配置下，MLJAR 的 MAE（{b['MAE']:.2f}）和 RMSE（{b['RMSE']:.2f}）低于 "
                f"TabularX，且 $R^2$ 为 {b['R2']:.4f}。该结果只描述本次小规模搜索，不构成普遍性能排名。")
    return (f"本次固定配置下，TabularX 与 MLJAR 的 Accuracy 分别为 {a['Accuracy']:.4f} 和 "
            f"{b['Accuracy']:.4f}，ROC-AUC 分别为 {a['ROC-AUC']:.4f} 和 {b['ROC-AUC']:.4f}。"
            "类别不均衡任务还应结合 Balanced Accuracy、Macro-F1 与混淆矩阵解读；该结果不构成普遍性能排名。")


def artifact_table(artifacts: pd.DataFrame, cfg: dict) -> str:
    rows = []
    for _, row in artifacts.iterrows():
        label = str(row["工件"])
        left, right = str(row["TabularX"]), str(row["MLJAR"])
        if cfg["task"] == "回归" and label == "混淆矩阵":
            left = right = "不适用"
        detail = DESCRIPTIONS.get(label, "按本次运行目录中的实际文件判定。")
        detail = detail.replace("@@FAIRNESS@@", cfg["fairness"])
        rows.append(f"  {esc(label)} & {status(left)} & {status(right)} & {detail}" + ROW_END)
    return "\n".join([
        r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.27\textwidth}"
        r">{\centering\arraybackslash}p{0.14\textwidth}"
        r">{\centering\arraybackslash}p{0.14\textwidth}"
        r">{\raggedright\arraybackslash}p{0.34\textwidth}}",
        r"  \caption{本次运行实际观察到的工件}\label{tab:artifacts}\\",
        r"  \toprule", "  工件 & TabularX & MLJAR & 说明" + ROW_END, r"  \midrule", r"  \endfirsthead",
        r"  \toprule", "  工件 & TabularX & MLJAR & 说明" + ROW_END, r"  \midrule", r"  \endhead",
        *rows, r"  \bottomrule", r"\end{longtable}",
    ])


def render(dataset: str, cfg: dict) -> str:
    output = OUTPUT_ROOT / dataset
    metrics = pd.read_json(output / "external_test_metrics.json")
    artifacts = pd.read_csv(output / "artifact_matrix.csv")
    return f"""% !TeX encoding = UTF-8
% !TeX program = xelatex
\\documentclass[degree=bachelor,language=chinese,fontset=fandol]{{thuthesis}}

\\usepackage{{booktabs,longtable,array,xcolor,hyperref,enumitem}}
\\hypersetup{{colorlinks=true,linkcolor=black,urlcolor=blue}}
\\setlist[itemize]{{leftmargin=2em,itemsep=0.25em}}
\\newcommand{{\\yes}}{{\\textcolor{{green!50!black}}{{已生成}}}}
\\newcommand{{\\no}}{{\\textcolor{{red!70!black}}{{未观察到}}}}
\\newcommand{{\\na}}{{\\textcolor{{gray}}{{不适用}}}}

\\begin{{document}}
\\begin{{titlepage}}
  \\centering
  \\vspace*{{2.3cm}}
  {{\\zihao{{1}}\\bfseries MLJAR 与 TabularX 完整工具包对比\\par}}
  \\vspace{{1.2cm}}
  {{\\zihao{{3}}{cfg['title']} 数据集实验报告\\par}}
  \\vspace{{2.2cm}}
  {{\\zihao{{4}}TabularX 课程项目\\par}}
  \\vfill
  {{\\zihao{{-4}}实验日期：2026 年 9 月 29 日\\par}}
  \\vspace{{1.2cm}}
\\end{{titlepage}}

\\frontmatter
\\tableofcontents
\\mainmatter
\\chapter{{实验概述}}
\\section{{数据集与任务}}
\\textbf{{数据集：{cfg['title']}（{cfg['source']}，{cfg['task']}）。}}
{cfg['description']}数据按固定随机种子 42 划分为 80\\% 开发集和 20\\% 外部测试集，外部测试集共 {cfg['test_size']} 条记录。

\\section{{比较范围与原则}}
实验比较 MLJAR 与 TabularX 两个完整工具包，范围包括预处理、模型搜索与选择、解释、公平性适用性、可信审计、模型持久化、报告和证据交付。双方使用同一数据划分，并运行 Random Forest、XGBoost、LightGBM 和 CatBoost。外部测试集不参与训练、调参和选模，只在最终模型锁定后评估一次。

性能指标只描述本次固定配置。主要比较对象是实际生成并能够核对的解释性与可信性工件。

\\chapter{{实验结果}}
\\section{{外部测试集表现}}
{metric_table(metrics, cfg)}

{interpretation(metrics, cfg)}

\\section{{可解释性与可信性工件盘点}}
{artifact_table(artifacts, cfg)}

“未观察到”只表示本次配置和显式调用范围内没有找到对应工件，不外推为软件在任何条件下都不支持。

\\chapter{{工具优缺点与结果解读}}
\\section{{TabularX}}
\\textbf{{优点：}}将测试集隔离、泄露与漂移审计、模型重载核验、报告生成和交付状态检查放入同一次运行的证据链。

\\textbf{{不足：}}本次未观察到 ALE；当前小规模配置只能说明流程可运行，不能代表最优模型性能。

\\section{{MLJAR}}
\\textbf{{优点：}}Explain 模式能够生成逐模型 Markdown、学习曲线、混淆矩阵、特征重要性与 SHAP 等常规解释工件。

\\textbf{{不足：}}本次运行目录中未观察到泄露与划分审计、漂移与群体覆盖、重载一致性、交付状态以及原生汇总 HTML 工件。

\\chapter{{结论}}
两种工具均完成主流模型训练和常规解释工件生成。MLJAR 的逐模型说明较成熟；TabularX 在常规模型报告之外补充可信审计、测试隔离、重载核验和交付状态证据。后续数据集继续复用相同的数据隔离、工件状态词和结果边界。

\\backmatter
\\chapter{{来源}}
\\begin{{itemize}}
  \\item 数据来源：{cfg['source']}
  \\item MLJAR supervised：\\url{{https://github.com/mljar/mljar-supervised}}
  \\item THUThesis：\\url{{https://github.com/tuna/thuthesis}}
\\end{{itemize}}
\\end{{document}}
"""


selected = sys.argv[1:] or list(DATASETS)
for key in selected:
    if key not in DATASETS:
        raise SystemExit(f"未知数据集: {key}")
    config = DATASETS[key]
    path = OUTPUT_ROOT / key / f"{config['title']}.tex"
    path.write_text(render(key, config), encoding="utf-8")
    print(path)
