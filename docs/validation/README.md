# MLJAR 与 TabularX 十数据集完整工具包对比实验

项目根路径：<https://github.com/gaozhemei-hub/ML26-gzm>。以下相对路径均以该仓库 `develop` 分支的根目录为起点。

本实验在十个公开表格数据集上运行 MLJAR Explain 与 TabularX，比较两种工具包在固定配置下的预测结果与实际工件交付。实验覆盖八个分类任务和两个回归任务，以及数值型、类别型、混合型、缺失值、类别不均衡和高维表格等数据条件。

每个数据集均使用随机种子 42 划分为 80% 开发集和 20% 外部测试集。两种工具接收相同的开发集与外部测试集，候选算法统一为随机森林、XGBoost、LightGBM 与 CatBoost。外部测试集不参与训练、调参和选模，只在最终模型锁定后评估一次。

本目录保存 HTML、PDF 与 TeX 报告、结构化指标、工件清单和状态汇总。为满足实验复现要求，个人提交仓库另在 `drafts/introduction/` 中保留十个 Notebook、统一运行器和专项验证脚本；正式实验结论仍以本目录的归档结果为准。

性能指标仅描述本次固定配置，不构成严格等预算的普遍性能排名。“未观测”表示本次运行目录中没有找到相应工件，不外推为工具在任何配置下均不支持。

## 综合结果

- [十数据集综合报告（PDF）](十数据集综合报告.pdf)
- [十数据集综合报告（HTML）](十数据集综合报告.html)
- [学校模板 TeX 源文件](十数据集综合报告.tex)
- [外部测试指标总表](ten_dataset_metrics.csv)
- [逐数据集工件总表](ten_dataset_artifacts.csv)
- [工件状态统计](ten_dataset_status.csv)

## 实验入口

| 数据集 | 任务 | Notebook | 对比报告 | 结构化结果 |
|---|---|---|---|---|
| Bank Marketing | 二分类 | [Notebook](../../drafts/introduction/benchmarks/bank-marketing/bank_marketing_mljar_vs_tabularx.ipynb) | [PDF](bank-marketing/实验报告-Bank%20Marketing.pdf) · [HTML](bank-marketing/comparison_report.html) · [TeX](bank-marketing/Bank%20Marketing.tex) | [指标](bank-marketing/external_test_metrics.json) · [工件清单](bank-marketing/artifact_matrix.csv) |
| Credit Scoring | 二分类 | [Notebook](../../drafts/introduction/benchmarks/credit-scoring/credit_scoring_mljar_vs_tabularx.ipynb) | [PDF](credit-scoring/实验报告-Credit%20Scoring.pdf) · [HTML](credit-scoring/comparison_report.html) · [TeX](credit-scoring/Credit%20Scoring.tex) | [指标](credit-scoring/external_test_metrics.json) · [工件清单](credit-scoring/artifact_matrix.csv) |
| Employee Attrition | 二分类 | [Notebook](../../drafts/introduction/benchmarks/employee-attrition/employee_attrition_mljar_vs_tabularx.ipynb) | [PDF](employee-attrition/实验报告-Employee%20Attrition.pdf) · [HTML](employee-attrition/comparison_report.html) · [TeX](employee-attrition/Employee%20Attrition.tex) | [指标](employee-attrition/external_test_metrics.json) · [工件清单](employee-attrition/artifact_matrix.csv) |
| Wine | 三分类 | [Notebook](../../drafts/introduction/benchmarks/wine/wine_mljar_vs_tabularx.ipynb) | [PDF](wine/实验报告-Wine.pdf) · [HTML](wine/comparison_report.html) · [TeX](wine/Wine.tex) | [指标](wine/external_test_metrics.json) · [工件清单](wine/artifact_matrix.csv) |
| House Prices | 回归 | [Notebook](../../drafts/introduction/benchmarks/house-prices/house_prices_mljar_vs_tabularx.ipynb) | [PDF](house-prices/实验报告-House%20Prices.pdf) · [HTML](house-prices/comparison_report.html) · [TeX](house-prices/House%20Prices.tex) | [指标](house-prices/external_test_metrics.json) · [工件清单](house-prices/artifact_matrix.csv) |
| Adult Income | 二分类 | [Notebook](../../drafts/introduction/benchmarks/adult/adult_mljar_vs_tabularx.ipynb) | [PDF](adult/对比报告-Adult%20Income.pdf) · [HTML](adult/comparison_report.html) · [学校模板 TeX](adult/Adult%20Income.tex) | [指标](adult/external_test_metrics.json) · [工件清单](adult/artifact_matrix.csv) · [运行摘要](adult/run_summary.json) |
| Breast Cancer Wisconsin | 二分类 | [Notebook](../../drafts/introduction/benchmarks/breast-cancer/breast_cancer_mljar_vs_tabularx.ipynb) | [PDF](breast-cancer/对比报告-Breast%20Cancer%20Wisconsin.pdf) · [HTML](breast-cancer/comparison_report.html) · [学校模板 TeX](breast-cancer/Breast%20Cancer%20Wisconsin.tex) | [指标](breast-cancer/external_test_metrics.json) · [工件清单](breast-cancer/artifact_matrix.csv) · [运行摘要](breast-cancer/run_summary.json) |
| Titanic | 二分类 | [Notebook](../../drafts/introduction/benchmarks/titanic/titanic_mljar_vs_tabularx.ipynb) | [PDF](titanic/对比报告-Titanic.pdf) · [HTML](titanic/comparison_report.html) · [学校模板 TeX](titanic/Titanic.tex) | [指标](titanic/external_test_metrics.json) · [工件清单](titanic/artifact_matrix.csv) · [运行摘要](titanic/run_summary.json) |
| Iris | 三分类 | [Notebook](../../drafts/introduction/benchmarks/iris/iris_mljar_vs_tabularx.ipynb) | [PDF](iris/对比报告-Iris.pdf) · [HTML](iris/comparison_report.html) · [学校模板 TeX](iris/Iris.tex) | [指标](iris/external_test_metrics.json) · [工件清单](iris/artifact_matrix.csv) · [运行摘要](iris/run_summary.json) |
| California Housing | 回归 | [Notebook](../../drafts/introduction/benchmarks/california-housing/california_housing_mljar_vs_tabularx.ipynb) | [PDF](california-housing/对比报告-California%20Housing.pdf) · [HTML](california-housing/comparison_report.html) · [学校模板 TeX](california-housing/California%20Housing.tex) | [指标](california-housing/external_test_metrics.json) · [工件清单](california-housing/artifact_matrix.csv) · [运行摘要](california-housing/run_summary.json) |

## 公平性设置与解释边界

Adult Income 和 Titanic 使用数据集中具有明确语义的性别字段进行分组审计，敏感字段不进入两种工具的模型输入。Employee Attrition 使用 Gender 进行审计。Bank Marketing 与 Credit Scoring 根据预先声明的年龄阈值构造探索性分组，因此相应结果只反映该分组定义下的差异，不代表普遍的公平性判断。其余数据集因缺少明确敏感字段、多分类或回归适用边界而标记为不适用。

MLJAR 在部分二分类任务中会进入公平性缓解流程，而 TabularX 当前主要生成审计证据。因此，相关任务的预测指标用于描述完整工具包的实际运行结果，不解释为完全相同训练协议下的性能差异。

## 复现与归档核对

从仓库根目录创建独立环境并安装工具包与实验依赖：

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ./TabularX-project
python -m pip install optuna shap joblib xgboost lightgbm catboost openml
```

macOS和Linux使用 `source .venv/bin/activate` 激活环境；Windows PowerShell使用 `.venv\Scripts\Activate.ps1`。实际依赖版本、公开数据版本和当前提交编号应随复现实验一并记录。

共享运行器位于 [`drafts/introduction/benchmarks/run_toolkit_benchmark.py`](../../drafts/introduction/benchmarks/run_toolkit_benchmark.py)，当前覆盖 Adult Income、Breast Cancer Wisconsin、Titanic、Iris 和 California Housing 五个数据集。例如：

```bash
python drafts/introduction/benchmarks/run_toolkit_benchmark.py adult
python drafts/introduction/benchmarks/run_toolkit_benchmark.py california-housing
```

其余五个数据集以及全部十个数据集的逐步流程保存在本页“实验入口”所列 Notebook 中，可使用 `python -m jupyterlab` 打开并依次执行。部分本地数据通过 `TABULARX_DATASETS_ROOT` 指定，OpenML或scikit-learn数据由Notebook按固定编号或接口读取。

运行产生的模型二进制和缓存保存在 `benchmark-output/`，默认不纳入版本控制；可复核的 Notebook、报告、指标与工件清单归档在本目录。复现或复核时，应先比较 `external_test_metrics.json` 或综合指标 CSV，核对任务类型、外部测试样本数和主要预测指标；再检查 `artifact_matrix.csv`，确认模型、解释、诊断、公平性和报告工件的实际状态；最后打开相应 HTML 或 PDF 报告核对人类可读结果。不同操作系统、硬件和依赖版本下不要求运行时间或浮点结果逐位相同，但任务类型、测试样本规模、主要指标范围和工件状态应能够解释并相互对应。

## 结论边界

跨数据集结果支持以下工程判断：两种工具均能够完成主流模型训练并生成常规解释工件；TabularX 的主要增量集中在测试隔离、运行审计、模型重载核验以及工件状态与失败原因的统一交付。该判断描述证据交付机制的工程差异，不等同于预测性能优势，也不证明模型在现实场景中公平。
