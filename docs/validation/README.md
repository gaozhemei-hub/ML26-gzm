# MLJAR 与 TabularX 十数据集完整工具包对比实验

本实验在十个公开表格数据集上运行 MLJAR Explain 与 TabularX，比较两种工具包在固定配置下的预测结果与实际工件交付。实验覆盖八个分类任务和两个回归任务，以及数值型、类别型、混合型、缺失值、类别不均衡和高维表格等数据条件。

每个数据集使用随机种子 42 划分为 80% 开发集和 20% 外部测试集。两种工具接收相同的开发集与外部测试集，候选算法统一为随机森林、XGBoost、LightGBM 与 CatBoost。外部测试集不参与训练、调参和选模，只在最终模型锁定后评估一次。

本目录保存实验结果归档，包括 HTML/TeX 报告、结构化指标、工件清单和状态汇总。当前仓库不再保留生成这些历史结果的 Notebook 和统一运行器，因此本页用于核查归档结果，不提供当前版本的一键重跑命令。

性能指标仅描述本次固定配置，不构成严格等预算的普遍性能排名。“未观测”表示本次运行目录中没有找到相应工件，不外推为工具在任何配置下均不支持。

## 综合结果

- [十数据集综合报告（HTML）](十数据集综合报告.html)
- [学校模板 TeX 源文件](十数据集综合报告.tex)
- [外部测试指标总表](ten_dataset_metrics.csv)
- [逐数据集工件总表](ten_dataset_artifacts.csv)
- [工件状态统计](ten_dataset_status.csv)

当前归档未包含综合 PDF 文件，因此不在此处提供 PDF 链接。

## 归档结果

| 数据集 | 任务 | 对比报告 | 结构化结果 |
|---|---|---|---|
| Bank Marketing | 二分类 | [HTML](bank-marketing/comparison_report.html) · [TeX](bank-marketing/Bank%20Marketing.tex) | [指标](bank-marketing/external_test_metrics.json) · [工件清单](bank-marketing/artifact_matrix.csv) |
| Credit Scoring | 二分类 | [HTML](credit-scoring/comparison_report.html) · [TeX](credit-scoring/Credit%20Scoring.tex) | [指标](credit-scoring/external_test_metrics.json) · [工件清单](credit-scoring/artifact_matrix.csv) |
| Employee Attrition | 二分类 | [HTML](employee-attrition/comparison_report.html) · [TeX](employee-attrition/Employee%20Attrition.tex) | [指标](employee-attrition/external_test_metrics.json) · [工件清单](employee-attrition/artifact_matrix.csv) |
| Wine | 三分类 | [HTML](wine/comparison_report.html) · [TeX](wine/Wine.tex) | [指标](wine/external_test_metrics.json) · [工件清单](wine/artifact_matrix.csv) |
| House Prices | 回归 | [HTML](house-prices/comparison_report.html) · [TeX](house-prices/House%20Prices.tex) | [指标](house-prices/external_test_metrics.json) · [工件清单](house-prices/artifact_matrix.csv) |
| Adult Income | 二分类 | [HTML](adult/comparison_report.html) · [TeX](adult/Adult%20Income.tex) | [指标](adult/external_test_metrics.json) · [工件清单](adult/artifact_matrix.csv) · [运行摘要](adult/run_summary.json) |
| Breast Cancer Wisconsin | 二分类 | [HTML](breast-cancer/comparison_report.html) · [TeX](breast-cancer/Breast%20Cancer%20Wisconsin.tex) | [指标](breast-cancer/external_test_metrics.json) · [工件清单](breast-cancer/artifact_matrix.csv) · [运行摘要](breast-cancer/run_summary.json) |
| Titanic | 二分类 | [HTML](titanic/comparison_report.html) · [TeX](titanic/Titanic.tex) | [指标](titanic/external_test_metrics.json) · [工件清单](titanic/artifact_matrix.csv) · [运行摘要](titanic/run_summary.json) |
| Iris | 三分类 | [HTML](iris/comparison_report.html) · [TeX](iris/Iris.tex) | [指标](iris/external_test_metrics.json) · [工件清单](iris/artifact_matrix.csv) · [运行摘要](iris/run_summary.json) |
| California Housing | 回归 | [HTML](california-housing/comparison_report.html) · [TeX](california-housing/California%20Housing.tex) | [指标](california-housing/external_test_metrics.json) · [工件清单](california-housing/artifact_matrix.csv) · [运行摘要](california-housing/run_summary.json) |

## 公平性设置与解释边界

Adult Income 和 Titanic 使用数据集中具有明确语义的性别字段进行分组审计，敏感字段不进入两种工具的模型输入。Employee Attrition 使用 Gender 进行审计。Bank Marketing 与 Credit Scoring 根据预先声明的年龄阈值构造探索性分组，因此相应结果只反映该分组定义下的差异，不代表普遍的公平性判断。其余数据集因缺少明确敏感字段、多分类或回归适用边界而标记为不适用。

MLJAR 在部分二分类任务中会进入公平性缓解流程，而 TabularX 当前主要生成审计证据。因此，相关任务的预测指标用于描述完整工具包的实际运行结果，不解释为完全相同训练协议下的性能差异。

## 如何使用归档

复核时，先比较 `external_test_metrics.json` 或综合指标 CSV，核对任务类型、外部测试样本数和主要预测指标；再检查 `artifact_matrix.csv`，确认模型、解释、诊断、公平性和报告工件的实际状态；最后打开对应 HTML 报告核对人类可读结果。

预测指标接近且工件状态一致可作为流程复核证据，不要求不同操作系统、硬件和依赖版本下的运行时间或浮点数结果逐位相同。归档结果不包含原始数据、虚拟环境和模型二进制文件。

## 结论边界

跨数据集结果支持以下工程判断：两种工具均能够完成主流模型训练并生成常规解释工件；TabularX 的主要增量集中在测试隔离、运行审计、模型重载核验以及工件状态与失败原因的统一交付。该判断描述证据交付机制的工程差异，不等同于预测性能优势，也不证明模型在现实场景中公平。
