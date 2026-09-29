# MLJAR 与 TabularX 完整工具包对比实验

本实验在五个公开表格数据集上运行 MLJAR Explain 与 TabularX，比较完整工具包的实际输出，而不是只比较单个模型分数。双方使用共同外部测试集，并覆盖预处理、模型搜索与选择、特征重要性、SHAP、学习曲线或调参轨迹、混淆矩阵、公平性适用性、可信审计、模型持久化、HTML/PDF 报告与证据交付状态。

性能指标只描述本次固定配置，不作为严格等预算的性能排名。“未观察到”表示本次运行目录中没有找到对应工件，不外推为工具在任何配置下都不支持。

## 实验入口

| 数据集 | 任务 | Notebook | 对比报告 | 结构化结果 |
|---|---|---|---|---|
| Bank Marketing | 二分类 | [Notebook](../../../introduction/benchmarks/bank-marketing/bank_marketing_mljar_vs_tabularx.ipynb) | [PDF](bank-marketing/实验报告-Bank%20Marketing.pdf) · [HTML](bank-marketing/comparison_report.html) · [TeX](bank-marketing/Bank%20Marketing.tex) | [指标](bank-marketing/external_test_metrics.json) · [工件清单](bank-marketing/artifact_matrix.csv) |
| Credit Scoring | 二分类 | [Notebook](../../../introduction/benchmarks/credit-scoring/credit_scoring_mljar_vs_tabularx.ipynb) | [PDF](credit-scoring/实验报告-Credit%20Scoring.pdf) · [HTML](credit-scoring/comparison_report.html) · [TeX](credit-scoring/Credit%20Scoring.tex) | [指标](credit-scoring/external_test_metrics.json) · [工件清单](credit-scoring/artifact_matrix.csv) |
| Employee Attrition | 二分类 | [Notebook](../../../introduction/benchmarks/employee-attrition/employee_attrition_mljar_vs_tabularx.ipynb) | [PDF](employee-attrition/实验报告-Employee%20Attrition.pdf) · [HTML](employee-attrition/comparison_report.html) · [TeX](employee-attrition/Employee%20Attrition.tex) | [指标](employee-attrition/external_test_metrics.json) · [工件清单](employee-attrition/artifact_matrix.csv) |
| Wine | 三分类 | [Notebook](../../../introduction/benchmarks/wine/wine_mljar_vs_tabularx.ipynb) | [PDF](wine/实验报告-Wine.pdf) · [HTML](wine/comparison_report.html) · [TeX](wine/Wine.tex) | [指标](wine/external_test_metrics.json) · [工件清单](wine/artifact_matrix.csv) |
| House Prices | 回归 | [Notebook](../../../introduction/benchmarks/house-prices/house_prices_mljar_vs_tabularx.ipynb) | [PDF](house-prices/实验报告-House%20Prices.pdf) · [HTML](house-prices/comparison_report.html) · [TeX](house-prices/House%20Prices.tex) | [指标](house-prices/external_test_metrics.json) · [工件清单](house-prices/artifact_matrix.csv) |

## 文件范围

仓库只保存可复核且适合版本控制的 Notebook、汇总报告、指标和工件清单。MLJAR/TabularX 的模型二进制、OpenML 缓存、临时文件和重复运行目录不纳入提交。

