# 正式交付文档

## 功能说明书

| 功能 | 说明书 | 主要内容 |
|---|---|---|
| 证据交付检查与局部补做 | [证据交付与局部补做说明书](evidence-delivery-guide.md) | 交付规则、状态检查、局部恢复、使用方法与验证边界 |
| 可信审计 | [可信审计整合说明书](trust-audit-integration-guide.md) | 数据泄露、划分完整性、分布漂移、群体覆盖与评估协议 |

## 实验依据

- [MLJAR 与 TabularX 完整工具包对比](validation/mljar-tabularx-comparison/README.md)：在 Bank Marketing、Credit Scoring、Employee Attrition、Wine 和 House Prices 五个公开表格数据集上运行两套工具，使用共同外部测试集，盘点预处理、模型选择、特征重要性、SHAP、学习曲线、混淆矩阵、公平性适用性、可信审计、持久化、报告和证据交付等实际工件。索引提供每个数据集的 [对比报告文件夹](validation/mljar-tabularx-comparison/) 与 [Notebook 文件夹](../introduction/benchmarks/)。实验用于核对工件覆盖和报告边界，不作为严格等预算的性能排名。
- [证据交付实验验证索引](validation/evidence-delivery/README.md)：提供真实运行报告、结构化结果及故障恢复记录。各批次保留对应代码阶段与结论边界，不将历史实验混同为当前版本的重复验证。

## 架构与任务背景

- [设计文档](设计/)：架构、数据结构、参数、异常、报告和流水线设计。
- [设计理念](philosophy.md)：项目设计背景。
- [原始选题要求](选题.md)：任务要求，不表示全部目标能力已经实现。
