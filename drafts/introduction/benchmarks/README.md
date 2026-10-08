# 十数据集实验复现入口

本目录保存论文中 MLJAR 与 TabularX 十数据集对比实验的可执行材料。实验结果、结构化指标、工件清单及 PDF/HTML 报告位于 [`docs/validation/`](../../../docs/validation/README.md)。

## 运行方式

共享运行器 `run_toolkit_benchmark.py` 覆盖 Adult Income、Breast Cancer Wisconsin、Titanic、Iris 和 California Housing。其余数据集使用相应子目录中的 Notebook 运行；十个数据集均保留 Notebook，便于检查数据来源、划分方式、模型配置和结果生成过程。

```bash
python drafts/introduction/benchmarks/run_toolkit_benchmark.py adult
python -m jupyterlab
```

Notebook 的历史执行输出已清除，以免提交本机绝对路径、临时缓存位置和环境警告。真实运行结果没有删除，统一归档在 `docs/validation/`。复跑时请记录当前 Git 提交编号、Python 与依赖版本、公开数据版本和随机种子。

## 结论边界

实验采用共同外部测试集，但两套工具的内部搜索机制和实际计算开销并不完全相同，因此结果用于比较本次运行中的预测指标与工件交付，不构成严格等预算的普遍性能排名。
