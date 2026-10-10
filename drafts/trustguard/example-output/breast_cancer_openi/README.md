# TrustGuard 正式示例结果

本目录保存一份由 TrustGuard 源码生成的完整、可审计示例，便于无需重新训练即可查看报告
结构，并核对机器可读工件。主实验在 OpenI 环境运行，使用 scikit-learn 乳腺癌数据集：

- 569 条样本，30 个数值特征，二分类任务；
- 随机森林、XGBoost、LightGBM、CatBoost；
- 每个模型执行 5 次超参数搜索；
- 根据验证集 ROC-AUC 选择 LightGBM；
- 模型锁定后仅访问测试集一次；
- 第 5 部分汇总 5 个成功 OpenML 数据集，并保留 `credit-g` 读取超时记录。

## 推荐查看顺序

1. `reports/trustguard.html`：浏览器可读的完整报告；
2. `reports/trustguard.pdf`：便于归档和分享的同内容报告；
3. `leakage_audit/summary.json`：六项泄露检查及执行状态；
4. `model_search_result.json`：四模型验证分数、最终模型和测试指标；
5. `benchmark_summary.json`：MLJAR 与 TrustGuard 的公开数据集工件对比；
6. `artifact_manifest.json`：当前目录全部工件的 SHA-256 清单。

## 结果边界

乳腺癌数据集不含患者标识或时间字段，因此实体分组泄露和时间穿越检查标记为“未执行”，
而不是“通过”。OpenML 对比实验关注解释与可信性工件是否生成，不把不同搜索策略下的分数
作为严格算法排名。原始数据没有复制到本目录。

该示例用于展示正式运行结果。若要复现完整流程，请从仓库根目录运行
`python -m trustguard.run_full_experiment --help`，并根据环境设置数据集、搜索次数和基准汇总路径。
