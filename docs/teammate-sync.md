# 本周同步：证据交付检查与局部补做

这周是在现有训练和报告流程上增补交付保障，不重新写 AutoML，不合并 TrustGuard。

## 原来有什么，我们新增什么

| 部分 | 原有能力 | 本次新增或修正 |
|---|---|---|
| 训练与报告 | 模型搜索、SHAP、公平性计算、HTML 报告 | 不将这些原有算法算作自己的创新 |
| 创新一：交付检查 | 能保存部分模型、图和报告 | 先声明该交付什么，逐项记录已生成、失败、缺失、未启用或不适用；报告存在不等于交付完整 |
| 创新二：局部补做 | 原流程缺少统一的逐项恢复入口 | 同一程序会话内，保持模型和数据不变，只补做可恢复的失败/缺失项，保留尝试历史 |
| 模型复用 | 保存处理后输入的模型 | 同时保存预处理器与模型，重载后直接接收原始表格，并核对预测一致性 |
| 公平性边界 | 分组与 DP/EO 计算 | 保留原始敏感分组，区分零差异与无法计算，明确目前只支持 0/1 二分类，正类为 1 |
| 测试集使用 | 原流程候选阶段就计算测试分数 | 验证集选模后保存锁定记录，只给选中单模型生成最终测试分数和诊断图 |

这里的创新是本项目相对于原有实现的工程增量，不主张学术首创。

## 实际验证了什么

- 217 项测试通过，包含搜索阶段禁止访问测试集、原始表格重载、公平性边界及故障恢复。
- Breast Cancer、Iris、Diabetes × 随机森林、XGBoost：6 个小规模组合，第三版 54 项检查通过；不是 54 个独立实验。
- 第四版 Iris、Diabetes 双候选真实运行：只给锁定模型生成测试分数，Voting 仅验证集比较。
- 本周补跑完整 Adult：48,842 行、14 特征，随机森林和 XGBoost 均成功。人为让 SHAP 首次失败，交付检查判为不完整；真实补做后完整，补做期间模型训练次数为零，原始表格重载预测一致。
- Adult 每个候选一次小规模调参，32 棵树、最大深度 5，SHAP 解释 24 个测试样本。这不是性能最优配置或性能排名实验。

Adult 的 sex 在本次仓库实现中既是模型输入也是审计分组；早期独立 v0.1 排除了该输入。两版设置不一致，不直接比较预测或公平性数字。

## 队友怎样使用

在仓库根目录、自己的 Python 虚拟环境中安装（下列为当前演示所需依赖）：

```bash
python -m pip install -e ./TabularX-project
python -m pip install optuna matplotlib shap xgboost joblib pytest
```

XGBoost 可能需要系统 OpenMP；具体按自己的系统安装。本项目尚未做跨平台环境锁定。

```python
from tabularx import AutoEstimator

# X 是原始 pandas DataFrame，y 是同序且索引对齐的标签 Series。
ae = AutoEstimator(task="classification", output_dir="outputs/my_run")
ae.param.model_search.models = ["random_forest"]
ae.param.model_search.n_jobs = 1
ae.param.model_search.n_trials = 1
# 可选：传入原始表格中的敏感字段；不会自动从模型输入排除它。
ae.fit(X, y, sensitive_features=["sex"])

print(ae.result.report_paths["evidence"])  # 打开交付检查网页
print(ae.evidence_.audit()["complete"])
# 在同一程序会话中，修复解释步骤的问题后再调用；已成功项不会重新训练。
ae.retry_evidence(["shap"])
```

不是所有状态都能补做。只补做清单中允许恢复的失败/缺失项；数据、模型或配置变化时应重新 fit。

```python
import joblib
pipeline = joblib.load(ae.output_dir / "pipeline.joblib")  # 仅加载可信文件
prediction = pipeline.predict(new_raw_dataframe)
```

预测重载与证据补做是不同能力：重载可使用保存的流水线；补做目前需要原有 ae 实例仍在当前程序中。

## 最快演示顺序（约两分钟）

1. 打开 Adult 的 before_recovery.json：模型已交付，但 SHAP 失败，所以整体不完整。
2. 打开同一运行的 evidence/delivery.html：查看 SHAP 先失败后成功的尝试历史。
3. 打开 reload_check.json：原始表格重载预测一致；summary.json 记录补做训练次数为零。
4. 打开 selection_lock.json：解释模型如何依据验证集确定。

真实运行产物保存在本地 outputs，未将 Adult 数据、模型文件或整包实验结果提交到代码仓库。
队友可自行复跑：

```bash
python introduction/selection_protocol_demo.py --output outputs/protocol-demo
python introduction/adult_evidence_demo.py --data /path/to/adult.arff.gz --output outputs/adult-demo
```

Adult 脚本读取 OpenML 1590 的完整 ARFF gzip 缓存，不会联网下载。综合恢复演示见 introduction/evidence_validation_suite.py。

## 与 TrustGuard 的边界

队友的 TrustGuard 负责可信建模审计；我们负责审计/解释等产物是否按要求交付以及失败后如何补做。修正自己分支中的基础计算和选模问题，不等于实现或合并了 TrustGuard。

这周不做正式汇报稿，也不合并队友报告。后续再商量把 TrustGuard 的输出接成一个独立证据生产者。

## 不能夸大的地方

- 完整只表示满足当前声明的交付集合，不代表课程全部要求完成；ALE、学习曲线等仍未纳入。
- 故障为人工注入，不是自然故障率或加速比实验。
- 同环境预测一致不代表跨环境完全复现；补做不支持程序退出后恢复。
- 公平指标不代表模型公平，更不代表完成公平性缓解。
- 数据概况仍读取全量输入，本次选模流程修复不等于全面防泄露审计。
