# LeakageAudit 独立扩展

此目录不修改原有 `tabularx/` 包。推荐接入点只有四行：

```python
from leakage_audit import LeakageAuditor, LeakageAuditConfig, write_audit_artifacts
auditor = LeakageAuditor(LeakageAuditConfig(group_column="patient_id"))
audit = auditor.preflight(X, y, task_type="binary")
split = split_data(X.drop(columns=audit.excluded_columns), y, profile)
audit = auditor.check_split(X, split.train_indices, split.valid_indices, split.test_indices, audit)
write_audit_artifacts(audit, output_dir)
```

训练结束后可额外调用：

```python
from leakage_audit import validate_selection_protocol
audit = validate_selection_protocol(search_result.model_results,
                                    search_result.primary_metric,
                                    search_result.optimization_direction,
                                    audit)
```

该调用会明确报告当前 TabularX 版本中“多个模型读取测试集并按测试指标选择”的问题；修复该问题应作为单独、可审阅的后续提交，而不是混在本扩展中。

## 安全模型搜索器（独立，不修改原模型）

`safe_model_search.py` 是一条并行、独立的建模路径：它复用 TabularX 已有的模型注册表、预处理和 Optuna 调参器，但不修改 `model_search.py`、`ensemble_search.py` 或任何队友的模型实现。

其严格协议为：

1. 每个候选仅使用 train / validation 调参；
2. 仅按 validation 主指标选择候选；
3. 锁定唯一候选后，用 train + validation 重训；
4. 仅该最终候选访问 test 一次。

默认候选为 Random Forest、XGBoost、LightGBM、CatBoost。缺失任一可选库时，
该候选会在 JSON 中记录失败原因，不会中断其他模型。

建议将完整四模型实验放在 OpenI，具体的环境、资源和任意 CSV 运行方式见
[`OPENI_SETUP.md`](OPENI_SETUP.md)。本机只做快速验证时可明确传入
`--models random_forest`，无需安装三个梯度提升库。
安全搜索器仍需要 Optuna，可安装 `python -m pip install -r
leakage_audit/requirements-safe-search.txt`。

可用以下命令跑项目现有 breast-cancer 教程数据（每个模型默认 3 个 Optuna trial）：

```powershell
$env:PYTHONPATH = "$PWD;$PWD/src"
python leakage_audit/demo_safe_model_search.py --dataset breast_cancer
```

输出 `safe_model_search.json` 会同时记录所有候选的训练/验证指标、最终被锁定的模型，以及唯一一次测试集指标。

可替换公开演示数据集：

```powershell
python leakage_audit/demo_safe_model_search.py --dataset iris
python leakage_audit/demo_safe_model_search.py --dataset diabetes
```

也可运行任意 CSV。CSV 中必须存在目标列，其他列将作为特征：

```powershell
python leakage_audit/demo_safe_model_search.py `
  --csv "D:\data\my_table.csv" `
  --target outcome `
  --models random_forest xgboost lightgbm catboost `
  --n-trials 20 `
  --output-dir .\my-run
```

## 运行项目现有示例数据

安装 TabularX 依赖后，在 `TabularX-project/` 根目录运行：

```powershell
$env:PYTHONPATH = "$PWD;$PWD/src"
python leakage_audit/demo_existing_dataset.py --output-dir ./leakage-demo-output
```

加入 `--inject-leakage` 可验证目标复制字段会被识别；加入
`--run-model-search` 则会额外审计现有模型搜索是否暴露测试集。
