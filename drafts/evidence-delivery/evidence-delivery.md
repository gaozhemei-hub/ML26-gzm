# 证据交付与局部恢复：实现与验证

## 分工

本模块位于 `TabularX-project/src/tabularx/core/evidence/`，由 `AutoEstimator.fit()`
保存阶段接入。没有复制或修改 feature/trust-audit 的 TrustGuard。

- TrustGuard：泄露、评估协议、分布和群体覆盖审计，数据/划分/环境记录及文件哈希。
- Evidence：预先声明应交付项目，结合执行结果检查实际文件，记录原因与历次尝试，恢复失败的可重试步骤。

本版尚未直接接通 TrustGuard；合并时应由上层把它作为证据生产者注册，不能把独立模块的
存在当作已运行审计。TrustGuard 的文件清单应在恢复产物和报告更新完成后重新封存。

## 默认交付要求

| 项目 | 条件 | 必需 |
|---|---|---|
| 最佳模型、候选分数、各阶段及总 HTML 报告 | 有建模结果 | 是 |
| SHAP 汇总图、重要性图、局部图 | 当前模型解释步骤 | 是，三张缺任意一张会记录失败 |
| 混淆矩阵 | 分类 | 是 |
| ROC | 当前二分类模型 | 是（目前注册模型支持概率预测） |
| 调参轨迹 | 有试验记录 | 是；无记录记不适用 |
| 公平性结构化结果 | 提供敏感属性且为当前实现支持的 0/1 二分类 | 是 |
| 公平性评估 | 未提供敏感属性 | 未启用，非必需 |
| 当前 DP/EO 生产者 | 回归、多分类、非 0/1 标签 | 不适用，注明实现边界 |
| PDF | 显式请求 | 是，否则未启用 |
| 集成模型报告 | 已有验证集集成结果 | 可选；默认不产生集成测试图 |

这是明确版本化的 v1 交付集合，不声称覆盖所有课程要求：ALE、学习曲线等
仍待实现；原始数据预处理器重载已在后续修复中完成。调参轨迹不是学习曲线。此处的“完整”仅指满足上述已声明要求。

状态：待执行、已生成、失败、缺失、未启用、不适用。必需但未启用的适用项仍导致不完整。
没有声明任何必需适用项时，不得宣称交付完整；完整率为 null。
必需项完整率 = 已生成的必需适用项 / 全部必需适用项。
报告成功不覆盖上游失败；逐项失败记录仍保留。

## 输出位置与兼容性

每次 fit 输出到 `<output_dir>/runs/<run_id>/`，而非复用旧目录。
通过 `ae.output_dir` 或 `ae.result.output_dir` 获取实际目录。
原来硬编码 `<output_dir>/report.html` 的脚本需要改为 `ae.result.report_paths['full']`。

新增入口：

- `evidence/delivery.json`：交付规则、状态、路径和尝试历史。
- `evidence/delivery.html`：中文验收页，总报告中有链接。
- `evidence/model_scores.json`：候选模型分数。
- `evidence/fairness.json`：启用时的公平性结构化结果。

不能依靠旧目录文件证明本次生成成功；生产者必须明确返回文件路径，且文件必须在本次运行目录内、存在且非空。
非空不是内容正确；本模块没有重复实现 SHA-256 校验。

## 使用

```python
from tabularx import AutoEstimator

ae = AutoEstimator(task="classification", output_dir="outputs/demo")
ae.fit(X, y, sensitive_features=["sex"])
print(ae.result.report_paths["evidence"])
print(ae.evidence_.audit()["complete"])

# 同一个 Python 会话内，依赖恢复可用后，补做失败的解释。
ae.retry_evidence(["shap"])
# 或重试所有当前可恢复的失败/缺失项目。
ae.retry_evidence()
```

恢复复用同一个已拟合最终模型、同一份处理后的数据和相同配置，不重新搜索或训练。
报告在上游恢复后重新渲染，已有成功证据不重算。数据/模型/配置变化时拒绝恢复，要求新建运行。
修改解释预算也属于配置变化，当前版本不支持在原运行内这样恢复。
训练失败不属于本模块的自动恢复范围；非最佳候选和集成图表仍由原有函数重建模型，因此明确禁止自动重试这些绘图步骤。
现有函数内部产生的某些图形错误只返回缺失路径，记录会说明“未返回图”，不会虚构底层异常原因。

## 边界与待完成修复

1. 仅同会话恢复，不支持重启进程后从清单执行代码。
2. 已新增 pipeline.joblib，保存拟合后的预处理器与模型，可直接接收原始 DataFrame；仅验证同环境重载。
3. 候选搜索与 Voting 比较仅使用训练/验证数据；主流程只对锁定单模型计算最终测试分数。
   锁定后的解释、诊断和重载检查仍使用测试输入，不等于物理上只调用一次预测。
4. 最终模型在交付前统一拟合一次，最终分数、解释、公平性与模型文件使用同一对象。
   主流程不再为非选中候选或 Voting 重建模型生成测试诊断图。不声称已经完成全生命周期泄露审计。
5. DP/EO 明确限制为 0/1 二分类（正类 1）；缺少分母和少于两个分组时记录 null，不以零替代。主流程按原始敏感属性分组；暂未扩展连续属性分箱、多分类公平指标或缓解算法。
6. 当前完整率是交付完整率，不是解释忠实性、公平性或跨环境复现通过率。

## 验证

安装 TabularX 基本依赖及 `matplotlib optuna shap pytest`，在仓库根目录执行：

```bash
PYTHONPATH=TabularX-project/src python -m pytest tests/test_evidence.py -q
PYTHONPATH=TabularX-project/src python drafts/introduction/evidence_recovery_demo.py --output outputs/evidence-demo
```

单元测试覆盖未生成、空文件、旧运行文件、删除、必需项禁用、任务不适用、恢复历史与上下文变更。
真实集成测试在 Breast Cancer 和 Diabetes 上执行随机森林建模，注入首次 SHAP 失败，
随后使用真实 SHAP 恢复，并拦截 fit() 确认恢复时没有再次训练。
这只是两数据集、一个模型族的初步验证，不构成跨模型全面稳健性或加速结论。

## 扩展验证（2026-09-22）

新增可重复运行的脚本 `drafts/introduction/evidence_validation_suite.py`：

```bash
PYTHONPATH=TabularX-project/src python drafts/introduction/evidence_validation_suite.py --output outputs/evidence-validation
```

依赖除前述基本包外，需要 `xgboost`。本地 macOS 的 XGBoost 需要可用的 OpenMP 动态库；
本次测试通过进程环境变量引用独立测试环境中 sklearn 自带的 libomp，没有修改系统库。

实验覆盖 Breast Cancer（二分类）、Iris（多分类）、Diabetes（回归），分别使用随机森林与
XGBoost，共 6 个组合；每次 1 个调参 trial，固定 8 棵树、最大深度 3，SHAP 最多解释 12 个样本。
这些是恢复机制的低成本验证参数，不用于预测性能比较。

每个组合保留 8 项检查结果：

1. 首次同时注入 SHAP 故障和统一报告渲染故障，检查模型仍正常交付。
2. 解释继续失败时，即使报告已恢复，也不能把整体标为完整。
3. 移除故障后，调用真实 SHAP 并重新生成报告，检查交付完整性。
4. 删除本次产生的一张解释图，检查识别为缺失并能补齐。
5. 删除本次保存的模型文件，直接从内存中的已拟合对象重新保存，核对字节一致。
6. 所有恢复期间拦截模型 fit，确认没有重新训练。
7. 比较恢复前后同一测试输入的预测。
8. 改变输入后，确认执行任何补做前就拒绝恢复。

本轮 6/6 组合通过各自 8 项检查，共 48 个检查结果通过，恢复时 fit 调用均为 0。
输出 `summary.html`（中文说明）、`summary.json`（逐项结果），以及各运行目录中的
`validation_snapshots/`（故障、持续失败、恢复、删除与最终状态）。失败组合也会保留在汇总中，
脚本退出状态为非零，不会只汇总成功案例。

这是两模型族、三个内置公开数据集的小规模故障注入验证；48 项检查不是 48 个独立数据集实验。
未测试自然失败率、完整重跑加速比、跨进程恢复或跨硬件复现。TrustGuard 未合并。

## 第三步：原始表格推断与公平性边界修复

- 持久化训练阶段拟合的填补、编码、缩放及日期转换状态，不保存训练行、不在预测时重新拟合。
- `best_model.joblib` 保留处理后输入兼容用法；新增 `pipeline.joblib` 接收原始 DataFrame。
- 新增必需项 `reload_check`：重载后对原始测试输入比较预测与概率，结果写入 `evidence/reload_check.json`。核对失败时证据包不完整。
- 修正展开特征后的特征名顺序，使解释名称与实际模型输入顺序一致。
- 公平性正类预测率始终来自离散预测，而非预测概率均值；原始敏感属性保留缺失值分组，未知独热编码不猜测类别。
- 分组缺少正/负样本时相关比率为 null；少于两个分组或任一所需比率未定义时相关差异为 null，HTML 说明无法计算原因。
- 完整交付表示产生了结果和边界说明，不表示所有公平指标可计算或模型公平。没有合并 TrustGuard。

第三版真实实验在原有八项检查之外增加原始表格重载检查：6/6 组合、54 项检查通过。
补充回归测试覆盖混合类型表格、缺失值、未知类别、列顺序变化、缺列拒绝、禁止预测时重新拟合及公平性未定义情况。

使用（在相同依赖环境且已安装本分支 tabularx 时）：

```python
import joblib
pipeline = joblib.load("本次运行目录/pipeline.joblib")
predictions = pipeline.predict(raw_dataframe)
```

仅加载可信的本地 joblib 文件。不同软件版本或硬件环境的完全复现尚未验证。
跨进程故障恢复与完整数据泄露审计仍是后续事项，本次不宣称已经完成。

## 第四步：验证集选模与最终测试分离

- `_run_model_tuning` 不再接收测试数据；候选调参和分数仅来自训练/验证集，支持串行和并行。
- `search_models` 的 `best_test_metrics` 和候选 `test_metrics` 现为 `None`，直到 AutoEstimator 锁定模型并完成最终测试。调用方不能再假设搜索步骤提供测试分数。
- Voting 仅在验证集比较，不访问测试集、不自动替换最优单模型。其报告只比较同一验证集上的指标。
- 拟合最终模型前写入 `evidence/selection_lock.json`：模型、参数、验证指标和选择依据；它同时成为交付清单的必需项。
- 仅选中单模型有测试分数及测试诊断图；非选中模型和 Voting 的测试诊断记为不适用。
- 模板流程图与文字同步修改，避免仍显示“按测试分数排序”或把验证/测试指标混合比较。

新增 `tests/test_selection_protocol.py`：用禁止访问测试属性的代理验证串行/并行搜索和 Voting，
并以两个候选模型验证 Iris 与 Diabetes 完整运行中的锁定记录、分数归属和诊断范围。
`drafts/introduction/selection_protocol_demo.py --output <目录>` 保留两组真实运行与中文结果页。

边界：profile/类型识别仍读取全量输入；诊断、解释和重载核对会在锁定后使用测试输入。
这里保证默认搜索阶段不读取测试数据、测试分数不决定候选选择，不宣称不存在任何数据泄露风险。
独立的旧 `save_voting_plots` 工具仍可被外部显式调用；AutoEstimator 默认流程不再调用它。


本周简明使用说明与创新对照见 [历史同步说明](teammate-sync.md)。Adult 完整数据验证脚本为 `drafts/introduction/adult_evidence_demo.py`。
