# 证据交付与局部补做说明书

本模块在 TabularX 的训练、解释与报告流程上增加交付检查和局部恢复能力：先声明本次运行应产生哪些证据，再逐项记录适用条件、生成状态、文件位置和失败原因；在同一程序会话内，可补做允许恢复的失败或缺失项目。

可信审计回答“数据和建模过程有哪些风险”，本模块回答“应交付的证据是否完整生成”。当前主流程已将画像、划分、预处理和模型选择的审计结果纳入报告与证据清单，详见[可信审计整合说明书](trust-audit-integration-guide.md)。交付完整不表示审计无风险。

## 1. 功能定位与工程贡献

| 部分 | 原有能力 | 本模块新增或修正 |
|---|---|---|
| 训练与解释 | 模型搜索、SHAP、公平性计算和 HTML 报告 | 复用已有算法，不将这些能力列为新增贡献 |
| 证据交付检查 | 保存模型、图和报告 | 预先声明必需性、适用性与启用条件；检查本次运行产物，报告缺口与原因 |
| 局部补做 | 缺少统一的逐项恢复入口 | 复用已拟合模型，只补做允许恢复的失败或缺失项，保留尝试历史并刷新关联报告 |
| 模型复用 | 保存处理后输入模型 | 同时保存拟合后的预处理器和模型，支持原始表格推断及同环境重载核对 |
| 评估协议 | 候选阶段计算测试指标 | 依据验证集选模，测试前锁定最终模型，仅为锁定模型生成最终测试指标与诊断 |
| 公平性边界 | 分组指标与公平性差异 | 保留原始敏感分组，明确任务边界；无法计算时记录空值及原因，不以零替代 |

工程贡献是相对于仓库原有实现的交付保障与恢复机制，不主张新增预测算法或学术首创。交付完整率与模型准确率、解释质量、公平性是不同指标。

## 2. 实现结构与交付规则

### 2.1 模块结构

| 位置 | 职责 |
|---|---|
| [core/evidence/delivery.py](../TabularX-project/src/tabularx/core/evidence/delivery.py) | 注册交付要求、执行生产者、检查文件、记录尝试历史、调度补做 |
| [core/evidence/estimator.py](../TabularX-project/src/tabularx/core/evidence/estimator.py) | 将模型、解释、公平性、审计和报告接入交付流程 |
| [estimator.py](../TabularX-project/src/tabularx/estimator.py) | 在拟合主流程中调用证据交付模块 |
| [core/inference.py](../TabularX-project/src/tabularx/core/inference.py) | 保存和复用拟合后的预处理状态 |

```text
声明交付要求 → 执行证据生产者 → 核对本次运行文件 → 生成交付清单
                                  ↓
                         发现失败或缺失项目
                                  ↓
               同会话补做 → 刷新关联报告 → 再次检查
```

交付规则由 Python 模块注册并执行，JSON 是状态记录与交换格式，不是可执行恢复程序。当前交付记录使用字典结构，不能将其等同于已建立独立的强类型公共 schema 协议层。

### 2.2 默认交付集合

| 证据 | 适用条件及要求 |
|---|---|
| 选模锁定、最终模型、预处理流水线、候选分数、重载核对 | 主流程必需项 |
| 审计汇总与模型选择协议审计 | 主流程必需项；有风险也应如实交付 |
| SHAP 汇总图、重要性图、局部图 | 当前解释步骤必需项，缺少预期图会记录失败 |
| 混淆矩阵、ROC 曲线 | 仅锁定分类模型生成混淆矩阵；二分类另需 ROC，回归不适用 |
| 调参轨迹 | 有试验记录时必需；没有记录时注明不适用 |
| 公平性结构化结果 | 提供敏感字段且符合当前 0/1 二分类实现时必需，正类为 1 |
| 各阶段及总 HTML 报告 | 必需项；报告存在不代表上游证据齐全 |
| PDF 报告 | 显式请求时必需，否则记为未启用 |
| 集成模型报告 | 有验证集集成结果时提供，可选；不生成集成模型测试诊断图 |

未提供敏感字段时，公平性未启用；回归、多分类或不支持的标签形式注明不适用。以上是当前已声明的交付集合，不代表 ALE、学习曲线等全部目标功能已经实现；调参轨迹不是学习曲线。

### 2.3 状态与完整性

| 状态 | 含义 |
|---|---|
| `pending`（待执行） | 已声明要求，尚未执行 |
| `generated`（已生成） | 生产者执行成功，返回的文件位于本次运行目录内、存在且非空 |
| `failed`（失败） | 执行异常或生产者明确报告失败，保留原因 |
| `missing`（缺失） | 预期文件缺少、为空，或生成后被删除等 |
| `not_enabled`（未启用） | 功能未开启，注明原因 |
| `not_applicable`（不适用） | 本任务不符合该证据的适用条件，注明原因 |

必需项完整率 = 已生成的必需适用项数 ÷ 全部必需适用项数。必需且适用的项目即使未启用，也不能视为完成；没有任何必需适用项时，完整率为 `null`，不判为完整。

文件存在且非空只说明通过交付检查，不验证内容科学正确，也不等于文件哈希校验。各次尝试分别保留状态、原因、路径与时间，成功补做不会抹去此前失败记录。

## 3. 使用方法

### 3.1 安装与运行

在仓库根目录、独立 Python 环境中执行：

```bash
python -m pip install -e ./TabularX-project
python -m pip install optuna matplotlib shap xgboost joblib pytest
```

XGBoost 可能需要系统 OpenMP，PDF 导出可能需要 WeasyPrint 的系统依赖，需按运行环境配置。

以下示例使用内置乳腺癌数据集，降低验证成本，不用于模型性能排名：

```python
from sklearn.datasets import load_breast_cancer
from tabularx import AutoEstimator

sample = load_breast_cancer(as_frame=True)
ae = AutoEstimator(task="classification", output_dir="outputs/example")
ae.param.model_search.models = ["random_forest"]
ae.param.model_search.n_jobs = 1
ae.param.model_search.n_trials = 1
ae.param.model_search.fixed_params = {
    "random_forest": {"n_estimators": 8, "max_depth": 3}
}
ae.param.shap.max_explain_samples = 12
ae.fit(sample.data, sample.target)

print(ae.result.report_paths["full"])      # 总报告
print(ae.result.report_paths["evidence"])  # 交付检查页面
print(ae.evidence_.audit()["complete"])    # 当前内存中的检查结果
```

自有数据中，X 应为原始 DataFrame，y 为同序且索引对齐的标签。可调用 `ae.fit(X, y, sensitive_features=["sex"])` 请求分组评估；字段必须存在，指定为敏感字段本身不会自动将其从模型输入中排除。回归使用 `task="regression"`。

### 3.2 补做失败或缺失项目

```python
# 保留原来的 ae 实例，在故障原因消除后执行。
# 没有待补做项时可直接返回当前检查结果。
status = ae.retry_evidence()

# 也可指定项目，但仅限当前失败/缺失且允许恢复的项：
# status = ae.retry_evidence(["shap"])
```

补做复用同一个已拟合模型、数据和配置，不重新搜索或训练。关联报告与重载核对可随补做刷新，因此不应理解为所有已成功项目都永不再执行。改变数据、模型或配置（包括解释预算）会被拒绝，需新建运行。训练失败不属于本模块自动恢复范围。

补做需要原程序会话仍在；程序退出后，仅凭 JSON 清单不能恢复执行。补做状态一致性修复见团队上游的[合并请求 #2](https://github.com/gaozhemei-hub/ML26-gzm/pulls)：将审计刷新纳入统一流程，避免报告重复刷新以及返回状态与落盘状态不一致。该修复已合并到 `develop`。

### 3.3 重载模型进行预测

```python
import joblib

pipeline = joblib.load(ae.output_dir / "pipeline.joblib")
predictions = pipeline.predict(sample.data.iloc[:5])
```

只加载可信来源的 joblib 文件。模型重载可在兼容环境的新进程中使用；它与必须保留原实例的证据补做是两种能力。

## 4. 输出文件与报告入口

每次运行使用独立目录 `<output_dir>/runs/<run_id>/`，避免将旧运行产物误认成本次结果。实际目录从 `ae.output_dir` 获取，报告路径从 `ae.result.report_paths` 获取。

| 输出 | 用途 |
|---|---|
| `report.html` | 模型、解释和审计总报告 |
| `reports/` | 各阶段报告及图形资源 |
| `evidence/delivery.html` | 查看交付缺口、失败原因和补做历史 |
| `evidence/delivery.json` | 结构化交付要求、状态、路径和尝试历史 |
| `evidence/audit.json` | 画像、划分、预处理和模型选择审计汇总 |
| `evidence/model_search_audit.json` | 评估协议审计 |
| `evidence/selection_lock.json` | 测试评估前的选模依据与锁定结果 |
| `evidence/model_scores.json` | 候选结果与选择指标 |
| `evidence/fairness.json` | 启用且适用时的公平性结果 |
| `evidence/reload_check.json` | 原始测试表格重载预测及概率一致性核对 |
| `best_model.joblib` | 接收处理后输入的模型 |
| `pipeline.joblib` | 接收原始表格的预处理器与模型 |
| `report.pdf` | 请求 PDF 且成功生成时的导出报告 |

## 5. 验证结果与复跑

完整产物和原始结果见[实验验证索引](../docs/validation/evidence-delivery/README.md)。不同批次对应不同代码阶段，不应当作同一版本的重复实验。

| 验证范围 | 观测结果 | 结论边界 |
|---|---|---|
| 早期实现自动测试 | 217 项通过 | 历史测试结果，不代表当前全部测试数 |
| Breast Cancer、Iris、Diabetes × 随机森林、XGBoost | 6 个组合、54 项检查通过，含故障检出、真实补做、删除后恢复、原始表格重载和禁止重训 | 小规模故障注入；54 项是检查数，不是独立实验数；该批次先于最终测试协议修复 |
| Iris、Diabetes 双候选运行 | 验证集选模后锁定单模型，仅锁定模型生成测试指标；Voting 只作验证集比较 | 验证选模协议及证据范围 |
| Adult 全量数据 | 48,842 行、14 特征，两候选模型运行成功；SHAP 首次故障被识别，真实补做后完整，恢复训练次数为 0，原始表格重载预测一致 | 每候选仅一次小规模调参，32 棵树、深度 5，解释 24 个测试样本；验证工程机制 |
| 补做状态一致性修复 | 包含可信审计测试在内的全套 419 项通过，新增 3 项回归用例 | 419 是整套测试数；PDF 以模拟导出器验证顺序，不验证实际排版 |

Adult 的正类测试 F1 为 0、召回率为 0，ROC AUC 约为 0.902，说明该配置不能证明正类识别效果良好。对应公平性差异为零也不能据此证明模型公平。本实验只支持交付与恢复机制的验证；该批次性别字段同时用于模型输入和审计分组，不与早期排除性别输入的独立原型直接比较。

仓库根目录复跑：

```bash
PYTHONPATH=.:TabularX-project/src python -m pytest tests drafts/trustguard/tests -q
PYTHONPATH=TabularX-project/src python drafts/introduction/evidence_validation_suite.py --output outputs/evidence-validation
PYTHONPATH=TabularX-project/src python drafts/introduction/selection_protocol_demo.py --output outputs/selection-protocol
PYTHONPATH=TabularX-project/src python drafts/introduction/adult_evidence_demo.py --data /path/to/adult.arff.gz --output outputs/adult-validation
```

Adult 数据需自行准备，归档不包含原始数据和虚拟环境。当前代码复跑的产物不保证与历史快照逐字节相同。

## 6. 功能边界

1. 交付完整仅表示满足当前声明的要求，不代表解释忠实、模型公平或审计无风险。
2. 仅支持同会话局部补做，不支持进程重启后的恢复，也不自动恢复模型训练失败。
3. 重载核对验证同环境预测一致性，不代表跨操作系统、依赖版本和硬件完全复现。
4. 当前 DP/EO 仅适用于实现支持的 0/1 二分类；少于两个分组或分母不足时记录空值，不提供公平性缓解算法。
5. 候选搜索使用训练与验证数据；锁定后的解释、诊断和重载核对仍使用测试输入，不表示测试数据只被读取一次。
6. 故障验证采用人工注入，尚未测量自然失败率、完整重跑加速比或跨硬件运行开销。
7. 当前仍待扩展 ALE、学习曲线、独立强类型交付协议及跨进程恢复等能力。
