# TabularX API 参考手册 (API Reference)

`TabularX` 是一个**面向可解释性与可复现性的自动化表格建模工具包**。本参考手册提供工具包所有公开模块、核心类、配置参数、结果契约、可信审计与证据交付接口的精确技术规格。

---

## 目录

1. [快速入门与端到端示例](#1-快速入门与端到端示例)
2. [顶级建模入口 (AutoEstimator)](#2-顶级建模入口-autoestimator)
3. [参数配置体系 (TabularXParams)](#3-参数配置体系-tabularxparams)
   - [ProfileConfig（画像与静态数据泄漏检查）](#31-profileconfig)
   - [SplitConfig（划分与完整性审计）](#32-splitconfig)
   - [PreprocessConfig（预处理与分布漂移审计）](#33-preprocessconfig)
   - [ModelSearchConfig（模型搜索与超参数调优）](#34-modelsearchconfig)
   - [ShapConfig（SHAP 解释配置）](#35-shapconfig)
   - [FairnessConfig（敏感属性公平性评估）](#36-fairnessconfig)
4. [结果契约与中间对象 (Result Schemas)](#4-结果契约与中间对象-result-schemas)
   - [AutoResult](#41-autoresult)
   - [DataProfile / ColumnFact](#42-dataprofile--columnfact)
   - [SplitResult](#43-splitresult)
   - [PreprocessedData](#44-preprocesseddata)
   - [ModelSearchResult / ModelRunResult](#45-modelsearchresult--modelrunresult)
   - [ShapResult / FairnessResult](#46-shapresult--fairnessresult)
5. [可信审计体系 (TrustGuard Audit API)](#5-可信审计体系-trustguard-audit-api)
   - [公共审计对象 (AuditFinding, AuditCheckResult, AuditReport)](#51-公共审计对象)
   - [四阶段审计产物](#52-四阶段审计产物)
6. [证据交付与推断 (Evidence Delivery & Inference)](#6-证据交付与推断-evidence-delivery--inference)
   - [EvidenceDelivery 与交付验收](#61-evidencedelivery-与交付验收)
   - [模型持久化与推断流水线 (pipeline.joblib)](#62-模型持久化与推断流水线)
7. [异常体系 (Exceptions)](#7-异常体系-exceptions)

---

## 1. 快速入门与端到端示例

```python
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from tabularx import AutoEstimator

# 1. 准备数据
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

# 2. 初始化高阶估计器
ae = AutoEstimator(
    task="classification",
    output_dir="./outputs/breast_cancer_run",
    pdf_report=False
)

# 3. 定制超参数（可选）
ae.param.model_search.models = ["random_forest", "xgboost"]
ae.param.model_search.n_trials = 10

# 4. 执行完整流水线（数据画像 → 划分 → 预处理 → 搜索调优 → 解释/公平性评估 → 证据交付）
ae.fit(X, y)

# 5. 查看输出与审计状态
print("最佳模型:", ae.result.searched.best_model_name)
print("交付验收报告路径:", ae.result.report_paths["evidence"])
print("交付是否完整:", ae.evidence_.audit()["complete"])

# 6. 端到端推断（接收原始 DataFrame，自动应用拟合好的预处理变换）
pipeline = joblib.load(ae.output_dir / "pipeline.joblib")
preds = pipeline.predict(X.head(5))
print("预测结果:", preds)
```

---

## 2. 顶级建模入口 (AutoEstimator)

### `tabularx.AutoEstimator`

高阶一键建模估计器，混入了 `EvidenceEstimatorMixin`，实现从配置、训练、解释、四阶段审计到文件级证据交付的闭环。

```python
class tabularx.AutoEstimator(
    task: str = "auto",
    output_dir: str | Path = "./tabularx_output",
    pdf_report: bool = False
)
```

#### 参数 (Parameters)

| 参数名 | 类型 | 默认值 | 详细说明 |
| :--- | :--- | :--- | :--- |
| `task` | `str` | `"auto"` | 建模任务类型。可选值：`"auto"`（自动根据目标变量推断）、`"classification"`（分类，支持二分类与多分类）、`"regression"`（回归）。 |
| `output_dir` | `str \| Path` | `"./tabularx_output"` | 实验工件、模型文件与 HTML/PDF 报告的根输出目录。每次 `fit()` 会自动在该目录下创建带独立哈希的 `runs/<run_id>/` 子目录。 |
| `pdf_report` | `bool` | `False` | 是否在生成 HTML 报告的同时导出 PDF 格式完整报告（需系统预装 `weasyprint`）。 |

#### 核心属性 (Attributes)

| 属性名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `param` | `TabularXParams` | 全流程集中参数配置管理器，包含各子阶段的 Config。 |
| `result_` / `result` | `AutoResult` | 建模执行完毕后的各阶段工件总入口。未调用 `fit()` 前为 `None`，通过属性访问器 `result` 在未拟合时抛出 `RuntimeError`。 |
| `output_dir` | `Path` | 当前次运行的绝对路径目录（解析为 `<output_dir>/runs/<run_id>`）。 |
| `evidence_` | `EvidenceDelivery` | 交付验收引擎实例，负责注册、校验和补救本次运行声明的所有交付工件。 |

#### 方法 (Methods)

##### `fit(X, y, sensitive_features=None) -> AutoEstimator`
执行完整的端到端训练与可信审计流程。

* **参数**：
  * `X` (`pd.DataFrame`): 输入特征矩阵，形状为 `(n_samples, n_features)`。
  * `y` (`pd.Series`): 目标变量，形状为 `(n_samples,)`，索引须与 `X` 对齐。
  * `sensitive_features` (`list[str] | None`, 可选): 原始表格中的敏感属性列名列表（如 `["sex", "race"]`），用于公平性评估审计。
* **返回**：
  * `self`: 拟合后的 `AutoEstimator` 实例。
* **行为保证**：
  * 自动在搜索与 Voting 阶段激活 `_SelectionDataView` 保护视图，**硬拦截任何访问测试集的行为**。
  * 在访问测试集前固化并持久化 `selection_lock.json`（选模锁定记录）。

##### `retry_evidence(keys=None) -> dict`
在当前活体 Python 会话中重试失败或缺失的交付项（如因依赖缺失导致的 SHAP 图生成失败）。

* **参数**：
  * `keys` (`list[str] | None`, 可选): 指定需要重试的证据项键名列表（如 `["shap"]`）。若为 `None`，则重试当前所有状态为 `failed` 或 `missing` 的可恢复项。
* **约束与保证**：
  * **绝不重新训练模型**：复用内存中已拟合的模型与预处理流水线。
  * **环境指纹校验**：通过 `_context_token()` 校验数据、模型和参数指纹；若环境发生改变则拒绝恢复，要求重新运行 `fit()`。

---

## 3. 参数配置体系 (TabularXParams)

所有阶段的配置集中于 `tabularx.core.params.TabularXParams` 实例中。可以通过 `ae.param.<stage>.<param_name> = value` 直接配置，也可以使用 `set_param(name, value)` 获得严格的范围与约束检查。

```python
# 示例：集中修改超参数
ae.param.split.test_size = 0.2
ae.param.model_search.models = ["lightgbm", "xgboost"]
ae.param.preprocess.apply_audit_drops = "high"
```

### 3.1 ProfileConfig

负责数据画像统计与 TrustGuard 静态数据泄漏预检。

| 参数名 | 类型 | 默认值 | 校验范围 / 选项 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `enable_leakage_audit` | `bool` | `True` | - | 是否执行静态数据泄漏与风险预检。 |
| `identifier_unique_ratio` | `float` | `0.98` | `[0.0, 1.0]` | 疑似标识符特征阈值：唯一值占比超过此值且列名类似 ID，标记为高危并建议排除。 |
| `target_proxy_threshold` | `float` | `0.98` | `[0.0, 1.0]` | 疑似目标代理特征评分阈值：单特征预测能力超过此阈值，标记为目标泄露风险。 |
| `id_columns` | `list[str]` | `[]` | - | 用户显式指定的 ID 列名列表，直接标记为标识符风险。 |
| `forbidden_features` | `list[str]` | `[]` | - | 用户声明预测时不可得的禁止特征列表。 |
| `text_unique_threshold` | `int` | `200` | `>= 1` | 文本列判定：唯一值超过此数量且占比高时判定为文本列。 |
| `int_categorical_threshold` | `int` | `20` | `>= 1` | 整数列判定：唯一取值数量小于等于此值视为类别列。 |
| `imbalance_ratio` | `float` | `0.1` | `(0.0, 1.0)` | 分类任务最小类别占比低于此值标记为类别不平衡。 |

### 3.2 SplitConfig

控制样本切分比例、切分策略及划分完整性审计。

| 参数名 | 类型 | 默认值 | 校验范围 / 选项 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `test_size` | `float` | `0.2` | `(0.0, 1.0)` | 测试集占总样本的比例。 |
| `validation_size` | `float` | `0.25` | `(0.0, 1.0)` | 验证集相对于训练集+验证集总量的比例。 |
| `random_state` | `int` | `42` | `>= 0` | 随机种子，确保切分索引完全复现。 |
| `split_strategy` | `str` | `"auto"` | `{"auto", "stratified", "random", "group", "temporal"}` | 数据划分策略。`auto` 会结合目标和感知列声明自动选择。 |
| `group_column` | `str \| None` | `None` | - | 实体分组列名（如患者 ID）。声明后用于感知切分并审计跨集合实体泄漏。 |
| `time_column` | `str \| None` | `None` | - | 时间列名。声明后用于时序切分并审计时间穿越。 |
| `enable_split_audit` | `bool` | `True` | - | 是否执行划分完整性审计（重复样本、跨集合重叠）。 |
| `duplicate_audit_max_rows` | `int` | `200000` | `>= 1` | 跨集合重复检测的最大行数，超出会自动抽样检测。 |

### 3.3 PreprocessConfig

控制特征工程处理方法及分布漂移审计与自动剔除。

| 参数名 | 类型 | 默认值 | 校验范围 / 选项 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `numeric_impute` | `str` | `"median"` | `{"median", "mean", "constant"}` | 数值型缺失值填充策略。 |
| `categorical_impute` | `str` | `"most_frequent"` | `{"most_frequent", "constant"}` | 类别型缺失值填充策略。 |
| `scale_method` | `str` | `"standard"` | `{"standard", "minmax", "robust"}` | 数值特征标准化缩放算法。 |
| `skew_method` | `str` | `"yeo-johnson"` | `{"yeo-johnson", "log1p"}` | 偏斜特征校正变换方法。 |
| `low_cardinality_encoding` | `str` | `"onehot"` | `{"onehot", "label"}` | 低基数类别编码策略。 |
| `high_cardinality_encoding`| `str` | `"target"` | `{"target", "frequency"}` | 高基数类别编码策略。 |
| `drop_group_column` | `bool` | `True` | - | 声明了 `group_column` 时是否自动将其从模型输入特征中排除。 |
| `apply_audit_drops` | `str` | `"off"` | `{"off", "high", "all"}` | 审计建议特征剔除联动开关。`"off"`：仅告警；`"high"`：自动剔除 high 与 critical 风险列；`"all"`：剔除所有建议排除列。 |
| `enable_drift_audit` | `bool` | `True` | - | 是否在训练集、验证集和测试集间执行 PSI / KS / TVD 分布漂移审计。 |
| `psi_warning` / `psi_high` | `float` | `0.10` / `0.25` | `0 <= warn <= high` | PSI（群体稳定性指标）中度风险与高危漂移阈值。 |
| `min_group_size` | `int` | `20` | `>= 1` | 敏感子群体在各数据子集中的最少样本数，过低标记为覆盖不足风险。 |

### 3.4 ModelSearchConfig

控制候选算法、Optuna 超参数搜索及集成选择。

| 参数名 | 类型 | 默认值 | 校验范围 / 选项 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `models` | `list[str]` | `["auto"]` | - | 待搜索的模型名称列表。`["auto"]` 会搜索注册表支持的所有算法。支持：`"random_forest"`, `"xgboost"`, `"lightgbm"`, `"catboost"`。 |
| `n_trials` | `int` | `50` | `>= 1` | 每个算法在验证集上用 Optuna 调优的试验轮数。 |
| `n_jobs` | `int` | `4` | `>= 1` | 模型间并行调优进程数。 |
| `primary_metric` | `str` | `"auto"` | - | 选模与超参优化主指标。`"auto"` 下回归默认 `rmse`，二分类默认 `roc_auc`，多分类默认 `f1_macro`。 |
| `device` | `str` | `"cpu"` | `{"cpu", "gpu", "auto"}` | 训练计算设备。 |
| `fixed_params` | `dict` | `{}` | - | 固定的算法超参数字典，指定参数跳过搜索。 |
| `search_space_overrides` | `dict` | `{}` | - | 覆盖特定算法超参数的搜索范围（`low`, `high`, `choices` 等）。 |

### 3.5 ShapConfig

控制基于测试集的 SHAP 可解释性样本预算与图表生成。

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `max_explain_samples` | `int` | `200` | 解释样本数上限。测试集样本超出此值时进行随机截断抽样。 |
| `n_background` | `int` | `100` | KernelExplainer 或无树结构模型时所使用的背景样本数量。 |
| `top_k_features` | `int` | `10` | 重点解释的重要特征统计数量及瀑布图最大显示特征数。 |

### 3.6 FairnessConfig

控制敏感特征子组表现与公平性度量阈值。

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `sensitive_features` | `list[str]` | `[]` | 敏感特征列表（也可在 `fit()` 中传入）。 |
| `dp_threshold` | `float` | `0.1` | 人口统计学奇偶性 (Demographic Parity) 差异告警阈值。 |
| `eo_tpr_threshold` | `float` | `0.1` | 均等化赔率 (Equalized Odds) TPR 差异告警阈值。 |
| `eo_fpr_threshold` | `float` | `0.1` | 均等化赔率 (Equalized Odds) FPR 差异告警阈值。 |

---

## 4. 结果契约与中间对象 (Result Schemas)

执行 `ae.fit()` 成功后，所有中间状态和工件均挂载在 `ae.result`（`AutoResult` 实例）上。

### 4.1 AutoResult

```python
@dataclass
class AutoResult:
    profile: DataProfile              # 阶段 1：数据画像工件
    split: SplitResult                # 阶段 2：数据集划分工件
    preprocessed: PreprocessedData    # 阶段 3：预处理与特征工件
    searched: ModelSearchResult       # 阶段 4：模型搜索与评估工件
    shap: ShapResult | None           # 可解释性结果
    fairness: FairnessResult | None   # 公平性评估结果
    output_dir: Path | None           # 输出目录路径
    report_paths: dict[str, str]      # 各独立与汇总 HTML 报告路径
```

#### 常见 `report_paths` 键名：
* `"full"`: 汇总全量交互式报告 (`report.html`)。
* `"evidence"`: 交付验收页面 (`evidence/delivery.html`)。
* `"evidence_json"`: 机器可读的结构化交付记录 (`evidence/delivery.json`)。
* `"profile"` / `"split"` / `"preprocess"` / `"model_search"` / `"explanation"`: 各阶段分项报告。

### 4.2 DataProfile / ColumnFact

* **`DataProfile`**：
  * `n_rows` (`int`), `n_cols` (`int`): 数据集行列规模。
  * `columns` (`dict[str, ColumnFact]`): 单列特征统计。
  * `target` (`TargetFact`): 目标变量属性与推断任务类型。
  * `audit` (`ProfileAuditResult`): 静态泄漏审计结果。
* **`ColumnFact`**：包含 `col_type`（数值/类别/文本/日期/ID等）、`n_missing`、`n_unique`、`mean`、`std`、`skew` 等统计指标。

### 4.3 SplitResult

* `train_indices` / `valid_indices` / `test_indices` (`np.ndarray`): 切分后的行索引数组。
* `split_strategy` (`str`): 实际执行的切分方式。
* `audit` (`SplitAuditResult`): 划分完整性审计结果。

### 4.4 PreprocessedData

* `X_train`, `X_valid`, `X_test` (`pd.DataFrame`): 变换后的数值特征矩阵。
* `y_train`, `y_valid`, `y_test` (`pd.Series`): 编码后的标签向量。
* `fitted_preprocessor` (`TransformerMixin`): 拟合好的 scikit-learn 复合预处理器。
* `audit` (`PreprocessAuditResult`): 分布漂移与敏感群体覆盖审计结果。
* `excluded_by_audit` (`list[str]`): 被审计联动规则实际剔除的特征列名。

### 4.5 ModelSearchResult / ModelRunResult

* **`ModelSearchResult`**：
  * `best_model_name` (`str`): 依据验证集指标选出的最优模型名称。
  * `best_model_params` (`dict`): 最优超参数组合。
  * `primary_metric` (`str`), `optimization_direction` (`"maximize" | "minimize"`): 选模优化目标。
  * `best_test_metrics` (`dict[str, float]`): 仅对锁定最优模型在测试集上计算的最终测试指标。
  * `ensemble_result` (`ModelRunResult | None`): Voting 组合模型的验证集诊断结果。
* **`ModelRunResult`**：单个模型的运行记录，含 `validation_metrics`、`optuna_trials`、`artifact_paths` 等。

### 4.6 ShapResult / FairnessResult

* **`ShapResult`**：
  * `feature_importance` (`dict[str, float]`): 全局平均绝对 SHAP 值。
  * `artifact_paths` (`dict[str, str]`): 包含 summary 图、条形图与局部瀑布图的相对文件路径。
* **`FairnessResult`**：
  * `feature_results` (`list[FairnessFeatureResult]`): 各指定敏感特征在各子群上的正类预测率、TPR、FPR 及子组差异。

---

## 5. 可信审计体系 (TrustGuard Audit API)

TabularX 的原生特色是将 **TrustGuard 可信审计契约** 深度集成在全生命周期建模中。公共数据结构位于 `tabularx.schema.audit`。

### 5.1 公共审计对象

#### `AuditSeverity` 与 `SEVERITY_ORDER`
风险等级定义为五个层级（按严重程度递增）：
`"info"` (0) < `"low"` (1) < `"medium"` (2) < `"high"` (3) < `"critical"` (4)。

#### `AuditFinding`
单项审计风险条目：

```python
@dataclass
class AuditFinding:
    code: str                  # 机器可读的唯一风险标识码 (如 "identifier_leakage")
    severity: AuditSeverity    # 严重程度 ("info" ~ "critical")
    title: str                 # 风险标题说明
    evidence: str              # 具体的数字或事实依据
    recommendation: str        # 处置建议
    columns: list[str]         # 涉事特征列名
    details: dict              # 附加调试上下文
```

#### `AuditCheckResult`
固定检查项判定：
* `code` (`str`): 检查项代号。
* `check` (`str`): 检查项中文名。
* `status` (`"通过" | "发现风险" | "未执行"`): 检查状态。
* `result` (`str`): 判定原因与结论。

### 5.2 四阶段审计产物

每个审计结果继承自 `AuditReport`，具备 `.risk_level`（最高风险等级）与 `.to_dict()` 序列化能力：

1. **`ProfileAuditResult`（画像静态预检）**：
   * 检查高基数 ID 标识符泄露、目标直接复制、高判别力目标代理特征。
   * 产出 `suggested_drop_columns`（建议排除特征清单）。
2. **`SplitAuditResult`（划分完整性审计）**：
   * 跨训练/验证/测试集的哈希重复样本检测 (`duplicate_count`)。
   * 感知分组切分下的实体跨集合重叠检测 (`group_overlap_count`)。
3. **`PreprocessAuditResult`（预处理分布与覆盖审计）**：
   * 记录逐特征的两两子集间分布漂移指标（PSI, KS, TVD）。
   * 检查敏感子群体在训练集/验证集/测试集中的绝对样本量覆盖。
4. **`ModelSearchAuditResult`（模型搜索评估协议）**：
   * 验证是否在访问测试集前完成单模型锁定 (`locked_before_test_evaluation`)。
   * 审计是否存在未锁定的候选模型或 Voting 集成窥探测试集 (`candidate_test_metrics_count`)。

所有阶段审计结果汇总为全局结构并持久化至 `evidence/audit.json` 和 `evidence/model_search_audit.json`。

---

## 6. 证据交付与推断 (Evidence Delivery & Inference)

### 6.1 EvidenceDelivery 与交付验收

`EvidenceDelivery` 确保自动化建模不是黑箱：**每次实验前声明交付契约，实验后逐项验收文件存在性与非空性，拒绝静默丢失工件**。

#### 交付状态分类
* `"pending"`: 待执行。
* `"generated"`: 生产步骤成功，声明的文件存在且非空。
* `"failed"`: 生产步骤执行抛出异常。
* `"missing"`: 生产步骤结束但声明的文件未落盘或大小为 0。
* `"not_enabled"`: 显式未启用（如未请求 PDF 导出）。
* `"not_applicable"`: 任务类型不适用（如回归任务不生成混淆矩阵与 ROC 图）。

#### 交付清单方法
* `ae.evidence_.audit() -> dict`:
  * 返回 `complete` (`bool`，必需项是否全部生成)、`required_count`、`generated_required_count` 与 `records` 详情。
  * 状态持久化写入 `evidence/delivery.json` 与中文验收报告 `evidence/delivery.html`。

### 6.2 模型持久化与推断流水线

每次拟合成功后，当前运行目录下固定生成两个模型工件：

#### 1. `pipeline.joblib` (端到端推断流水线 - 推荐)
包含拟合完毕的预处理器和模型对象（`sklearn.pipeline.Pipeline`）。
```python
import joblib

# 直接输入未经任何处理的原始 DataFrame
pipeline = joblib.load("runs/<run_id>/pipeline.joblib")
predictions = pipeline.predict(raw_new_df)
```
* **一致性核对机制**：在交付保存阶段，系统自动执行 `reload_check`，比较测试集通过 `pipeline` 预测的结果与内存最优模型直接预测的结果；如果预测概率或类别不匹配，交付清单判定为失败并报警。

#### 2. `best_model.joblib` (纯模型对象)
仅包含最底层算法估计器（如 `XGBClassifier` 或 `RandomForestRegressor`），接收经 `preprocessed.X_test` 类似格式的变换后数值矩阵。

---

## 7. 异常体系 (Exceptions)

所有自定义异常位于 `tabularx.exceptions`。异常分层清晰，便于上层精细化捕获：

```text
TabularXError (基类，统一记录发生阶段 stage)
 ├── DataError (数据类问题，通常需调整输入数据或配置)
 │    ├── InvalidDataError / EmptyDataError / MismatchedLengthError
 │    ├── EmptyTargetError / TaskTypeError / InsufficientDataError
 ├── ProfileError (画像阶段分析异常)
 ├── SplitError (划分异常)
 │    ├── InvalidSplitError
 │    ├── StratifyError (类别样本数不足以分层切分)
 │    ├── SplitColumnMissingError (未找到声明的 group_column 或 time_column)
 │    └── TimeParseError (时序解析失败)
 ├── PreprocessError (预处理异常)
 │    ├── TextColumnNotSupportedError
 │    └── TransformError
 ├── ModelSearchError (模型搜索与调优异常)
 │    ├── ModelNotAvailableError (缺少第三方算法库依赖，如 xgboost)
 │    ├── ModelTrainingError (底层算法拟合失败)
 │    └── OptunaError (超参搜索空间异常)
 └── ShapError / FairnessError (解释与公平性计算异常)
```

#### 异常捕获示例
```python
from tabularx import AutoEstimator
from tabularx.exceptions import DataError, SplitError, TabularXError

ae = AutoEstimator()
try:
    ae.fit(X, y)
except SplitError as e:
    print(f"数据划分阶段发生错误: {e}")
except DataError as e:
    print(f"输入数据不符合规范: {e}")
except TabularXError as e:
    print(f"TabularX 流水线异常 [{e.stage}]: {e.message}")
```
