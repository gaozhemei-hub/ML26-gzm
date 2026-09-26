# 可信审计整合

可信审计与“证据交付检查与局部补做（`feature/gzm`）”相互补充，共同完善实验可信性。可信审计负责发现数据、划分、分布和模型选择中的风险；证据交付模块负责检查审计结果及其他必需工件是否完整生成，并补做可恢复的失败项。两部分共用报告与 `evidence` 输出体系：一个回答“有哪些可信性风险”，另一个回答“相关证据是否交付完整”。

## 1. 分工与定位

### 本周增量：可信审计整合

原独立 TrustGuard（`feature/trust-audit`）用单独入口完成泄露、评估协议、复现证据、漂移和群体覆盖检查。本周没有把旧目录原样并入主程序，而是将可复用能力拆入 TabularX 的画像、划分、预处理、模型选择和报告流程。

- **先看整合内容与使用边界**：[可信审计整合说明书](https://github.com/gaozhemei-hub/ML26-gzm/blob/develop/docs/trust-audit-integration-guide.md)
- **看画像与划分审计代码**：[画像预检](https://github.com/gaozhemei-hub/ML26-gzm/blob/develop/TabularX-project/src/tabularx/core/profile.py) · [划分审计](https://github.com/gaozhemei-hub/ML26-gzm/blob/develop/TabularX-project/src/tabularx/core/split.py)
- **看漂移、群体覆盖与审计排除**：[预处理审计](https://github.com/gaozhemei-hub/ML26-gzm/blob/develop/TabularX-project/src/tabularx/core/preprocess.py)
- **看公共审计结构与报告聚合**：[审计契约](https://github.com/gaozhemei-hub/ML26-gzm/blob/develop/TabularX-project/src/tabularx/schema/audit.py) · [总流程接入](https://github.com/gaozhemei-hub/ML26-gzm/blob/develop/TabularX-project/src/tabularx/core/evidence/estimator.py)
- **看专项测试**：[划分审计测试](https://github.com/gaozhemei-hub/ML26-gzm/blob/develop/tests/test_split.py) · [预处理审计测试](https://github.com/gaozhemei-hub/ML26-gzm/blob/develop/tests/test_preprocess_audit.py) · [报告审计测试](https://github.com/gaozhemei-hub/ML26-gzm/blob/develop/tests/test_audit_report.py) · [评估协议测试](https://github.com/gaozhemei-hub/ML26-gzm/blob/develop/tests/test_selection_protocol.py)
- **看相关提交**：[画像与总报告](https://github.com/gaozhemei-hub/ML26-gzm/commits/develop) · [划分完整性](https://github.com/gaozhemei-hub/ML26-gzm/commits/develop) · [漂移与群体覆盖](https://github.com/gaozhemei-hub/ML26-gzm/commits/develop) · [评估协议](https://github.com/gaozhemei-hub/ML26-gzm/commits/develop)

整合后的职责分为三块：

| 功能块 | 接入阶段 | 解决的问题 |
|---|---|---|
| 数据泄露防控 | `profile`、`split`、`model-search` | 特征是否提前包含答案，样本关系是否跨集合，测试信息是否参与选模 |
| 复现与可核查证据 | `profile`、`split`、`evidence` | 能否确认输入、划分策略、锁定模型和保存流水线，重载后预测是否一致 |
| 分布漂移与群体覆盖 | `preprocess` | 验证集/测试集是否偏离训练集，敏感群体在各集合中是否覆盖充分 |

三块能力复用公共审计契约，统一输出风险等级、发现项、检查状态和建议动作。各阶段先生成结构化结果，再渲染阶段报告，最后聚合到总报告和 `evidence/audit.json`。

## 2. 本周同步对照表

这周在现有训练、解释和报告流程上增加可信审计与交付保障，没有重新实现 AutoML。旧版 TrustGuard 的可复用能力已拆入主流程，但不是把旧目录逐文件原样合并。

| 部分 | 原有能力 | 本次新增或修正 |
|---|---|---|
| 训练与报告 | 模型搜索、SHAP、公平性计算、HTML 报告 | 不将这些原有算法算作本次新增 |
| 数据泄露防控 | 常规画像、随机/分层切分 | 增加目标复制、目标代理、ID、禁用特征、跨集合重复、实体泄露和时间穿越检查，并支持 group/temporal 切分 |
| 审计策略落地 | 风险提示与模型输入相互独立 | 增加 `drop_group_column` 和 `apply_audit_drops`，记录被排除字段及原因 |
| 漂移与群体覆盖 | 缺少统一的集合差异检查 | 增加 PSI/KS、TVD、未见类别和敏感群体覆盖，并用样本量显著性门槛降低小样本误报 |
| 评估协议 | 候选阶段曾计算测试指标 | 改为验证集选模，测试前保存锁定记录，只给锁定单模型生成最终测试指标和诊断图 |
| 复现与审计证据 | 有随机种子、模型和部分报告 | 增加数据指纹、独立运行目录、审计 JSON、锁定记录、流水线保存和重载一致性核对 |
| 交付检查 | 能保存部分模型、图和报告 | 先声明应交付内容，逐项记录已生成、失败、缺失、未启用或不适用；报告存在不等于交付完整 |
| 局部补做 | 缺少统一的逐项恢复入口 | 同一程序会话内保持模型和数据不变，只补做可恢复的失败/缺失项，并保留尝试历史 |
| 模型复用 | 保存处理后输入模型 | 同时保存预处理器与模型，重载后直接接收原始表格，并核对预测一致性 |
| 公平性边界 | 分组与 DP/EO 计算 | 保留原始敏感分组，区分零差异与无法计算；当前 DP/EO 仅支持 0/1 二分类，正类为 1 |

这里的创新是相对于仓库原有实现的工程增量，不主张学术首创。

## 3. 可信审计整合的具体内容

具体功能按照建模生命周期组织：

```text
建模前：画像预检 → 结构感知切分 → 划分完整性审计
                         ↓
建模中：审计排除 → 分布/群体覆盖审计 → 验证集选模 → 最终测试
                         ↓
运行后：聚合审计结论 → 保存模型流水线 → 重载核对 → 交付检查
```

### 3.1 建模前：数据泄露防控

数据泄露防控从划分前开始，先检查字段，再检查样本之间的结构关系。

#### 3.1.1 画像阶段：静态泄露预检

`core/profile.py` 在划分前执行 `audit_profile_preflight()`。该检查默认启用且只观察、不修改数据，覆盖：

- ID 或高唯一性字段；
- 特征与目标逐行一致；
- 单特征判别力异常高或名称具有后验含义的目标代理；
- 用户声明为预测时不可获得的禁用特征。

主要配置位于 `ProfileConfig`：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `enable_leakage_audit` | `True` | 启用静态泄露预检 |
| `identifier_unique_ratio` | `0.98` | 高唯一性字段阈值 |
| `target_proxy_threshold` | `0.98` | 疑似目标代理阈值 |
| `id_columns` | `[]` | 用户明确声明的 ID 列 |
| `forbidden_features` | `[]` | 预测时不可获得的字段 |

结果进入 `DataProfile.audit`，并在画像报告中展示检查状态、风险证据和处理建议。

受控风险用例表明，画像预检能够识别高唯一性 ID、目标复制、疑似目标代理和用户声明的禁用特征。

#### 3.1.2 划分阶段：完整性审计与结构感知切分

`core/split.py` 将“数据有什么结构”和“实际怎样切分”分开：

- `group_column`、`time_column` 声明实体和时间语义；
- `split_strategy` 决定使用 `auto`、`stratified`、`random`、`group` 或 `temporal`。

`auto` 的规则是：有分组列时使用 `group`，有时间列时使用 `temporal`，否则分类使用 `stratified`、回归使用 `random`。

划分后执行 `audit_split()`，检查：

- 原始特征行是否跨 train/validation/test 重复；
- 同一患者、用户或设备是否跨集合；
- 三个集合的时间区间是否重叠或倒置；
- 画像已经发现疑似实体列，但没有声明 `group_column` 的情况。

主要配置位于 `SplitConfig`：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `split_strategy` | `"auto"` | 选择实际切分策略 |
| `group_column` | `None` | 患者、用户、设备等实体列 |
| `time_column` | `None` | 事件时间列 |
| `enable_split_audit` | `True` | 启用划分完整性审计 |
| `duplicate_audit_max_rows` | `200000` | 重复检测上限，超过时抽样并标注 |

未声明分组列或时间列时，相应检查显示“未执行”，不会误写为“通过”。

划分测试表明，重复样本、实体重叠和时间穿越能够被检出；改用 `group` 或 `temporal` 切分后，相应风险不再出现。

#### 3.1.3 预处理阶段：将审计建议落地

画像和划分审计默认只提供证据。是否删除字段由 `PreprocessConfig` 显式控制：

- `drop_group_column=True`：实体分组列默认不进入模型；
- `apply_audit_drops="off"`：默认只告警；
- `apply_audit_drops="high"`：排除画像阶段 high/critical 级建议列；
- `apply_audit_drops="all"`：执行全部带列名的排除建议。

被排除字段记录为 `dropped_by_audit`，附带排除原因，也不会再被拟合预处理器要求。排除动作本身是 info 级记录，不会抬高风险等级。

测试确认，被排除字段不会进入模型输入，也不会继续参与漂移检查。

### 3.2 建模中：漂移、群体覆盖与选模协议

完成划分后，系统继续检查各集合是否可比，并防止测试分数参与候选模型选择。

#### 3.2.1 数值特征漂移

`core/preprocess.py` 以训练集为基准，分别比较 validation 和 test：

- PSI 衡量分箱比例变化；
- 经验 KS 衡量累积分布最大差异；
- 先判断 KS 是否超过当前样本量对应的临界值，再按 PSI 分级；
- PSI 分箱数随样本量自动下调，连两个箱都无法支撑时返回 `None`。

该顺序用于减少小样本随机波动造成的误报，`max_psi` 只统计通过显著性门槛的记录。

#### 3.2.2 类别特征漂移

- TVD 衡量类别比例的整体变化；
- 未见类别比例衡量 validation/test 中属于训练阶段未知类别的样本比例；
- TVD 先经过与样本量相关的显著性判断，再进入风险分级。

#### 3.2.3 敏感群体覆盖

调用 `fit(..., sensitive_features=[...])` 后，系统统计每个群体在 train、validation、test 中的样本量和占比。完全缺失的群体保留为 0 条记录，低于 `min_group_size` 的组合标记为风险。

群体覆盖不是 Demographic Parity 或 Equalized Odds。它用于说明公平性指标是否具有足够的样本基础。

已被审计排除的字段不再参与漂移检查。特别是分组切分中的实体列，各集合类别不重叠属于预期行为；继续计算会产生“未见率 100%”的假风险。

主要配置位于 `PreprocessConfig`：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `enable_drift_audit` | `True` | 启用漂移和群体覆盖审计 |
| `psi_warning` / `psi_high` | `0.10 / 0.25` | PSI 告警和高危阈值 |
| `tvd_warning` / `tvd_high` | `0.10 / 0.25` | TVD 告警和高危阈值 |
| `min_group_size` | `20` | 单个集合中群体最小样本量 |
| `drift_bins` | `10` | PSI 分箱数上限 |

漂移不等于泄露，也不表示模型一定失效。系统给出证据和建议，由用户结合时间、实体、采样和业务背景判断。

漂移测试能够识别数值分布变化、类别比例变化、未见类别以及敏感群体缺失或样本不足；样本量门槛降低了小样本随机波动造成的误报。

#### 3.2.4 从验证集选模到最终测试

评估协议属于数据泄露防控在模型选择阶段的延伸：

1. 候选模型只使用训练集和验证集调参、比较和排序；
2. 最终模型在测试评估前写入 `evidence/selection_lock.json`；
3. 只有锁定的单模型获得最终测试指标和测试诊断图；
4. Voting 只在验证集比较，不访问测试集，也不会替换最佳单模型；
5. 协议结果写入 `evidence/model_search_audit.json`。

这里保证的是“测试分数不决定候选选择”。锁定后的诊断、解释和重载核对仍使用测试输入，因此不应表述为测试数据在整个生命周期中只被物理读取一次。

评估协议测试确认，候选模型只按验证集结果排序，只有锁定模型生成最终测试指标和诊断图。

### 3.3 运行后：复现证据与审计交付

整合后的复现能力沿用现有 `evidence` 交付链路，不再生成旧版独立目录 `reproducibility_ledger/`。

当前记录和核对内容包括：

1. `profile` 对 X、y 的列结构、类型、索引和值计算 SHA-256 数据指纹；
2. `split` 记录随机种子、实际切分策略、分组或时间边界，结果对象保留三组索引；
3. 每次 `fit()` 使用独立的 `<output_dir>/runs/<run_id>/`，防止旧运行文件冒充本次产物；
4. `evidence/selection_lock.json` 固化最终模型在测试评估前的选择依据；
5. `best_model.joblib` 保存处理后输入模型，`pipeline.joblib` 保存拟合后的预处理器和最终模型；
6. `evidence/reload_check.json` 比较内存模型与重载流水线在原始测试表格上的预测及概率；
7. `evidence/delivery.json` 记录交付项状态、路径、失败原因和尝试历史；
8. `evidence/audit.json` 聚合画像、划分、预处理和模型选择阶段的审计结果。

这些证据用于回答“输入是否一致、使用了什么切分方式、锁定了哪个模型、保存流水线能否复用”。数据指纹不会复制原始数据。

当前边界：

- 没有重复实现旧版 `artifact_manifest.json` 的全目录 SHA-256 清单；
- 尚未形成旧版 `split_indices.npz` 式的独立划分索引交付文件；
- `reload_check` 验证同环境预测一致性，不代表跨操作系统、依赖版本或硬件完全复现；
- 只应加载可信来源的 joblib 文件。

### 3.4 报告和文件怎样对应

| 输出 | 内容 |
|---|---|
| `reports/profile.html` | 静态泄露预检、疑似标识符和画像风险 |
| `reports/split.html` | 切分策略、跨集合重复、实体泄露和时间穿越 |
| `reports/preprocess.html` | 审计排除、数值/类别漂移和敏感群体覆盖 |
| `reports/model_search.html` | 验证集选模和候选结果范围 |
| `report.html` | 分阶段风险卡片、发现汇总、建议动作和审计链路图 |
| `evidence/audit.json` | 四阶段结构化审计汇总 |
| `evidence/model_search_audit.json` | 评估协议审计 |
| `evidence/selection_lock.json` | 最终模型锁定证据 |
| `evidence/reload_check.json` | 保存流水线重载一致性 |
| `evidence/delivery.json` | 全部交付项的生成状态和失败原因 |

总报告封面展示可信审计综合风险等级。聚合报告不会覆盖上游失败，各阶段发现项、未执行项和失败原因仍然保留。

报告测试确认，各阶段审计结果能够写入阶段报告、总报告和 `evidence/audit.json`，未执行项和失败原因不会被误写为“通过”。

## 4. 可信审计验证结果

[划分审计](https://github.com/gaozhemei-hub/ML26-gzm/blob/develop/tests/test_split.py)新增 76 项测试，[预处理审计](https://github.com/gaozhemei-hub/ML26-gzm/blob/develop/tests/test_preprocess_audit.py)新增 73 项测试，[报告审计](https://github.com/gaozhemei-hub/ML26-gzm/blob/develop/tests/test_audit_report.py)新增 21 项测试。[对应提交节点](https://github.com/gaozhemei-hub/ML26-gzm/commits/develop)的全量测试共 396 项，全部通过。

## 5. 功能边界

1. 自动规则不能判断全部业务语义，ID、实体、时间和预测时不可得字段应由用户声明。
2. `apply_audit_drops` 默认是 `off`；自动排除必须显式开启。
3. 分布漂移表示集合差异，不等于数据泄露或模型必然失效。
4. 群体覆盖说明公平性指标的样本基础，不代表模型公平，也不提供偏差缓解算法。
5. 评估协议保证候选排序不使用测试分数，不代表锁定后的解释、诊断和重载核对不读取测试输入。
6. 当前复现证据主要验证输入身份、运行隔离和同环境重载一致性，不宣称跨环境完全复现。
7. 旧版 `trustguard/` 可作为历史实现和对比实验材料；`develop` 主流程应以 `profile/split/preprocess/model-search/evidence` 的整合结果为准。
