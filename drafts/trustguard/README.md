# TabularX TrustGuard：可信建模保障层

## 1. 出发点：泄露风险

**泄露风险**是指真实预测时不可获得的信息进入特征、预处理、训练、模型选择或最终评估过程，
导致模型提前“知道答案”或测试结果参与开发决策，进而高估模型性能。

Kapoor 与 Narayanan 的论文 *Leakage and the Reproducibility Crisis in
Machine-learning-based Science* 将泄露归纳为八种类型，并提出 Model Info Sheets，要求围绕
数据构造、特征可用时间、样本依赖关系和评估过程提供可核查说明。[1]

泄露不仅是“目标列被复制进特征”。常见风险包括：

- **特征泄露**：目标复制、目标代理或预测时尚未产生的未来信息进入特征；
- **样本泄露**：相同记录或同一患者、用户、设备出现在不同数据集合中；
- **预处理泄露**：使用验证集或测试集拟合插补、标准化、编码或特征选择；
- **评估泄露**：反复查看测试结果，并据此调整模型、参数、特征或阈值。

数据漂移、群体覆盖不足和缺少复现证据不一定属于泄露，但同样会削弱实验结论的可信度。
因此，TrustGuard 从泄露检查扩展为数据、评估、复现和分布有效性四层保障，回答：

**模型分数是否建立在正确的数据、正确的划分、正确的评估协议和可复核的实验记录上？**

四个模块分别处理实验生命周期中的四类问题：

| 阶段 | 模块 | 核心问题 | 主要证据 |
|---|---|---|---|
| 训练前/数据划分后 | `LeakageAuditor` | 数据本身有没有泄露？ | 检查状态、风险证据与处理建议 |
| 模型选择期间 | `EvaluationProtocolGuard` | 有没有使用测试集选择模型？ | 候选模型、锁定模型和测试访问日志 |
| 实验全过程/完成后 | `ReproducibilityLedger` | 能否确认使用了相同的数据、划分、配置和环境？ | 数据与划分哈希、环境记录和工件清单 |
| 数据划分后/结果解释前 | `DistributionAudit` | 验证集、测试集是否偏离训练集？敏感群体是否覆盖充分？ | 漂移指标和群体样本量 |

综合报告前四部分在主实验数据集上展示四个模块的运行结果。第五部分在多个公开数据集上对比
MLJAR Explain 与 TrustGuard 默认生成的工件，并通过注入已知泄露风险，验证 TrustGuard
能否将其检出。这一实验用于说明 TrustGuard 功能的独特性和设计必要性，而不是比较两个工具
的预测分数。

整体逻辑如下：

```text
检查数据和划分是否泄露
          ↓
约束候选模型只能依据验证集选择
          ↓
记录数据、划分、配置、环境和输出证据
          ↓
检查数据漂移与敏感群体覆盖
          ↓
通过公开数据集和风险注入验证模块有效性
```

TrustGuard 是一个可独立运行的扩展，不修改现有模型注册表和训练逻辑。各模块先生成结构化
审计结果，再统一渲染为 HTML/PDF 报告；全部工件保存在独立目录中，不覆盖已有模型和主报告。

> [1] Kapoor, S., & Narayanan, A. (2023). *Leakage and the reproducibility crisis in
> machine-learning-based science*. Patterns, 4(9), 100804.
> https://doi.org/10.1016/j.patter.2023.100804

## 2. 四个模块的职责与解读

### 2.1 LeakageAuditor：数据泄露审计（`trustguard/leakage/`）

**所处阶段：训练前和数据划分后。**

**解决的问题：数据内容或划分关系是否让模型接触了不应获得的信息？**

该模块固定检查六类风险：

1. **目标复制**：某个特征是否与目标列完全或近乎完全相同；
2. **目标代理**：某个特征是否与目标异常相关，可能间接编码答案；
3. **ID/高唯一性字段**：特征是否接近“一行一个值”，可能导致记忆样本或隐藏实体信息；
4. **跨集合重复**：相同特征记录是否同时存在于 train、validation 或 test；
5. **实体分组泄露**：同一患者、用户或设备是否跨越不同数据集合；
6. **时间穿越**：训练数据的时间是否晚于验证集或测试集，或者使用了预测时不可获得的未来信息。

实体分组泄露需要 `group_column`，时间穿越需要 `time_column`。未提供时，报告标记为
“未执行”并说明原因，不会把未检查误写成“通过”。

输出包括：

- `leakage_audit/summary.json`：总体风险等级和结构化结果；
- `leakage_audit/findings.csv`：逐条风险、证据、严重程度和建议；
- 综合 HTML/PDF 中的“检查项结果表”：六项检查的通过、发现风险或未执行状态。

**如何解读：**“通过”表示自动规则未发现风险；“发现风险”要求检查字段和切分；“未执行”
表示缺少必要配置。自动检查不能替代业务判断，模块也不会静默删除特征或修改数据。

### 2.2 EvaluationProtocolGuard：评估协议守卫

**所处阶段：候选模型比较、超参数搜索和最终测试期间。**

**解决的问题：是否提前或反复查看测试集，并根据测试结果选择模型？** 如果多个候选模型都
在测试集上评分后再选最高者，测试集已经变成验证集，最终分数会带有选择偏差。

TrustGuard 将三个数据集合的职责写成可执行规则：

| 数据集合 | 职责 | 使用方式 |
|---|---|---|
| train | 拟合模型及预处理参数 | 可用于训练 |
| validation | 比较算法、超参数、特征和阈值 | 可用于模型选择 |
| test | 估计锁定模型面对未知数据的最终性能 | 模型锁定后仅评估一次 |

`EvaluationProtocolGuard` 使用状态机约束以下顺序：

```text
记录候选模型的验证结果
          ↓
根据验证指标锁定唯一模型
          ↓
只授权该模型访问一次测试集
          ↓
评估完成，不允许换模型或二次测试
```

以下操作会抛出异常：未锁定模型就测试、改测其他模型、二次测试、锁定后加入新候选。
正常运行会生成 `evaluation_protocol/test_access_log.json`，记录候选模型、锁定模型、测试
访问次数、事件顺序和合规状态。

`trustguard/leakage/safe_model_search.py` 复用现有预处理和模型注册表，但候选排名只使用验证
指标。交叉验证或 Optuna 也可使用，前提是搜索过程不访问最终测试集。

**如何解读：**“测试集访问 1 次、已完成且合规”表示模型在测试前已经锁定。整合时应让最终
测试统一经过 `authorize_final_test()`，避免其他代码绕过守卫。

### 2.3 ReproducibilityLedger：可复现实验证据包

**所处阶段：贯穿实验全过程，并在实验完成后封存证据。**

**解决的问题：以后能否确认两次运行使用了相同的数据、划分、配置、环境和输出文件？**
随机种子不能记录数据变化、重新划分、依赖升级或报告被修改，因此还需要完整证据链。

该模块记录：

- 完整数据的 SHA-256 指纹，覆盖列顺序、数据类型、索引、特征值和目标值；
- train/validation/test 的具体索引、样本数以及每组索引的 SHA-256；
- 配置、随机种子和运行命令；
- Python、操作系统以及 NumPy、pandas、scikit-learn、Optuna 和各模型库版本；
- 输出目录中每个工件的路径、大小和 SHA-256。

原始数据不会复制到报告目录；数据指纹只用于判断输入是否一致，不能恢复原始数据。

输出包括：

- `reproducibility_ledger/reproducibility_ledger.json`：数据、配置和环境证据；
- `reproducibility_ledger/split_indices.npz`：可复用的数据划分索引；
- `artifact_manifest.json`：全部输出工件的 SHA-256 清单。

可以调用 `verify_artifact_manifest()` 检查文件是否被修改、删除或额外加入。

**如何解读：**相同数据指纹表示输入一致，相同划分哈希表示集合包含相同样本；工件清单
校验失败表示文件生成后发生变化。该模块提供复核证据，不承诺跨硬件结果逐位相同。

### 2.4 DistributionAudit：漂移与群体覆盖审计

**所处阶段：数据划分后、解释模型指标和公平性结果之前。**

**解决的问题：验证集和测试集是否偏离训练集，敏感群体是否有足够样本支持稳定结论？**

该模块以训练集为基准，分别比较 train vs validation 和 train vs test：

- **数值特征 PSI**：按训练集区间比较各区间比例，衡量整体分布稳定性；
- **经验 KS 统计量**：衡量两组经验累积分布之间的最大距离；
- **类别特征 TVD**：衡量类别比例整体变化，取值越大表示分布差异越大；
- **未见类别比例**：验证集或测试集中属于训练阶段从未出现类别的样本比例；
- **敏感群体覆盖**：记录每个群体在 train、validation、test 中的样本量和占比，包括 0 条的完全缺失情况。

默认中等/高风险阈值为：PSI 0.10/0.25、KS 0.10/0.20、TVD 或未见类别比例 0.10/0.25；
敏感群体少于 20 条标记为高风险。可通过 `DistributionAuditConfig` 调整。

输出包括：

- `distribution_audit/summary.json`：总体风险与全部结构化记录；
- `distribution_audit/feature_drift.csv`：每个特征的完整漂移明细；
- `distribution_audit/group_coverage.csv`：敏感群体在不同划分中的覆盖情况。

**如何解读：**漂移不等于泄露，也不要求自动删除特征；它提示检查划分、时间变化、样本
代表性和编码。群体覆盖不替代 Demographic Parity 或 Equalized Odds，而是说明公平性指标
是否有足够样本支持。未提供 `sensitive_features` 时，该项标记为不适用。

## 3. 快速运行：四个模块

在仓库根目录运行。TabularX 的 `pyproject.toml` 位于 `TabularX-project/`，因此必须明确
安装这个子目录，而不是在仓库根目录执行 `pip install -e .`。

```powershell
python -m pip install -e .\TabularX-project
python -m pip install -r trustguard/leakage/requirements-safe-search.txt
$env:PYTHONPATH = "$PWD;$PWD\TabularX-project\src"

python -m trustguard.demo_trustguard `
  --dataset breast_cancer `
  --pdf-report `
  --output-dir .\outputs\trustguard-smoke
```

若需要一次完成随机森林、XGBoost、LightGBM、CatBoost 安全搜索以及四类可信性审计，使用
完整实验入口：

```powershell
python -m trustguard.run_full_experiment `
  --dataset breast_cancer `
  --n-trials 5 `
  --benchmark-summary .\outputs\trustguard-benchmark\benchmark_summary.json `
  --pdf-report `
  --output-dir .\outputs\trustguard-full-experiment
```

该命令会将四模型验证结果、最终测试指标、主实验数据集说明和 MLJAR 对比结论直接写入
`reports/trustguard.html` 与 `reports/trustguard.pdf`，无需运行后手工改写报告。

HTML 报告无需额外参数，默认写入 `reports/trustguard.html`。PDF 使用与 TabularX 主报告
相同的 WeasyPrint 依赖；Windows + Conda 推荐安装方式如下，可同时补齐 GTK/Pango 原生库：

```powershell
conda install -n tabularx --override-channels -c conda-forge weasyprint -y
```

任意 CSV 的最小命令如下；将 `label` 改为真实目标列名：

```powershell
python -m trustguard.demo_trustguard `
  --csv "D:\data\my_table.csv" `
  --target label `
  --group-column patient_id `
  --time-column visit_time `
  --sensitive-features sex race `
  --output-dir .\outputs\my-table-trustguard
```

`group_column`、`time_column` 和 `sensitive_features` 均为可选。对于包含患者、用户、
设备或时间语义的数据，应明确提供相应列名；系统无法仅凭列名完全推断业务含义。

## 4. MLJAR 对比实验

本实验的目的不是声称各工具使用了完全相同的搜索预算后谁分数更高，而是比较：

1. 各工具默认产出了哪些解释性工件；
2. 是否输出了泄露风险、测试访问、复现证据和漂移/覆盖工件；
3. 在注入泄露字段后，是否能明确报警或阻断。

安装实验依赖：

```bash
python -m pip install -r trustguard/requirements-benchmark.txt
```

先用内置乳腺癌数据做 MLJAR 冒烟测试：

```bash
python trustguard/run_mljar_baseline.py \
  --dataset breast_cancer \
  --total-time-limit 120 \
  --output-dir outputs/mljar-breast-cancer
```

使用任意 CSV：

```bash
python trustguard/run_mljar_baseline.py \
  --csv data/your_dataset.csv \
  --target label \
  --total-time-limit 300 \
  --output-dir outputs/mljar-your-dataset
```

脚本会运行 MLJAR 的 `Explain` 模式，并保存 `mljar_results/`、运行元数据和
`mljar_artifact_inventory.json`。MLJAR 已提供自动文档、SHAP 和公平性能力；该实验将
这些既有工件与 TrustGuard 生成的可信性工件一并记录。

### 4.1 CC-18 的 5–10 数据集批量实验

CC-18 是 OpenML 的 72 个分类任务基准套件。OpenML 的 task 同时包含数据、目标和
划分元数据；这里保留 task ID 以保证来源可追溯。请注意：本项目当前的 TabularX
默认采用自身的 train/validation/test 划分，因此本实验应称为“工件生成与风险审计
对比”，而不是严格意义上完全统一划分的算法排名。

先生成一个 5 数据集清单：

```bash
python -c "from trustguard import create_cc18_manifest; print(create_cc18_manifest('outputs/cc18_manifest.json', limit=5))"
```

然后运行完整对比套件，而不是只运行 MLJAR。对每个数据集，程序会依次生成：

1. MLJAR Explain 基线及其工件清单；
2. 正常数据上的 TrustGuard 四模块工件；
3. 注入目标复制和唯一 ID 后的 TrustGuard 工件；
4. `benchmark_summary.json` 以及包含真实对比结果的 HTML/PDF 综合报告。

```bash
python -m trustguard.run_benchmark_suite \
  --manifest outputs/cc18_manifest.json \
  --total-time-limit 120 \
  --pdf-report \
  --output-dir outputs/trustguard-benchmark
```

批量脚本会把单个数据集的下载或训练失败写入汇总，而不会中断其余数据集。最终报告第
5 节会明确列出：MLJAR 解释工件覆盖、MLJAR 可信性工件覆盖、TrustGuard 四类工件覆盖，
以及泄露注入是否被实际检测。更多实验边界见 `BACKGROUND_AND_BENCHMARK_PLAN.md`。

只需要单独验证 MLJAR 能否启动时，仍可运行：

```bash
python -m trustguard.run_mljar_baseline --dataset breast_cancer --total-time-limit 120 --output-dir outputs/mljar-smoke
```

### 4.2 综合报告如何阅读

`reports/trustguard.html` 和 `reports/trustguard.pdf` 不只是列出数值，而是对五部分采用统一
结构：**内容 → 结果 → 解读 → 建议**。

- 泄露审计展示每条发现的证据与处理建议；
- 评估协议区分“尚未训练”“验证中”“已锁定”“最终评估完成”，不会伪造测试访问；
- 复现证据展示数据和划分哈希、命令、环境与工件完整性含义；
- 漂移报告正文优先展示最高风险记录，完整明细保留在 CSV，避免 PDF 被长表淹没；
- MLJAR 对比只有在真实基准完成后才给出结论；普通冒烟报告会明确标注“尚未执行”。

## 5. 对自动化表格建模流程的贡献

TrustGuard 将数据、评估和报告的可信性要求落实为本地可检查的工件，与 TabularX 的模型
训练、解释和公平性模块形成互补。

1. **提升自动模型选择的可信度。** 评估协议守卫将“验证集选模型、测试集最终评估一次”
   固化为可执行规则，避免测试指标参与候选模型筛选。
2. **补全强制实验报告。** 泄露审计、测试访问日志、复现实验证据和漂移/群体覆盖报告与
   原有的模型指标、SHAP、学习曲线、混淆矩阵共同构成完整实验报告。
3. **强化可复现与本地优先。** 数据指纹、划分索引、环境版本、配置和工件哈希均保存为
   JSON、CSV、NPZ 等本地可读文件，支持版本控制、复跑和完整性校验。
4. **增强公平性结论的解释边界。** 原有公平性模块衡量群体间性能与公平性指标；TrustGuard
   额外报告敏感群体在不同划分中的样本覆盖，避免在极小群体上过度解读公平性结果。
5. **支持基线研究和公开数据集验证。** MLJAR 基线脚本、CC-18 清单生成器和工件盘点器
   将竞品研究与 5–10 个公开表格数据集实验转化为可重复执行的代码流程。

## 6. 实验报告目录结构

```text
run-output/
├── leakage_audit/                 # 模块 1 的运行产物：泄露发现与处理建议
├── evaluation_protocol/           # 模块 2：测试集访问记录
├── reproducibility_ledger/        # 模块 3：数据、划分、环境和配置证据
├── distribution_audit/            # 模块 4：特征漂移与群体覆盖
├── artifact_manifest.json         # 全部输出文件校验和
├── trustguard_run_summary.json    # 四模块入口索引
└── reports/                       # 与 TabularX 现有报告目录保持一致
    ├── trustguard.html            # 默认生成的可信建模综合报告
    └── trustguard.pdf             # 使用 --pdf-report 时生成
```

这种结构正好体现“模型报告”和“可信性报告”并列：前者解释模型，后者解释为什么应当
相信模型的评估结论。

报告实现遵循 TabularX 现有的“纯渲染 + 上层写文件”约定：
`render_trustguard_report()` 只接收结构化结果并返回 HTML 字符串，
`write_trustguard_report()` 才负责创建文件。这样既能独立运行，也便于后续由主流程统一调用，
且不会覆盖根目录的 `report.html` 或 `report.pdf`。

仓库中的 `trustguard/example-output/breast_cancer_openi/` 保存了一份体积受控的正式运行
示例，包括四模型结果、四类可信性工件、MLJAR 对比汇总以及 HTML/PDF 报告。该目录不含
模型缓存和原始数据，适合随源码一同进行版本控制。示例的实验参数和阅读顺序见该目录的
`README.md`。

## 7. OpenI 运行流程

在 OpenI 调试环境的项目根目录依次执行：

```bash
python -m pip install -e ./TabularX-project
python -m pip install -r trustguard/requirements-benchmark.txt
export PYTHONPATH="$PWD:$PWD/TabularX-project/src"

# 先验证独立模块；全部通过后再消耗时间运行 MLJAR。
python -m unittest discover -s trustguard/tests -v
python -m trustguard.demo_trustguard \
  --dataset breast_cancer \
  --pdf-report \
  --output-dir outputs/trustguard-smoke

# 固化 5 个 CC-18 任务并运行正式对比。
python -c "from trustguard import create_cc18_manifest; print(create_cc18_manifest('outputs/cc18_manifest.json', limit=5))"
python -m trustguard.run_benchmark_suite \
  --manifest outputs/cc18_manifest.json \
  --total-time-limit 120 \
  --pdf-report \
  --output-dir outputs/trustguard-benchmark
```

正式实验完成后应保存以下三个入口文件：

- `outputs/trustguard-benchmark/benchmark_summary.json`：逐数据集机器可读结果；
- `outputs/trustguard-benchmark/reports/trustguard.html`：完整网页报告；
- `outputs/trustguard-benchmark/reports/trustguard.pdf`：用于提交和归档的综合报告。

如果 5 个数据集的 120 秒预算运行稳定，可将清单提高到 10 个任务，或把单数据集预算提高
到 300 秒。实验报告必须保留失败数据集及错误原因，不能只展示成功样本。
