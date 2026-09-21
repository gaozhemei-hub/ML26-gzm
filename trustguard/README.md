# TabularX TrustGuard：可信建模保障层

## 1. 研究动机与设计依据

表格模型取得较高分数，并不自动意味着评估结论可信。若一个特征在预测时尚不可获得，
同一患者同时出现在训练集和测试集，预处理过程提前使用了测试集信息，或测试集被反复用于
模型选择，评估分数就可能被系统性高估。此时即使 SHAP、特征重要性、混淆矩阵和学习曲线
都已生成，模型解释仍然建立在不可靠的实验基础上。

Kapoor 与 Narayanan 的论文 *Leakage and the Reproducibility Crisis in
Machine-learning-based Science* 系统研究了机器学习中的数据泄露与复现问题。作者汇总的既有
审查覆盖 17 个研究领域、至少 294 篇受到泄露影响的论文，并将泄露划分为八种类型；论文还
提出 Model Info Sheets，要求研究者围绕数据构造、特征可用时间、样本依赖关系和评估过程
提供可核查的说明。[1]

TrustGuard 参考上述思想，但不声称仅凭自动规则就能覆盖论文中的所有泄露类型。它把其中
可程序化检查的风险转化为运行前和划分后的审计，把不能可靠自动判断的业务语义转化为
Model Info Sheet 问题，并进一步记录测试集访问、数据划分、运行环境与输出工件。其核心
问题是：

> 这次模型分数是否建立在正确的数据、正确的划分和正确的评估协议上？

本目录是一个可独立运行的扩展，不修改 `src/tabularx/`、模型注册表或既有训练逻辑。
泄露审计实现位于 `trustguard/leakage/`，其余可信性模块位于 `trustguard/` 根目录，便于
独立测试、复用和集成。

实现遵循 TabularX 的设计约定：审计计算与文件写入分离，结构化结果先产生、报告随后
渲染；`pipeline.py` 只负责编排，`reporting.py` 只把结果渲染为 HTML 字符串，模板位于
`templates/`，所有 TrustGuard 产物写入独立子目录，不覆盖已有模型和主报告。

> [1] Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in
> machine-learning-based science. *Patterns, 4*(9), 100804.
> https://doi.org/10.1016/j.patter.2023.100804

## 2. 目标与功能实现

| 可信建模目标 | 基础建模流水线的常见输出 | TrustGuard 的补强 |
|---|---|---|
| 自动化预处理与模型选择 | 已有模型搜索、Optuna、预处理 | 强制验证集选择，避免按测试集分数选模型 |
| 强制可读实验报告 | 已有 HTML/PDF、模型报告、图表 | 新增泄露、测试访问、数据漂移、复现证据等风险工件 |
| 可复现 | 已有随机种子、数据指纹和参数对象 | 导出划分索引、环境版本、运行配置和工件 SHA-256 清单 |
| 公平性感知评估 | 已有公平性指标 | 检查敏感群体在 train/valid/test 中是否样本过少，避免公平性结论不稳定 |
| 本地优先、版本控制友好 | 已输出本地报告 | 全部 TrustGuard 工件为本地 JSON/CSV/NPZ，便于 Git/LFS 或归档 |

## 3. 四个模块

### 1. LeakageAuditor：数据泄露审计（`trustguard/leakage/`）

- 检查目标复制、疑似目标代理、高唯一性 ID；
- 检查跨 train/validation/test 的重复记录；
- 支持用户声明患者/用户等 `group_column`，发现同一实体跨集合；
- 支持用户声明 `time_column`，发现未来时间进入训练的时间穿越；
- 输出 `leakage_audit/summary.json`、`findings.csv` 与说明文件。

综合报告不会只展示异常列表，而会固定列出六项检查的执行状态：目标复制、目标代理、
ID/高唯一性字段、跨集合重复、实体分组泄露和时间穿越。状态分为“通过”“发现风险”与
“未执行”；缺少 `group_column` 或 `time_column` 时会明确说明原因，不会把“不适用”误写成
“通过”。

### 2. EvaluationProtocolGuard：评估协议守卫

机器学习中，验证集用于选模型，测试集只能在最终模型锁定后评估一次。
`EvaluationProtocolGuard` 是一个状态机：候选模型可记录验证集表现；锁定验证集最佳
模型后，只有该模型可进行一次最终测试。违反顺序会抛出异常而不是悄悄继续运行。

输出：`evaluation_protocol/test_access_log.json`。

`trustguard/leakage/safe_model_search.py` 是该原则的可运行实现：它复用 TabularX 的
预处理和模型注册表，但只在最终锁定模型后访问测试集。

### 3. ReproducibilityLedger：可复现实验证据包

它不会复制原始数据，从而避免把隐私数据意外写进报告；而会写出：

- 完整数据的 SHA-256 指纹（含列顺序、类型、索引和值）；
- train/validation/test 索引压缩文件及每个索引数组哈希；
- 配置、随机种子、运行命令；
- Python、操作系统和关键依赖版本；
- 输出目录中每一个工件的 SHA-256 清单。

输出：`reproducibility_ledger/reproducibility_ledger.json`、`split_indices.npz`、
`artifact_manifest.json`。可以调用 `verify_artifact_manifest()` 验证工件是否被改动。

### 4. DistributionAudit：漂移与群体覆盖审计

该模块比较 train 与 validation/test：

- 数值特征：PSI、经验 KS 统计量；
- 类别特征：总变差距离、测试集中未见类别比例；
- 敏感特征：每个群体在各集合的样本量和占比。

输出：`distribution_audit/summary.json`、`feature_drift.csv`、`group_coverage.csv`。
高风险不自动删除特征，而是提示检查切分策略、时间漂移、数据覆盖或编码方案。

## 4. 快速运行：四个模块

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

## 5. MLJAR 对比实验

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

### 5.1 CC-18 的 5–10 数据集批量实验

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

### 5.2 综合报告如何阅读

`reports/trustguard.html` 和 `reports/trustguard.pdf` 不只是列出数值，而是对五部分采用统一
结构：**内容 → 结果 → 解读 → 建议**。

- 泄露审计展示每条发现的证据与处理建议；
- 评估协议区分“尚未训练”“验证中”“已锁定”“最终评估完成”，不会伪造测试访问；
- 复现证据展示数据和划分哈希、命令、环境与工件完整性含义；
- 漂移报告正文优先展示最高风险记录，完整明细保留在 CSV，避免 PDF 被长表淹没；
- MLJAR 对比只有在真实基准完成后才给出结论；普通冒烟报告会明确标注“尚未执行”。

## 6. 对自动化表格建模流程的贡献

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

## 7. 实验报告目录结构

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

## 8. OpenI 运行流程

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
