# TabularX 设计哲学（MLJAR 源码观察）

## 1. 注册表 + 能力声明机制

- 每个模型族在注册时声明自己的 `required_preprocessing`（需要的预处理清单）
- 预处理决策 = 数据事实（profile）∩ 模型能力声明（registry）
- 不同模型族预处理需求不同（Linear/NN 要 scale，XGBoost/RF 不要，CatBoost 连类别转换都不要）
- 注册表按任务分桶：`registry = {binary: {}, multiclass: {}, regression: {}}`

## 2. 预处理是"数据事实"与"模型能力"两层决策

- 第一步：`DataInfo.compute(X, y, task)` → 数据事实（与模型无关）：逐列判 categorical/datetime/text/scale/missing_values/empty_column
- 第二步：`PreprocessingTuner.get(required_preprocessing, data_info, task)` → 决策（模型与数据求交集）
- **profile 只报事实，不做决策**；scale 判据：`|mean|>0.5 or std>1.5`

## 3. 划分先于预处理

- 划分是实验设计（组织实验），预处理是数据变换（变换数据）
- 预处理统计量**只能来自训练集**，否则数据泄露
- MLJAR 用 `validation_strategy` 统一管理划分参数，划分索引落盘（`split_train_indices.npy`）

## 4. 产物自包含 + 元数据驱动

- 每个模型目录有 `framework.json`：含预处理策略、验证策略、模型参数、训练时间、分数
- 顶层有 `data_info.json`（列信息）、`params.json`（运行参数）、`leaderboard.csv`（全部模型对比）
- **一个目录 = 一次完整实验快照**

## 5. 按数据规模动态选择默认策略

- 小数据全选所有模型族，中数据去掉神经网络，大数据只留快速模型
- 用户可显式指定 `algorithms` 覆盖默认

## 6. 公平性审计已内置

- `fairness/` 模块含 `metrics.py`（Demographic Parity 等）、`certificate.py`、`plots.py`、`report.py`
- 差异化卖点应聚焦：**强制解释工件 + 显式决策可追溯 + 本地文档化**

## 7. 架构分层

- `algorithms/`（注册表 + 各模型实现）
- `tuner/`（Optuna 搜索 + 预处理决策 + 数据信息）
- `preprocessing/`（预处理执行：缺失/scale/编码/text/datetime）
- `validation/`（划分策略）
- `fairness/`（公平性审计）
- `apps/`（报告生成与发布）
- 编排层（`base_automl.py` / `automl.py`）只做组装，不写业务逻辑
