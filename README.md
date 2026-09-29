# ML26-qyr

正式交付材料统一从 [docs 文档目录](docs/README.md) 阅读。

## 证据交付检查与局部补做

在训练、解释和报告流程中声明应交付的证据，按适用条件检查生成状态、文件与失败原因；在同一程序会话内补做可恢复的失败或缺失项，复用已拟合模型并保留尝试历史。可信审计识别实验风险，本模块检查相关证据是否完整交付。

- **功能、使用方法与边界**：[证据交付与局部补做说明书](docs/evidence-delivery-guide.md)
- **真实实验与原始结果**：[实验验证索引](docs/validation/evidence-delivery/README.md)
- **核心实现**：[交付检查与调度](TabularX-project/src/tabularx/core/evidence/delivery.py) · [主流程接入](TabularX-project/src/tabularx/core/evidence/estimator.py)
- **验证代码**：[交付恢复测试](tests/test_evidence.py) · [选模协议测试](tests/test_selection_protocol.py)

交付完整不等于模型准确、公平或跨环境完全复现。本模块的贡献是交付检查与恢复机制，不将已有训练、SHAP 和公平性算法列为新增贡献。文末原始选题为任务目标，不表示全部能力已经完成。

## 本周增量：可信审计整合

原独立 TrustGuard（`feature/trust-audit`）用单独入口完成泄露、评估协议、复现证据、漂移和群体覆盖检查。本周没有把旧目录原样并入主程序，而是将可复用能力拆入 TabularX 的画像、划分、预处理、模型选择和报告流程。

- **先看整合内容与使用边界**：[可信审计整合说明书](https://github.com/gaozhemei-hub/ML26-gzm/blob/develop/docs/trust-audit-integration-guide.md)
- **看画像与划分审计代码**：[画像预检](https://github.com/gaozhemei-hub/ML26-gzm/blob/develop/TabularX-project/src/tabularx/core/profile.py) · [划分审计](https://github.com/gaozhemei-hub/ML26-gzm/blob/develop/TabularX-project/src/tabularx/core/split.py)
- **看漂移、群体覆盖与审计排除**：[预处理审计](https://github.com/gaozhemei-hub/ML26-gzm/blob/develop/TabularX-project/src/tabularx/core/preprocess.py)
- **看公共审计结构与报告聚合**：[审计契约](https://github.com/gaozhemei-hub/ML26-gzm/blob/develop/TabularX-project/src/tabularx/schema/audit.py) · [总流程接入](https://github.com/gaozhemei-hub/ML26-gzm/blob/develop/TabularX-project/src/tabularx/core/evidence/estimator.py)
- **看专项测试**：[划分审计测试](https://github.com/gaozhemei-hub/ML26-gzm/blob/develop/tests/test_split.py) · [预处理审计测试](https://github.com/gaozhemei-hub/ML26-gzm/blob/develop/tests/test_preprocess_audit.py) · [报告审计测试](https://github.com/gaozhemei-hub/ML26-gzm/blob/develop/tests/test_audit_report.py) · [评估协议测试](https://github.com/gaozhemei-hub/ML26-gzm/blob/develop/tests/test_selection_protocol.py)
- **看相关提交**：[画像与总报告](https://github.com/gaozhemei-hub/ML26-gzm/commits/develop) · [划分完整性](https://github.com/gaozhemei-hub/ML26-gzm/commits/develop) · [漂移与群体覆盖](https://github.com/gaozhemei-hub/ML26-gzm/commits/develop) · [评估协议](https://github.com/gaozhemei-hub/ML26-gzm/commits/develop)


选题：题目10

## 题目-TabularX：面向可解释性与可复现性的自动化表格建模工具包

当前表格数据仍然是工业界和许多学科中最常见的数据形式。但大多数AutoML工具将模型训练视为一个黑箱，用户难以了解数据预处理细节、模型选择依据和最终决策原因。TabularX旨在填补这一空白：在保持高预测精度的同时，强制生成完整的、可读的实验报告和解释工件。你的核心任务是构建一个Python工具包，实现以下功能：自动化预处理与模型选择：支持分类和回归任务，集成XGBoost、LightGBM、CatBoost、随机森林等主流算法，并自动进行超参数调优。强制生成可复现实验报告：每次运行后，工具必须自动生成一个本地文件夹，内含：最佳模型、所有尝试过的模型与验证分数、数据预处理流水线、特征重要性排序、SHAP/ALE解释图、学习曲线和混淆矩阵。公平性感知评估模块：允许用户指定敏感特征（如性别、种族），自动计算并报告模型在不同子组间的性能差异和公平性指标（如Demographic Parity、Equalized Odds）。一键导出功能：支持将解释报告导出为PDF或HTML格式，方便与非技术人员分享。

入门路径：研究MLJAR AutoML的mljar-supervised包的报告生成机制和设计哲学。在5-10个公开表格数据集（如OpenML的CC-18基准）上运行现有AutoML工具，记录它们在“可解释性工件生成”方面的不足。基于scikit-learn和Optuna搭建流水线，设计模块化架构。编写详细的API文档和3个以上Jupyter Notebook教程。

创新点贡献：首创“强制可解释工件”机制：不是“可选”功能，而是工具运行后的默认产出，从根本上改变AutoML的使用习惯。原生集成公平性审计：无需额外代码，自动检测并报告模型在敏感属性上的偏见。本地优先的文档化设计：所有报告、模型、配置均保存在本地可读文件夹中，版本控制友好。
