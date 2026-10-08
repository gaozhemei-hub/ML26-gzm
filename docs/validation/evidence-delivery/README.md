# 证据交付实验验证索引

本目录保存不同代码阶段的真实运行产物，用于核查交付检查、故障恢复及选模协议。脚本在 drafts/introduction/，功能说明见[报告与证据交付](../../features/06-报告与证据交付.md)。

| 验证 | 范围与结果 | 原始汇总 | 网页入口（下载后打开） |
|---|---|---|---|
| 交付与恢复 | 3 数据集 × 2 模型，6 个组合、54 项检查通过 | [JSON](evidence_validation_v3/summary.json) | [HTML](evidence_validation_v3/summary.html) |
| 选模与测试分离 | Iris、Diabetes 双候选运行通过 | [JSON](selection_protocol_v4/summary.json) | [HTML](selection_protocol_v4/summary.html) |
| Adult 补充验证 | 48,842 行；注入解释失败后恢复，无重新训练 | [JSON](adult_evidence_v5/summary.json) | [HTML](adult_evidence_v5/summary.html) |
| 自动测试 | 217 项通过 | [测试输出](test-results.txt) | — |

## 文件里有什么

- 各运行的交付清单、失败/恢复尝试历史、选模记录（第四版及 Adult）、重载核对。
- 模型和预处理流水线、真实 SHAP/诊断图、各阶段及总报告。
- 第三版的故障注入快照；Adult 的首次失败快照。

模型文件仅供同依赖环境下复用，joblib 文件只应在信任来源时加载。原始数据集、虚拟环境、登录凭据未包含。
网页在代码托管平台可能只显示源码；请下载该目录或拉取仓库后打开 summary.html，保持目录结构即可浏览相对链接。

## 怎么理解这些记录

第三版、第四版是开发迭代编号，不能理解为独立产品版本；第三版是在测试集流程修复之前运行的，用于验证交付恢复。第四版及 Adult 展示修复后的流程。不同批次不能当作完全相同代码的重复实验。

实验故障是主动注入，不能推断自然失败率或加速比。Adult 的性别字段保留为模型输入和审计分组，不与早期排除性别输入的 v0.1 数字直接比较。完整交付不等于模型公平或跨环境完全复现。

这些是历史快照，包含故意删除/恢复的记录和暂时失败的状态；以每批 summary.json 理解实验结果。归档文本中的本机工作区路径已替换为 `<LOCAL_WORKSPACE>`，模型文件和相对路径保留。
