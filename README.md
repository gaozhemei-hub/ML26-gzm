# TabularX

本仓库是团队课程项目的个人提交与复现镜像。默认的 `develop` 分支完整保留团队共同开发的代码、功能文档、十数据集实验结果和复现材料，便于独立检查和复现。

个人提交与复现仓库固定入口：<https://github.com/gaozhemei-hub/ML26-gzm>。

TabularX 是一个面向表格数据的自动化建模 Python 工具包，支持分类和回归任务。项目提供统一的建模入口，并将实验报告、模型和结构化运行记录保存到本地。

项目的研究背景、方法和整体实现请参阅论文。本文档聚焦于项目安装、最小使用示例、个人贡献范围和文档入口。

## 个人贡献范围

**Zmei_IR** 在团队项目中主要完成以下工作：

- **证据交付机制：** 设计并实现证据登记、生成状态、适用条件、失败原因、尝试历史和完整性检查，并提供 JSON 与 HTML 两种交付结果；实现失败工件的同会话局部补做，修复补做后的状态刷新一致性。
- **建模流程正确性：** 完善验证集选模与外部测试隔离，保存可接收原始表格的完整预处理—模型流水线并核对重载预测一致性，同时明确二分类公平性指标的适用边界。
- **实验与成果整理：** 负责 Adult Income、Breast Cancer Wisconsin、Titanic、Iris 和 California Housing 五个数据集的 MLJAR—TabularX 对比实验，并参与十数据集结果汇总、证据交付功能说明、实验分析、论文及跨平台复现材料整理。

仓库中的功能页面介绍团队版本的完整能力；个人工作范围以上述内容及 Git 提交记录为准。

## 快速开始

要求 Python 3.10 或更高版本。建议先创建并激活虚拟环境，然后在仓库根目录安装本地包：

```bash
python -m pip install -e ./TabularX-project
```

下面以 scikit-learn 内置数据集为例，演示基本调用：

```python
from sklearn.datasets import load_breast_cancer
from tabularx import AutoEstimator

sample = load_breast_cancer(as_frame=True)
X, y = sample.data, sample.target

estimator = AutoEstimator(
    task="classification",
    output_dir="./outputs/breast-cancer",
)
estimator.param.model_search.models = ["random_forest"]
estimator.param.model_search.n_trials = 2
estimator.fit(X, y)

print("最佳模型：", estimator.result.searched.best_model_name)
print("报告路径：", estimator.result.report_paths["full"])
print("运行目录：", estimator.output_dir)
```

模型搜索还支持 XGBoost、LightGBM 和 CatBoost。使用这些模型时，需要另外安装对应依赖，例如：

```bash
python -m pip install xgboost lightgbm catboost
```

使用 Notebook 教程前，请确保已安装项目依赖以及 Notebook 环境（如 JupyterLab）。

## 文档

- [功能概览](docs/features/README.md)：了解建模工作流、可信审计、证据交付和能力边界。
- [API 参考手册](docs/api_reference.md)：公开类、配置参数、结果对象、审计、证据交付和异常接口。
- [教学 Notebook](docs/demo/)：三个循序渐进的使用示例。
- [实验验证](docs/validation/README.md)：十个公开表格数据集上的 MLJAR 与 TabularX 工具包对比。
- [设计文档](docs/design/)：架构、数据契约、参数、异常、报告渲染和流水线设计。

## 项目结构

```text
TabularX-project/   Python 包源码与项目配置
 tests/             自动化测试
 docs/              API、教学示例、设计文档与实验验证材料
 drafts/            开发过程材料
   introduction/    十数据集实验与专项验证的复现脚本
   trustguard/       可信审计的历史实现与对比实验材料
```

## 开发测试

在 `TabularX-project/` 目录安装包和开发依赖：

```bash
cd TabularX-project
python -m pip install -e ".[dev]"
cd ..
python -m pytest
```

实验脚本所需的可选模型和绘图库依赖需按具体用途单独安装。
