# TabularX

TabularX 是一个面向表格数据的自动化建模 Python 工具包，支持分类和回归任务。项目提供统一的建模入口，并将实验报告、模型和结构化运行记录保存到本地。

项目的研究背景、方法和完整功能贡献请参阅论文。本文档聚焦于项目安装、最小使用示例和文档入口。

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
 introduction/      十数据集实验与专项验证的复现脚本
 drafts/            开发过程中的历史记录和演示材料
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
