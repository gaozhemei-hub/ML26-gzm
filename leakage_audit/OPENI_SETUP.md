# 在 OpenI 上运行完整版

这个扩展适合采用“本地轻量验证、OpenI 完整建模”的分工。本文件只新增
环境和运行说明，不修改原项目的 `src/tabularx/` 代码。

## 资源建议

- 中小型 CSV（约数万行、数百列）：CPU 4 核、16 GB 内存；
- 更大的数据集或较多 Optuna trial：CPU 8 核、32 GB 内存；
- 本安全搜索器默认按 CPU 运行。GPU 不是必需条件，只有数据很大时才值得
  另行配置模型的 GPU 参数。

安装包主要消耗的是容器磁盘；训练时的内存主要由数据规模、树数量、并行线程和
`--n-trials` 决定。建议先用 `--n-trials 3` 验证流程，再提高到 20 或更多。

## 1. 准备工作区和数据

在 OpenI 创建 Notebook 或训练任务，选择 Python 3.11（或兼容的 Python 3.10+）
环境。将本项目上传/克隆到工作区，并将 CSV 数据上传到例如 `data/` 目录。CSV 应：

- 每行代表一个样本；
- 有一列明确的目标变量，例如 `label`、`outcome`；
- 不要把目标列同时作为特征列；
- 对有患者/用户/时间语义的数据，后续在审计配置中声明 `group_column` 或时间列。

## 2. 创建环境并安装完整依赖

在 OpenI 终端进入项目根目录后执行：

```bash
conda create -n tabularx python=3.11 -y
conda activate tabularx
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r leakage_audit/requirements-full.txt
```

验证四种模型均可导入：

```bash
python -c "import xgboost, lightgbm, catboost, optuna; print('full environment ready')"
```

如 OpenI 镜像没有 conda，可创建 `venv` 后执行相同的两条 `pip install` 命令。

## 3. 运行任意 CSV

以下命令只通过验证集选择模型；测试集只在最终锁定一个模型后评估一次：

```bash
export PYTHONPATH="$PWD:$PWD/src"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

python leakage_audit/demo_safe_model_search.py \
  --csv data/your_dataset.csv \
  --target label \
  --models random_forest xgboost lightgbm catboost \
  --n-trials 20 \
  --output-dir outputs/experiment-001
```

将 `label` 换成你的真实目标列名。分类或回归任务由 TabularX 的数据画像自动判断。
运行完成后下载 `outputs/experiment-001/safe_model_search.json`；其中会保存每个候选的
验证指标、被锁定的模型和唯一一次测试指标。

## 4. 先在本地快速验证（可选）

本机不需要安装三个梯度提升库。只跑随机森林即可检查数据格式和流程：

```powershell
$env:PYTHONPATH = "$PWD;$PWD/src"
python -m pip install -r leakage_audit/requirements-safe-search.txt
python leakage_audit/demo_safe_model_search.py `
  --csv "D:\data\your_dataset.csv" `
  --target label `
  --models random_forest `
  --n-trials 3 `
  --output-dir .\local-smoke-test
```

之后将同一份 CSV 上传至 OpenI，使用第 3 步的完整命令即可。
