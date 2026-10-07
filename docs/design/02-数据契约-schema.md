# 数据契约（Schema）

## 为什么需要数据契约

各阶段的函数（profile、split、preprocess……）如果直接用 dict 或 tuple 传数据，接口会变得隐晦——调用方需要读源码才知道 dict 里有哪些 key、value 是什么类型。

schema 层的存在就是为了解决这个问题：**用显式的 dataclass 定义每个阶段的输入/输出协议**。读 dataclass 的定义就能知道"这个阶段产出什么"，不需要翻实现代码。

## 两条设计原则

### 1. 不存原始数据

Schema 对象只存元信息（统计量、索引、标签），不存 `X`、`y` 的实际值。例如 `SplitResult` 存的是 `train_indices`（行号数组），而不是 `X_train`。这样做的原因：

- **避免内存翻倍**：一份数据在多个 schema 之间复制，大数据集下不可接受
- **保持职责单一**：Schema 是"契约"不是"缓存"，它只管说"数据被分成了哪些子集"，不管实际数据长什么样
- **下游按需取数据**：Preprocess 阶段通过索引从原始 `X, y` 切片，不经过多余的中间拷贝

### 2. 用 Evidence 链记录判断依据

每个阶段的每个重要判断都产生一个 `Evidence` 对象：

```python
@dataclass
class Evidence:
    metric: str                # 证据名，如 "n_unique"
    value: float | int | str   # 证据值，如 2
    note: str | None = None    # 自然语言说明
```

Evidence 的设计哲学是**存数据快照，不存规则文本**。规则文本是模板的事，不在内存中携带。这样可以避免规则变化时 schema 也需要改版。

## 现有契约

| 契约 | 所在阶段 | 核心字段 | 不含什么 |
|---|---|---|---|
| `DataProfile` | Profile 输出 | n_rows, n_cols, columns(ColumnFact列表), target(TargetFact), fingerprint | X, y 实际数据 |
| `ColumnFact` | Profile 的列 | col_name, dtype, n_missing, tags, 统计量(mean/std...), evidences | 列的实际值数组 |
| `TargetFact` | Profile 的目标 | task_type, n_classes, class_balance, tags, evidences | y 的实际值数组 |
| `SplitResult` | Split 输出 | train/valid/test_indices, 比例, random_state, stratify, evidences | X, y, 划分后的子集数据 |

## 为什么用 dataclass 而非 TypedDict / pydantic

- **dataclass 足够简单**：TabularX 不需要序列化校验（`pydantic`）或者模式匹配（`TypedDict` 也不够）
- **纯数据容器**：没有方法、没有继承，就是字段的集合
- **Python 原生**：不需要额外依赖，IDE 类型提示开箱即用