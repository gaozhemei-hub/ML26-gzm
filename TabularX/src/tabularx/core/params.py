"""TabularX 参数配置——各阶段的配置参数集中定义。

所有阈值全大写命名展示（overview 时），字段用小写+下划线。
每个 Config 提供：set_param（校验）、get_comments（注释）、overview（总览）、to_dict（序列化）。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, ClassVar


# ── 辅助：参数校验用元组 ─────────────────────────────────────
_Range = tuple[float | int | None, float | int | None]  # (lo, hi)，None 表示无限制


# ═══════════════════════════════════════════════════════════════
# Profile 阶段参数
# ═══════════════════════════════════════════════════════════════

@dataclass
class ProfileConfig:
    """Profile 阶段参数配置。

    所有阈值在构造时做 __post_init__ 校验，之后也可通过 set_param() 修改带校验，
    或直接用 = 赋值（但 = 赋值跳过校验，请谨慎）。
    """

    # ── 列类型判定 ──────────────────────────────────────────
    text_unique_threshold: int = 200
    """文本列判定：唯一值超过此数量 → 可能为文本列"""
    text_unique_ratio: float = 0.5
    """文本列判定：唯一值占比（唯一值数/行数）超过此值 → 文本列"""
    int_categorical_threshold: int = 20
    """整数列判定：唯一值数量 ≤ 此值 → 类别列（而非数值列）"""

    # ── 预处理决策 ──────────────────────────────────────────
    scale_mean_threshold: float = 0.5
    """Scale 判据：|mean| > 此值 → 需要标准化"""
    scale_std_threshold: float = 1.5
    """Scale 判据：std > 此值 → 需要标准化"""
    high_cardinality_threshold: int = 50
    """高基数判定：唯一值 > 此值 → high_cardinality（建议目标编码）"""
    skew_threshold: float = 1.0
    """偏斜判定：|skew| > 此值 → skewed（建议 log 变换或 Yeo-Johnson）"""
    imbalance_ratio: float = 0.1
    """类别不平衡：最小类别占比 < 此值 → imbalanced（建议 F1 评估）"""

    # ── 数据量下限 ──────────────────────────────────────────
    min_samples_regression: int = 10
    """回归任务最少样本数：低于此值 → InsufficientDataError"""
    min_samples_per_class: int = 2
    """分类任务每类最少样本数：低于此值 → InsufficientDataError"""

    # ── 元数据（不参与 __init__ / __eq__ / __repr__ / asdict）──
    _COMMENTS: ClassVar[dict[str, str]] = {
        "text_unique_threshold": "文本列判定：唯一值超过此数量 → 可能为文本列",
        "text_unique_ratio": "文本列判定：唯一值占比（唯一值数/行数）超过此值 → 文本列",
        "int_categorical_threshold": "整数列判定：唯一值数量 ≤ 此值 → 类别列",
        "scale_mean_threshold": "Scale 判据：|mean| > 此值 → 需要标准化",
        "scale_std_threshold": "Scale 判据：std > 此值 → 需要标准化",
        "high_cardinality_threshold": "高基数判定：唯一值 > 此值 → high_cardinality",
        "skew_threshold": "偏斜判定：|skew| > 此值 → skewed",
        "imbalance_ratio": "类别不平衡：最小类别占比 < 此值 → imbalanced",
        "min_samples_regression": "回归任务最少样本数（低于此值报错）",
        "min_samples_per_class": "分类任务每类最少样本数（低于此值报错）",
    }
    _RANGES: ClassVar[dict[str, _Range]] = {
        "text_unique_threshold": (1, None),
        "text_unique_ratio": (0.0, 1.0),
        "int_categorical_threshold": (1, None),
        "scale_mean_threshold": (0.0, None),
        "scale_std_threshold": (0.0, None),
        "high_cardinality_threshold": (1, None),
        "skew_threshold": (0.0, None),
        "imbalance_ratio": (0.0, 1.0),
        "min_samples_regression": (1, None),
        "min_samples_per_class": (1, None),
    }

    def __post_init__(self) -> None:
        """构造时自动校验所有参数。"""
        for name, (lo, hi) in self._RANGES.items():
            val = getattr(self, name)
            if lo is not None and val < lo:
                raise ValueError(
                    f"{name} 不能小于 {lo}，收到 {val}。"
                    f"  {self.get_comments(name)}"
                )
            if hi is not None and val > hi:
                raise ValueError(
                    f"{name} 不能大于 {hi}，收到 {val}。"
                    f"  {self.get_comments(name)}"
                )

    # ── 实例方法 ────────────────────────────────────────────

    def set_param(self, name: str, value: Any) -> None:
        """设置参数值（带范围校验）。

        Args:
            name: 参数名（与字段名一致，如 "scale_mean_threshold"）。
            value: 参数值。

        Raises:
            ValueError: 参数名不存在、值不在有效范围内。
        """
        if name not in self._COMMENTS:
            raise ValueError(
                f"未知参数 {name!r}。可用参数：\n"
                + "\n".join(f"  {k}" for k in self._COMMENTS)
            )

        # 范围校验（类型错误会在后续使用中自然暴露）
        lo, hi = self._RANGES[name]
        if lo is not None and value < lo:
            raise ValueError(
                f"{name} 不能小于 {lo}，收到 {value}。"
                f"  {self.get_comments(name)}"
            )
        if hi is not None and value > hi:
            raise ValueError(
                f"{name} 不能大于 {hi}，收到 {value}。"
                f"  {self.get_comments(name)}"
            )

        setattr(self, name, value)

    def get_comments(self, name: str) -> str:
        """返回指定参数的中文注释。"""
        if name not in self._COMMENTS:
            raise ValueError(
                f"未知参数 {name!r}。可用参数：\n"
                + "\n".join(f"  {k}" for k in self._COMMENTS)
            )
        return self._COMMENTS[name]

    def overview(self) -> str:
        """返回本阶段所有参数的中文总览（含当前值）。"""
        lines = []
        for name, comment in self._COMMENTS.items():
            val = getattr(self, name)
            lines.append(f"  {name:>30} = {str(val):>5}   #{comment}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """返回参数字典（用于序列化/保存实验配置）。"""
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if not f.name.startswith("_")
        }


# ═══════════════════════════════════════════════════════════════
# Split 阶段参数（暂定，后续补充）
# ═══════════════════════════════════════════════════════════════

@dataclass
class SplitConfig:
    """Split 阶段参数配置。"""

    test_size: float = 0.2
    """测试集占比（占总样本的比例）"""
    validation_size: float = 0.25
    """验证集占比（相对于去掉测试集后剩余部分的比例）"""
    random_state: int = 42
    """随机种子（保证划分可复现）"""
    min_samples_split: int = 5
    """训练/验证/测试集每集最少样本数，任一子集少于该值则报错"""

    _COMMENTS: ClassVar[dict[str, str]] = {
        "test_size": "测试集占比（占总样本的比例）",
        "validation_size": "验证集占比（相对于去掉测试集后剩余部分的比例）",
        "random_state": "随机种子（保证划分可复现）",
        "min_samples_split": "训练/验证/测试集每集最少样本数",
    }
    _RANGES: ClassVar[dict[str, _Range]] = {
        "test_size": (0.0, 1.0),
        "validation_size": (0.0, 1.0),
        "random_state": (0, None),
        "min_samples_split": (1, None),
    }

    def __post_init__(self) -> None:
        for name, (lo, hi) in self._RANGES.items():
            val = getattr(self, name)
            if lo is not None and val < lo:
                raise ValueError(
                    f"{name} 不能小于 {lo}，收到 {val}。"
                    f"  {self.get_comments(name)}"
                )
            if hi is not None and val > hi:
                raise ValueError(
                    f"{name} 不能大于 {hi}，收到 {val}。"
                    f"  {self.get_comments(name)}"
                )
        # 推导检查：by-construction train_ratio > 0
        derived_train_ratio = (
            1 - self.test_size - self.validation_size * (1 - self.test_size)
        )
        if derived_train_ratio <= 0:
            raise ValueError(
                f"test_size ({self.test_size}) + validation_size ({self.validation_size})"
                f" 组合导致训练集占比为 {derived_train_ratio:.4f}，应为正数。"
                f"  test_size 与 validation_size 之和过大，请至少减小其中之一。"
            )

    def set_param(self, name: str, value: Any) -> None:
        if name not in self._COMMENTS:
            raise ValueError(
                f"未知参数 {name!r}。可用参数：\n"
                + "\n".join(f"  {k}" for k in self._COMMENTS)
            )
        lo, hi = self._RANGES[name]
        if lo is not None and value < lo:
            raise ValueError(
                f"{name} 不能小于 {lo}，收到 {value}。"
                f"  {self.get_comments(name)}"
            )
        if hi is not None and value > hi:
            raise ValueError(
                f"{name} 不能大于 {hi}，收到 {value}。"
                f"  {self.get_comments(name)}"
            )
        # 推导检查：修改 test_size / validation_size 时确保 train_ratio > 0
        if name in ("test_size", "validation_size"):
            tmp_test = value if name == "test_size" else self.test_size
            tmp_valid = value if name == "validation_size" else self.validation_size
            derived_train_ratio = 1 - tmp_test - tmp_valid * (1 - tmp_test)
            if derived_train_ratio <= 0:
                raise ValueError(
                    f"test_size ({tmp_test}) + validation_size ({tmp_valid})"
                    f" 组合导致训练集占比为 {derived_train_ratio:.4f}，应为正数。"
                    f"  test_size 与 validation_size 之和过大，请至少减小其中之一。"
                )
        setattr(self, name, value)

    def get_comments(self, name: str) -> str:
        if name not in self._COMMENTS:
            raise ValueError(f"未知参数 {name!r}")
        return self._COMMENTS[name]

    def overview(self) -> str:
        lines = []
        for name, comment in self._COMMENTS.items():
            val = getattr(self, name)
            lines.append(f"  {name:>30} = {str(val):>5}   #{comment}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if not f.name.startswith("_")
        }


# ═══════════════════════════════════════════════════════════════
# Preprocess 阶段参数（暂定，后续补充）
# ═══════════════════════════════════════════════════════════════

@dataclass
class PreprocessConfig:
    """Preprocess 阶段参数配置。"""

    # ── 缺失值填补 ──────────────────────────────────────────
    numeric_impute: str = "median"
    """数值列填补方法："median" | "mean" | "constant"（使用 impute_constant_value）"""
    categorical_impute: str = "most_frequent"
    """类别列填补方法："most_frequent" | "constant"（使用 impute_constant_value）"""
    impute_constant_value: float = 0.0
    """当数值/类别填补方法为 constant 时使用的填充值"""

    # ── 标准化 ──────────────────────────────────────────────
    scale_method: str = "standard"
    """标准化方法："standard"（StandardScaler） | "minmax"（MinMaxScaler） | "robust"（RobustScaler）"""

    # ── 偏度校正 ────────────────────────────────────────────
    skew_method: str = "yeo-johnson"
    """偏度校正方法："yeo-johnson"（推荐，支持非正数）| "log1p"（仅正数）"""

    # ── 类别编码 ────────────────────────────────────────────
    low_cardinality_encoding: str = "onehot"
    """低基数类别列编码："onehot"（OneHotEncoder）| "label"（OrdinalEncoder）"""
    high_cardinality_encoding: str = "target"
    """高基数类别列编码："target"（目标编码）| "frequency"（频率编码）"""
    max_onehot_categories: int = 20
    """低于此值的类别列使用 low_cardinality_encoding，超过的使用 high_cardinality_encoding"""

    # ── 特殊列 ──────────────────────────────────────────────
    handle_text: str = "drop"
    """文本列处理："drop"（丢弃）| "keep"（保留，当前 keep 会报错）"""
    handle_datetime: str = "extract"
    """日期列处理："extract"（抽特征）| "drop"（丢弃）| "keep"（保留）"""
    datetime_features: str = "basic"
    """日期特征提取级别："basic"（年/月/日）| "full"（+星期/季度）"""

    _COMMENTS: ClassVar[dict[str, str]] = {
        "numeric_impute": "数值列填补方法：median / mean / constant",
        "categorical_impute": "类别列填补方法：most_frequent / constant",
        "impute_constant_value": "数值/类别填补为 constant 时的填充值",
        "scale_method": "标准化方法：standard / minmax / robust",
        "skew_method": "偏度校正方法：yeo-johnson（推荐）或 log1p",
        "low_cardinality_encoding": "低基数类别列编码：onehot 或 label",
        "high_cardinality_encoding": "高基数类别列编码：target 或 frequency",
        "max_onehot_categories": "低/高基数编码切换阈值（唯一值 ≤ 此值用 low_cardinality_encoding）",
        "handle_text": "文本列处理：drop（丢弃）或 keep（当前 keep 会报错）",
        "handle_datetime": "日期列处理：extract / drop / keep",
        "datetime_features": "日期特征提取级别：basic（年/月/日）或 full（+星期/季度）",
    }
    _RANGES: ClassVar[dict[str, _Range]] = {
        "impute_constant_value": (None, None),
        "max_onehot_categories": (1, None),
    }
    _CHOICES: ClassVar[dict[str, set[str]]] = {
        "numeric_impute": {"median", "mean", "constant"},
        "categorical_impute": {"most_frequent", "constant"},
        "scale_method": {"standard", "minmax", "robust"},
        "skew_method": {"yeo-johnson", "log1p"},
        "low_cardinality_encoding": {"onehot", "label"},
        "high_cardinality_encoding": {"target", "frequency"},
        "handle_text": {"drop", "keep"},
        "handle_datetime": {"extract", "drop", "keep"},
        "datetime_features": {"basic", "full"},
    }

    def __post_init__(self) -> None:
        """构造时自动校验所有参数。"""
        for name, (lo, hi) in self._RANGES.items():
            val = getattr(self, name)
            if lo is not None and val < lo:
                raise ValueError(
                    f"{name} 不能小于 {lo}，收到 {val}。"
                    f"  {self.get_comments(name)}"
                )
            if hi is not None and val > hi:
                raise ValueError(
                    f"{name} 不能大于 {hi}，收到 {val}。"
                    f"  {self.get_comments(name)}"
                )
        # 枚举选项校验
        for name, choices in self._CHOICES.items():
            val = getattr(self, name)
            if val not in choices:
                raise ValueError(
                    f"{name} 必须为 {choices} 之一，收到 {val!r}。"
                    f"  {self.get_comments(name)}"
                )

    def set_param(self, name: str, value: Any) -> None:
        if name not in self._COMMENTS:
            raise ValueError(f"未知参数 {name!r}。可用：{list(self._COMMENTS)}")
        lo, hi = self._RANGES.get(name, (None, None))
        if lo is not None and value < lo:
            raise ValueError(f"{name} 不能小于 {lo}，收到 {value}。  {self.get_comments(name)}")
        if hi is not None and value > hi:
            raise ValueError(f"{name} 不能大于 {hi}，收到 {value}。  {self.get_comments(name)}")
        choices = self._CHOICES.get(name)
        if choices is not None and value not in choices:
            raise ValueError(f"{name} 必须为 {choices} 之一，收到 {value!r}。  {self.get_comments(name)}")
        setattr(self, name, value)

    def get_comments(self, name: str) -> str:
        if name not in self._COMMENTS:
            raise ValueError(f"未知参数 {name!r}")
        return self._COMMENTS[name]

    def overview(self) -> str:
        lines = []
        for name, comment in self._COMMENTS.items():
            val = getattr(self, name)
            lines.append(f"  {name:>30} = {str(val):>5}   #{comment}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if not f.name.startswith("_")
        }


# ═══════════════════════════════════════════════════════════════
# Learning（模型搜索）阶段参数
# ═══════════════════════════════════════════════════════════════

@dataclass
class ModelSearchConfig:
    """ModelSearch 阶段参数配置。"""

    # ── 并行与搜索 ──────────────────────────────────────────
    n_jobs: int = 4
    """并行进程数（同时运行多少个模型的 Optuna）"""
    n_trials: int = 50
    """每个模型 Optuna 搜索的 trial 数量"""
    random_state: int = 42
    """随机种子"""

    # ── 模型选择 ────────────────────────────────────────────
    models: list[str] = field(default_factory=lambda: ["auto"])
    """使用的模型列表。默认 ["auto"] 表示全部可用模型。
       可指定子集，如 ["random_forest", "xgboost"]。
       每个模型自动匹配当前任务类型（回归/分类）。"""
    top_k_models: int = 3
    """保留验证集前 K 个模型进入 Voting 组合"""

    # ── 评估指标 ────────────────────────────────────────────
    primary_metric: str = "auto"
    """模型选择的主指标。"auto" 根据 task_type 自动选择：
       回归默认 "rmse"，二分类默认 "roc_auc"，多分类默认 "f1_macro"。"""

    # ── 最终训练 ────────────────────────────────────────────
    retrain_on_train_valid: bool = True
    """选出最佳模型后，是否用 train+valid 重新训练再评估 test"""

    # ── 保存控制 ────────────────────────────────────────────
    save_models: bool = True
    """是否保存所有模型的最终版本到本地"""

    # ── 搜索空间定制 ────────────────────────────────────────
    fixed_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    """固定参数（不参与搜索），按模型名 → 参数名 → 值 索引。
       被列出的参数将跳过 trial.suggest_*() 调用，直接使用固定值。
       示例:
         {"random_forest": {"max_depth": 8, "n_estimators": 200}}"""
    search_space_overrides: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    """搜索范围覆盖，按模型名 → 参数名 → 覆盖字段 三层索引。
       可覆盖字段: low, high, step, log, choices。
       示例:
         {"xgboost": {"n_estimators": {"high": 1000},
                       "learning_rate": {"low": 1e-4, "high": 0.5}}}"""

    _COMMENTS: ClassVar[dict[str, str]] = {
        "n_jobs": "并行进程数（同时跑几个模型的 Optuna）",
        "n_trials": "每个模型 Optuna 搜索的 trial 数量",
        "random_state": "随机种子",
        "models": "使用的模型列表，默认为全部可用模型",
        "top_k_models": "保留验证集前 K 个模型进入 Voting 组合",
        "primary_metric": "模型选择的主指标，auto 根据 task_type 自动选择",
        "retrain_on_train_valid": "选出最佳模型后是否用 train+valid 重训再评估 test",
        "save_models": "是否保存所有模型的最终版本到本地",
        "fixed_params": "固定参数（不参与搜索），按模型名索引",
        "search_space_overrides": "搜索范围覆盖，按模型名 → 参数名 → 字段 三层索引",
    }
    _RANGES: ClassVar[dict[str, _Range]] = {
        "n_jobs": (1, None),
        "n_trials": (1, None),
        "random_state": (0, None),
        "top_k_models": (1, None),
    }

    def __post_init__(self) -> None:
        """构造时自动校验所有参数。"""
        for name, (lo, hi) in self._RANGES.items():
            val = getattr(self, name)
            if lo is not None and val < lo:
                raise ValueError(
                    f"{name} 不能小于 {lo}，收到 {val}。"
                    f"  {self.get_comments(name)}"
                )
        # models 校验
        models = self.models
        if not isinstance(models, list) or len(models) == 0:
            raise ValueError(
                f"models 必须为非空列表，收到 {models!r}。"
                f"  使用 ['auto'] 表示全部模型。"
            )
        if self.primary_metric not in ("auto", "rmse", "mae", "r2",
                                         "roc_auc", "f1_macro", "f1", "accuracy",
                                         "balanced_accuracy"):
            raise ValueError(
                f"不支持的 primary_metric {self.primary_metric!r}。"
                f"  使用 'auto' 自动选择，或指定具体指标。"
            )

    def set_param(self, name: str, value: Any) -> None:
        if name not in self._COMMENTS:
            raise ValueError(f"未知参数 {name!r}。可用：{list(self._COMMENTS)}")
        lo, hi = self._RANGES.get(name, (None, None))
        if lo is not None and value < lo:
            raise ValueError(f"{name} 不能小于 {lo}，收到 {value}。  {self.get_comments(name)}")
        if hi is not None and value > hi:
            raise ValueError(f"{name} 不能大于 {hi}，收到 {value}。  {self.get_comments(name)}")
        setattr(self, name, value)

    def get_comments(self, name: str) -> str:
        if name not in self._COMMENTS:
            raise ValueError(f"未知参数 {name!r}")
        return self._COMMENTS[name]

    def overview(self) -> str:
        lines = []
        for name, comment in self._COMMENTS.items():
            val = getattr(self, name)
            lines.append(f"  {name:>30} = {str(val):>5}   #{comment}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if not f.name.startswith("_")
        }


# ═══════════════════════════════════════════════════════════════
# 全局参数容器
# ═══════════════════════════════════════════════════════════════

@dataclass
class TabularXParams:
    """TabularX 全局参数容器——聚合所有阶段的配置。

    Usage:
        tx = TabularX()
        tx.params.profile.scale_mean_threshold = 1.0          # 直接赋值
        tx.params.set_param("profile", "skew_threshold", 0.5)  # 带校验
        print(tx.params.overview())                            # 总览
        print(tx.params.get_comments("profile", "scale_mean_threshold"))  # 注释
    """

    profile: ProfileConfig = field(default_factory=ProfileConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model_search: ModelSearchConfig = field(default_factory=ModelSearchConfig)

    # ── 总览 ──────────────────────────────────────────────────

    def overview(self) -> str:
        """返回所有阶段参数的中文总览。"""
        sep = "=" * 60
        lines = [sep, "  TabularX 参数总览", sep]
        for label, cfg in [("Profile", self.profile),
                           ("Split", self.split),
                           ("Preprocess", self.preprocess),
                           ("ModelSearch", self.model_search)]:
            lines.append(f"\n【{label} 阶段】")
            lines.append(cfg.overview())
        lines.append(sep)
        return "\n".join(lines)

    # ── 按阶段参数名设置（带校验） ────────────────────────────

    def set_param(self, stage: str, name: str, value: Any) -> None:
        """按阶段设置参数值（带校验）。

        Args:
            stage: 阶段名 "profile" / "split" / "preprocess" / "model_search"。
            name: 参数名。
            value: 参数值。

        Raises:
            ValueError: 阶段名不存在、参数名不存在。
            TypeError: 参数类型不匹配。
        """
        stage_map = {
            "profile": self.profile,
            "split": self.split,
            "preprocess": self.preprocess,
            "model_search": self.model_search,
        }
        if stage not in stage_map:
            raise ValueError(
                f"未知阶段 {stage!r}。可用：{list(stage_map)}"
            )
        stage_map[stage].set_param(name, value)

    # ── 查询注释 ──────────────────────────────────────────────

    def get_comments(self, stage: str, name: str) -> str:
        """返回指定阶段中某参数的中文注释。

        Args:
            stage: 阶段名 "profile" / "split" / "preprocess" / "model_search"。
            name: 参数名。

        Returns:
            中文注释字符串。

        Raises:
            ValueError: 阶段名或参数名不存在。
        """
        stage_map = {
            "profile": self.profile,
            "split": self.split,
            "preprocess": self.preprocess,
            "model_search": self.model_search,
        }
        if stage not in stage_map:
            raise ValueError(
                f"未知阶段 {stage!r}。可用：{list(stage_map)}"
            )
        return stage_map[stage].get_comments(name)

    # ── 序列化 ─────────────────────────────────────────────────

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """返回所有阶段参数字典（用于保存实验配置）。"""
        return {
            "profile": self.profile.to_dict(),
            "split": self.split.to_dict(),
            "preprocess": self.preprocess.to_dict(),
            "model_search": self.model_search.to_dict(),
        }

    # ── 重置 ──────────────────────────────────────────────────

    def reset(self) -> None:
        """重置所有参数为默认值。"""
        self.profile = ProfileConfig()
        self.split = SplitConfig()
        self.preprocess = PreprocessConfig()
        self.model_search = ModelSearchConfig()


__all__ = [
    # Profile
    "ProfileConfig",
    # Split
    "SplitConfig",
    # Preprocess
    "PreprocessConfig",
    # Learning
    "ModelSearchConfig",
    # 全局容器
    "TabularXParams",
]