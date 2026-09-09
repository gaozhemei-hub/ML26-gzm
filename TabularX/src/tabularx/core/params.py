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

    skew_method: str = "yeojohnson"
    """偏度校正方法："yeojohnson"（推荐，支持非正数）| "log1p"（仅正数）"""

    _COMMENTS: ClassVar[dict[str, str]] = {
        "skew_method": "偏度校正方法：yeojohnson（推荐）或 log1p",
    }
    _RANGES: ClassVar[dict[str, _Range]] = {}

    def set_param(self, name: str, value: Any) -> None:
        if name not in self._COMMENTS:
            raise ValueError(f"未知参数 {name!r}。可用：{list(self._COMMENTS)}")
        if name == "skew_method" and value not in ("yeojohnson", "log1p"):
            raise ValueError(f"skew_method 必须为 'yeojohnson' 或 'log1p'，收到 {value!r}")
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

    # ── 总览 ──────────────────────────────────────────────────

    def overview(self) -> str:
        """返回所有阶段参数的中文总览。"""
        sep = "=" * 60
        lines = [sep, "  TabularX 参数总览", sep]
        for label, cfg in [("Profile", self.profile),
                           ("Split", self.split),
                           ("Preprocess", self.preprocess)]:
            lines.append(f"\n【{label} 阶段】")
            lines.append(cfg.overview())
        lines.append(sep)
        return "\n".join(lines)

    # ── 按阶段参数名设置（带校验） ────────────────────────────

    def set_param(self, stage: str, name: str, value: Any) -> None:
        """按阶段设置参数值（带校验）。

        Args:
            stage: 阶段名 "profile" / "split" / "preprocess"。
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
            stage: 阶段名 "profile" / "split" / "preprocess"。
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
        }

    # ── 重置 ──────────────────────────────────────────────────

    def reset(self) -> None:
        """重置所有参数为默认值。"""
        self.profile = ProfileConfig()
        self.split = SplitConfig()
        self.preprocess = PreprocessConfig()


__all__ = [
    # Profile
    "ProfileConfig",
    # Split
    "SplitConfig",
    # Preprocess
    "PreprocessConfig",
    # 全局容器
    "TabularXParams",
]