"""报告渲染模块：纯渲染，零 IO（不负责写文件）。

核心设计：
- 模板注册表（TEMPLATES）：将"报告类型（枚举）"映射到"模板文件名"
- render_report()：dataclass/对象 + 报告类型 → HTML 字符串（纯函数，无副作用）

职责边界：
- 本模块只做"数据 → HTML 字符串"的渲染，不做任何计算、不写文件
- 写文件由 pipeline（或上层调用方）负责：html = render_report(...); path.write_text(html)
- 这样做的好处：纯函数可测（断言输出字符串），IO 逻辑集中在上层
"""

from __future__ import annotations

from enum import Enum
from importlib import resources
from pathlib import Path
from typing import Any, TypeVar

from jinja2 import Environment, FileSystemLoader, select_autoescape

# 包内模板目录（静态资源），使用 importlib.resources 定位，不依赖运行时路径
_TEMPLATES_PKG = "tabularx.templates"


class ReportType(str, Enum):
    """报告类型注册表：枚举值即模板文件名（无后缀的）。"""

    PROFILE = "profile_report.html.jinja2"
    SPLIT = "split_report.html.jinja2"
    PREPROCESS = "preprocess_report.html.jinja2"


# 模板文件名 → 渲染上下文 key（模板里用什么变量名访问数据）
TEMPLATE_CONTEXT_KEY: dict[ReportType, str] = {
    ReportType.PROFILE: "data",
    ReportType.SPLIT: "data",
    ReportType.PREPROCESS: "data",
}


def _default_template_dir() -> Path:
    """返回包内模板目录的绝对路径。

    优先用 reports.py 旁边的 templates/（源码布局最可靠），
    回退到 importlib.resources（wheel 安装布局）。
    """
    src_layout = Path(__file__).resolve().parent / "templates"
    if src_layout.is_dir():
        return src_layout
    try:
        res = resources.files(_TEMPLATES_PKG)
        if res is not None and res.is_dir():
            return Path(str(res))
    except (ModuleNotFoundError, AttributeError):
        pass
    raise FileNotFoundError(
        f"找不到模板目录 templates/（已在 {src_layout} 和包资源中查找）"
    )


# Jinja2 Environment 是线程安全的（只读模板），可缓存复用
_TEMPLATE_ENV: Environment | None = None


def _get_env(template_dir: str | Path | None = None) -> Environment:
    """创建（或复用）Jinja2 Environment，模板目录可注入（便于测试）。

    autoescape 对所有模板启用（不只是 .html/.xml）：我们的模板统一用
    .jinja2 扩展名，select_autoescape 按扩展名判断会漏掉它们，导致 XSS 风险。
    """
    env_kwargs = {
        "autoescape": True,  # 全部转义；模板需要原样输出时显式用 |safe
    }
    global _TEMPLATE_ENV
    if template_dir is not None:
        return Environment(loader=FileSystemLoader(str(template_dir)), **env_kwargs)
    if _TEMPLATE_ENV is None:
        _TEMPLATE_ENV = Environment(
            loader=FileSystemLoader(str(_default_template_dir())), **env_kwargs
        )
    return _TEMPLATE_ENV


T = TypeVar("T")


def render_template(
    template_file: str,
    data: Any,
    *,
    context_key: str = "data",
    template_dir: str | Path | None = None,
) -> str:
    """底层通用渲染：模板文件名（任意）+ 数据对象 → HTML 字符串（纯函数，无 IO）。

    这是最底层的能力：接受任意模板文件名（不经过注册表），
    供 render_report（注册表入口）和高级/测试场景使用。

    Args:
        template_file: 模板文件名（如 custom.html.jinja2）。
        data: 要渲染的数据对象（dataclass / dict / 任意有属性的对象）。
        context_key: 模板中访问数据的变量名（默认 "data"，即模板里用 {{ data.x }}）。
        template_dir: 可选注入模板目录（默认包内 templates/）。

    Returns:
        渲染好的 HTML 字符串。

    Raises:
        jinja2.TemplateNotFound: 模板文件不存在。
    """
    env = _get_env(template_dir)
    template = env.get_template(template_file)
    return template.render(**{context_key: data})


def render_report(
    report_type: ReportType,
    data: Any,
    *,
    template_dir: str | Path | None = None,
) -> str:
    """通用渲染函数（注册表入口）：报告类型 + 数据对象 → HTML 字符串（纯函数，无 IO）。

    通过 ReportType 枚举从注册表选择模板，正常使用只需这一层。

    Args:
        report_type: 报告类型（枚举，决定用哪个模板）。
        data: 要渲染的数据对象（dataclass / dict / 任意有属性的对象）。
        template_dir: 可选注入模板目录（默认包内 templates/）。

    Returns:
        渲染好的 HTML 字符串。

    Raises:
        jinja2.TemplateNotFound: 注册表指向的模板文件不存在。
    """
    return render_template(
        report_type.value,
        data,
        context_key=TEMPLATE_CONTEXT_KEY[report_type],
        template_dir=template_dir,
    )


def render_profile_report(data: Any, *, template_dir: str | Path | None = None) -> str:
    """便捷函数：渲染数据概况报告（profile）。

    Args:
        data: DataProfile 实例（或任何含 n_rows/n_cols/columns/target/fingerprint 的对象）。
        template_dir: 可选注入模板目录。

    Returns:
        HTML 字符串。
    """
    return render_report(ReportType.PROFILE, data, template_dir=template_dir)


def render_split_report(data: Any, *, template_dir: str | Path | None = None) -> str:
    """便捷函数：渲染数据划分报告（split）。

    Args:
        data: SplitResult 实例（或任何含 n_train/n_valid/n_test 等属性的对象）。
        template_dir: 可选注入模板目录。

    Returns:
        HTML 字符串。
    """
    return render_report(ReportType.SPLIT, data, template_dir=template_dir)


__all__ = [
    "ReportType",
    "TEMPLATE_CONTEXT_KEY",
    "render_report",
    "render_profile_report",
    "render_split_report",
    "render_preprocess_report",
    "render_template",
]


def render_preprocess_report(data: Any, *, template_dir: str | Path | None = None) -> str:
    """便捷函数：渲染数据预处理报告（preprocess）。

    Args:
        data: PreprocessedData 实例（或任何含 n_features_in/n_features_out/transforms 的对象）。
        template_dir: 可选注入模板目录。

    Returns:
        HTML 字符串。
    """
    return render_report(ReportType.PREPROCESS, data, template_dir=template_dir)
