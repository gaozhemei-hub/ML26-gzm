"""TrustGuard report rendering and writing, kept separate to match TabularX style."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

from jinja2 import Environment, FileSystemLoader


_SEVERITY_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def _display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _report_view(data: dict[str, Any], max_feature_rows: int) -> dict[str, Any]:
    leakage = data.get("leakage", {})
    distribution = data.get("distribution", {})
    protocol = data.get("protocol", {})
    ledger = data.get("ledger", {})
    ledger_view = dict(ledger)
    ledger_view.setdefault("data", {})
    ledger_view.setdefault("split", {})
    ledger_view.setdefault("environment", {})
    feature_rows = list(distribution.get("feature_records", []))
    feature_rows.sort(
        key=lambda row: (
            _SEVERITY_ORDER.get(row.get("severity", "info"), 0),
            max(
                float(row.get("psi") or 0),
                float(row.get("ks_statistic") or 0),
                float(row.get("total_variation") or 0),
                float(row.get("unseen_rate") or 0),
            ),
        ),
        reverse=True,
    )
    levels = [leakage.get("risk_level", "info"), distribution.get("risk_level", "info")]
    risk = max(levels, key=lambda level: _SEVERITY_ORDER.get(level, 0))
    protocol_status = protocol.get("status", "complete" if protocol.get("compliant") else "not_started")
    protocol_labels = {
        "not_started": "尚未开始模型评估",
        "validation_in_progress": "正在验证候选模型",
        "selection_locked": "模型已锁定，等待最终测试",
        "complete": "已完成且合规",
    }
    if risk in {"high", "critical"}:
        overall = "存在需要在训练或发布前处理的高风险证据；当前分数不应直接作为最终结论。"
    elif risk == "medium":
        overall = "未发现已确认的严重泄露，但存在需要人工复核的漂移或疑似风险。"
    else:
        overall = "自动规则未发现显著风险；仍需结合字段产生时间和业务流程完成人工确认。"
    leakage_interpretation = (
        "存在高风险或严重泄露发现，应按建议移除/重构字段或重新划分数据后再训练。"
        if leakage.get("risk_level") in {"high", "critical"}
        else "未发现自动规则能够确认的高风险泄露；这不等于业务语义已被证明安全，应继续回答报告中的模型信息问题。"
    )
    protocol_interpretations = {
        "not_started": "本次是数据审计运行，并未执行模型选择，因此没有虚构测试集访问记录；实际训练时应把同一个守卫对象传入流水线。",
        "validation_in_progress": "候选模型仍在验证阶段，此时访问测试集会被阻止。",
        "selection_locked": "唯一模型已经由验证集锁定，下一步只允许进行一次最终测试评估。",
        "complete": "模型按验证集完成选择，测试集只被锁定模型访问一次，协议记录完整。",
    }
    distribution_interpretation = (
        "存在高风险漂移或群体样本不足。应检查划分策略、时间变化、类别编码和群体代表性，并在修正后重新评估。"
        if distribution.get("risk_level") in {"high", "critical"}
        else "未发现高风险分布差异。中等风险指标仍应结合样本量和业务变化解释，不能只凭阈值判断模型可部署。"
    )
    return {
        "risk": risk,
        "leakage": leakage,
        "distribution": distribution,
        "protocol": protocol,
        "ledger": ledger_view,
        "findings": leakage.get("findings", []),
        "feature_rows": feature_rows[:max_feature_rows],
        "feature_total": len(feature_rows),
        "feature_truncated": len(feature_rows) > max_feature_rows,
        "coverage_rows": distribution.get("group_coverage", []),
        "protocol_status_label": protocol_labels.get(protocol_status, protocol_status),
        "overall_interpretation": overall,
        "leakage_interpretation": leakage_interpretation,
        "protocol_interpretation": protocol_interpretations.get(protocol_status, "请检查评估协议日志。"),
        "distribution_interpretation": distribution_interpretation,
        "benchmark": data.get("benchmark"),
    }


def render_trustguard_report(data: dict[str, Any], *, max_feature_rows: int = 15) -> str:
    """Pure function: structured results to self-contained HTML, with no I/O."""
    if max_feature_rows < 1:
        raise ValueError("max_feature_rows 必须至少为 1。")
    template_dir = Path(__file__).resolve().parent / "templates"
    environment = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=True,
    )
    environment.filters["display"] = _display
    template = environment.get_template("trustguard_report.html.jinja2")
    return template.render(data=_report_view(data, max_feature_rows))


def _import_weasyprint():
    """Import WeasyPrint, including a Windows-Conda DLL-name compatibility shim."""
    if os.name != "nt":
        import weasyprint
        return weasyprint

    # Conda-forge names several GTK DLLs without the ``lib`` prefix, while
    # WeasyPrint probes Unix-style aliases first. Resolve only those known
    # aliases to absolute DLL paths during import, then restore cffi unchanged.
    dll_dir = Path(sys.prefix) / "Library" / "bin"
    if not dll_dir.is_dir():
        import weasyprint
        return weasyprint

    import cffi

    aliases = {
        "libgobject-2.0-0": "gobject-2.0-0.dll",
        "gobject-2.0-0": "gobject-2.0-0.dll",
        "libpango-1.0-0": "pango-1.0-0.dll",
        "pango-1.0-0": "pango-1.0-0.dll",
        "libharfbuzz-0": "harfbuzz.dll",
        "harfbuzz": "harfbuzz.dll",
        "libfontconfig-1": "fontconfig-1.dll",
        "fontconfig-1": "fontconfig-1.dll",
        "libpangoft2-1.0-0": "pangoft2-1.0-0.dll",
        "pangoft2-1.0-0": "pangoft2-1.0-0.dll",
    }
    original_dlopen = cffi.FFI.dlopen

    def resolved_dlopen(ffi, name, flags=0):
        candidate = dll_dir / aliases.get(name, "")
        if name in aliases and candidate.is_file():
            name = str(candidate)
        return original_dlopen(ffi, name, flags)

    cffi.FFI.dlopen = resolved_dlopen
    try:
        import weasyprint
        return weasyprint
    finally:
        cffi.FFI.dlopen = original_dlopen


def write_trustguard_report(data: dict[str, Any], output_dir: str | Path, *, pdf_report: bool = False) -> dict[str, str]:
    """Write ``reports/trustguard.html`` and optionally a PDF beside it."""
    reports_dir = Path(output_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    html_path = reports_dir / "trustguard.html"
    html_path.write_text(render_trustguard_report(data), encoding="utf-8")
    paths = {"html": str(html_path)}
    if pdf_report:
        try:
            weasyprint = _import_weasyprint()
            pdf_path = reports_dir / "trustguard.pdf"
            weasyprint.HTML(filename=str(html_path)).write_pdf(pdf_path)
            paths["pdf"] = str(pdf_path)
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "PDF 导出需要可用的 WeasyPrint/GTK 运行库；"
                "Windows + Conda 请使用 conda-forge 安装 WeasyPrint。"
            ) from exc
    return paths
