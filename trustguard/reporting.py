"""TrustGuard report rendering and writing, kept separate to match TabularX style."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
from typing import Any

from jinja2 import Environment, FileSystemLoader


_SEVERITY_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
_SEVERITY_LABELS = {"info": "提示", "low": "低", "medium": "中等", "high": "高", "critical": "严重"}
_MODEL_LABELS = {
    "random_forest": "随机森林",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}
_SPLIT_LABELS = {
    "train_indices": "训练集",
    "valid_indices": "验证集",
    "test_indices": "测试集",
    "train": "训练集",
    "validation": "验证集",
    "valid": "验证集",
    "test": "测试集",
    "train_vs_validation": "训练集 vs 验证集",
    "train_vs_test": "训练集 vs 测试集",
}
_ARTIFACT_LABELS = {
    "report": "报告",
    "feature_importance": "特征重要性",
    "shap": "SHAP",
    "confusion_matrix": "混淆矩阵",
    "learning_curve": "学习曲线",
    "fairness": "公平性",
    "leakage_audit": "泄露审计",
    "test_access_log": "测试访问日志",
    "reproducibility_ledger": "复现证据",
    "distribution_audit": "分布审计",
}
_METRIC_LABELS = {
    "accuracy": "准确率", "balanced_accuracy": "平衡准确率", "f1": "F1",
    "f1_macro": "宏平均 F1", "roc_auc": "ROC-AUC", "precision": "精确率",
    "recall": "召回率", "rmse": "RMSE", "mae": "MAE", "r2": "R²",
}
_TASK_LABELS = {"binary": "二分类", "multiclass": "多分类", "regression": "回归", "auto": "自动识别"}


def _display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _number(value: Any) -> str:
    """Compact, stable number formatting for HTML and PDF tables."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _translated_summary(value: Any) -> str:
    text = str(value or "—")
    for source, target in _ARTIFACT_LABELS.items():
        text = text.replace(source, target)
    return text.replace(":", "：").replace(", ", "、")


def _leakage_checks(leakage: dict[str, Any], ledger: dict[str, Any]) -> list[dict[str, str]]:
    if leakage.get("check_results"):
        return leakage["check_results"]
    findings = leakage.get("findings", [])
    codes = {item.get("code") for item in findings}
    checked = set(leakage.get("checked_stages", []))
    config = ledger.get("config", {})

    def row(code: str, name: str, passed: str) -> dict[str, str]:
        count = sum(item.get("code") == code for item in findings)
        return {"code": code, "check": name, "status": "发现风险" if count else "通过", "result": f"发现 {count} 项风险。" if count else passed}

    rows = [
        row("target_copy", "目标复制", "未发现特征与目标逐行相同。"),
        row("possible_target_proxy", "目标代理", "未发现超过阈值的疑似目标代理特征。"),
        row("identifier_like_column", "ID/高唯一性字段", "未发现疑似 ID 或高唯一性字段。"),
    ]
    rows.append(
        row("cross_split_duplicates", "跨集合重复", "重复数为 0。")
        if "split_integrity" in checked
        else {"code": "cross_split_duplicates", "check": "跨集合重复", "status": "未执行", "result": "尚未执行数据划分完整性检查。"}
    )
    for code, name, key in (("cross_split_groups", "实体分组泄露", "group_column"), ("temporal_overlap", "时间穿越", "time_column")):
        if not config.get(key):
            rows.append({"code": code, "check": name, "status": "未执行", "result": f"未指定 {key}，无法执行该检查。"})
        else:
            rows.append(row(code, name, f"字段 {config[key]!r} 未发现异常。"))
    return rows


def _model_view(model_result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not model_result:
        return None
    metric_key = model_result.get("primary_metric", "主要指标")
    metric = _METRIC_LABELS.get(metric_key, metric_key)
    candidates = []
    for item in model_result.get("candidates", []):
        validation = item.get("validation_metrics", {})
        candidates.append({
            "model": _MODEL_LABELS.get(item.get("model_name"), item.get("model_name", "—")),
            "status": "失败" if item.get("error") else "成功",
            "primary_value": validation.get(metric_key),
            "accuracy": validation.get("accuracy"),
            "f1": validation.get("f1"),
            "roc_auc": validation.get("roc_auc"),
            "error": item.get("error"),
        })
    selected_key = model_result.get("selected_model_name")
    selected = _MODEL_LABELS.get(selected_key, selected_key or "—")
    metrics = model_result.get("final_test_metrics") or {}
    successful = [row for row in candidates if row["status"] == "成功"]
    interpretation = "模型搜索尚未产生可解释的成功候选结果。"
    if successful:
        primary_value = next((row["primary_value"] for row in successful if row["model"] == selected), None)
        metric_text = f"，验证集 {metric}={_number(primary_value)}" if primary_value is not None else ""
        test_text = "、".join(f"{_METRIC_LABELS.get(name, name)}={_number(value)}" for name, value in metrics.items()) or "未记录"
        interpretation = (
            f"{selected} 根据验证集 {metric} 被选为最终模型{metric_text}。"
            f"模型锁定后才访问一次测试集，最终测试指标为：{test_text}。"
            "测试结果没有参与候选模型排名。"
        )
    return {
        "primary_metric": metric,
        "direction": "越大越好" if model_result.get("direction") == "maximize" else "越小越好",
        "selected_model": selected,
        "candidates": candidates,
        "final_test_metrics": metrics,
        "interpretation": interpretation,
    }


def _benchmark_view(benchmark: dict[str, Any] | None) -> dict[str, Any] | None:
    if not benchmark:
        return None
    rows = []
    for source in benchmark.get("rows", []):
        row = dict(source)
        # Older benchmark summaries did not persist ``status`` on successful rows.
        success = source.get("status", "success") == "success"
        row["status_label"] = "成功" if success else "失败"
        row["mljar_explanation"] = _translated_summary(source.get("mljar_explanation"))
        row["mljar_trust_artifacts"] = _translated_summary(source.get("mljar_trust_artifacts"))
        row["trustguard_artifacts"] = _translated_summary(source.get("trustguard_artifacts"))
        row["detection_label"] = "成功" if source.get("injected_leakage_detected") else ("未检测" if success else "未执行")
        coverage_count = source.get("group_coverage_count")
        raw_interpretation = str(source.get("interpretation", ""))
        if coverage_count is None and source.get("dataset") == "adult":
            match = re.search(r"(\d+)\s+subgroup records", raw_interpretation)
            coverage_count = int(match.group(1)) if match else None
        if success and source.get("injected_leakage_detected"):
            interpretation = "目标复制和高唯一性 ID 风险均被识别。"
            if coverage_count:
                interpretation += f"同时生成 {coverage_count} 条敏感群体覆盖记录。"
        elif success:
            interpretation = "注入风险未全部命中，需要检查数据类型和审计规则。"
        else:
            error = source.get("error_message") or raw_interpretation
            interpretation = "运行失败，错误记录已保留。"
            if "timed out" in error.lower() or "timeout" in error.lower() or "超时" in error:
                interpretation = "OpenML 数据接口在训练前读取超时；该记录被保留，未计入成功数据集结论。"
        row["interpretation_cn"] = interpretation
        rows.append(row)
    successful = [row for row in rows if row["status_label"] == "成功"]
    detected_count = sum(row["detection_label"] == "成功" for row in successful)
    return {
        **benchmark,
        "rows": rows,
        "interpretation_cn": (
            f"共成功完成 {len(successful)} 个公开数据集。MLJAR Explain 的工件盘点结果与 TrustGuard 的可信性工件互补；"
            f"TrustGuard 在 {len(successful)} 个成功数据集上均生成四类可信性工件，并在 {detected_count} 个数据集上识别全部注入风险。"
            "失败数据集及其原因被如实保留，没有从汇总中删除。"
        ),
    }


def _report_view(data: dict[str, Any], max_feature_rows: int) -> dict[str, Any]:
    leakage = data.get("leakage", {})
    distribution = data.get("distribution", {})
    protocol = data.get("protocol", {})
    ledger = data.get("ledger", {})
    ledger_view = dict(ledger)
    ledger_view.setdefault("data", {})
    ledger_view.setdefault("split", {})
    ledger_view.setdefault("environment", {})
    ledger_view.setdefault("config", {})
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
    model = _model_view(data.get("model_result"))
    benchmark = _benchmark_view(data.get("benchmark"))
    key_conclusions = []
    if model and model["candidates"]:
        succeeded = [row for row in model["candidates"] if row["status"] == "成功"]
        key_conclusions.append(f"共完成 {len(succeeded)} 个候选模型的验证集评估，最终选择 {model['selected_model']}。")
    if protocol_status == "complete":
        key_conclusions.append(f"最终测试集访问 {protocol.get('test_access_count', 0)} 次，评估协议合规。")
    if benchmark:
        key_conclusions.append(f"MLJAR 对比实验成功完成 {benchmark.get('success_count', 0)} 个公开数据集，失败 {benchmark.get('failure_count', 0)} 个。")
        detected = sum(row.get("detection_label") == "成功" for row in benchmark.get("rows", []))
        key_conclusions.append(f"{detected} 个成功数据集均识别出目标复制和高唯一性 ID 注入风险，具体证据见第 5 部分。")
    return {
        "risk": risk,
        "leakage": leakage,
        "distribution": distribution,
        "protocol": protocol,
        "ledger": ledger_view,
        "findings": leakage.get("findings", []),
        "leakage_checks": _leakage_checks(leakage, ledger_view),
        "feature_rows": feature_rows[:max_feature_rows],
        "feature_total": len(feature_rows),
        "feature_truncated": len(feature_rows) > max_feature_rows,
        "coverage_rows": distribution.get("group_coverage", []),
        "protocol_status_label": protocol_labels.get(protocol_status, protocol_status),
        "overall_interpretation": overall,
        "leakage_interpretation": leakage_interpretation,
        "protocol_interpretation": protocol_interpretations.get(protocol_status, "请检查评估协议日志。"),
        "distribution_interpretation": distribution_interpretation,
        "dataset": data.get("dataset") or {},
        "model": model,
        "benchmark": benchmark,
        "key_conclusions": key_conclusions,
        "risk_label": _SEVERITY_LABELS.get(risk, risk),
        "task_label": _TASK_LABELS.get((data.get("dataset") or {}).get("task_type"), (data.get("dataset") or {}).get("task_type", "—")),
        "coverage_message": (
            "本次数据集未指定敏感特征，因此群体覆盖审计不适用。"
            if not ledger_view.get("config", {}).get("sensitive_features")
            else "已指定敏感特征，但没有生成群体覆盖记录，请检查字段配置。"
        ),
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
    environment.filters["number"] = _number
    environment.filters["severity"] = lambda value: _SEVERITY_LABELS.get(value, value)
    environment.filters["split_name"] = lambda value: _SPLIT_LABELS.get(value, value)
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
    # Jinja control-flow indentation can leave spaces on otherwise blank lines.
    # Strip line endings so ``git diff --check`` stays clean on every platform.
    rendered = render_trustguard_report(data)
    rendered = "\n".join(line.rstrip() for line in rendered.splitlines()) + "\n"
    html_path.write_text(rendered, encoding="utf-8")
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
