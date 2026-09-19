"""A guard that exposes test-set-driven model selection as an audit finding."""
from __future__ import annotations

from .audit import AuditFinding, AuditResult


def validate_selection_protocol(model_results, primary_metric: str, direction: str, result: AuditResult | None = None) -> AuditResult:
    """Verify the selected-by-validation model is not replaced using test scores.

    Accepts existing TabularX ``ModelRunResult`` objects by duck typing, so no
    change to their schema is required.
    """
    result = result or AuditResult()
    result.checked_stages.append("model_selection")
    successful = [r for r in model_results if getattr(r, "status", "success") == "success"]
    if not successful: return result
    reverse = direction == "maximize"
    best_valid = sorted(successful, key=lambda r: getattr(r, "validation_metrics", {}).get(primary_metric, float("-inf") if reverse else float("inf")), reverse=reverse)[0]
    with_test = [r for r in successful if getattr(r, "test_metrics", None)]
    if len(with_test) > 1:
        best_test = sorted(with_test, key=lambda r: r.test_metrics.get(primary_metric, float("-inf") if reverse else float("inf")), reverse=reverse)[0]
        if best_test.model_name != best_valid.model_name:
            result.findings.append(AuditFinding("test_set_model_selection", "critical", "测试集驱动模型选择", f"验证集最佳为 {best_valid.model_name!r}，测试集最佳为 {best_test.model_name!r}。", "锁定验证集最佳模型后，只允许该模型访问测试集一次。", [best_valid.model_name, best_test.model_name]))
        else:
            result.findings.append(AuditFinding("test_set_exposed_to_multiple_models", "high", "多个候选模型访问测试集", f"有 {len(with_test)} 个模型产生测试集指标。", "即使排名恰好一致，也应在选择前阻断候选模型访问测试集。"))
    return result
