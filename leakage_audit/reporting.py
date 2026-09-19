"""Local, dependency-free audit artifact writer."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .audit import AuditResult


def write_audit_artifacts(result: AuditResult, output_dir: str | Path) -> dict[str, str]:
    root = Path(output_dir) / "leakage_audit"
    root.mkdir(parents=True, exist_ok=True)
    summary = root / "summary.json"
    summary.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    findings = root / "findings.csv"
    with findings.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=["severity", "code", "title", "columns", "evidence", "recommendation"])
        writer.writeheader()
        for item in result.findings:
            writer.writerow({"severity": item.severity, "code": item.code, "title": item.title, "columns": ", ".join(item.columns), "evidence": item.evidence, "recommendation": item.recommendation})
    infosheet = root / "model_info_sheet.md"
    infosheet.write_text("# 模型信息表：泄露风险人工确认\n\n" + "\n".join(f"{i}. {q}" for i, q in enumerate(result.model_info_questions, 1)), encoding="utf-8")
    return {"summary": str(summary), "findings": str(findings), "model_info_sheet": str(infosheet)}
