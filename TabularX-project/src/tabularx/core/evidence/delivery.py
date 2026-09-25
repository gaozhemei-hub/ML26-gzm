"""Explicit requirements, attempt history, file acceptance and local retry."""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from typing import Callable
from uuid import uuid4


@dataclass
class EvidenceOutcome:
    paths: list[str] = field(default_factory=list)
    error: str | None = None


class EvidenceDelivery:
    """Each producer must explicitly declare its outputs; no filename guessing.

    Generated means a successful producer returned existing nonempty files.
    It does not mean scientific validity or reproducibility. Recovery callbacks
    live only in memory. No arbitrary code is loaded from a saved manifest.
    """

    def __init__(self, base_dir: str | Path, task: str):
        self.run_id = uuid4().hex
        self.root = Path(base_dir).resolve() / "runs" / self.run_id
        self.root.mkdir(parents=True, exist_ok=False)
        self.task = task
        self.records: dict[str, dict] = {}
        self._callbacks: dict[str, Callable] = {}
        self._refresh: set[str] = set()

    def register(self, key, producer=None, *, required=True, applicable=True,
                 enabled=True, reason="", recoverable=True, refresh=False, condition=""):
        if key in self.records:
            raise ValueError(f"Duplicate evidence requirement: {key}")
        status = "pending"
        if not applicable:
            status = "not_applicable"
        elif not enabled:
            status = "not_enabled"
        if status != "pending" and not reason:
            raise ValueError("Excluded requirements need an explicit reason")
        self.records[key] = dict(required=required, applicable=applicable,
                                 condition=condition or f"Declared for task={self.task}",
                                 enabled=enabled, status=status, reason=reason,
                                 paths=[], attempts=[], recoverable=recoverable)
        if producer is not None:
            self._callbacks[key] = producer
        if refresh:
            self._refresh.add(key)

    def execute(self, key):
        row = self.records[key]
        if not row["applicable"] or not row["enabled"]:
            return
        started = datetime.now(timezone.utc).isoformat()
        paths = []
        try:
            if key not in self._callbacks:
                raise RuntimeError("No producer registered")
            outcome = self._callbacks[key]()
            if not isinstance(outcome, EvidenceOutcome):
                raise TypeError("Producer must return EvidenceOutcome")
            for name in outcome.paths:
                path = Path(name)
                if not path.is_absolute():
                    path = self.root / path
                path = path.resolve()
                paths.append(path.relative_to(self.root).as_posix())
            if outcome.error:
                status, reason = "failed", outcome.error
            elif not paths or not self._files_valid(paths):
                status, reason = "missing", "Expected output absent, empty, or outside this run"
            else:
                status, reason = "generated", "Producer completed; declared files present and nonempty"
        except Exception as exc:
            status, reason = "failed", f"{type(exc).__name__}: {exc}"
        row.update(status=status, reason=reason, paths=paths)
        row["attempts"].append(dict(started_at=started,
            ended_at=datetime.now(timezone.utc).isoformat(), status=status,
            reason=reason, paths=list(paths)))
        self.write()

    def _files_valid(self, paths):
        for name in paths:
            path = (self.root / name).resolve()
            if not path.is_relative_to(self.root) or not path.is_file() or path.stat().st_size == 0:
                return False
        return True

    def audit(self):
        for row in self.records.values():
            if row["status"] == "generated" and not self._files_valid(row["paths"]):
                row.update(status="missing", reason="Declared output deleted, empty, or invalid after generation")
        required = [r for r in self.records.values() if r["required"] and r["applicable"]]
        delivered = sum(r["status"] == "generated" for r in required)
        return dict(schema_version="1.0", run_id=self.run_id, task=self.task,
                    complete=bool(required) and all(r["status"] == "generated" for r in required),
                    required_count=len(required), generated_required_count=delivered,
                    completeness=delivered / len(required) if required else None,
                    boundary="Delivery acceptance only; not scientific validity or reproducibility",
                    records=deepcopy(self.records))

    def retry(self, keys=None):
        self.audit()
        selected = list(dict.fromkeys(keys)) if keys is not None else [
            k for k, r in self.records.items()
            if r["status"] in {"failed", "missing"} and r["recoverable"]]
        # Validate the whole request before starting any work.
        for key in selected:
            row = self.records[key]
            if row["status"] not in {"failed", "missing"} or not row["recoverable"]:
                raise ValueError(f"{key} is not a recoverable failed/missing item")
        for key in selected:
            if key not in self._refresh:
                self.execute(key)
        if selected:
            # Reports may mention failed steps, so these are refresh dependencies,
            # not blocking dependencies. Re-render them after recovery.
            for key in self.records:
                if key in self._refresh and self.records[key]["enabled"]:
                    self.execute(key)
        return self.write()

    def write(self):
        payload = self.audit()
        folder = self.root / "evidence"
        folder.mkdir(exist_ok=True)
        target = folder / "delivery.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)
        labels = {"pending": "待执行", "generated": "已生成", "failed": "失败",
                  "missing": "缺失", "not_enabled": "未启用", "not_applicable": "不适用"}
        names = {"selection_lock": "验证集选模锁定记录", "model": "模型与原始表格推断流水线", "reload_check": "重载预测一致性核对", "shap": "SHAP 解释图（三类）", "fairness": "公平性评估",
                 "audit": "可信审计结论（画像预检与划分审计）",
                 "model_scores": "候选模型分数", "profile_report": "数据概况报告",
                 "split_report": "数据划分报告", "preprocess_report": "预处理报告",
                 "search_report": "模型搜索报告", "explanation_report": "解释与公平性报告",
                 "full_report": "统一网页报告", "pdf_report": "PDF 报告",
                 "ensemble_diagnostics": "集成模型诊断图", "ensemble_report": "集成模型报告"}
        reasons = {"Producer completed; declared files present and nonempty": "生成步骤成功，声明的文件存在且非空",
                   "Expected output absent, empty, or outside this run": "预期文件缺失、为空或不属于本次运行",
                   "Declared output deleted, empty, or invalid after generation": "原已生成的文件已被删除、清空或路径失效",
                   "Sensitive attributes were not supplied": "本次未提供敏感属性",
                   "Current DP/EO producer requires binary 0/1 labels": "当前公平性指标实现仅适用于标签为 0/1 的二分类",
                   "Confusion matrix / ROC do not apply to regression": "回归任务不适用混淆矩阵和 ROC",
                   "PDF export not requested": "本次未请求导出 PDF"}
        rows = ""
        for key, r in self.records.items():
            prefix_names = {"diagnostics": "诊断图", "trials": "调参轨迹", "model_report": "模型报告"}
            prefix, _, suffix = key.partition("/")
            name = names.get(key, prefix_names.get(prefix, prefix) + ("：" + suffix if suffix else ""))
            history = "".join(f'<li>{escape(a["started_at"])}：{labels[a["status"]]} — '
                f'{escape(reasons.get(a["reason"], a["reason"]))}</li>' for a in r["attempts"])
            rows += "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in
                [name, "是" if r["required"] else "否", labels[r["status"]], reasons.get(r["reason"], r["reason"]),
                 ", ".join(r["paths"])]) + f'<td><details><summary>{len(r["attempts"])} 次（展开）</summary><ol>{history}</ol></details></td></tr>'
        (folder / "delivery.html").write_text(
            '<!doctype html><html lang="zh"><meta charset="utf-8"><title>证据交付验收</title>'
            '<style>body{font-family:sans-serif;margin:32px;color:#193c46}table{border-collapse:collapse}'
            'td,th{padding:10px;border:1px solid #ccd7db;text-align:left}th{background:#eaf2f4}</style>'
            f'<h1>证据交付：{"完整" if payload["complete"] else "不完整"}</h1>'
            f'<p>运行标识：{self.run_id}；必需项已交付 {payload["generated_required_count"]}/{payload["required_count"]}</p>'
            '<p>本页仅检查交付状态，不代表模型、解释或公平性结论正确，也不证明实验完全可复现。</p>'
            '<table><tr><th>证据项目</th><th>必需</th><th>状态</th><th>原因</th><th>文件</th><th>尝试次数</th></tr>'
            + rows + '</table></html>', encoding="utf-8")
        return payload
