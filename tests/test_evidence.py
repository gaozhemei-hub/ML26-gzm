"""Delivery semantics and failure recovery, including actual fitted models."""
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes

from tabularx import AutoEstimator
from tabularx.core.evidence import EvidenceDelivery, EvidenceOutcome
from tabularx.schema.evidence import EvidenceManifest
from tabularx.schema.learning import ModelRunResult, ModelSearchResult


def writer(delivery, name):
    def produce():
        path = delivery.root / name
        path.write_text("real output", encoding="utf-8")
        return EvidenceOutcome([str(path)])
    return produce


def test_manifest_json_round_trip_and_version_validation():
    payload = {
        "schema_version": "1.0", "run_id": "run-1", "task": "binary",
        "complete": True, "required_count": 1, "generated_required_count": 1,
        "completeness": 1.0, "boundary": "delivery only",
        "records": {"plot": {"required": True, "applicable": True,
            "condition": "always", "enabled": True, "status": "generated",
            "reason": "ok", "paths": ["plot.png"], "recoverable": True,
            "attempts": [{"started_at": "t0", "ended_at": "t1", "status": "generated",
                          "reason": "ok", "paths": ["plot.png"]}]}},
    }
    parsed = EvidenceManifest.from_dict(payload)
    assert parsed.records["plot"].attempts[0].status == "generated"
    assert parsed.to_dict() == payload
    payload["schema_version"] = "99"
    with pytest.raises(ValueError, match="schema_version"):
        EvidenceManifest.from_dict(payload)


def test_missing_from_start_not_hidden_by_present_files(tmp_path):
    d = EvidenceDelivery(tmp_path, "regression")
    d.register("report", writer(d, "report.html"))
    d.register("required_explanation", lambda: EvidenceOutcome())
    d.register("confusion_matrix", applicable=False, reason="Regression")
    d.register("fairness", required=False, enabled=False, reason="No sensitive attributes")
    d.execute("report")
    d.execute("required_explanation")
    audit = d.write()
    contract = d.audit_contract()
    assert not audit["complete"]
    assert contract.complete is False
    assert contract.records["required_explanation"].status == "missing"
    assert audit["completeness"] == .5
    assert audit["records"]["required_explanation"]["status"] == "missing"
    assert audit["records"]["fairness"]["status"] == "not_enabled"
    assert audit["records"]["confusion_matrix"]["status"] == "not_applicable"


def test_attempt_history_recovery_and_deleted_file(tmp_path):
    d = EvidenceDelivery(tmp_path, "binary")
    def transient():
        if not d.records["explanation"]["attempts"]:
            raise RuntimeError("Injected failure")
        return writer(d, "plot.png")()
    d.register("explanation", transient)
    d.register("report", writer(d, "report.html"), refresh=True)
    d.execute("explanation")
    d.execute("report")
    assert not d.audit()["complete"]
    assert d.retry()["complete"]
    assert [a["status"] for a in d.records["explanation"]["attempts"]] == ["failed", "generated"]
    (d.root / "plot.png").unlink()
    assert d.audit()["records"]["explanation"]["status"] == "missing"
    assert d.retry()["complete"]


def test_empty_foreign_and_stale_files_rejected(tmp_path):
    previous = EvidenceDelivery(tmp_path, "binary")
    stale = previous.root / "old.png"
    stale.write_text("old")
    d = EvidenceDelivery(tmp_path, "binary")
    d.register("foreign", lambda: EvidenceOutcome([str(stale)]))
    d.execute("foreign")
    assert d.records["foreign"]["status"] == "failed"
    empty = d.root / "empty.png"
    empty.touch()
    d.register("empty", lambda: EvidenceOutcome([str(empty)]))
    d.execute("empty")
    assert d.records["empty"]["status"] == "missing"


def test_disabled_required_item_is_not_complete(tmp_path):
    d = EvidenceDelivery(tmp_path, "binary")
    d.register("required", enabled=False, reason="Disabled")
    assert not d.audit()["complete"]


def test_empty_contract_is_not_a_success(tmp_path):
    d = EvidenceDelivery(tmp_path, "binary")
    assert not d.audit()["complete"]
    assert d.audit()["completeness"] is None


def test_duplicate_retry_keys_do_not_repeat_work(tmp_path):
    d = EvidenceDelivery(tmp_path, "binary")
    d.register("plot", lambda: EvidenceOutcome(error="temporary failure"))
    d.execute("plot")
    d._callbacks["plot"] = writer(d, "plot.png")
    assert d.retry(["plot", "plot"])["complete"]
    assert len(d.records["plot"]["attempts"]) == 2


def test_persistent_failure_never_becomes_complete_and_history_is_visible(tmp_path):
    d = EvidenceDelivery(tmp_path, "binary")
    d.register("plot", lambda: EvidenceOutcome(error="<still unavailable>"))
    d.execute("plot")
    assert not d.retry()["complete"]
    assert [a["status"] for a in d.records["plot"]["attempts"]] == ["failed", "failed"]
    page = (d.root / "evidence" / "delivery.html").read_text()
    assert "&lt;still unavailable&gt;" in page
    assert "<still unavailable>" not in page
    assert "2 次（展开）" in page


def test_new_session_cannot_recover_without_context(tmp_path):
    ae = AutoEstimator(output_dir=tmp_path)
    with pytest.raises(RuntimeError, match="No recoverable"):
        ae.retry_evidence(["shap"])


def test_invalid_retry_rejected_before_any_producer_runs(tmp_path):
    d = EvidenceDelivery(tmp_path, "binary")
    d.register("failed", lambda: EvidenceOutcome(error="offline"))
    d.register("good", writer(d, "good.txt"))
    d.execute("failed")
    d.execute("good")
    with pytest.raises(ValueError):
        d.retry(["failed", "good"])
    assert len(d.records["failed"]["attempts"]) == 1


def test_report_failure_isolated_and_audit_snapshots_stable(tmp_path):
    d = EvidenceDelivery(tmp_path, "binary")
    d.register("model", writer(d, "model.bin"))
    d.register("report", lambda: EvidenceOutcome(error="renderer unavailable"), refresh=True)
    d.execute("model")
    d.execute("report")
    before = d.audit()
    d._callbacks["report"] = writer(d, "report.html")
    assert d.retry(["report"])["complete"]
    assert before["records"]["report"]["status"] == "failed"
    assert len(d.records["model"]["attempts"]) == 1


def test_ranking_ignores_better_test_score():
    a = ModelRunResult("a", "binary", "success", {}, 1,
                       validation_metrics={"roc_auc": .9}, test_metrics={"roc_auc": .6})
    b = ModelRunResult("b", "binary", "success", {}, 1,
                       validation_metrics={"roc_auc": .7}, test_metrics={"roc_auc": .99})
    result = ModelSearchResult("binary", "roc_auc", "maximize", model_results=[a, b])
    assert result.ranking[0].model_name == "a"


def estimator(tmp_path, task):
    ae = AutoEstimator(task=task, output_dir=tmp_path)
    ae.param.model_search.models = ["random_forest"]
    ae.param.model_search.n_trials = 1
    ae.param.model_search.n_jobs = 1
    ae.param.model_search.fixed_params = {"random_forest": {"n_estimators": 8, "max_depth": 3}}
    ae.param.shap.max_explain_samples = 12
    return ae


@pytest.mark.parametrize("task,loader", [("classification", load_breast_cancer), ("regression", load_diabetes)])
def test_real_pipeline_failure_retry_never_refits(tmp_path, task, loader):
    data = loader(as_frame=True)
    ae = estimator(tmp_path, task)
    # Inject only the first SHAP attempt; all modeling and report generation real.
    with patch.object(ae, "_run_shap", side_effect=RuntimeError("Injected SHAP outage")):
        ae.fit(data.data, data.target)
    assert ae.evidence_.records["shap"]["status"] == "failed"
    assert ae.result.evidence.records["shap"].status == "failed"
    assert ae.evidence_.records["full_report"]["status"] == "generated"
    assert ae.evidence_.records["model"]["status"] == "generated"
    assert not ae.evidence_.audit()["complete"]
    with patch.object(type(ae._evidence_model), "fit", side_effect=AssertionError("Unexpected model refit")):
        audit = ae.retry_evidence(["shap"])
    assert audit["records"]["shap"]["status"] == "generated", audit["records"]["shap"]
    assert ae.result.evidence.records["shap"].status == "generated"
    assert audit["complete"], audit
    assert len(audit["records"]["shap"]["attempts"]) == 2
    assert len(audit["records"]["model"]["attempts"]) == 1
    if task == "regression":
        assert audit["records"]["diagnostics/random_forest"]["status"] == "not_applicable"
    # Changed context must not silently reuse old evidence.
    ae.param.shap.max_explain_samples += 1
    with pytest.raises(RuntimeError, match="changed"):
        ae.retry_evidence()
