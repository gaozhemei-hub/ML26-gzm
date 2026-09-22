"""Real public-data smoke experiment with one explicit injected SHAP failure.

Run from repository root with PYTHONPATH=TabularX-project/src.
This is a recovery demonstration, not a model-performance benchmark.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from sklearn.datasets import load_breast_cancer, load_diabetes
from tabularx import AutoEstimator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, task, loader in [("breast_cancer", "classification", load_breast_cancer),
                                ("diabetes", "regression", load_diabetes)]:
        data = loader(as_frame=True)
        ae = AutoEstimator(task=task, output_dir=root / name)
        ae.param.model_search.models = ["random_forest"]
        ae.param.model_search.n_trials = 1
        ae.param.model_search.n_jobs = 1
        ae.param.model_search.fixed_params = {"random_forest": {"n_estimators": 8, "max_depth": 3}}
        ae.param.shap.max_explain_samples = 12
        started = perf_counter()
        with patch.object(ae, "_run_shap", side_effect=RuntimeError("Experiment: injected SHAP failure")):
            ae.fit(data.data, data.target)
        fit_seconds = perf_counter() - started
        before = ae.evidence_.audit()
        # Preserve the pre-recovery snapshot beside the evolving delivery report.
        (ae.output_dir / "before_recovery.json").write_text(json.dumps(before, ensure_ascii=False, indent=2))
        model_bytes = (ae.output_dir / "best_model.joblib").read_bytes()
        started = perf_counter()
        with patch.object(type(ae._evidence_model), "fit", side_effect=AssertionError("Recovery must not refit")) as fit_spy:
            after = ae.retry_evidence(["shap"])
            recovery_fit_calls = fit_spy.call_count
        recovery_seconds = perf_counter() - started
        rows.append(dict(dataset=name, task=task, samples=len(data.data),
            features=data.data.shape[1], run_dir=str(ae.output_dir),
            initial_fit_seconds=fit_seconds, recovery_seconds=recovery_seconds,
            recovery_model_fit_calls=recovery_fit_calls,
            saved_model_unchanged=model_bytes == (ae.output_dir / "best_model.joblib").read_bytes(),
            before_complete=before["complete"], after_complete=after["complete"],
            shap_attempts=len(after["records"]["shap"]["attempts"])))
    payload = dict(scope="2 public datasets, RF, 1 trial; one deliberately injected SHAP failure per run; not a speedup benchmark", results=rows)
    path = root / "recovery_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
