"""Adapter for AutoEstimator; TrustGuard remains an independent producer."""
from dataclasses import asdict
import json
import re
import warnings

import joblib

from .delivery import EvidenceDelivery, EvidenceOutcome


class EvidenceEstimatorMixin:
    def _context_token(self):
        r = self.result_
        return joblib.hash((self._evidence_model, r.preprocessed, self.param.to_dict(),
                           self.task, self._sensitive_features, self.output_dir, self.pdf_report, self._raw_test, self._selection_lock,
                           r.searched.best_model_name, r.searched.best_model_params))

    def retry_evidence(self, keys=None):
        """Retry failed/missing producers in this live estimator, never search/fit.

        Changed data, fitted model or configuration requires a new run. A saved
        delivery.json is not sufficient for recovery after a process restart.
        """
        if not getattr(self, "evidence_", None) or not hasattr(self, "_evidence_context"):
            raise RuntimeError("No recoverable fitted run in this session")
        if self._context_token() != self._evidence_context:
            raise RuntimeError("Model/data/configuration changed; start a new fit instead")
        result = self.evidence_.retry(keys)
        if self.result_ is not None:
            self.result_.evidence = self.evidence_.audit_contract()
        return result

    def _save_evidence(self):
        from tabularx.core import reports
        from tabularx.core.learning import save_model_plots, save_trial_trajectory_plot, save_voting_plots
        from tabularx.core.learning.metrics import compute_all_metrics

        result = self.result_
        delivery = self.evidence_
        root = delivery.root
        reports_dir = root / "reports"
        assets = reports_dir / "assets"
        (reports_dir / "models").mkdir(parents=True, exist_ok=True)
        assets.mkdir(exist_ok=True)
        from copy import deepcopy
        # Freeze the validation decision before final fitting and held-out evaluation.
        self._selection_lock = dict(model_name=result.searched.best_model_name,
            parameters=deepcopy(result.searched.best_model_params),
            metric=result.searched.primary_metric,
            validation_metrics=deepcopy(result.searched.best_validation_metrics),
            basis="validation only; voting is a validation-only diagnostic")
        (root / "evidence").mkdir(exist_ok=True)
        (root / "evidence" / "selection_lock.json").write_text(
            json.dumps(self._selection_lock, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        self._evidence_model = None
        model_error = None
        try:
            fitted = self._get_best_model(result.searched, result.preprocessed)
            if fitted is None:
                raise RuntimeError("No successful model; downstream evidence blocked")
            self._evidence_model, self._evidence_spec = fitted
            # The selected model is fitted once here and reused by every selected-
            # model producer. Report its actual metrics, not another refit's scores.
            model = self._evidence_model
            p = result.preprocessed
            prediction = model.predict(p.X_test)
            probability = model.predict_proba(p.X_test) if hasattr(model, "predict_proba") else None
            metrics = compute_all_metrics(task_type=self.task, y_true=p.y_test,
                                          y_pred=prediction, y_prob=probability)
            result.searched.best_test_metrics = metrics
            for run in result.searched.succeeded_results:
                if run.model_name == result.searched.best_model_name:
                    run.test_metrics = metrics
        except Exception as exc:
            model_error = f"{type(exc).__name__}: {exc}"

        def require_model():
            if model_error:
                raise RuntimeError(model_error)

        def dump_json(path, data):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)), encoding="utf-8")
            return EvidenceOutcome([str(path)])

        def save_model():
            require_model()
            from sklearn.pipeline import Pipeline
            if result.preprocessed.fitted_preprocessor is None:
                raise RuntimeError("Fitted preprocessing state unavailable")
            path = root / "best_model.joblib"
            joblib.dump(self._evidence_model, path)
            pipeline_path = root / "pipeline.joblib"
            joblib.dump(Pipeline([("preprocess", result.preprocessed.fitted_preprocessor),
                                  ("model", self._evidence_model)]), pipeline_path)
            return EvidenceOutcome([str(path), str(pipeline_path)])

        def reload_check():
            require_model()
            import numpy as np
            if delivery.records["model"]["status"] != "generated":
                raise RuntimeError("Saved model/pipeline unavailable")
            pipeline = joblib.load(root / "pipeline.joblib")
            expected = self._evidence_model.predict(result.preprocessed.X_test)
            actual = pipeline.predict(self._raw_test)
            equal = (np.allclose(actual, expected, rtol=1e-10, atol=1e-12)
                     if self.task == "regression" else np.array_equal(actual, expected))
            probability_equal = None
            if hasattr(self._evidence_model, "predict_proba"):
                probability_equal = bool(np.allclose(pipeline.predict_proba(self._raw_test),
                    self._evidence_model.predict_proba(result.preprocessed.X_test), rtol=1e-10, atol=1e-12))
            output = dump_json(root / "evidence" / "reload_check.json", dict(
                passed=bool(equal and probability_equal is not False), samples=len(actual),
                prediction_equal=bool(equal), probability_equal=probability_equal,
                scope="Same environment; raw held-out table; no refitting; not cross-environment reproducibility"))
            if not equal or probability_equal is False:
                output.error = "Raw-table reloaded predictions differ from fitted model"
            return output

        def shap():
            require_model()
            result.shap = None
            result.shap = self._run_shap(result.searched, result.preprocessed)
            paths = [str(reports_dir / p) for p in result.shap.artifact_paths.values()]
            if result.shap.error:
                return EvidenceOutcome(paths, result.shap.error)
            expected = {"shap_summary", "shap_importance", "shap_waterfall"}
            missing = expected - result.shap.artifact_paths.keys()
            if missing:
                return EvidenceOutcome(paths, "SHAP plots not returned: " + ", ".join(sorted(missing)))
            return EvidenceOutcome(paths)

        def fairness():
            require_model()
            result.fairness = None
            result.fairness = self._run_fairness(result.searched, result.preprocessed,
                                                 self._sensitive_features)
            output = dump_json(root / "evidence" / "fairness.json", asdict(result.fairness))
            if result.fairness.error:
                output.error = result.fairness.error
            elif any(f.is_dropped for f in result.fairness.feature_results):
                output.error = "Some requested sensitive features could not be evaluated"
            return output

        sections = []

        def html(key, title, path, renderer, data, required=True):
            def produce():
                path.write_text(renderer(data()), encoding="utf-8")
                return EvidenceOutcome([str(path)])
            delivery.register(key, produce, required=required, refresh=True)
            sections.append((key, title, renderer, data))

        delivery.register("selection_lock", lambda: dump_json(root / "evidence" / "selection_lock.json", self._selection_lock))
        delivery.register("model", save_model, recoverable=not bool(model_error))
        delivery.register("reload_check", reload_check, refresh=True, recoverable=not bool(model_error))
        delivery.register("shap", shap, recoverable=not bool(model_error))
        # Existing DP/EO implementation is binary 0/1 only. Do not label other
        # tasks as fairly evaluated merely because a JSON/HTML file was written.
        supported = self.task == "binary" and set(result.preprocessed.y_test.unique()) <= {0, 1}
        delivery.register("fairness", fairness, required=bool(self._sensitive_features),
            applicable=supported, enabled=bool(self._sensitive_features),
            condition="Sensitive attributes supplied; binary task; labels encoded 0/1",
            reason=("Current DP/EO producer requires binary 0/1 labels" if not supported
                    else "Sensitive attributes were not supplied" if not self._sensitive_features else ""),
            recoverable=not bool(model_error))
        delivery.register("model_scores", lambda: dump_json(root / "evidence" / "model_scores.json",
            {"selection_metric": result.searched.primary_metric,
             "selected_model": result.searched.best_model_name,
             "models": [asdict(r) for r in result.searched.model_results]}))

        for run in result.searched.succeeded_results:
            selected = run.model_name == result.searched.best_model_name
            def diagnostics(run=run, selected=selected):
                if selected:
                    require_model()
                save_model_plots(run, assets, result.searched, result.preprocessed,
                                 self.task, fitted_model=self._evidence_model if selected else None)
                required_keys = ["confusion_matrix"]
                if self.task == "binary":
                    required_keys.append("roc_curve")
                return EvidenceOutcome([str(reports_dir / run.artifact_paths[k])
                    for k in required_keys if k in run.artifact_paths],
                    "Diagnostic output missing" if any(k not in run.artifact_paths for k in required_keys) else None)
            delivery.register("diagnostics/" + run.model_name, diagnostics,
                applicable=selected and self.task in ("binary", "multiclass"),
                condition="Classification: confusion matrix; binary: also ROC",
                reason=("Only the locked selected model receives held-out diagnostics" if not selected else
                        "Confusion matrix / ROC do not apply to regression" if self.task == "regression" else ""),
                recoverable=selected and not bool(model_error))
            def trials(run=run):
                save_trial_trajectory_plot(run, assets, result.searched.primary_metric)
                path = run.artifact_paths.get("trial_trajectory")
                return EvidenceOutcome([str(reports_dir / path)] if path else [])
            delivery.register("trials/" + run.model_name, trials,
                              applicable=bool(run.optuna_trials), reason="No trial records" if not run.optuna_trials else "")

        html("profile_report", "数据概况", reports_dir / "profile.html", reports.render_profile_report, lambda: result.profile)
        html("split_report", "数据划分", reports_dir / "split.html", reports.render_split_report, lambda: result.split)
        html("preprocess_report", "数据预处理", reports_dir / "preprocess.html", reports.render_preprocess_report, lambda: result.preprocessed)
        html("search_report", "模型搜索", reports_dir / "model_search.html", reports.render_model_search_report, self._build_model_search_data)
        if result.searched.ensemble_result is not None:
            er = result.searched.ensemble_result
            def voting_plots():
                save_voting_plots(er, assets, result.searched, result.preprocessed, self.task)
                return EvidenceOutcome([str(reports_dir / p) for p in er.artifact_paths.values()])
            delivery.register("ensemble_diagnostics", voting_plots, required=False, recoverable=False,
                applicable=False, reason="Voting is compared on validation only; no held-out diagnostics")
            def voting_data():
                return dict(run=er, primary_metric=result.searched.primary_metric,
                    optimization_direction=result.searched.optimization_direction,
                    best_single_validation=(result.searched.best_validation_metrics or {}).get(result.searched.primary_metric),
                    is_classification=self.task in ("binary", "multiclass"), artifact_dir="../assets",
                    has_confusion_matrix="confusion_matrix" in er.artifact_paths,
                    has_roc_curve="roc_curve" in er.artifact_paths,
                    has_voting_comparison="voting_comparison" in er.artifact_paths,
                    succeeded_results=result.searched.succeeded_results)
            html("ensemble_report", "组合模型", reports_dir / "models" / "voting.html",
                 reports.render_voting_report, voting_data, required=False)
        for run in result.searched.succeeded_results:
            def detail(run=run):
                return dict(run=run, target_name=str(result.preprocessed.y_test.name or "target"),
                    primary_metric=result.searched.primary_metric,
                    optimization_direction=result.searched.optimization_direction,
                    is_classification=self.task in ("binary", "multiclass"), artifact_dir="../assets",
                    has_confusion_matrix="confusion_matrix" in run.artifact_paths,
                    has_roc_curve="roc_curve" in run.artifact_paths,
                    has_trial_trajectory="trial_trajectory" in run.artifact_paths)
            html("model_report/" + run.model_name, "模型：" + run.model_name,
                 reports_dir / "models" / (run.model_name.replace(" ", "_") + ".html"),
                 reports.render_model_detail_report, detail)
        html("explanation_report", "解释与公平性", reports_dir / "explanation.html",
             reports.render_explanation_report, self._build_explanation_data)

        def full_report():
            from datetime import datetime
            items = []
            for key, title, renderer, data in sections:
                if delivery.records[key]["status"] != "generated":
                    from html import escape
                    body = "<p>报告未生成：" + escape(delivery.records[key]["reason"]) + "</p>"
                else:
                    body = renderer(data(), body_only=True)
                    body = body.replace('src="../assets/', 'src="reports/assets/').replace('src="assets/', 'src="reports/assets/')
                items.append(dict(id=key.replace("/", "_"), title=title, body=body))
            path = root / "report.html"
            content = reports.render_full_report(dict(
                generated_at=datetime.now().isoformat(), task_type=self.task,
                best_model_name=result.searched.best_model_name,
                dataset_info=dict(n_rows=result.profile.n_rows, n_cols=result.profile.n_cols), sections=items))
            banner = '<p style="padding:12px;background:#fff3cd">本报告生成成功不代表证据包完整。请查看 <a href="evidence/delivery.html">证据交付验收</a>。</p>'
            content = re.sub(r"(<body[^>]*>)", lambda m: m.group(1) + banner, content, count=1)
            path.write_text(content, encoding="utf-8")
            return EvidenceOutcome([str(path)])

        def pdf():
            if delivery.records["full_report"]["status"] != "generated":
                raise RuntimeError("HTML report unavailable")
            import weasyprint
            path = root / "report.pdf"
            weasyprint.HTML(filename=str(root / "report.html")).write_pdf(path)
            return EvidenceOutcome([str(path)])

        delivery.register("full_report", full_report, refresh=True)
        delivery.register("pdf_report", pdf, required=self.pdf_report, enabled=self.pdf_report,
                          reason="PDF export not requested" if not self.pdf_report else "", refresh=True)
        delivery.write()  # Pending obligations are persisted before producers run.
        for key in delivery.records:
            delivery.execute(key)
        self._evidence_context = self._context_token()
        result.report_paths = {"profile": str(reports_dir / "profile.html"),
            "split": str(reports_dir / "split.html"), "preprocess": str(reports_dir / "preprocess.html"),
            "model_search": str(reports_dir / "model_search.html"),
            "explanation": str(reports_dir / "explanation.html"), "full": str(root / "report.html"),
            "evidence": str(root / "evidence" / "delivery.html"),
            "evidence_json": str(root / "evidence" / "delivery.json")}
        summary = delivery.write()
        result.evidence = delivery.audit_contract()
        if not summary["complete"]:
            warnings.warn("Evidence delivery incomplete; inspect result.report_paths['evidence']", stacklevel=2)
