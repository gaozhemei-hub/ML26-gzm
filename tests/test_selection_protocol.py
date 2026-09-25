"""Tests enforce no held-out access during candidate/ensemble selection."""
import json
from unittest.mock import patch
import pytest
from sklearn.datasets import load_iris, load_diabetes
from tabularx import AutoEstimator, build_profile, split_data, preprocess
from tabularx.core.learning import search_models, build_voting
from tabularx.core.params import ModelSearchConfig

class SelectionOnly:
    def __init__(self, data): self.data = data
    def __getattr__(self, name):
        if name in ('X_test','y_test') or name.lower().startswith('test_'):
            raise AssertionError('held-out access during selection')
        return getattr(self.data,name)

@pytest.mark.parametrize('jobs',[1,2])
def test_search_and_voting_never_read_heldout(jobs):
    d=load_diabetes(as_frame=True)
    p=build_profile(d.data,d.target,task_type='regression')
    split=split_data(d.data,d.target,profile=p)
    data=SelectionOnly(preprocess(d.data,d.target,profile=p,split_result=split))
    cfg=ModelSearchConfig(models=['random_forest','xgboost'],n_trials=1,n_jobs=jobs,
        fixed_params={m:{'n_estimators':3,'max_depth':2} for m in ['random_forest','xgboost']})
    search=search_models(data,p,cfg)
    assert search.n_models_succeeded==2
    assert search.best_test_metrics is None
    assert all(r.test_metrics is None for r in search.model_results)
    voting=build_voting(search,data,'regression')
    assert voting.status=='success' and voting.test_metrics is None
    assert voting.validation_metrics


def test_selection_data_view_rejects_test_attributes():
    from tabularx.estimator import _SelectionDataView

    class Data:
        X_train = object()
        X_valid = object()
        X_test = object()

    guarded = _SelectionDataView(Data())
    assert guarded.X_train is not None
    assert guarded.X_valid is not None
    with pytest.raises(AssertionError, match="X_test"):
        _ = guarded.X_test
    with pytest.raises(AssertionError, match="test_indices"):
        _ = guarded.test_indices
    assert guarded.blocked_accesses == ["X_test", "test_indices"]


def test_protocol_audit_flags_unlocked_candidate_test_metrics():
    from tabularx.core.evidence.estimator import EvidenceEstimatorMixin
    from tabularx.schema.estimator import AutoResult
    from tabularx.schema.learning import ModelRunResult, ModelSearchResult

    candidate = ModelRunResult(model_name="candidate", task_type="regression", status="success",
        best_params={}, n_trials=1, validation_metrics={"r2":0.4}, test_metrics={"r2":0.5})
    selected = ModelRunResult(model_name="selected", task_type="regression", status="success",
        best_params={}, n_trials=1, validation_metrics={"r2":0.8})
    searched = ModelSearchResult(task_type="regression", primary_metric="r2", optimization_direction="maximize",
        model_results=[candidate, selected], best_model_name="selected", best_model_params={},
        best_validation_metrics={"r2":0.8}, best_test_metrics={"r2":0.7})
    estimator = EvidenceEstimatorMixin()
    estimator.result_ = type("Result", (), {"searched":searched})()
    estimator._selection_lock={"model_name":"selected", "locked_before_test_evaluation":True}
    estimator._selection_guard_enabled=True
    estimator._final_test_evaluation_succeeded=True
    protocol=estimator._build_model_search_audit()
    assert protocol.protocol_status == "violated"
    assert protocol.candidate_test_metrics_count == 1
    assert protocol.risk_level == "critical"
    assert any(f.code == "unlocked_candidate_test_evaluation" for f in protocol.findings)


def test_protocol_audit_flags_voting_test_metrics():
    from tabularx.core.evidence.estimator import EvidenceEstimatorMixin
    from tabularx.schema.learning import ModelRunResult, ModelSearchResult

    selected = ModelRunResult(model_name="selected", task_type="regression", status="success",
        best_params={}, n_trials=1, validation_metrics={"r2":0.8})
    voting = ModelRunResult(model_name="voting", task_type="regression", status="success",
        best_params={}, n_trials=0, validation_metrics={"r2":0.82}, test_metrics={"r2":0.75})
    searched = ModelSearchResult(task_type="regression", primary_metric="r2", optimization_direction="maximize",
        model_results=[selected], best_model_name="selected", best_model_params={},
        best_validation_metrics={"r2":0.8}, best_test_metrics={"r2":0.7}, ensemble_result=voting)
    estimator = EvidenceEstimatorMixin()
    estimator.result_ = type("Result", (), {"searched":searched})()
    estimator._selection_lock={"model_name":"selected", "locked_before_test_evaluation":True}
    estimator._selection_guard_enabled=True
    estimator._final_test_evaluation_succeeded=True
    protocol=estimator._build_model_search_audit()
    assert protocol.protocol_status == "violated"
    assert protocol.voting_test_metrics_present is True
    assert protocol.risk_level == "high"
    assert any(f.code == "voting_test_evaluation" for f in protocol.findings)

@pytest.mark.parametrize('loader,task',[(load_iris,'classification'),(load_diabetes,'regression')])
def test_only_locked_model_gets_test_scores_and_plots(tmp_path,loader,task):
    d=loader(as_frame=True)
    ae=AutoEstimator(task=task,output_dir=tmp_path)
    ae.param.model_search.models=['random_forest','xgboost']
    ae.param.model_search.n_trials=1
    ae.param.model_search.n_jobs=1
    ae.param.model_search.fixed_params={m:{'n_estimators':3,'max_depth':2} for m in ae.param.model_search.models}
    ae.param.shap.max_explain_samples=6
    import tabularx.core.learning as learning
    original=learning.save_model_plots
    seen=[]
    def guarded(run,*args,**kwargs):
        assert (ae.output_dir/'evidence/selection_lock.json').exists()
        assert run.model_name==ae.result.searched.best_model_name
        assert kwargs['fitted_model'] is ae._evidence_model
        seen.append(run.model_name)
        return original(run,*args,**kwargs)
    original_fit=ae._get_best_model
    def locked_fit(search,processed):
        lock=json.loads((ae.output_dir/'evidence/selection_lock.json').read_text())
        assert lock['model_name']==search.best_model_name
        assert all(r.test_metrics is None for r in search.model_results)
        return original_fit(search,processed)
    with patch.object(ae,'_get_best_model',side_effect=locked_fit), \
         patch.object(learning,'save_model_plots',side_effect=guarded), \
         patch.object(learning,'save_voting_plots',side_effect=AssertionError('No ensemble test plots')):
        ae.fit(d.data,d.target)
    search=ae.result.searched
    assert ae.evidence_.audit()['complete']
    assert ae._selection_lock['locked_before_test_evaluation'] is True
    assert ae._selection_lock['test_evaluation_succeeded'] is True
    assert ae._selection_lock['final_test_scope'] == 'locked_model_only'
    protocol = json.loads((ae.output_dir/'evidence/model_search_audit.json').read_text())
    assert protocol['candidate_test_metrics_count'] == 0
    assert protocol['voting_test_metrics_present'] is False
    assert protocol['selection_guard_enabled'] is True
    assert protocol['protocol_status'] == 'passed'
    assert protocol['locked_model_test_evaluated'] is True
    assert [r.model_name for r in search.model_results if r.test_metrics]==[search.best_model_name]
    if search.ensemble_result is not None:
        assert search.ensemble_result.test_metrics is None

    assert search.best_test_metrics
    model_report=(ae.output_dir/'reports/model_search.html').read_text()
    assert '④ 评估协议审计' in model_report
    assert '候选模型测试指标数：</b> 0' in model_report
    if search.ensemble_result is not None:
        assert '测试集：未评估（Voting 仅用于验证集比较）' in model_report

    assert '审计链路' in (ae.output_dir/'report.html').read_text()
    assert len(seen)==(1 if task=='classification' else 0)
    if search.ensemble_result is not None:
        report=(ae.output_dir/'reports/models/voting.html').read_text()
        assert '仅为验证集比较' in report
