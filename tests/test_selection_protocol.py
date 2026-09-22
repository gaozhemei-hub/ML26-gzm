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
        if name in ('X_test','y_test'):
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
    assert [r.model_name for r in search.model_results if r.test_metrics]==[search.best_model_name]
    assert search.ensemble_result.test_metrics is None
    assert search.best_test_metrics
    assert len(seen)==(1 if task=='classification' else 0)
    report=(ae.output_dir/'reports/models/voting.html').read_text()
    assert '仅为验证集比较' in report
