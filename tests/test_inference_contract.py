"""Raw-table reload and fairness boundary regression tests."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from tabularx import build_profile, split_data, preprocess
from tabularx.core.learning.fairness import evaluate_fairness


def test_mixed_raw_table_roundtrip(tmp_path):
    X = pd.DataFrame({'number': np.arange(120, dtype=float), 'category': ['a','b','c']*40,
                      'flag': [True,False]*60, 'date': pd.date_range('2020-01-01', periods=120), 'constant':1})
    X.loc[::9, 'number'] = np.nan
    X.loc[::11, 'category'] = np.nan
    y = pd.Series([0,1]*60)
    profile = build_profile(X,y,task_type='classification')
    split = split_data(X,y,profile=profile)
    result = preprocess(X,y,profile=profile,split_result=split)
    for name in ('train','valid','test'):
        raw = X.iloc[getattr(split,name+'_indices')]
        pd.testing.assert_frame_equal(result.fitted_preprocessor.transform(raw), getattr(result,'X_'+name), check_dtype=False)
    model = RandomForestClassifier(n_estimators=5,random_state=42).fit(result.X_train,result.y_train)
    path = tmp_path/'pipeline.joblib'
    joblib.dump(Pipeline([('preprocess',result.fitted_preprocessor),('model',model)]),path)
    loaded = joblib.load(path)
    raw = X.iloc[split.test_indices]
    np.testing.assert_array_equal(loaded.predict(raw[list(reversed(raw.columns))]),model.predict(result.X_test))
    np.testing.assert_allclose(loaded.predict_proba(raw),model.predict_proba(result.X_test))
    with pytest.raises(ValueError,match='Missing raw'):
        loaded.predict(raw.drop(columns='number'))
    with pytest.raises(RuntimeError,match='Frozen'):
        loaded.steps[0][1].fit(raw)
    unseen = raw.copy(); unseen['category'] = 'never-seen'
    assert np.isfinite(loaded.predict_proba(unseen)).all()


def evaluate(labels, predicted, groups, task='binary', probability=None):
    return evaluate_fairness(pd.DataFrame({'group':groups}), pd.Series(labels),
        np.array(predicted), probability, [], ['group'], task)


def test_undefined_rates_not_zero_and_missing_group_retained():
    result = evaluate([0,0,1,1],[0,1,1,0],['a','a',None,None])
    feature = result.feature_results[0]
    assert sum(g.n_samples for g in feature.groups) == 4
    assert feature.groups[0].tpr is None
    assert feature.groups[1].fpr is None
    assert feature.demographic_parity_diff == 0
    assert feature.equalized_odds_tpr_diff is None
    assert feature.eo_tpr_violation is None


def test_dp_uses_decisions_not_probability_mean():
    feature = evaluate([0,1,0,1],[1,1,0,0],['a','a','b','b'],
        probability=np.array([[.4,.6],[.4,.6],[.6,.4],[.6,.4]])).feature_results[0]
    assert feature.demographic_parity_diff == 1


@pytest.mark.parametrize('task,labels',[('regression',[1.,2.]),('multiclass',[0,1]),('binary',['no','yes'])])
def test_unsupported_tasks_explicit(task, labels):
    assert evaluate(labels,labels,['a','b'],task).error


def test_one_group_not_perfect_fairness():
    feature = evaluate([0,1],[0,1],['a','a']).feature_results[0]
    assert feature.demographic_parity_diff is None
    assert feature.dp_violation is None


def test_undefined_fairness_report_is_readable():
    from tabularx.core import reports
    result = evaluate([0,0,1,1],[0,1,1,0],['a','a','b','b'])
    html = reports.render_explanation_report(dict(task_type='binary', target_name='target',
        best_model_name='example', shap_available=False, shap_error=None,
        fairness_enabled=True, fairness_error=None, fairness_sensitive_features=['group'],
        fairness_feature_results=result.feature_results))
    assert '无法计算' in html and '无法判断' in html
    assert '公平性通过' not in html
