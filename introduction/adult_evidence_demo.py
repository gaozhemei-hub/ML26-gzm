"""Full Adult cached OpenML 1590: delivery/recovery validation, not a performance benchmark."""
import argparse,gzip,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.io import arff
from tabularx import AutoEstimator


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    with gzip.open(args.data,'rt') as f: data,_=arff.loadarff(f)
    frame=pd.DataFrame(data)
    for col in frame.select_dtypes(include='object'):
        frame[col]=frame[col].map(lambda v:v.decode().strip() if isinstance(v,bytes) else v).replace('?',np.nan)
    y=frame.pop('class').map({'<=50K':0,'>50K':1})
    assert frame.shape==(48842,14) and not y.isna().any()
    ae=AutoEstimator(task='classification',output_dir=root/'runs_adult')
    ae.param.model_search.models=['random_forest','xgboost'];ae.param.model_search.n_jobs=1
    ae.param.model_search.n_trials=1
    ae.param.model_search.fixed_params={m:{'n_estimators':32,'max_depth':5,'random_state':42} for m in ae.param.model_search.models}
    ae.param.shap.max_explain_samples=24
    with patch.object(ae,'_run_shap',side_effect=RuntimeError('演示注入：解释生成暂时失败')):
        ae.fit(frame,y,sensitive_features=['sex'])
    before=ae.evidence_.audit()
    (root/'before_recovery.json').write_text(json.dumps(before,ensure_ascii=False,indent=2))
    with patch.object(type(ae._evidence_model),'fit',side_effect=AssertionError('Recovery must not train')) as fit_spy:
        after=ae.retry_evidence(['shap'])
    search=ae.result.searched
    summary=dict(dataset='Adult Income',source='Local cache of OpenML 1590',rows=len(frame),features=14,
        sensitive_feature='sex',sensitive_used_as_predictor=True,
        comparability='New repository implementation and settings; not directly comparable to earlier v0.1 numbers, which excluded sex.',
        models=ae.param.model_search.models,trials_per_model=1,trees=32,max_depth=5,shap_samples=24,
        selected=search.best_model_name,test_metrics=search.best_test_metrics,
        initial_failure_detected=not before['complete'] and before['records']['shap']['status']=='failed',
        recovered=after['complete'],recovery_fit_calls=fit_spy.call_count,
        reload_check=json.loads((ae.output_dir/'evidence/reload_check.json').read_text()),
        fairness=json.loads((ae.output_dir/'evidence/fairness.json').read_text()),
        delivery=str((ae.output_dir/'evidence/delivery.html').relative_to(root)),
        report=str((ae.output_dir/'report.html').relative_to(root)))
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (root/'summary.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>Adult：本周队友同步</title><style>body{{font:18px/1.8 system-ui;max-width:1000px;margin:40px auto;color:#173e48;background:#f5f8fa}}section{{background:white;padding:24px;margin:20px 0;border-radius:12px}}a{{color:#087f8c}}</style>
<h1>Adult：证据交付与局部恢复验证</h1><section><b>完整 48,842 行 · 14 个特征 · 随机森林与 XGBoost</b><p>每个模型一次小规模调参，32 棵树、最大深度 5；解释 24 个测试样本。验证工程流程，不用于性能排名。</p></section>
<section><h2>实际结果</h2><p>解释失败被识别：{summary['initial_failure_detected']}；真实补做后交付完整：{after['complete']}；恢复时训练次数：{fit_spy.call_count}；原始表格重载预测一致：{summary['reload_check']['passed']}。</p><p><a href="{summary['delivery']}">打开交付清单与补做历史</a> · <a href="{summary['report']}">打开完整报告</a> · <a href="summary.json">实验设置和原始结果</a></p></section>
<section><h2>和之前实验的区别</h2><p>本次为队友仓库上的新实现，性别同时用于分组审计和模型输入；早期 v0.1 实验排除了性别输入。因此不将两次性能或公平性数字直接比较。这里验证交付和恢复，并不证明模型公平。</p><p>故障为主动注入；恢复限于同一程序会话。同环境重载一致不等于跨环境完全复现。</p></section>''',encoding='utf-8')
    print(root/'summary.html')
    assert summary['initial_failure_detected'] and after['complete'] and fit_spy.call_count==0 and summary['reload_check']['passed']
if __name__=='__main__':main()
