"""Small real experiments documenting validation-only selection and final evaluation."""
import argparse
import json
from pathlib import Path
from html import escape
from sklearn.datasets import load_iris, load_diabetes
from tabularx import AutoEstimator


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    root=Path(parser.parse_args().output).resolve();root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name,loader,task in [('Iris',load_iris,'classification'),('Diabetes',load_diabetes,'regression')]:
        data=loader(as_frame=True)
        ae=AutoEstimator(task=task,output_dir=root/name)
        ae.param.model_search.models=['random_forest','xgboost']
        ae.param.model_search.n_trials=1;ae.param.model_search.n_jobs=1
        ae.param.model_search.fixed_params={m:{'n_estimators':8,'max_depth':3} for m in ae.param.model_search.models}
        ae.param.shap.max_explain_samples=12
        ae.fit(data.data,data.target)
        search=ae.result.searched
        tested=[r.model_name for r in search.model_results if r.test_metrics is not None]
        complete=ae.evidence_.audit()['complete']
        row=dict(dataset=name,task=task,selected=search.best_model_name,tested_models=tested,
            voting_test_metrics=search.ensemble_result.test_metrics,
            passed=complete and tested==[search.best_model_name] and search.ensemble_result.test_metrics is None,
            selection_lock=str((ae.output_dir/'evidence/selection_lock.json').relative_to(root)),
            delivery=str((ae.output_dir/'evidence/delivery.html').relative_to(root)),
            full_report=str((ae.output_dir/'report.html').relative_to(root)))
        rows.append(row)
    (root/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    table=''.join('<tr><td>'+escape(r['dataset'])+'</td><td>'+escape(r['selected'])+'</td><td>'+str(len(r['tested_models']))+'</td><td>未测试（仅验证集比较）</td><td>'+('通过' if r['passed'] else '未通过')+f'</td><td><a href="{r["selection_lock"]}">选模记录</a> · <a href="{r["delivery"]}">交付检查</a> · <a href="{r["full_report"]}">完整报告</a></td></tr>' for r in rows)
    (root/'summary.html').write_text('''<!doctype html><meta charset="utf-8"><title>先选模，再测试｜流程修复验证</title>
<style>body{font:17px/1.75 system-ui;max-width:1150px;margin:40px auto;color:#183f4a;background:#f5f8fa}h1{font-size:30px}section{background:white;padding:24px;margin:20px 0;border-radius:12px}table{width:100%;border-collapse:collapse}td,th{padding:12px;text-align:left;border-bottom:1px solid #dbe5e9}th{background:#123e49;color:white}a{color:#087f8c}</style>
<h1>先选模，再测试：测试集使用流程修复</h1><section><b>训练集调参 → 验证集选模 → 保存锁定记录 → 最终拟合 → 锁定模型测试与解释</b>
<p>候选搜索与 Voting 比较不再接收测试数据。未被选择的模型不生成测试分数或测试诊断图；报告不会将“未测试”填成零。</p></section>
<section><h2>两组真实实验</h2><p>每组比较随机森林与 XGBoost；各运行一次小规模调参，8 棵树，最大深度 3。这些实验检查流程正确性，不比较工具性能。</p>
<table><tr><th>数据集</th><th>锁定模型</th><th>获得测试分数的模型数</th><th>Voting</th><th>结果</th><th>真实记录</th></tr>'''+table+'''</table></section>
<section><h2>与两个创新点的关系</h2><p>选模锁定记录现在进入证据交付清单；报告或解释失败后仍可局部补做，不重新选模或训练。</p>
<h2>研究边界</h2><p>这是主训练、选模和报告路径的修复，不是完整的数据泄露审计。数据概况与类型识别仍会读取全量输入；解释、诊断和重载核对仍会在锁定之后使用测试输入，不声称物理上只调用一次预测。不合并队友的 TrustGuard，也未推送远程。</p></section>''',encoding='utf-8')
    print(root/'summary.html')
    if not all(r['passed'] for r in rows):raise SystemExit(1)
if __name__=='__main__':main()
