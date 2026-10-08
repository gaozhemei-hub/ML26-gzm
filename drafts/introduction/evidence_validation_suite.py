"""Expanded recovery experiments. Faults are deliberately injected, not natural outages.

Three bundled public datasets x two model families; no remote data downloads.
The suite retains unsuccessful cases and returns a nonzero exit code on failure.
"""
import argparse
from html import escape
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from sklearn.datasets import load_breast_cancer, load_iris, load_diabetes
from tabularx import AutoEstimator
from tabularx.core import reports


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_case(root, name, task, loader, model_name):
    data = loader(as_frame=True)
    ae = AutoEstimator(task=task, output_dir=root / name / model_name)
    ae.param.model_search.models = [model_name]
    ae.param.model_search.n_jobs = 1
    ae.param.model_search.n_trials = 1
    ae.param.model_search.fixed_params = {model_name: {"n_estimators": 8, "max_depth": 3}}
    ae.param.shap.max_explain_samples = 12
    start = perf_counter()
    with patch.object(ae, "_run_shap", side_effect=RuntimeError("实验注入：解释步骤暂时不可用")), \
         patch.object(reports, "render_full_report", side_effect=RuntimeError("实验注入：报告渲染失败")):
        ae.fit(data.data, data.target)
    fit_seconds = perf_counter() - start
    snapshots = ae.output_dir / "validation_snapshots"
    snapshots.mkdir()
    before = ae.evidence_.audit()
    dump(snapshots / "01_initial_failures.json", before)
    checks = []

    def check(label, passed, detail):
        checks.append(dict(check=label, passed=bool(passed), detail=detail))

    check("局部失败被识别，模型仍交付", not before["complete"]
          and before["records"]["model"]["status"] == "generated"
          and before["records"]["shap"]["status"] == "failed"
          and before["records"]["full_report"]["status"] == "failed",
          "首次同时注入解释失败和统一报告渲染失败")
    reload_record = json.loads((ae.output_dir / "evidence" / "reload_check.json").read_text())
    check("原始表格重载预测一致", reload_record["passed"],
          "保存预处理器与模型，重载后直接接收原始测试表格；同时核对预测概率（如适用）")
    model_path = ae.output_dir / "best_model.joblib"
    original_model_bytes = model_path.read_bytes()
    original_prediction = ae._evidence_model.predict(ae.result.preprocessed.X_test)
    started = perf_counter()
    with patch.object(type(ae._evidence_model), "fit", side_effect=AssertionError("恢复期间禁止重新训练")) as fit_spy:
        with patch.object(ae, "_run_shap", side_effect=RuntimeError("实验注入：解释步骤仍不可用")):
            persistent = ae.retry_evidence(["shap"])
        dump(snapshots / "02_still_failed.json", persistent)
        check("持续失败不会误报完整", not persistent["complete"]
              and persistent["records"]["shap"]["status"] == "failed"
              and persistent["records"]["full_report"]["status"] == "generated",
              "报告恢复成功不覆盖仍存在的解释失败")

        restored = ae.retry_evidence(["shap"])
        dump(snapshots / "03_recovered.json", restored)
        check("真实解释及报告恢复", restored["complete"]
              and restored["records"]["shap"]["status"] == "generated",
              "不再注入故障，实际执行 SHAP 和报告渲染")

        # Only files created by this experiment are deleted.
        image_paths = restored["records"]["shap"]["paths"]
        if image_paths:
            (ae.output_dir / image_paths[0]).unlink()
            missing = ae.evidence_.audit()
            dump(snapshots / "04_image_missing.json", missing)
            repaired = ae.retry_evidence(["shap"])
            check("解释文件删除后检出并补齐", missing["records"]["shap"]["status"] == "missing"
                  and repaired["records"]["shap"]["status"] == "generated" and repaired["complete"],
                  "删除本轮真实生成的一张解释图，再局部补做")
        else:
            check("解释文件删除后检出并补齐", False, "没有可用于删除实验的解释图")

        model_path.unlink()
        model_missing = ae.evidence_.audit()
        model_restored = ae.retry_evidence(["model"])
        check("模型文件可由内存对象重新保存", model_missing["records"]["model"]["status"] == "missing"
              and model_restored["records"]["model"]["status"] == "generated"
              and model_path.read_bytes() == original_model_bytes,
              "模型没有重新训练；该恢复仍需要存活的 Python 会话")
        recovery_fit_calls = fit_spy.call_count

    check("恢复不调用模型训练", recovery_fit_calls == 0, f"恢复阶段 fit 调用次数：{recovery_fit_calls}")
    check("恢复前后预测一致", (ae._evidence_model.predict(ae.result.preprocessed.X_test) == original_prediction).all(),
          "使用同一测试输入比较；不等于跨环境复现")
    previous_attempts = sum(len(r["attempts"]) for r in ae.evidence_.records.values())
    previous = ae.result.preprocessed.X_test.iloc[0, 0]
    ae.result.preprocessed.X_test.iloc[0, 0] = previous + 1
    rejected = False
    try:
        ae.retry_evidence()
    except RuntimeError as exc:
        rejected = "changed" in str(exc)
    finally:
        ae.result.preprocessed.X_test.iloc[0, 0] = previous
    check("输入变化后拒绝沿用旧运行", rejected and previous_attempts == sum(
        len(r["attempts"]) for r in ae.evidence_.records.values()), "在执行任何补做之前拒绝变化的上下文")
    audit = ae.evidence_.write()
    dump(snapshots / "05_final.json", audit)
    return dict(dataset=name, model=model_name, task=ae.task, samples=len(data.data),
                features=data.data.shape[1], run_dir=str(ae.output_dir),
                fit_seconds=fit_seconds, recovery_checks_seconds=perf_counter()-started,
                recovery_fit_calls=recovery_fit_calls, passed=all(c["passed"] for c in checks), checks=checks)


def render_summary(root, payload):
    names = {"breast_cancer": "乳腺癌", "iris": "鸢尾花", "diabetes": "糖尿病"}
    tasks = {"binary": "二分类", "multiclass": "多分类", "regression": "回归"}
    rows = ""
    detail = ""
    for row in payload["results"]:
        passed = sum(c["passed"] for c in row.get("checks", []))
        run = row.get("run_dir")
        link = '<a href="' + escape((Path(run) / "evidence" / "delivery.html").relative_to(root).as_posix(), quote=True) + '">查看验收页</a>' if run else "未生成"
        rows += "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in [names[row["dataset"]],
            row["model"], tasks.get(row.get("task"), row.get("task", "—")),
            "通过" if row["passed"] else "未通过", f'{passed}/{len(row.get("checks", []))}',
            row.get("recovery_fit_calls", "—")]) + f"<td>{link}</td></tr>"
        detail += f'<h3>{names[row["dataset"]]} / {escape(row["model"])}</h3><ul>'
        for c in row.get("checks", []):
            detail += f'<li>{"✓" if c["passed"] else "✗"} {escape(c["check"])}：{escape(c["detail"])}</li>'
        if "error" in row:
            detail += f'<li>未完成：{escape(row["error"])}</li>'
        detail += "</ul>"
    html = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>证据交付与恢复实验</title>
    <style>body{font-family:system-ui,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#183b45;line-height:1.7}
    table{border-collapse:collapse;width:100%}td,th{padding:12px;text-align:left;border-bottom:1px solid #dce5e7}th{background:#183b45;color:white}
    .card{background:#eef5f6;padding:20px;margin:20px 0}h1{font-size:30px}a{color:#087c80}li{margin:6px}</style>
    <h1>证据交付与局部恢复：扩展验证</h1>
    <p>本页展示你新增的“交付检查”和“失败后局部补做”。未合并队友的报告。</p>'''
    html += f'<div class="card"><b>本轮结果：{payload["passed_cases"]}/{len(payload["results"])} 个组合通过全部检查。</b>'
    html += '<p>覆盖三个公开数据集、两种模型，以及二分类、多分类和回归。每个组合仅进行一次小规模训练，再注入明确故障。</p></div>'
    html += '<table><tr><th>数据集</th><th>模型</th><th>任务</th><th>结果</th><th>检查项</th><th>恢复时训练次数</th><th>真实产物</th></tr>' + rows + '</table>'
    html += '<div class="card"><b>如何理解：</b>训练、解释图和普通报告是原有能力；“发现应该交付但没有交付的内容，区分失败与不适用，并只补做失败部分”是本次新增能力。'
    html += '<p>这是人为故障注入实验，不是自然故障率统计；没有测量相对完整重跑的加速比。不证明预测性能、公平性或跨环境复现。</p></div>'
    html += '<h2>逐项实验记录</h2>' + detail + '<p>边界：同一 Python 会话；已核对同环境原始表格重载预测；仍不代表跨环境复现。<a href="summary.json">原始汇总记录</a></p></html>'
    (root / "summary.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for name, task, loader in [("breast_cancer", "classification", load_breast_cancer),
                                ("iris", "classification", load_iris),
                                ("diabetes", "regression", load_diabetes)]:
        for model in ["random_forest", "xgboost"]:
            try:
                row = run_case(root, name, task, loader, model)
            except Exception as exc:
                row = dict(dataset=name, model=model, passed=False, checks=[], error=f"{type(exc).__name__}: {exc}")
            results.append(row)
            payload = dict(scope="3 public datasets × 2 models; 1 HPO trial, 8 trees, SHAP 12 samples; injected failures; no speedup claims",
                           passed_cases=sum(r["passed"] for r in results), results=results)
            dump(root / "summary.json", payload)
            render_summary(root, payload)
            print(name, model, row["passed"], row.get("error", ""), flush=True)
    print(root / "summary.html")
    raise SystemExit(0 if all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    main()
