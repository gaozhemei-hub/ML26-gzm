"""使用 sklearn 默认数据集生成 profile / split / preprocess 报告，保存到 .inbox/reports/。

运行方式：
    cd 大作业/ && PYTHONPATH=TabularX/src python tests/gen_profile_reports.py
"""

from pathlib import Path

from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris

# 确保 TabularX 在导入路径上
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "TabularX" / "src"))

from tabularx import build_profile, preprocess, split_data
from tabularx.core.reports import render_preprocess_report, render_profile_report, render_split_report


# 目标目录
OUTPUT_DIR = Path(__file__).resolve().parent.parent / ".inbox" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 数据集清单
DATASETS = {
    "breast_cancer": load_breast_cancer,
    "iris": load_iris,
    "diabetes": load_diabetes,
}

for name, loader in DATASETS.items():
    print(f"正在处理 {name} ...", end=" ")

    # 加载数据
    data = loader(as_frame=True)
    X, y = data.data, data.target

    # 构建 profile
    profile = build_profile(X, y)

    # ── profile 报告 ────────────────────────────────────────
    html = render_profile_report(profile)
    out_path = OUTPUT_DIR / f"{name}_profile_report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"✓ profile → {out_path.relative_to(OUTPUT_DIR.parent.parent)}")

    # ── split 报告 ──────────────────────────────────────────
    split_result = split_data(X, y, profile)
    html_split = render_split_report(split_result)
    split_path = OUTPUT_DIR / f"{name}_split_report.html"
    split_path.write_text(html_split, encoding="utf-8")
    print(f"✓ split   → {split_path.relative_to(OUTPUT_DIR.parent.parent)}")

    # ── preprocess 报告 ─────────────────────────────────────
    pp_result = preprocess(X, y, profile, split_result)
    html_pp = render_preprocess_report(pp_result)
    pp_path = OUTPUT_DIR / f"{name}_preprocess_report.html"
    pp_path.write_text(html_pp, encoding="utf-8")
    print(f"✓ preproc → {pp_path.relative_to(OUTPUT_DIR.parent.parent)}")

print(f"\n全部完成，报告保存在 {OUTPUT_DIR}")