"""
prepare_figures.py — Copy analysis plots into docs/figures/ for the LaTeX report.

Run this after analyze.py has generated the plots in run_outputs/_plots/.

Usage:
    python docs/prepare_figures.py
"""

import shutil
from pathlib import Path

HERE     = Path(__file__).parent
SRC      = HERE.parent / "run_outputs" / "_plots"
DST      = HERE / "figures"

FIGURES = [
    "confusion_matrix.png",
    "acc_curves.png",
    "f1_curves.png",
    "precision_curves.png",
    "recall_curves.png",
    "heatmap.png",
    "per_class.png",
    "metrics_summary.png",
    "loss_curves.png",
]

DST.mkdir(exist_ok=True)

print(f"Source : {SRC}")
print(f"Output : {DST}\n")

for fig in FIGURES:
    src = SRC / fig
    dst = DST / fig
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  OK    {fig}")
    else:
        print(f"  SKIP  {fig}  (not found — run analyze.py first)")

print("\nDone. Compile the report with:")
print("  cd docs && pdflatex report_en.tex")
