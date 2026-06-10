"""
prepare_figures.py — Copy analysis plots into figures/ for the LaTeX report.

Run this after 03_02_analyze.py (training curves) and 03_03_ensemble.py
(ensemble confusion matrices), 03_05_per_fruit.py have generated their plots.

Usage:
    python prepare_figures.py
"""

import shutil
from pathlib import Path

HERE   = Path(__file__).parent
ROOT   = HERE.parent.parent              # project root
PLOTS  = ROOT / "run_outputs" / "_plots"
ENS    = ROOT / "run_outputs" / "_ensemble"
PFRUIT = ROOT / "run_outputs" / "_per_fruit"
DST    = HERE / "figures"

# (source_dir, source_name, dest_name)
FIGURES = [
    (PLOTS,  "confusion_matrix.png",            "confusion_matrix.png"),
    (PLOTS,  "acc_curves.png",                  "acc_curves.png"),
    (PLOTS,  "f1_curves.png",                   "f1_curves.png"),
    (PLOTS,  "precision_curves.png",            "precision_curves.png"),
    (PLOTS,  "recall_curves.png",               "recall_curves.png"),
    (PLOTS,  "heatmap.png",                     "heatmap.png"),
    (PLOTS,  "per_class.png",                   "per_class.png"),
    (PLOTS,  "metrics_summary.png",             "metrics_summary.png"),
    (PLOTS,  "loss_curves.png",                 "loss_curves.png"),
    # Ensemble confusion matrices (bar chart + CM panel)
    (ENS,    "ensemble_KFR-FS_all_mean.png",    "ensemble_kfr.png"),
    (ENS,    "ensemble_MFR-FS_all_mean.png",    "ensemble_mfr.png"),
    # Per-fruit domain-shift breakdown on own photos
    (PFRUIT, "per_fruit.png",                   "per_fruit.png"),
]

DST.mkdir(exist_ok=True)

print(f"Output : {DST}\n")

for src_dir, src_name, dst_name in FIGURES:
    src = src_dir / src_name
    dst = DST / dst_name
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  OK    {dst_name:<24} <- {src_dir.name}/{src_name}")
    else:
        print(f"  SKIP  {dst_name:<24} (not found: {src})")

print("\nDone. Compile the report with:")
print("  pdflatex report_en.tex")
