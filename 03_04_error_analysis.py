"""
03_04_error_analysis.py — Per-image error analysis on own_dataset (state mode).

Instead of ranking models by their average F1, this flips the table and looks
at each photo: across all evaluated models, how many get it wrong?

  - An image missed by few models   → easy; individual model noise.
  - An image missed by most/all      → intrinsically hard; errors are
    correlated, so NO ensemble can fix it. These define the accuracy ceiling.

Reads the per-image predictions saved by 03_01_evaluate.py
(run_outputs/<run>/eval/own_dataset_state_preds.json) — no GPU needed.

Usage:
    python 03_04_error_analysis.py
    python 03_04_error_analysis.py --top 20 --no-plot
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RUN_DIR  = Path("run_outputs")
OUT_DIR  = RUN_DIR / "_error_analysis"
EVAL_TAG = "own_dataset_state"
CLASSES  = ["fresh", "rotten"]


def load_preds():
    """Return list of per-model prediction dicts {file: record}."""
    models = []
    for run_dir in sorted(RUN_DIR.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        p = run_dir / "eval" / f"{EVAL_TAG}_preds.json"
        e = run_dir / "eval" / f"{EVAL_TAG}.json"
        if not (p.exists() and e.exists()):
            continue
        with open(e) as f:
            if json.load(f).get("model_classes") != CLASSES:
                continue
        with open(p) as f:
            recs = json.load(f)
        models.append({r["file"]: r for r in recs})
    return models


def per_image_stats(models):
    """For each image, aggregate how many models got it wrong."""
    files = set(models[0].keys())
    for m in models[1:]:
        files &= set(m.keys())
    files = sorted(files)
    n_models = len(models)

    stats = []
    for f in files:
        wrong = [m[f] for m in models if not m[f]["correct"]]
        n_wrong = len(wrong)
        true = models[0][f]["true_label"]
        # average confidence the wrong models had in their (wrong) answer
        conf_wrong = float(np.mean([w["confidence"] for w in wrong])) if wrong else 0.0
        stats.append({
            "file"      : f,
            "true_label": true,
            "n_wrong"   : n_wrong,
            "n_models"  : n_models,
            "err_rate"  : round(n_wrong / n_models, 4),
            "conf_wrong": round(conf_wrong, 4),
        })
    stats.sort(key=lambda s: s["n_wrong"], reverse=True)
    return stats, n_models


def make_plot(stats, n_models, top, out_path):
    rates = [s["err_rate"] * 100 for s in stats]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                   gridspec_kw={"width_ratios": [1, 1.3]})

    # left — distribution of per-image error rate
    ax1.hist(rates, bins=np.arange(0, 101, 10), color="#4C72B0",
             edgecolor="white")
    ax1.set_xlabel("% of models that miss the image")
    ax1.set_ylabel("number of images")
    ax1.set_title(f"Per-image difficulty ({len(stats)} images, {n_models} models)")

    # right — the hardest images
    hard = stats[:top]
    labels = [f"{s['file']}  ({s['true_label'][:1].upper()})" for s in hard]
    vals   = [s["err_rate"] * 100 for s in hard]
    colors = ["#55A868" if s["true_label"] == "fresh" else "#C44E52"
              for s in hard]
    y = range(len(hard))
    ax2.barh(list(y), vals, color=colors)
    ax2.set_yticks(list(y))
    ax2.set_yticklabels(labels, fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel("% of models wrong")
    ax2.set_xlim(0, 100)
    ax2.set_title(f"Top {top} hardest images  (green=fresh, red=rotten)")
    for i, v in zip(y, vals):
        ax2.text(v + 1, i, f"{v:.0f}", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=15,
                    help="How many hardest images to list.")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    print("\n── Per-image error analysis — own_dataset (state) ──\n")
    models = load_preds()
    if len(models) < 2:
        print(f"Need at least 2 evaluated models, found {len(models)}.")
        return
    stats, n_models = per_image_stats(models)

    # difficulty buckets
    buckets = {"easy (0-10%)": 0, "mild (10-50%)": 0,
               "hard (50-90%)": 0, "unanimous (>90%)": 0}
    for s in stats:
        r = s["err_rate"]
        if r <= 0.10:   buckets["easy (0-10%)"]     += 1
        elif r <= 0.50: buckets["mild (10-50%)"]    += 1
        elif r <= 0.90: buckets["hard (50-90%)"]    += 1
        else:           buckets["unanimous (>90%)"] += 1

    n_unanimous = sum(1 for s in stats if s["n_wrong"] == n_models)
    ceiling = (1 - n_unanimous / len(stats)) * 100

    print(f"Images: {len(stats)} | Models: {n_models}\n")
    print("  Difficulty distribution:")
    for k, v in buckets.items():
        print(f"    {k:<18} {v:3d} images")
    print(f"\n  Images EVERY model fails : {n_unanimous}  "
          f"(no ensemble can fix these)")
    print(f"  Accuracy ceiling         : {ceiling:.1f}%  "
          f"(best possible if all fixable images were fixed)")

    print(f"\n  Top {args.top} hardest images:")
    print(f"  {'file':<18}{'true':>8}{'models_wrong':>14}{'conf_wrong':>12}")
    for s in stats[:args.top]:
        print(f"  {s['file']:<18}{s['true_label']:>8}"
              f"{s['n_wrong']:>8}/{s['n_models']:<5}{s['conf_wrong']*100:>10.1f}%")

    # save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "n_images"        : len(stats),
        "n_models"        : n_models,
        "buckets"         : buckets,
        "n_unanimous_fail": n_unanimous,
        "accuracy_ceiling": round(ceiling, 2),
        "per_image"       : stats,
    }
    jpath = OUT_DIR / "error_analysis.json"
    with open(jpath, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {jpath}")

    if not args.no_plot:
        ppath = OUT_DIR / "error_analysis.png"
        make_plot(stats, n_models, args.top, ppath)
        print(f"  Saved: {ppath}")


if __name__ == "__main__":
    main()
