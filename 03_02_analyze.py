"""
analyze.py — Visualize training results for the Freshness Classifier.

Plots generated:
  1. acc_curves.png       — Accuracy over epochs per dataset
  2. f1_curves.png        — F1 macro over epochs per dataset
  3. precision_curves.png — Precision macro over epochs per dataset
  4. recall_curves.png    — Recall macro over epochs per dataset
  5. loss_curves.png      — Train loss + Val loss over epochs
  6. metrics_summary.png  — Bar chart: acc / F1 / precision / recall per condition
  7. heatmap.png          — Condition × Backbone matrix (acc and F1)
  8. confusion_matrix.png — Confusion matrix for each run (final epoch)
  9. per_class.png        — Per-class F1 / precision / recall bars

Output: run_outputs/_plots/

Usage:
    python analyze.py
    python analyze.py --dataset mendeley_fruitvision
"""

import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────

CONDITIONS = ["frozen", "layer4", "head_frozen", "head_layer4"]
COND_COLOR = {
    "frozen"     : "#4C72B0",
    "layer4"     : "#DD8452",
    "head_frozen": "#55A868",
    "head_layer4": "#C44E52",
}
COND_LABEL = {
    "frozen"     : "C1",
    "layer4"     : "C2",
    "head_frozen": "C3",
    "head_layer4": "C4",
}
BACKBONE_LS = {
    "resnet18"          : "-",
    "resnet34"          : "--",
    "resnet50"          : "-.",
    "mobilenet_v3_small": ":",
    "efficientnet_b0"   : (0, (5, 1)),
    "efficientnet_b2"   : (0, (3, 1, 1, 1)),
}
BB_CODE = {
    "resnet18"          : "R18",
    "resnet34"          : "R34",
    "resnet50"          : "R50",
    "mobilenet_v3_small": "MN3",
    "efficientnet_b0"   : "EB0",
    "efficientnet_b2"   : "EB2",
}
DS_CODE = {
    "kaggle_fruits_quality"     : "KFQ",
    "mendeley_fruits"           : "MFR",
    "mendeley_lemon_varieties"  : "MLM",
    "mendeley_fruitvision"      : "MFV",
    "kaggle_fruits_fresh_rotten": "KFR",
    "kaggle_fresh_stale"        : "KFS",
    "own_dataset"               : "OWN",
}


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def scan_runs(root="run_outputs", dataset_filter=None):
    raw = []
    for folder in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(folder) or os.path.basename(folder) == "_plots":
            continue
        mp = os.path.join(folder, "metrics.json")
        if not os.path.exists(mp):
            continue
        with open(mp) as f:
            m = json.load(f)
        m["_folder"] = folder
        raw.append(m)

    if dataset_filter:
        raw = [r for r in raw if r.get("dataset") == dataset_filter]

    # Deduplicate: keep run with most epochs per (dataset, fruits, condition, backbone)
    best = {}
    for r in raw:
        key = (r.get("dataset"), _fruits_tag(r), r.get("condition"), r.get("backbone_name"))
        if key not in best or _max_epoch(r) > _max_epoch(best[key]):
            best[key] = r

    return list(best.values())


def _max_epoch(r):
    hist = r.get("acc_history", [])
    return max((h["epoch"] for h in hist), default=0)


def _fruits_tag(r):
    folder = os.path.basename(r.get("_folder", ""))
    parts  = folder.split("_")
    ds_len = len(r.get("dataset", "").split("_"))
    remaining = parts[ds_len:]
    tag_parts = []
    stops = {"frozen", "layer4", "head"}
    for p in remaining:
        if p in stops or p.startswith("head"):
            break
        tag_parts.append(p)
    return "_".join(tag_parts) or "all"


def _run_label(r):
    bb   = BB_CODE.get(r.get("backbone_name", ""), r.get("backbone_name", "")[:4])
    cond = COND_LABEL.get(r.get("condition", ""), r.get("condition", ""))
    ds   = DS_CODE.get(r.get("dataset", ""), r.get("dataset", "")[:4].upper())
    return f"{ds}-{bb}-{cond}"


# ─────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────

def print_summary(runs):
    print(f"\n{'─'*105}")
    print(f"  {'Dataset':<25} {'Fruits':<12} {'Condition':<14} {'Backbone':<22} "
          f"{'Acc':>7}  {'F1':>7}  {'Prec':>7}  {'Recall':>7}  {'Ep':>4}")
    print(f"{'─'*105}")
    for r in sorted(runs, key=lambda x: x.get("best_f1", x.get("best_acc", 0)), reverse=True):
        acc  = f"{r.get('best_acc',0)*100:.1f}%"
        f1   = f"{r.get('best_f1',0)*100:.1f}%"
        prec = f"{r.get('best_precision',0)*100:.1f}%"
        rec  = f"{r.get('best_recall',0)*100:.1f}%"
        ep   = str(_max_epoch(r))
        print(f"  {r.get('dataset',''):<25} {_fruits_tag(r):<12} {r.get('condition',''):<14} "
              f"{r.get('backbone_name',''):<22} {acc:>7}  {f1:>7}  {prec:>7}  {rec:>7}  {ep:>4}")
    print(f"{'─'*105}")
    print(f"  Total runs: {len(runs)}\n")


# ─────────────────────────────────────────────────────────────
# Plot 1a — Accuracy curves
# ─────────────────────────────────────────────────────────────

def _plot_metric_curves(runs, out_dir, metric_key, metric_label, filename, marker):
    """Generic curve plotter — 2×3 grid of datasets, one metric."""
    datasets = sorted({r["dataset"] for r in runs if r.get("acc_history")})
    if not datasets:
        return

    n_cols = 3
    n_rows = (len(datasets) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 4.5 * n_rows), squeeze=False)

    for idx, ds in enumerate(datasets):
        row, col = divmod(idx, n_cols)
        ax      = axes[row][col]
        ds_runs = [r for r in runs if r["dataset"] == ds and r.get("acc_history")]

        for r in ds_runs:
            hist   = r["acc_history"]
            epochs = [h["epoch"] for h in hist]
            vals   = [h.get(metric_key, h["acc"]) * 100 for h in hist]
            color  = COND_COLOR.get(r.get("condition", ""), "gray")
            ls     = BACKBONE_LS.get(r.get("backbone_name", "resnet18"), "-")
            label  = _run_label(r)

            ax.plot(epochs, vals, color=color, linestyle=ls,
                    linewidth=1.8, marker=marker, markersize=3, label=label)

        title = DS_CODE.get(ds, ds.replace("mendeley_","").replace("_"," ").title())
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"{metric_label} (%)")
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=40)

    # Hide unused subplots
    for idx in range(len(datasets), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle(f"Validation {metric_label} over Training",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_dir, filename)


def plot_acc_curves(runs, out_dir):
    _plot_metric_curves(runs, out_dir,
                        metric_key="acc", metric_label="Accuracy",
                        filename="acc_curves.png", marker="o")


# ─────────────────────────────────────────────────────────────
# Plot 1b — F1 curves
# ─────────────────────────────────────────────────────────────

def plot_f1_curves(runs, out_dir):
    _plot_metric_curves(runs, out_dir,
                        metric_key="f1_macro", metric_label="F1 Macro",
                        filename="f1_curves.png", marker="s")


# ─────────────────────────────────────────────────────────────
# Plot 1c — Precision curves
# ─────────────────────────────────────────────────────────────

def plot_precision_curves(runs, out_dir):
    _plot_metric_curves(runs, out_dir,
                        metric_key="precision_macro", metric_label="Precision",
                        filename="precision_curves.png", marker="^")


# ─────────────────────────────────────────────────────────────
# Plot 1d — Recall curves
# ─────────────────────────────────────────────────────────────

def plot_recall_curves(runs, out_dir):
    _plot_metric_curves(runs, out_dir,
                        metric_key="recall_macro", metric_label="Recall",
                        filename="recall_curves.png", marker="v")


# ─────────────────────────────────────────────────────────────
# Plot 2 — Train loss + Val loss curves
# ─────────────────────────────────────────────────────────────

def plot_loss_curves(runs, out_dir):
    datasets = sorted({r["dataset"] for r in runs if r.get("acc_history")})
    has_val_loss = any("val_loss" in h for r in runs for h in r.get("acc_history", []))
    if not datasets or not has_val_loss:
        return

    n_cols = 3
    n_rows = (len(datasets) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 5 * n_rows), squeeze=False)

    for idx, ds in enumerate(datasets):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        ds_runs = [r for r in runs if r["dataset"] == ds and r.get("acc_history")]

        for r in ds_runs:
            hist   = [h for h in r["acc_history"] if "val_loss" in h]
            if not hist:
                continue
            epochs    = [h["epoch"] for h in hist]
            val_loss  = [h["val_loss"] for h in hist]
            train_loss= [h.get("train_loss", None) for h in hist]
            color     = COND_COLOR.get(r.get("condition", ""), "gray")
            ls        = BACKBONE_LS.get(r.get("backbone_name", "resnet18"), "-")
            label     = _run_label(r)

            ax.plot(epochs, val_loss, color=color, linestyle=ls,
                    linewidth=1.8, marker="o", markersize=3, label=f"{label} (val)")
            if any(v is not None for v in train_loss):
                ax.plot(epochs, train_loss, color=color, linestyle=":",
                        linewidth=1.2, alpha=0.6, label=f"{label} (train)")

        title = DS_CODE.get(ds, ds.replace("mendeley_","").replace("_"," ").title())
        ax.set_title(f"{title} — Loss", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.25)

    for idx in range(len(datasets), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle("Train Loss vs Val Loss", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_dir, "loss_curves.png")


# ─────────────────────────────────────────────────────────────
# Plot 3 — Metrics summary bar chart
# ─────────────────────────────────────────────────────────────

def plot_metrics_summary(runs, out_dir):
    datasets = sorted({r["dataset"] for r in runs})
    if not datasets:
        return

    metrics_keys = ["best_acc", "best_f1", "best_precision", "best_recall"]
    metric_labels = ["Accuracy", "F1 Macro", "Precision", "Recall"]
    metric_colors = ["#4C72B0", "#C44E52", "#55A868", "#DD8452"]

    n_cols = 3
    n_rows = (len(datasets) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(8 * n_cols, 6 * n_rows), squeeze=False)

    for idx, ds in enumerate(datasets):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        ds_runs = [r for r in runs if r["dataset"] == ds]
        conds   = [c for c in CONDITIONS if any(r.get("condition") == c for r in ds_runs)]

        x     = np.arange(len(conds))
        width = 0.18
        offsets = np.linspace(-(len(metrics_keys)-1)/2, (len(metrics_keys)-1)/2, len(metrics_keys)) * width

        for mi, (key, mlabel, mcolor) in enumerate(zip(metrics_keys, metric_labels, metric_colors)):
            vals = []
            for cond in conds:
                best = max(
                    (r.get(key, 0) for r in ds_runs if r.get("condition") == cond),
                    default=0
                )
                vals.append(best * 100)
            ax.bar(x + offsets[mi], vals, width, label=mlabel,
                   color=mcolor, alpha=0.85, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels([COND_LABEL[c] for c in conds], fontsize=9)
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 108)
        ax.set_title(DS_CODE.get(ds, ds.replace("mendeley_","").replace("_"," ").title()), fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
        ax.axhline(100 / (ds_runs[0].get("num_classes", 2) if ds_runs else 2),
                   color="gray", linestyle="--", linewidth=1, alpha=0.4, label="Random")

    for idx in range(len(datasets), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle("Best Acc / F1 / Precision / Recall per Condition", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_dir, "metrics_summary.png")


# ─────────────────────────────────────────────────────────────
# Plot 4 — Heatmap condition × backbone (Acc and F1)
# ─────────────────────────────────────────────────────────────

def _draw_heatmap_cell(ax, runs, ds, metric_key, metric_label, backbones):
    """Draw one condition×backbone heatmap cell for a given dataset and metric."""
    ds_code = DS_CODE.get(ds, ds[:4].upper())
    matrix  = np.full((len(CONDITIONS), len(backbones)), np.nan)

    for r in runs:
        if r["dataset"] != ds:
            continue
        c = r.get("condition", "")
        b = r.get("backbone_name", "")
        if c in CONDITIONS and b in backbones:
            val = r.get(metric_key, 0) * 100
            i, j = CONDITIONS.index(c), backbones.index(b)
            if np.isnan(matrix[i, j]) or val > matrix[i, j]:
                matrix[i, j] = val

    vmin = max(0, np.nanmin(matrix) - 5) if not np.all(np.isnan(matrix)) else 0
    im   = ax.imshow(matrix, cmap="YlGn", aspect="auto", vmin=vmin, vmax=100)
    plt.colorbar(im, ax=ax, shrink=0.7)

    short_bb = [BB_CODE.get(b, b[:4]) for b in backbones]
    ax.set_xticks(range(len(backbones))); ax.set_xticklabels(short_bb, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(CONDITIONS))); ax.set_yticklabels([COND_LABEL[c] for c in CONDITIONS], fontsize=9)
    ax.set_title(ds_code, fontsize=11)

    for i in range(len(CONDITIONS)):
        for j in range(len(backbones)):
            val = matrix[i, j]
            if not np.isnan(val):
                txt_color = "white" if val > (vmin + (100 - vmin) * 0.7) else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=9, color=txt_color, fontweight="bold")
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=11, color="#aaaaaa")


def plot_heatmap(runs, out_dir):
    """Two separate figures (Accuracy / F1 Macro), each a 3×2 grid of datasets."""
    datasets  = sorted({r["dataset"] for r in runs})
    backbones = sorted({r.get("backbone_name", "") for r in runs})
    if not datasets or not backbones:
        return

    n_cols = 2
    n_rows = 3

    for metric_key, metric_label, fname in [
        ("best_acc", "Accuracy",  "heatmap_acc.png"),
        ("best_f1",  "F1 Macro",  "heatmap_f1.png"),
    ]:
        cell_w = max(4, len(backbones) * 1.8)
        cell_h = len(CONDITIONS) * 1.2 + 1
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(cell_w * n_cols, cell_h * n_rows),
                                 squeeze=False)

        for idx, ds in enumerate(datasets):
            row, col = divmod(idx, n_cols)
            _draw_heatmap_cell(axes[row][col], runs, ds, metric_key, metric_label, backbones)

        for idx in range(len(datasets), n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row][col].set_visible(False)

        fig.suptitle(f"Condition × Backbone Heatmap — {metric_label}",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(h_pad=4.0)
        plt.subplots_adjust(top=0.92)
        _save(fig, out_dir, fname)


# ─────────────────────────────────────────────────────────────
# Plot 5 — Confusion matrices
# ─────────────────────────────────────────────────────────────

def _draw_confusion(ax, r, show_xlabel=True, show_ylabel=True):
    """Draw a normalized confusion matrix — no colorbar, no title."""
    cm  = np.array(r["final_confusion_matrix"])
    cls = r.get("class_names", [str(i) for i in range(cm.shape[0])])
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm_norm, row_sums, where=row_sums > 0)

    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(cls)))
    ax.set_yticks(range(len(cls)))
    # x-tick labels (class names) only on bottom row
    ax.set_xticklabels(cls if show_xlabel else [""] * len(cls),
                       rotation=30, ha="right", fontsize=8)
    # y-tick labels (class names) only on left column
    ax.set_yticklabels(cls if show_ylabel else [""] * len(cls), fontsize=8)

    if show_xlabel: ax.set_xlabel("Predicted", fontsize=8)
    # "True" ylabel handled externally as dataset code

    for i in range(len(cls)):
        for j in range(len(cls)):
            pct = cm_norm[i, j]
            ax.text(j, i, f"{pct*100:.0f}%\n({cm[i,j]})",
                    ha="center", va="center", fontsize=8,
                    color="white" if pct > 0.6 else "black")


def plot_confusion_matrices(runs, out_dir):
    """One PNG per backbone. Grid: rows=datasets, cols=C1-C4.
    Shared column headers (C1-C4) on top row, shared row labels (dataset codes) on left."""
    valid = [r for r in runs if r.get("final_confusion_matrix")]
    if not valid:
        return

    datasets  = sorted({r["dataset"] for r in valid})
    backbones = sorted({r.get("backbone_name","") for r in valid})

    for bb in backbones:
        bb_code = BB_CODE.get(bb, bb[:4])
        bb_runs = [r for r in valid if r.get("backbone_name") == bb]
        ds_list = [ds for ds in datasets if any(r["dataset"] == ds for r in bb_runs)]
        if not ds_list:
            continue

        n_rows = len(ds_list)
        n_cols = len(CONDITIONS)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4.5 * n_cols, 5.2 * n_rows),
                                 squeeze=False)
        fig.suptitle(f"Confusion Matrices — {bb_code}", fontsize=14,
                     fontweight="bold")

        for row, ds in enumerate(ds_list):
            ds_code = DS_CODE.get(ds, ds[:4].upper())
            for col, cond in enumerate(CONDITIONS):
                ax = axes[row][col]
                match = [r for r in bb_runs
                         if r["dataset"] == ds and r.get("condition") == cond]

                # Column header — only top row
                if row == 0:
                    ax.set_title(COND_LABEL.get(cond, cond), fontsize=13,
                                 fontweight="bold", pad=10)

                if match:
                    show_x = (row == n_rows - 1)
                    show_y = (col == 0)   # class names on left column
                    _draw_confusion(ax, match[0],
                                    show_xlabel=show_x, show_ylabel=show_y)
                    # Dataset code as row label — left column only, above class names
                    if col == 0:
                        ax.set_ylabel(f"{ds_code}", fontsize=12, fontweight="bold",
                                      labelpad=12, rotation=0, ha="right", va="center")
                    if row != n_rows - 1:
                        ax.set_xlabel("")
                else:
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                            transform=ax.transAxes, fontsize=14, color="gray")
                    ax.axis("off")

        plt.tight_layout(pad=1.5, h_pad=2.5, w_pad=1.5)
        plt.subplots_adjust(top=0.94, bottom=0.06)
        _save(fig, out_dir, f"confusion_matrix_{bb_code}.png")


# ─────────────────────────────────────────────────────────────
# Plot 6 — Per-class metrics
# ─────────────────────────────────────────────────────────────

def plot_per_class(runs, out_dir):
    """One PNG per backbone. Grid: rows = datasets, cols = conditions (C1-C4)."""
    valid = [r for r in runs if r.get("acc_history") and
             any("f1_per_class" in h for h in r["acc_history"])]
    if not valid:
        return

    datasets  = sorted({r["dataset"] for r in valid})
    backbones = sorted({r.get("backbone_name","") for r in valid})

    for bb in backbones:
        bb_code = BB_CODE.get(bb, bb[:4])
        bb_runs = [r for r in valid if r.get("backbone_name") == bb]
        ds_list = [ds for ds in datasets if any(r["dataset"] == ds for r in bb_runs)]
        if not ds_list:
            continue

        n_rows = len(ds_list)
        n_cols = len(CONDITIONS)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(6 * n_cols, 5.5 * n_rows), squeeze=False)
        fig.suptitle(f"Per-Class Metrics — {bb_code} (final epoch)",
                     fontsize=14, fontweight="bold")

        for row, ds in enumerate(ds_list):
            ds_code = DS_CODE.get(ds, ds[:4].upper())
            for col, cond in enumerate(CONDITIONS):
                ax    = axes[row][col]
                match = [r for r in bb_runs
                         if r["dataset"] == ds and r.get("condition") == cond]

                # Shared column header — top row only
                if row == 0:
                    ax.set_title(COND_LABEL.get(cond, cond), fontsize=13,
                                 fontweight="bold", pad=10)

                if not match:
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                            transform=ax.transAxes, fontsize=14, color="gray")
                    ax.axis("off")
                    continue

                r    = match[0]
                hist = [h for h in r["acc_history"] if "f1_per_class" in h]
                if not hist:
                    ax.set_visible(False)
                    continue

                last  = hist[-1]
                cls   = r.get("class_names", [])
                f1s   = last.get("f1_per_class", [])
                precs = last.get("precision_per_class", [])
                recs  = last.get("recall_per_class", [])

                x     = np.arange(len(cls))
                width = 0.25
                ax.bar(x - width, [v*100 for v in f1s],   width, label="F1",        color="#C44E52", alpha=0.85)
                ax.bar(x,         [v*100 for v in precs],  width, label="Precision", color="#4C72B0", alpha=0.85)
                ax.bar(x + width, [v*100 for v in recs],   width, label="Recall",    color="#55A868", alpha=0.85)

                ax.set_xticks(x)
                ax.set_xticklabels(cls, fontsize=8, rotation=20, ha="right")
                ax.set_ylim(0, 108)
                ax.grid(True, axis="y", alpha=0.25)

                # Shared row label — left column only
                if col == 0:
                    ax.set_ylabel(f"{ds_code}\nScore (%)", fontsize=11,
                                  fontweight="bold", labelpad=8)
                else:
                    ax.set_ylabel("")

                # Legend only on first cell per row
                if col == 0:
                    ax.legend(fontsize=7, loc="upper right")

        plt.tight_layout(pad=1.5, h_pad=2.5, w_pad=1.5)
        plt.subplots_adjust(top=0.94, bottom=0.06)
        _save(fig, out_dir, f"per_class_{bb_code}.png")


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────

def _save(fig, out_dir, filename):
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",  type=str, default=None)
    parser.add_argument("--runs_dir", type=str, default="run_outputs")
    args = parser.parse_args()

    runs = scan_runs(root=args.runs_dir, dataset_filter=args.dataset)

    if not runs:
        print("No runs found in run_outputs/.")
        raise SystemExit(0)

    out_dir = os.path.join(args.runs_dir, "_plots")
    os.makedirs(out_dir, exist_ok=True)

    print_summary(runs)
    plot_acc_curves(runs, out_dir)
    plot_f1_curves(runs, out_dir)
    plot_precision_curves(runs, out_dir)
    plot_recall_curves(runs, out_dir)
    plot_loss_curves(runs, out_dir)
    plot_metrics_summary(runs, out_dir)
    plot_heatmap(runs, out_dir)
    plot_confusion_matrices(runs, out_dir)
    plot_per_class(runs, out_dir)

    print(f"\nDone. All plots saved to {out_dir}/")
