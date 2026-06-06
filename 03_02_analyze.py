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
    "mobilenet_v3_small": ":",
    "efficientnet_b0"   : (0, (5, 1)),
    "efficientnet_b2"   : (0, (3, 1, 1, 1)),
}
BB_CODE = {
    "resnet18"          : "R18",
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
    return f"{bb}-{cond}"


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
                        metric_key="f1_macro", metric_label="F1 Score",
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
        ax.legend(fontsize=6, loc="upper left")
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
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, axis="y", alpha=0.25)
        ax.axhline(100 / (ds_runs[0].get("num_classes", 2) if ds_runs else 2),
                   color="gray", linestyle="--", linewidth=1, alpha=0.4, label="Random")

    for idx in range(len(datasets), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle("Training Condition Comparison", fontsize=13, fontweight="bold")
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

    n_cols = 3
    n_rows = 2

    for metric_key, metric_label, fname in [
        ("best_acc",       "Accuracy",  "heatmap_acc.png"),
        ("best_f1",        "F1 Score",  "heatmap_f1.png"),
        ("best_precision", "Precision", "heatmap_precision.png"),
        ("best_recall",    "Recall",    "heatmap_recall.png"),
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

        fig.suptitle(f"{metric_label} — Condition × Backbone Heatmap",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(h_pad=4.0)
        plt.subplots_adjust(top=0.92, bottom=0.08)
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
    cm_norm  = np.divide(cm_norm, row_sums, out=np.zeros_like(cm_norm), where=row_sums > 0)

    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(cls)))
    ax.set_yticks(range(len(cls)))
    # x-tick labels always visible (needed for formalin and multi-class)
    ax.set_xticklabels(cls, rotation=0, ha="center", fontsize=7)
    # y-tick labels always visible
    ax.set_yticklabels(cls, fontsize=7)

    if show_xlabel: ax.set_xlabel("Predicted", fontsize=7)
    # "True" ylabel handled externally as condition/dataset code

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

        # Rows = conditions, Cols = datasets
        n_rows = len(CONDITIONS)
        n_cols = len(ds_list)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(3.8 * n_cols, 4.2 * n_rows),
                                 squeeze=False)
        fig.suptitle(f"Confusion Matrices — {bb_code}", fontsize=14,
                     fontweight="bold")

        for row, cond in enumerate(CONDITIONS):
            for col, ds in enumerate(ds_list):
                ds_code = DS_CODE.get(ds, ds[:4].upper())
                ax = axes[row][col]
                match = [r for r in bb_runs
                         if r["dataset"] == ds and r.get("condition") == cond]

                # Column header (dataset code) — only top row
                if row == 0:
                    ax.set_title(ds_code, fontsize=13, fontweight="bold", pad=10)

                if match:
                    show_x = (row == n_rows - 1)
                    show_y = (col == 0)
                    _draw_confusion(ax, match[0],
                                    show_xlabel=show_x, show_ylabel=show_y)
                    # Condition code as row label — left column only
                    if col == 0:
                        ax.set_ylabel(COND_LABEL.get(cond, cond),
                                      fontsize=12, fontweight="bold",
                                      labelpad=12, rotation=0, ha="right", va="center")
                    if row != n_rows - 1:
                        ax.set_xlabel("")
                else:
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                            transform=ax.transAxes, fontsize=14, color="gray")
                    ax.axis("off")

        plt.tight_layout(pad=2.5, h_pad=5.0, w_pad=3.5)
        plt.subplots_adjust(top=0.91, bottom=0.06)
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

        # Rows = conditions, Cols = datasets
        n_rows = len(CONDITIONS)
        n_cols = len(ds_list)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(6 * n_cols, 6.5 * n_rows), squeeze=False)
        fig.suptitle(f"Per-Class Metrics — {bb_code} (final epoch)",
                     fontsize=14, fontweight="bold")

        for row, cond in enumerate(CONDITIONS):
            for col, ds in enumerate(ds_list):
                ds_code = DS_CODE.get(ds, ds[:4].upper())
                ax    = axes[row][col]
                match = [r for r in bb_runs
                         if r["dataset"] == ds and r.get("condition") == cond]

                # Column header (dataset code) — top row only
                if row == 0:
                    ax.set_title(ds_code, fontsize=13, fontweight="bold", pad=10)

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
                ax.set_xticklabels(cls, fontsize=8, rotation=0, ha="center")
                ax.set_ylim(0, 108)
                ax.grid(True, axis="y", alpha=0.25)

                if col == 0:
                    ax.set_ylabel("Score (%)", fontsize=9)
                else:
                    ax.set_ylabel("")

                # Legend only on first cell per row
                if col == 0:
                    ax.legend(fontsize=7, loc="upper right")

        plt.tight_layout(pad=2.5, h_pad=5.0, w_pad=3.0)
        plt.subplots_adjust(top=0.93, bottom=0.05, left=0.08)

        # Row labels using figure coordinates
        for row, cond in enumerate(CONDITIONS):
            y_pos = 1.0 - (row + 0.5) / n_rows
            fig.text(0.01, y_pos, COND_LABEL.get(cond, cond),
                     fontsize=13, fontweight="bold",
                     ha="left", va="center", rotation=0)

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
# Eval mode — load eval/own_dataset_state.json files
# ─────────────────────────────────────────────────────────────

def scan_eval_runs(root="run_outputs", eval_dataset="own_dataset",
                   label_mode="state"):
    """Load all eval JSON files for a given dataset and label mode."""
    fname = f"{eval_dataset}_{label_mode}.json"
    results = []
    for run_dir in sorted(Path(root).iterdir()):
        if not run_dir.is_dir() or run_dir.name == "_plots":
            continue
        jpath = run_dir / "eval" / fname
        if not jpath.exists():
            continue
        with open(jpath) as f:
            e = json.load(f)

        # Load training F1 from metrics.json if available
        f1_train = None
        mpath = run_dir / "metrics.json"
        if mpath.exists():
            with open(mpath) as f:
                m = json.load(f)
            f1_train = m.get("best_f1")
            bb   = m.get("backbone_name", "")
            cond = m.get("condition", "")
        else:
            bb   = ""
            cond = ""

        results.append({
            "model_run"   : run_dir.name,
            "backbone"    : bb,
            "condition"   : cond,
            "bb_code"     : BB_CODE.get(bb, bb[:4]),
            "cond_code"   : COND_LABEL.get(cond, cond),
            "ds_code"     : DS_CODE.get(e.get("eval_dataset",""), "?"),
            "f1_eval"     : round(e.get("f1_macro", 0) * 100, 1),
            "acc_eval"    : round(e.get("acc", 0) * 100, 1),
            "recall_eval" : round(e.get("recall_macro", 0) * 100, 1),
            "prec_eval"   : round(e.get("precision_macro", 0) * 100, 1),
            "mcc_eval"    : round(e.get("mcc", 0), 3),
            "auc_eval"    : round(e.get("auc_roc", 0), 4) if e.get("auc_roc") else None,
            "conf_correct": round(e.get("conf_avg_correct", 0) * 100, 1) if e.get("conf_avg_correct") else None,
            "conf_wrong"  : round(e.get("conf_avg_wrong", 0) * 100, 1) if e.get("conf_avg_wrong") else None,
            "f1_train"    : round(f1_train * 100, 1) if f1_train else None,
            "drop"        : round((f1_train - e.get("f1_macro", 0)) * 100, 1) if f1_train else None,
        })

    results.sort(key=lambda r: r["f1_eval"], reverse=True)
    return results


def plot_eval_ranking(evals, out_dir):
    """Bar chart: F1_eval per model, grouped by backbone, sorted descending."""
    if not evals:
        return

    n = len(evals)
    fig, ax = plt.subplots(figsize=(max(14, n * 0.45), 6))

    labels = [f"{r['bb_code']}-{r['cond_code']}" for r in evals]
    f1s    = [r["f1_eval"] for r in evals]
    colors = [COND_COLOR.get(r["condition"], "#888888") for r in evals]

    bars = ax.bar(range(n), f1s, color=colors, alpha=0.85, edgecolor="white")

    # Value labels on bars
    for i, (bar, val) in enumerate(zip(bars, f1s)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8,
                color="white")

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("F1 Score on own_dataset (%)")
    ax.set_ylim(0, 105)
    ax.axhline(90, color=COND_COLOR["head_frozen"], linestyle="--",
               linewidth=1, alpha=0.5, label="90% reference")
    ax.grid(True, axis="y", alpha=0.2)

    # Legend for conditions
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.85)
               for c in COND_COLOR.values()]
    ax.legend(handles, [COND_LABEL[c] for c in COND_COLOR],
              fontsize=9, loc="upper right")

    fig.suptitle(f"Eval Ranking — F1 on own_dataset  ({len(evals)} models)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(pad=1.5)
    _save(fig, out_dir, "01_eval_ranking.png")


def plot_eval_domain_shift(evals, out_dir):
    """Side-by-side bars: F1_train vs F1_eval with drop annotation."""
    valid = [r for r in evals if r["f1_train"] is not None]
    if not valid:
        return

    # Sort by drop ascending (best generalizers first)
    valid.sort(key=lambda r: r["drop"])
    n = len(valid)

    fig, ax = plt.subplots(figsize=(max(14, n * 0.55), 6))
    x = np.arange(n)
    w = 0.38

    ax.bar(x - w/2, [r["f1_train"] for r in valid], w,
           label="F1 Train", color="#4C72B0", alpha=0.85)
    ax.bar(x + w/2, [r["f1_eval"] for r in valid], w,
           label="F1 Eval (own_dataset)", color=COND_COLOR["head_layer4"],
           alpha=0.85)

    # Drop annotations
    for i, r in enumerate(valid):
        drop = r["drop"]
        col  = "#e74c3c" if drop > 15 else "#f39c12" if drop > 5 else "#2ecc71"
        ax.text(i, max(r["f1_train"], r["f1_eval"]) + 1.2,
                f"−{drop:.0f}", ha="center", fontsize=8, color=col,
                fontweight="bold")

    labels = [f"{r['bb_code']}-{r['cond_code']}" for r in valid]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("F1 Score (%)")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.2)

    fig.suptitle("Domain Shift — F1 Train vs F1 Eval  (sorted by drop ↑ = best generalizer)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(pad=1.5)
    _save(fig, out_dir, "03_eval_domain_shift.png")


def plot_eval_heatmap(evals, out_dir):
    """Heatmap: backbone × condition, value = F1_eval."""
    backbones  = sorted({r["backbone"]  for r in evals if r["backbone"]})
    conditions = [c for c in CONDITIONS if any(r["condition"] == c for r in evals)]

    if not backbones or not conditions:
        return

    fig, ax = plt.subplots(figsize=(len(conditions) * 2.2, len(backbones) * 1.5 + 1))

    matrix = np.full((len(backbones), len(conditions)), np.nan)
    for r in evals:
        if r["backbone"] in backbones and r["condition"] in conditions:
            i = backbones.index(r["backbone"])
            j = conditions.index(r["condition"])
            val = r["f1_eval"]
            if np.isnan(matrix[i, j]) or val > matrix[i, j]:
                matrix[i, j] = val

    vmin = max(0, np.nanmin(matrix) - 5) if not np.all(np.isnan(matrix)) else 0
    im = ax.imshow(matrix, cmap="YlGn", aspect="auto", vmin=vmin, vmax=100)
    plt.colorbar(im, ax=ax, label="F1 Eval (%)", shrink=0.8)

    bb_labels   = [BB_CODE.get(b, b[:4]) for b in backbones]
    cond_labels = [COND_LABEL.get(c, c) for c in conditions]
    ax.set_xticks(range(len(conditions))); ax.set_xticklabels(cond_labels, fontsize=11)
    ax.set_yticks(range(len(backbones)));  ax.set_yticklabels(bb_labels, fontsize=11)

    for i in range(len(backbones)):
        for j in range(len(conditions)):
            val = matrix[i, j]
            if not np.isnan(val):
                col = "white" if val > (vmin + (100 - vmin) * 0.7) else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=10, color=col, fontweight="bold")
            else:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=11, color="#aaaaaa")

    fig.suptitle("F1 Score on own_dataset — Backbone × Condition",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(pad=1.5)
    _save(fig, out_dir, "05_eval_heatmap.png")


def plot_eval_confidence(evals, out_dir):
    """Scatter: conf_correct vs conf_wrong per backbone, sized by F1_eval."""
    valid = [r for r in evals
             if r["conf_correct"] is not None and r["conf_wrong"] is not None]
    if not valid:
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    for r in valid:
        col  = COND_COLOR.get(r["condition"], "#888888")
        size = max(30, r["f1_eval"] ** 2 * 0.015)
        ax.scatter(r["conf_wrong"], r["conf_correct"],
                   color=col, s=size, alpha=0.75, edgecolors="white",
                   linewidths=0.8)
        ax.annotate(f"{r['bb_code']}-{r['cond_code']}",
                    (r["conf_wrong"], r["conf_correct"]),
                    fontsize=7, color="#cccccc",
                    xytext=(4, 2), textcoords="offset points")

    # Reference lines
    ax.axhline(90, color="#2ecc71", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(60, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Conf Wrong (%) — lower is better", fontsize=11)
    ax.set_ylabel("Conf Correct (%) — higher is better", fontsize=11)
    ax.set_xlim(40, 105); ax.set_ylim(60, 105)
    ax.grid(True, alpha=0.2)

    handles = [plt.scatter([], [], color=c, s=60, alpha=0.85)
               for c in COND_COLOR.values()]
    ax.legend(handles, [COND_LABEL[c] for c in COND_COLOR],
              fontsize=9, loc="lower right")

    fig.suptitle("Confidence Calibration on own_dataset\n"
                 "(ideal: top-left — certain when right, uncertain when wrong)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(pad=1.5)
    _save(fig, out_dir, "07_eval_confidence.png")


def plot_eval_by_dataset(evals, out_dir):
    """Option C: 2×N grid — row1=ranking, row2=domain_shift, cols=training datasets."""

    def get_ds_code(r):
        for ds, code in DS_CODE.items():
            if r["model_run"].startswith(ds):
                return code
        return "?"

    for r in evals:
        r["_ds_code"] = get_ds_code(r)

    ds_list = sorted({r["_ds_code"] for r in evals if r["_ds_code"] != "?"})
    n_cols  = len(ds_list)

    fig, axes = plt.subplots(2, n_cols,
                             figsize=(5 * n_cols, 10), squeeze=False)
    fig.suptitle("Eval Results by Training Dataset  "
                 "(row 1 = F1 Ranking · row 2 = Domain Shift)",
                 fontsize=14, fontweight="bold")

    for col, ds_code in enumerate(ds_list):
        subset = [r for r in evals if r["_ds_code"] == ds_code]

        # ── Row 0: ranking ──
        ax1 = axes[0][col]
        subset_s = sorted(subset, key=lambda r: r["f1_eval"], reverse=True)
        labels = [f"{r['bb_code']}-{r['cond_code']}" for r in subset_s]
        f1s    = [r["f1_eval"] for r in subset_s]
        colors = [COND_COLOR.get(r["condition"], "#888888") for r in subset_s]
        bars = ax1.bar(range(len(subset_s)), f1s, color=colors,
                       alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, f1s):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                     f"{val:.0f}", ha="center", fontsize=7, color="white")
        ax1.set_xticks(range(len(subset_s)))
        ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax1.set_ylim(0, 105)
        ax1.set_title(ds_code, fontsize=13, fontweight="bold")
        ax1.grid(True, axis="y", alpha=0.2)
        if col == 0:
            ax1.set_ylabel("F1 Eval (%)", fontsize=10)

        # ── Row 1: domain shift ──
        ax2 = axes[1][col]
        valid = sorted([r for r in subset if r["f1_train"] is not None],
                       key=lambda r: r["drop"] if r["drop"] else 999)
        x = np.arange(len(valid)); w = 0.38
        ax2.bar(x - w/2, [r["f1_train"] for r in valid], w,
                label="Train", color="#4C72B0", alpha=0.85)
        ax2.bar(x + w/2, [r["f1_eval"]  for r in valid], w,
                label="Eval",  color=COND_COLOR["head_layer4"], alpha=0.85)
        for i, r in enumerate(valid):
            if r["drop"] is not None:
                c = "#e74c3c" if r["drop"] > 15 else "#f39c12" if r["drop"] > 5 else "#2ecc71"
                ax2.text(i, max(r["f1_train"], r["f1_eval"]) + 1.2,
                         f"−{r['drop']:.0f}", ha="center", fontsize=7,
                         color=c, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{r['bb_code']}-{r['cond_code']}" for r in valid],
                             rotation=45, ha="right", fontsize=7)
        ax2.set_ylim(0, 110)
        ax2.grid(True, axis="y", alpha=0.2)
        if col == 0:
            ax2.set_ylabel("F1 (%)", fontsize=10)
            ax2.legend(fontsize=8)
        elif col == n_cols - 1:
            ax2.legend(fontsize=8)

    plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=1.5)
    _save(fig, out_dir, "04_eval_by_dataset.png")


def plot_eval_grid(evals, out_dir):
    """Option E: grid backbone × condition for ranking and domain shift."""
    backbones  = [b for b in ["resnet18","mobilenet_v3_small",
                               "efficientnet_b0","efficientnet_b2"]
                  if any(r["backbone"] == b for r in evals)]
    conditions = [c for c in CONDITIONS
                  if any(r["condition"] == c for r in evals)]

    if not backbones or not conditions:
        return

    n_rows = len(backbones)
    n_cols = len(conditions)

    for metric, ylabel, title_suffix, fname in [
        ("f1_eval",  "F1 Eval (%)",   "F1 on own_dataset",    "06_eval_grid_f1.png"),
        ("drop",     "Drop (%)",       "Domain Shift (drop)",  "02_eval_grid_drop.png"),
    ]:
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
        fig.suptitle(f"Backbone × Condition Grid — {title_suffix}",
                     fontsize=14, fontweight="bold")

        for row, bb in enumerate(backbones):
            bb_code = BB_CODE.get(bb, bb[:4])
            for col, cond in enumerate(conditions):
                ax = axes[row][col]
                subset = [r for r in evals
                          if r["backbone"] == bb and r["condition"] == cond
                          and r[metric] is not None]
                subset.sort(key=lambda r: -r["f1_eval"])

                if row == 0:
                    ax.set_title(COND_LABEL.get(cond, cond), fontsize=12,
                                 fontweight="bold")
                if col == 0:
                    ax.text(-0.25, 0.5, bb_code, transform=ax.transAxes,
                            fontsize=12, fontweight="bold", ha="right",
                            va="center", rotation=0)

                if not subset:
                    ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                            ha="center", va="center", color=C_GRAY, fontsize=13)
                    ax.axis("off")
                    continue

                # One bar per training dataset
                ds_codes = [DS_CODE.get(r["model_run"].split("_all_")[0]
                            if "_all_" in r["model_run"] else "", "?")
                            for r in subset]
                vals  = [r[metric] if r[metric] is not None else 0 for r in subset]
                col_c = COND_COLOR.get(cond, "#888888")
                bars  = ax.bar(range(len(subset)), vals, color=col_c,
                               alpha=0.85, edgecolor="white")
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.5,
                            f"{val:.0f}", ha="center", fontsize=8, color="white")
                ax.set_xticks(range(len(subset)))
                ax.set_xticklabels(ds_codes, rotation=30, ha="right", fontsize=8)
                if metric == "drop":
                    ax.set_ylim(0, 40)
                else:
                    ax.set_ylim(0, 105)
                ax.grid(True, axis="y", alpha=0.2)
                if col == 0:
                    ax.set_ylabel(ylabel, fontsize=9)

        plt.tight_layout(pad=2.0, h_pad=3.0)
        _save(fig, out_dir, fname)


def plot_eval_summary(evals, out_dir):
    """Print console summary of eval results."""
    print(f"\n{'─'*90}")
    print(f"  {'Model':<45} {'F1_eval':>8} {'F1_train':>9} {'Drop':>7} "
          f"{'AUC':>7} {'MCC':>7}")
    print(f"{'─'*90}")
    for r in evals[:20]:
        drop_str = f"-{r['drop']:.1f}" if r["drop"] is not None else "n/a"
        auc_str  = f"{r['auc_eval']:.3f}" if r["auc_eval"] else "n/a"
        f1t_str  = f"{r['f1_train']:.1f}%" if r["f1_train"] else "n/a"
        print(f"  {r['model_run']:<45} {r['f1_eval']:>7.1f}% "
              f"{f1t_str:>9} {drop_str:>7} {auc_str:>7} {r['mcc_eval']:>7.3f}")
    print(f"{'─'*90}")
    print(f"  Total models evaluated: {len(evals)}\n")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

PLOT_MENU = {
    1: ("acc_curves",      plot_acc_curves),
    2: ("f1_curves",       plot_f1_curves),
    3: ("precision_curves",plot_precision_curves),
    4: ("recall_curves",   plot_recall_curves),
    5: ("loss_curves",     plot_loss_curves),
    6: ("metrics_summary", plot_metrics_summary),
    7: ("heatmap",         plot_heatmap),
    8: ("confusion_matrix",plot_confusion_matrices),
    9: ("per_class",       plot_per_class),
}

EVAL_PLOT_MENU = {
    1: ("eval_ranking",      plot_eval_ranking),       # → 01_eval_ranking.png
    2: ("eval_grid_drop",    plot_eval_grid),           # → 02_eval_grid_drop.png + 06_eval_grid_f1.png
    3: ("eval_domain_shift", plot_eval_domain_shift),  # → 03_eval_domain_shift.png
    4: ("eval_by_dataset",   plot_eval_by_dataset),    # → 04_eval_by_dataset.png
    5: ("eval_heatmap",      plot_eval_heatmap),       # → 05_eval_heatmap.png
    6: ("eval_confidence",   plot_eval_confidence),    # → 07_eval_confidence.png
}


def parse_plots(raw, menu):
    """Parse --plots argument: '1,3', '1-5', 'all' → set of ints."""
    if raw is None or raw.strip().lower() == "all":
        return set(menu.keys())
    selected = set()
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            selected.update(range(int(a), int(b) + 1))
        else:
            selected.add(int(part))
    return selected & set(menu.keys())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset",  type=str, default=None,
                        help="Filter by dataset name (training mode only)")
    parser.add_argument("--runs_dir", type=str, default="run_outputs")
    parser.add_argument("--plots",    type=str, default=None,
                        help="Training mode: '1,3', '1-5', or 'all'.\n"
                             "  1=acc  2=f1  3=prec  4=recall  5=loss\n"
                             "  6=summary  7=heatmap  8=confusion  9=per_class\n"
                             "Eval mode (--eval): '1,3', or 'all'.\n"
                             "  1=ranking  2=domain_shift  3=heatmap  4=confidence")
    parser.add_argument("--eval",     action="store_true",
                        help="Eval mode: analyze cross-dataset evaluation results")
    parser.add_argument("--eval-dataset", type=str, default="own_dataset",
                        help="Dataset name used in evaluation (default: own_dataset)")
    parser.add_argument("--eval-mode",    type=str, default="state",
                        help="Label mode used in evaluation (default: state)")
    args = parser.parse_args()

    if args.eval:
        # ── Eval mode ──
        evals = scan_eval_runs(
            root=args.runs_dir,
            eval_dataset=args.eval_dataset,
            label_mode=args.eval_mode
        )
        if not evals:
            print(f"No eval results found for {args.eval_dataset} ({args.eval_mode}).")
            raise SystemExit(0)

        out_dir = os.path.join(args.runs_dir, "_plots", "eval")
        os.makedirs(out_dir, exist_ok=True)

        # Clean old numbered eval plots to avoid stale files from previous runs
        import glob as _glob
        for old_png in _glob.glob(os.path.join(out_dir, "0?_eval_*.png")):
            os.remove(old_png)

        selected = parse_plots(args.plots, EVAL_PLOT_MENU)

        plot_eval_summary(evals, out_dir)
        print(f"\nGenerating eval plots: {sorted(selected)}")
        print(f"  " + "  ".join(f"{n}={EVAL_PLOT_MENU[n][0]}"
                                 for n in sorted(selected)))

        for n in sorted(selected):
            name, fn = EVAL_PLOT_MENU[n]
            print(f"\n[{n}] {name}")
            fn(evals, out_dir)

        print(f"\nDone. Eval plots saved to {out_dir}/")

    else:
        # ── Training mode ──
        runs = scan_runs(root=args.runs_dir, dataset_filter=args.dataset)
        if not runs:
            print("No runs found in run_outputs/.")
            raise SystemExit(0)

        out_dir = os.path.join(args.runs_dir, "_plots")
        os.makedirs(out_dir, exist_ok=True)

        selected = parse_plots(args.plots, PLOT_MENU)

        print_summary(runs)
        print(f"\nGenerating plots: {sorted(selected)}")
        print(f"  " + "  ".join(f"{n}={PLOT_MENU[n][0]}"
                                  for n in sorted(selected)))

        for n in sorted(selected):
            name, fn = PLOT_MENU[n]
            print(f"\n[{n}] {name}")
            fn(runs, out_dir)

        print(f"\nDone. Plots saved to {out_dir}/")
