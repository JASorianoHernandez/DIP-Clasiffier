"""
evaluate.py — Cross-dataset evaluation for the Freshness Classifier.

Loads a best_model.pt from any completed training run and evaluates it
on any dataset — including datasets the model has never seen.

This is useful to measure how well a model trained on internet images
generalises to real photos (own_dataset) or to other fruit collections.

Usage:
    python evaluate.py
"""

import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix,
    roc_auc_score, roc_curve,
    matthews_corrcoef,
)

import importlib as _il
from _dataset import get_loaders, get_loaders_nested
from _backbone import get_backbone, BACKBONE_REGISTRY
_train        = _il.import_module("02_01_train")
ProjectionHead = _train.ProjectionHead
DATASETS       = _train.DATASETS
CONFIG         = _train.CONFIG

BASE_OUT = Path("run_outputs")

# ─────────────────────────────────────────────────────────────
# Scan available trained models
# ─────────────────────────────────────────────────────────────

def find_trained_models():
    """Return list of (run_dir, metrics) for runs that have best_model.pt."""
    models = []
    for run_dir in sorted(BASE_OUT.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "_plots":
            continue
        model_path   = run_dir / "best_model.pt"
        metrics_path = run_dir / "metrics.json"
        if model_path.exists() and metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            models.append((run_dir, metrics))
    return models


# ─────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────

def load_model(run_dir: Path, metrics: dict, device: str):
    """Reconstruct and load the model from best_model.pt."""
    backbone_name = metrics["backbone_name"]
    condition     = metrics["condition"]
    num_classes   = metrics["num_classes"]
    use_head      = metrics["use_head"]
    unfreeze      = metrics["unfreeze_layers"]

    # Build architecture
    backbone = get_backbone(device, backbone_name=backbone_name,
                            pretrained=False, unfreeze_layers=0)

    if use_head:
        proj   = ProjectionHead(in_dim=backbone.out_dim,
                                out_dim=CONFIG["embedding_dim"]).to(device)
        linear = nn.Linear(CONFIG["embedding_dim"], num_classes).to(device)
    else:
        proj   = None
        linear = nn.Linear(backbone.out_dim, num_classes).to(device)

    # Load weights
    ckpt = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    backbone.load_state_dict(ckpt["backbone"])
    linear.load_state_dict(ckpt["linear"])
    if proj is not None and ckpt.get("proj") is not None:
        proj.load_state_dict(ckpt["proj"])

    backbone.eval()
    linear.eval()
    if proj is not None:
        proj.eval()

    return backbone, proj, linear, unfreeze


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(backbone, proj, linear, loader, device):
    all_preds, all_labels, all_probs = [], [], []
    n_images = 0
    t0 = time.perf_counter()

    for images, labels in loader:
        images = images.to(device)
        feats  = backbone(images)
        emb    = proj(feats) if proj is not None else feats
        logits = linear(emb)
        probs  = torch.softmax(logits, dim=1)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        n_images += images.size(0)

    elapsed_ms   = (time.perf_counter() - t0) * 1000
    ms_per_image = elapsed_ms / n_images if n_images > 0 else 0

    probs_arr = np.array(all_probs)
    n_classes = probs_arr.shape[1]

    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec  = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    mcc  = matthews_corrcoef(all_labels, all_preds)
    cm   = confusion_matrix(all_labels, all_preds)
    f1_per   = f1_score(all_labels, all_preds, average=None, zero_division=0).tolist()
    prec_per = precision_score(all_labels, all_preds, average=None, zero_division=0).tolist()
    rec_per  = recall_score(all_labels, all_preds, average=None, zero_division=0).tolist()

    # AUC-ROC
    try:
        if n_classes == 2:
            auc = roc_auc_score(all_labels, probs_arr[:, 1])
        else:
            auc = roc_auc_score(all_labels, probs_arr,
                                multi_class="ovr", average="macro")
    except ValueError:
        auc = None  # fails if only 1 class present in small val sets

    # Confidence stats
    conf_correct = [float(max(p)) for p, pred, true
                    in zip(all_probs, all_preds, all_labels) if pred == true]
    conf_wrong   = [float(max(p)) for p, pred, true
                    in zip(all_probs, all_preds, all_labels) if pred != true]

    return {
        "acc"                  : round(float(acc),  4),
        "f1_macro"             : round(float(f1),   4),
        "precision_macro"      : round(float(prec), 4),
        "recall_macro"         : round(float(rec),  4),
        "mcc"                  : round(float(mcc),  4),
        "auc_roc"              : round(float(auc),  4) if auc is not None else None,
        "inference_ms_per_img" : round(ms_per_image, 2),
        "conf_avg_correct"     : round(float(np.mean(conf_correct)), 4) if conf_correct else None,
        "conf_avg_wrong"       : round(float(np.mean(conf_wrong)),   4) if conf_wrong   else None,
        "f1_per_class"         : [round(v, 4) for v in f1_per],
        "precision_per_class"  : [round(v, 4) for v in prec_per],
        "recall_per_class"     : [round(v, 4) for v in rec_per],
        "confusion_matrix"     : cm.tolist(),
        # raw per-image data (used for plots and _preds.json, not saved in metrics)
        "_preds"  : all_preds,
        "_labels" : all_labels,
        "_probs"  : all_probs,
    }


# ─────────────────────────────────────────────────────────────
# Per-image predictions builder
# ─────────────────────────────────────────────────────────────

def build_per_image_preds(results: dict, val_loader,
                          eval_classes: list, model_classes: list) -> list:
    """
    Build a list of per-image prediction records.
    Extracts filenames from the val_loader's underlying dataset.

    Uses eval_classes to decode true labels (dataset side)
    and model_classes to decode predicted labels + prob keys (model side).

    Returns list of dicts:
      { file, true_label, pred_label, correct, confidence, all_probs }
    """
    subset       = val_loader.dataset
    base_dataset = subset.dataset
    indices      = subset.indices

    records = []
    for i, (pred, true, probs) in enumerate(
        zip(results["_preds"], results["_labels"], results["_probs"])
    ):
        sample_path = Path(base_dataset.samples[indices[i]][0])

        # true label comes from eval dataset classes
        true_name = eval_classes[int(true)] if int(true) < len(eval_classes) else str(true)
        # predicted label comes from model classes
        pred_name = model_classes[int(pred)] if int(pred) < len(model_classes) else str(pred)

        confidence = round(float(max(probs)), 4)
        all_probs  = {model_classes[j]: round(float(p), 4)
                      for j, p in enumerate(probs)
                      if j < len(model_classes)}

        records.append({
            "file"       : sample_path.name,
            "true_label" : true_name,
            "pred_label" : pred_name,
            "correct"    : true_name == pred_name,
            "confidence" : confidence,
            "probs"      : all_probs,
        })

    return records


# ─────────────────────────────────────────────────────────────
# Print results
# ─────────────────────────────────────────────────────────────

def print_results(results: dict, model_classes: list, eval_classes: list):
    auc_str  = f"{results['auc_roc']:.4f}" if results.get("auc_roc") else "n/a"
    print(f"\n{'='*55}")
    print(f"  Acc             : {results['acc']*100:.2f}%")
    print(f"  F1 macro        : {results['f1_macro']*100:.2f}%")
    print(f"  Precision       : {results['precision_macro']*100:.2f}%")
    print(f"  Recall          : {results['recall_macro']*100:.2f}%")
    print(f"  MCC             : {results['mcc']:.4f}  (-1 worst / 0 random / +1 perfect)")
    print(f"  AUC-ROC         : {auc_str}  (0.5 random / 1.0 perfect)")
    print(f"  Inference speed : {results['inference_ms_per_img']:.1f} ms/image")
    if results.get("conf_avg_correct") is not None:
        print(f"  Conf correct    : {results['conf_avg_correct']*100:.1f}%  avg confidence on correct preds")
    if results.get("conf_avg_wrong") is not None:
        print(f"  Conf wrong      : {results['conf_avg_wrong']*100:.1f}%  avg confidence on wrong preds")
    print(f"\n  Per-class F1:")
    for cls, f1 in zip(eval_classes, results["f1_per_class"]):
        print(f"    {cls:<20} {f1*100:.2f}%")
    print(f"\n  Confusion matrix (rows=true, cols=predicted):")
    print(f"  Classes: {eval_classes}")
    for row in results["confusion_matrix"]:
        print(f"    {row}")
    print(f"{'='*55}")


# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────

def save_eval_plots(results: dict, eval_classes: list, model_classes: list,
                    run_dir: Path, ds_name: str, label_mode: str, model_num: int):
    """
    Generate and save evaluation plots for a single model:
      1. Confusion matrix (normalized)
      2. ROC curve (binary only)
      3. Confidence histogram (correct vs wrong)
      4. Metrics bar chart — Overall (Acc/F1/Prec/Recall/MCC) + Per-class breakdown
    """
    preds     = np.array(results["_preds"])
    labels    = np.array(results["_labels"])
    probs_arr = np.array(results["_probs"])
    n_classes = len(eval_classes)
    is_binary = n_classes == 2

    run_name = run_dir.parent.name  # run_dir is .../run_name/eval/
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"#{model_num} {run_name}\nEval: {ds_name} ({label_mode})",
                 fontsize=10, y=1.01)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Confusion matrix ──
    ax1 = fig.add_subplot(gs[0, 0])
    cm  = np.array(results["confusion_matrix"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax1.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax1.set_xticks(range(n_classes)); ax1.set_xticklabels(eval_classes, rotation=30, ha="right", fontsize=8)
    ax1.set_yticks(range(n_classes)); ax1.set_yticklabels(eval_classes, fontsize=8)
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("True")
    ax1.set_title("Confusion Matrix (normalized)")
    for i in range(n_classes):
        for j in range(n_classes):
            ax1.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                     color="white" if cm_norm[i,j] > 0.5 else "black", fontsize=9)
    plt.colorbar(im, ax=ax1, fraction=0.046)

    # ── 2. ROC curve (binary only) ──
    ax2 = fig.add_subplot(gs[0, 1])
    if is_binary and results.get("auc_roc") is not None:
        fpr, tpr, _ = roc_curve(labels, probs_arr[:, 1])
        ax2.plot(fpr, tpr, lw=2, label=f"AUC = {results['auc_roc']:.3f}")
        ax2.plot([0, 1], [0, 1], "k--", lw=1, label="Random (0.500)")
        ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curve")
        ax2.legend(fontsize=9); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02)
    else:
        ax2.text(0.5, 0.5, "ROC curve\n(binary only)",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=11)
        ax2.set_title("ROC Curve")

    # ── 3. Confidence histogram ──
    ax3 = fig.add_subplot(gs[1, 0])
    conf_correct = [float(max(p)) for p, pred, true
                    in zip(results["_probs"], preds, labels) if pred == true]
    conf_wrong   = [float(max(p)) for p, pred, true
                    in zip(results["_probs"], preds, labels) if pred != true]
    bins = np.linspace(0, 1, 21)
    if conf_correct:
        ax3.hist(conf_correct, bins=bins, alpha=0.7, label=f"Correct ({len(conf_correct)})", color="#2ecc71")
    if conf_wrong:
        ax3.hist(conf_wrong,   bins=bins, alpha=0.7, label=f"Wrong ({len(conf_wrong)})",   color="#e74c3c")
    ax3.set_xlabel("Confidence"); ax3.set_ylabel("Count")
    ax3.set_title("Confidence Distribution")
    ax3.legend(fontsize=9)

    # ── 4. Per-class + Overall metrics ──
    ax4        = fig.add_subplot(gs[1, 1])
    n_per      = len(results["f1_per_class"])
    bar_labels = (model_classes if len(model_classes) == n_per else eval_classes)

    # Overall values (aggregate)
    overall_f1   = results["f1_macro"]   * 100
    overall_prec = results["precision_macro"] * 100
    overall_rec  = results["recall_macro"]    * 100
    overall_acc  = results["acc"]        * 100

    # Build combined x positions: [Overall gap] [per-class bars]
    # gap of 0.8 between Overall and per-class groups
    gap       = 0.8
    w         = 0.2
    x_overall = np.array([0.0])
    x_per     = np.arange(n_per) + gap + 1.0
    x_acc     = np.array([0.6])  # Accuracy sits next to Overall

    # Overall group
    ax4.bar(x_overall - w*1.5, [overall_f1],   w, color="#3498db", label="F1")
    ax4.bar(x_overall - w*0.5, [overall_prec], w, color="#e67e22", label="Precision")
    ax4.bar(x_overall + w*0.5, [overall_rec],  w, color="#9b59b6", label="Recall")
    ax4.bar(x_overall + w*1.5, [overall_acc],  w, color="#2ecc71", label="Accuracy")

    # Per-class group
    ax4.bar(x_per - w, [v*100 for v in results["f1_per_class"]],        w, color="#3498db")
    ax4.bar(x_per,     [v*100 for v in results["precision_per_class"]], w, color="#e67e22")
    ax4.bar(x_per + w, [v*100 for v in results["recall_per_class"]],    w, color="#9b59b6")

    # x labels
    all_x      = np.concatenate([x_overall, x_per])
    all_labels = ["Overall"] + list(bar_labels)
    ax4.set_xticks(all_x)
    ax4.set_xticklabels(all_labels, rotation=20, ha="right", fontsize=8)

    # Divider line between Overall and per-class
    ax4.axvline(x=x_overall[0] + gap * 0.5, color="gray",
                linestyle="--", linewidth=0.8, alpha=0.6)

    # MCC annotation
    mcc_str = f"MCC: {results['mcc']:.3f}"
    ax4.text(0.98, 0.97, mcc_str, transform=ax4.transAxes,
             ha="right", va="top", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    ax4.set_ylabel("%")
    ax4.set_ylim(0, 115)
    ax4.set_title("Metrics — Overall & Per-class")
    ax4.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.92))

    plt.tight_layout()
    out_path = run_dir / f"{ds_name}_{label_mode}_plots.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


# ─────────────────────────────────────────────────────────────
# Recommendations
# ─────────────────────────────────────────────────────────────

def recommend_models(models: list, top_n: int = 3) -> list:
    """
    Return indices (0-based) of the top_n recommended models.
    Criteria (in order of priority):
      1. Has a valid best_f1 (not n/a)
      2. Trained on state mode (2 classes: fresh/rotten) — most general
      3. Highest best_f1
      4. Condition C4 preferred over C3
    """
    scored = []
    for i, (run_dir, m) in enumerate(models):
        f1      = m.get("best_f1", 0.0)
        classes = m.get("class_names", [])
        cond    = m.get("condition", "")
        if f1 <= 0:
            continue
        # prefer state mode (2 classes) for cross-dataset generalization
        class_score = 1 if len(classes) == 2 else 0
        cond_score  = 1 if "layer4" in cond else 0
        scored.append((f1 + class_score * 0.1 + cond_score * 0.01, i))

    scored.sort(reverse=True)
    return [idx for _, idx in scored[:top_n]]


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    device = (
        "cuda"  if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"\n── Freshness Classifier — Cross-Dataset Evaluation ──")
    print(f"   Device: {device}\n")

    # ── Step 1: select models ──
    models = find_trained_models()
    if not models:
        print("No trained models found in run_outputs/.")
        return

    recommended = recommend_models(models, top_n=3)

    print("Available trained models:")
    for i, (run_dir, m) in enumerate(models, 1):
        best_f1 = m.get("best_f1", 0.0)
        f1_str  = f"{best_f1*100:.1f}%" if best_f1 > 0 else "n/a"
        tag     = ""
        if (i - 1) == recommended[0]:
            tag = "  ★★★ recommended"
        elif (i - 1) == recommended[1]:
            tag = "  ★★  recommended"
        elif (i - 1) == recommended[2]:
            tag = "  ★   recommended"
        print(f"  {i:2d}. {run_dir.name}{tag}")
        print(f"       classes={m['class_names']}  "
              f"backbone={m['backbone_name']}  "
              f"cond={m['condition']}  "
              f"best_f1={f1_str}")

    print(f"\n  Tip: select multiple with commas (e.g. 1,3,7), a range (e.g. 1-3), or 'all'")
    raw = input(f"Select model(s) [1]: ").strip() or "1"

    try:
        parse_selection = _train.parse_selection
        indices = parse_selection(raw, len(models))
        selected_models = [(i, models[i-1][0], models[i-1][1]) for i in indices]
    except (ValueError, Exception) as e:
        print(f"Invalid selection: {e}")
        return

    # ── Step 2: select eval dataset ──
    print("\nDataset to evaluate on:")
    for i, (name, path, desc) in DATASETS.items():
        exists = "✓" if Path(path).exists() else "✗"
        print(f"  {i}. [{exists}] {name:<30} {desc}")

    raw = input("\nSelect dataset [1]: ").strip() or "1"
    try:
        ds_key = int(raw)
        ds_name, ds_path, _ = DATASETS[ds_key]
    except (ValueError, KeyError):
        print("Invalid selection.")
        return

    if not Path(ds_path).exists():
        print(f"\n  [ERROR] Dataset not found: {ds_path}")
        print(f"  Run first: python prepare_datasets.py")
        return

    # ── Step 3: select label mode ──
    print("\nLabel mode for evaluation:")
    print("  1. state        (fresh / rotten / formalin)")
    print("  2. fruit_state  (apple_fresh / apple_rotten / ...)")
    raw = input("Select mode [1]: ").strip() or "1"
    label_mode = "fruit_state" if raw == "2" else "state"

    # ── Step 4: load eval dataset ──
    p = Path(ds_path)
    has_split = (p / "train").exists() and (p / "test").exists()

    # Pure evaluation sets (never used for training) are scored on ALL their
    # images. Splitting them 80/20 would throw away 80% of held-out real
    # photos for no reason. In-distribution datasets keep the 0.2 val split
    # so we only score the portion the model did not train on.
    PURE_EVAL_DATASETS = {"own_dataset"}
    eval_val_split = 1.0 if ds_name in PURE_EVAL_DATASETS else 0.2
    if eval_val_split == 1.0:
        print(f"  Pure eval set '{ds_name}': scoring on ALL images (val_split=1.0)")

    if has_split:
        _, val_loader, num_classes, eval_classes = get_loaders(
            ds_path, batch_size=32, num_workers=4,
            val_split=eval_val_split, seed=CONFIG["seed"],
        )
    else:
        _, val_loader, num_classes, eval_classes = get_loaders_nested(
            ds_path, batch_size=32, num_workers=4,
            val_split=eval_val_split, seed=CONFIG["seed"],
            label_mode=label_mode,
        )

    # ── Step 5: compatibility check ──
    compatible   = [(n, d, m) for n, d, m in selected_models
                    if sorted(m["class_names"]) == sorted(eval_classes)]
    incompatible = [(n, d, m) for n, d, m in selected_models
                    if sorted(m["class_names"]) != sorted(eval_classes)]

    if incompatible:
        print(f"\n{'─'*55}")
        print(f"  ⚠  Compatibility check — dataset classes: {eval_classes}")
        print(f"\n  Compatible   ({len(compatible)}):")
        for n, d, m in compatible:
            print(f"    #{n:>2}  {d.name}")
        print(f"\n  Incompatible ({len(incompatible)}) — results would be meaningless:")
        for n, d, m in incompatible:
            print(f"    #{n:>2}  {d.name}  (classes: {m['class_names']})")

        compat_nums = ", ".join(f"#{n}" for n, _, _ in compatible)
        print(f"\n  Compatible models that will be applied: {compat_nums}")
        print(f"\n  Options:")
        print(f"    1. Run only compatible models (recommended)")
        print(f"    2. Re-select models manually")
        print(f"    3. Run all anyway (incompatible results will be incorrect)")
        choice = input("  Choose [1]: ").strip() or "1"

        if choice == "2":
            print(f"\n  Compatible model numbers: "
                  f"{', '.join(str(n) for n, _, _ in compatible)}")
            raw2 = input("  Re-select model(s): ").strip()
            try:
                parse_selection = _train.parse_selection
                indices2       = parse_selection(raw2, len(models))
                selected_models = [(i, models[i-1][0], models[i-1][1]) for i in indices2]
                compatible      = [(n, d, m) for n, d, m in selected_models
                                   if sorted(m["class_names"]) == sorted(eval_classes)]
                incompatible    = []
            except (ValueError, Exception) as e:
                print(f"  Invalid selection: {e}")
                return
        elif choice == "3":
            pass   # run all, including incompatible
        else:
            selected_models = compatible  # default: compatible only

    # ── Step 6: evaluate each selected model ──
    all_results = []

    for model_num, run_dir, metrics in selected_models:
        model_classes = metrics["class_names"]

        print(f"\n{'─'*55}")
        print(f"  Model #{model_num}: {run_dir.name}")
        print(f"  Classes : {model_classes}")

        if sorted(model_classes) != sorted(eval_classes):
            print(f"  ⚠  Running with class mismatch — results are for reference only.")

        print(f"  Evaluating on '{ds_name}' ({label_mode} mode) ...")

        backbone, proj, linear, _ = load_model(run_dir, metrics, device)
        results = evaluate(backbone, proj, linear, val_loader, device)

        print_results(results, model_classes, eval_classes)

        # ── output folder: run_dir/eval/ ──
        eval_dir = run_dir / "eval"
        eval_dir.mkdir(exist_ok=True)

        # ── save aggregated metrics ──
        metrics_out = {
            "model_run"      : run_dir.name,
            "model_classes"  : model_classes,
            "eval_dataset"   : ds_name,
            "eval_classes"   : eval_classes,
            "label_mode"     : label_mode,
            "device"         : device,
            **{k: v for k, v in results.items()
               if not k.startswith("_")},
        }
        out_path = eval_dir / f"{ds_name}_{label_mode}.json"
        with open(out_path, "w") as f:
            json.dump(metrics_out, f, indent=2)
        print(f"  Saved  : {out_path.relative_to(BASE_OUT)}")

        # ── save per-image predictions ──
        per_image  = build_per_image_preds(results, val_loader, eval_classes, model_classes)
        preds_path = eval_dir / f"{ds_name}_{label_mode}_preds.json"
        with open(preds_path, "w") as f:
            json.dump(per_image, f, indent=2)
        n_wrong = sum(1 for r in per_image if not r["correct"])
        print(f"  Saved  : {preds_path.relative_to(BASE_OUT)}  "
              f"({len(per_image)} images, {n_wrong} wrong)")

        # ── save plots ──
        plot_path = save_eval_plots(results, eval_classes, model_classes,
                                    eval_dir, ds_name, label_mode, model_num)
        print(f"  Saved  : {plot_path.relative_to(BASE_OUT)}")

        all_results.append((model_num, run_dir.name, results))

    # ── Step 6: summary if multiple models ──
    if len(all_results) > 1:
        print(f"\n{'='*75}")
        print(f"  SUMMARY — {ds_name} ({label_mode})")
        print(f"{'='*75}")
        print(f"  {'#':>3}  {'Model':<38} {'Acc':>7} {'F1':>7} {'MCC':>7} {'AUC':>7} {'ms/img':>7}")
        print(f"  {'─'*3}  {'─'*38} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
        for num, name, r in all_results:
            short   = name[:37]
            auc_str = f"{r['auc_roc']:>6.3f}" if r.get("auc_roc") else "   n/a"
            print(f"  {num:>3}.  {short:<38} "
                  f"{r['acc']*100:>6.1f}% "
                  f"{r['f1_macro']*100:>6.1f}% "
                  f"{r['mcc']:>7.3f} "
                  f"{auc_str} "
                  f"{r['inference_ms_per_img']:>6.1f}ms")
        print(f"{'='*75}")


if __name__ == "__main__":
    main()
