"""
03_03_ensemble.py — Combine several trained models into an ensemble and
evaluate it on own_dataset (state mode: fresh / rotten).

The ensemble is built directly from the per-image probabilities saved by
03_01_evaluate.py (run_outputs/<run>/eval/own_dataset_state_preds.json),
so NO GPU and NO model reloading are needed — this is a pure post-processing
step over predictions that already exist.

Goal of this stage:
    Test whether combining diverse models reduces the domain shift observed
    on real photos (own_dataset) compared to the best single model.

Selection presets and combination methods are chosen interactively, or set
via flags for a non-interactive run.

Usage:
    python 03_03_ensemble.py
    python 03_03_ensemble.py --preset all --method mean
    python 03_03_ensemble.py --preset topk --topk 5 --method weighted --no-plot
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Windows consoles default to cp1252 — make box-drawing chars safe to print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, roc_auc_score,
    matthews_corrcoef,
)

# ── Config ────────────────────────────────────────────────────
RUN_DIR  = Path("run_outputs")
OUT_DIR  = RUN_DIR / "_ensemble"
EVAL_TAG = "own_dataset_state"          # label_mode = state
CLASSES  = ["fresh", "rotten"]          # index 0 / 1, positive = rotten

# ── Short ID mappings (shared with 04_01) ─────────────────────
DS_CODE = {
    "kaggle_fruits_quality"     : "KFQ",
    "mendeley_fruits"           : "MFR",
    "mendeley_lemon_varieties"  : "MLM",
    "mendeley_fruitvision"      : "MFV",
    "kaggle_fruits_fresh_rotten": "KFR",
    "kaggle_fresh_stale"        : "KFS",
    "own_dataset"               : "OWN",
}
BB_CODE = {
    "resnet18"          : "R18",
    "mobilenet_v3_small": "MN3",
    "efficientnet_b0"   : "EB0",
    "efficientnet_b2"   : "EB2",
}
COND_CODE = {
    "frozen"     : "C1",
    "layer4"     : "C2",
    "head_frozen": "C3",
    "head_layer4": "C4",
}


def short_id(ds, bb, cond):
    return (f"{DS_CODE.get(ds, ds[:3].upper())}-"
            f"{BB_CODE.get(bb, bb[:3].upper())}-"
            f"{COND_CODE.get(cond, cond)}-ST")


# ── Discovery ─────────────────────────────────────────────────

def load_members():
    """
    Scan run_outputs/ for models that have a state-mode evaluation on
    own_dataset. Returns a list of member records with metadata, validation
    F1 (from training), own_dataset F1 (from eval), and per-image probs.
    """
    members = []
    for run_dir in sorted(RUN_DIR.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue

        preds_path = run_dir / "eval" / f"{EVAL_TAG}_preds.json"
        eval_path  = run_dir / "eval" / f"{EVAL_TAG}.json"
        mpath      = run_dir / "metrics.json"
        if not (preds_path.exists() and eval_path.exists() and mpath.exists()):
            continue

        with open(eval_path)  as f: ev = json.load(f)
        with open(mpath)      as f: tr = json.load(f)
        with open(preds_path) as f: pr = json.load(f)

        # only binary fresh/rotten models can be combined here
        if ev.get("model_classes") != CLASSES:
            continue

        ds   = tr.get("dataset", "")
        bb   = tr.get("backbone_name", "resnet18")
        cond = tr.get("condition", "")

        preds = {
            p["file"]: {"true_label": p["true_label"], "probs": p["probs"]}
            for p in pr
        }

        members.append({
            "run"     : run_dir.name,
            "id"      : short_id(ds, bb, cond),
            "ds"      : ds,
            "bb"      : bb,
            "cond"    : cond,
            "val_f1"  : tr.get("best_f1", 0.0),     # F1 on source val set
            "eval_f1" : ev.get("f1_macro", 0.0),    # F1 on own_dataset
            "preds"   : preds,
        })
    return members


# ── Combination ───────────────────────────────────────────────

def combine(members, method):
    """
    Combine member probabilities into one ensemble prediction per image.

    method:
      mean     — simple average of softmax vectors
      weighted — average weighted by each model's validation F1
      adaptive — per-image weighting by each model's own confidence
                 (max prob); uncertain models contribute less
    """
    # images present in every member (intersection, aligned by filename)
    files = set(members[0]["preds"].keys())
    for m in members[1:]:
        files &= set(m["preds"].keys())
    files = sorted(files)

    if method == "weighted":
        base_w = np.array([max(m["val_f1"], 1e-6) for m in members])
    else:
        base_w = np.ones(len(members))

    y_true, y_pred, y_prob = [], [], []
    for fname in files:
        vecs, ws = [], []
        for m, bw in zip(members, base_w):
            p = m["preds"][fname]["probs"]
            vec = [p[CLASSES[0]], p[CLASSES[1]]]
            vecs.append(vec)
            ws.append(max(vec) if method == "adaptive" else bw)
        vecs = np.array(vecs)                       # [N, 2]
        ws   = np.array(ws, dtype=float)
        ws   = ws / ws.sum()
        ens  = (vecs * ws[:, None]).sum(axis=0)     # [2]

        true = members[0]["preds"][fname]["true_label"]
        y_true.append(CLASSES.index(true))
        y_pred.append(int(np.argmax(ens)))
        y_prob.append(ens[1])                       # P(rotten)

    return files, np.array(y_true), np.array(y_pred), np.array(y_prob)


def metrics_from(y_true, y_pred, y_prob):
    out = {
        "n_images" : int(len(y_true)),
        "acc"      : round(accuracy_score(y_true, y_pred), 4),
        "f1_macro" : round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro"   : round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "mcc"      : round(matthews_corrcoef(y_true, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    try:
        out["auc_roc"] = round(roc_auc_score(y_true, y_prob), 4)
    except ValueError:
        out["auc_roc"] = None
    return out


# ── Selection presets ─────────────────────────────────────────

def parse_selection(user_input, n):
    s = user_input.strip().lower()
    if s == "all":
        return list(range(1, n + 1))
    sel = set()
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            sel.update(range(int(a), int(b) + 1))
        elif part:
            sel.add(int(part))
    bad = [x for x in sel if x < 1 or x > n]
    if bad:
        raise ValueError(f"Out of range: {bad}")
    return sorted(sel)


def choose_members(members, args):
    """Interactive (or flag-driven) selection of which members to ensemble."""
    if args.preset == "all":
        return members, "all"

    if args.preset == "topk":
        k = args.topk
        ranked = sorted(members, key=lambda m: m["eval_f1"], reverse=True)[:k]
        return ranked, f"top{k}"

    # interactive menu
    print("\nEnsemble model selection:")
    print(f"  1. All available models ({len(members)})")
    print( "  2. Top-K by own_dataset F1")
    print( "  3. By source dataset + condition  (the 4 backbones)")
    print( "  4. By backbone + condition        (across datasets)")
    print( "  5. Manual selection")
    choice = input("Select [1]: ").strip() or "1"

    if choice == "1":
        return members, "all"

    if choice == "2":
        k = int(input("K [5]: ").strip() or "5")
        ranked = sorted(members, key=lambda m: m["eval_f1"], reverse=True)[:k]
        return ranked, f"top{k}"

    if choice == "3":
        groups = {}
        for m in members:
            groups.setdefault((m["ds"], m["cond"]), []).append(m)
        groups = {k: v for k, v in groups.items() if len(v) >= 2}
        keys = sorted(groups)
        for i, (ds, cond) in enumerate(keys, 1):
            print(f"  {i}. {DS_CODE.get(ds, ds)}-{COND_CODE.get(cond, cond)}  "
                  f"({len(groups[(ds, cond)])} models)")
        idx = int(input("Select group [1]: ").strip() or "1") - 1
        ds, cond = keys[idx]
        return groups[(ds, cond)], f"arch_{DS_CODE.get(ds, ds)}_{COND_CODE.get(cond, cond)}"

    if choice == "4":
        groups = {}
        for m in members:
            groups.setdefault((m["bb"], m["cond"]), []).append(m)
        groups = {k: v for k, v in groups.items() if len(v) >= 2}
        keys = sorted(groups)
        for i, (bb, cond) in enumerate(keys, 1):
            print(f"  {i}. {BB_CODE.get(bb, bb)}-{COND_CODE.get(cond, cond)}  "
                  f"({len(groups[(bb, cond)])} models)")
        idx = int(input("Select group [1]: ").strip() or "1") - 1
        bb, cond = keys[idx]
        return groups[(bb, cond)], f"source_{BB_CODE.get(bb, bb)}_{COND_CODE.get(cond, cond)}"

    # manual
    for i, m in enumerate(members, 1):
        print(f"  {i:3d}. {m['id']:<18} own_F1={m['eval_f1']*100:5.1f}%  val_F1={m['val_f1']*100:5.1f}%")
    raw = input("Select (e.g. 1,3,5 or 1-4): ").strip()
    picks = parse_selection(raw, len(members))
    return [members[i - 1] for i in picks], "manual"


def choose_method(args):
    if args.method:
        return args.method
    print("\nCombination method:")
    print("  1. Mean softmax            (simple average)")
    print("  2. Weighted by validation F1")
    print("  3. Confidence-adaptive     (per-image confidence weighting)")
    c = input("Select [1]: ").strip() or "1"
    return {"1": "mean", "2": "weighted", "3": "adaptive"}[c]


# ── Plot ──────────────────────────────────────────────────────

def make_plot(members, ens, out_path):
    """Bar chart: each member's own_dataset F1 vs the ensemble F1."""
    labels = [m["id"] for m in members] + ["ENSEMBLE"]
    vals   = [m["eval_f1"] * 100 for m in members] + [ens["f1_macro"] * 100]
    colors = ["#4C72B0"] * len(members) + ["#C44E52"]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 4.5))
    bars = ax.bar(range(len(labels)), vals, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("F1 macro on own_dataset (%)")
    ax.set_ylim(0, 108)
    best = max(m["eval_f1"] for m in members) * 100
    ax.axhline(best, ls="--", lw=1, color="grey",
               label=f"best single = {best:.1f}%")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}",
                ha="center", va="bottom", fontsize=7)
    ax.set_title("Ensemble vs individual models")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["all", "topk"],
                    help="Non-interactive selection preset.")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--method", choices=["mean", "weighted", "adaptive"])
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    print("\n── Model Ensemble — own_dataset (state) ──\n")
    members = load_members()
    if len(members) < 2:
        print(f"Need at least 2 evaluated models, found {len(members)}.")
        print("Run 03_01_evaluate.py on own_dataset first.")
        return
    print(f"Found {len(members)} models with own_dataset evaluation.")

    chosen, tag = choose_members(members, args)
    if len(chosen) < 2:
        print("Need at least 2 models in the ensemble.")
        return
    method = choose_method(args)

    files, y_true, y_pred, y_prob = combine(chosen, method)
    ens = metrics_from(y_true, y_pred, y_prob)

    # ── Report ──
    best_member = max(chosen, key=lambda m: m["eval_f1"])
    delta = (ens["f1_macro"] - best_member["eval_f1"]) * 100

    print(f"\n{'='*64}")
    print(f"  Ensemble: {len(chosen)} models | method={method} | {len(files)} images")
    print(f"{'='*64}")
    print(f"  {'Model':<20}{'own_F1':>9}{'val_F1':>9}")
    for m in sorted(chosen, key=lambda x: x["eval_f1"], reverse=True):
        print(f"  {m['id']:<20}{m['eval_f1']*100:>8.1f}%{m['val_f1']*100:>8.1f}%")
    print(f"  {'-'*38}")
    print(f"  {'ENSEMBLE':<20}{ens['f1_macro']*100:>8.1f}%")
    print(f"\n  Best single model : {best_member['id']} "
          f"({best_member['eval_f1']*100:.1f}%)")
    print(f"  Ensemble F1       : {ens['f1_macro']*100:.1f}%")
    print(f"  Delta vs best     : {delta:+.1f} pts")
    print(f"  Ensemble ACC/MCC  : {ens['acc']*100:.1f}% / {ens['mcc']:.3f}")
    auc_str = f"{ens['auc_roc']:.4f}" if ens["auc_roc"] is not None else "n/a"
    print(f"  Ensemble AUC-ROC  : {auc_str}")

    # ── Save ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "tag"          : tag,
        "method"       : method,
        "n_members"    : len(chosen),
        "members"      : [
            {"id": m["id"], "run": m["run"],
             "val_f1": m["val_f1"], "eval_f1": m["eval_f1"]}
            for m in chosen
        ],
        "best_single"  : {"id": best_member["id"], "f1": best_member["eval_f1"]},
        "ensemble"     : ens,
        "delta_f1_pts" : round(delta, 2),
    }
    json_path = OUT_DIR / f"ensemble_{tag}_{method}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {json_path}")

    if not args.no_plot:
        png_path = OUT_DIR / f"ensemble_{tag}_{method}.png"
        make_plot(chosen, ens, png_path)
        print(f"  Saved: {png_path}")


if __name__ == "__main__":
    main()
