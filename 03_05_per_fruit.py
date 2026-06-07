"""
03_05_per_fruit.py — Per-fruit breakdown of own_dataset evaluation (state mode).

own_dataset now holds more than one fruit (strawberry, banana, ...). The state
classifier is fruit-agnostic (fresh / rotten), but a model may handle one fruit
far better than another — especially if its training dataset never contained
that fruit. This splits each model's predictions by fruit (parsed from the
filename code, e.g. FR_SB_001 → strawberry, RT_BN_007 → banana) and reports
F1 per fruit, so we can see who generalizes across fruits and who is lopsided.

Reads the per-image predictions saved by 03_01_evaluate.py — no GPU needed.

Usage:
    python 03_05_per_fruit.py
    python 03_05_per_fruit.py --no-plot
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import f1_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RUN_DIR  = Path("run_outputs")
OUT_DIR  = RUN_DIR / "_per_fruit"
EVAL_TAG = "own_dataset_state"
CLASSES  = ["fresh", "rotten"]

FRUIT_CODE = {"SB": "strawberry", "BN": "banana"}

DS_CODE = {"kaggle_fruits_quality":"KFQ","mendeley_fruits":"MFR",
           "mendeley_lemon_varieties":"MLM","mendeley_fruitvision":"MFV",
           "kaggle_fruits_fresh_rotten":"KFR","kaggle_fresh_stale":"KFS"}
BB_CODE = {"resnet18":"R18","mobilenet_v3_small":"MN3",
           "efficientnet_b0":"EB0","efficientnet_b2":"EB2"}
COND_CODE = {"frozen":"C1","layer4":"C2","head_frozen":"C3","head_layer4":"C4"}

# Which public datasets actually contain bananas (for the "did it ever see this
# fruit in training?" annotation). Verified by inspecting data/: KFR/KFS/MFV
# have banana folders; KFQ has banana images mixed into its flat fresh/rotten
# split (12 files named *banana*). Only MFR (peach/pomegranate/strawberry) and
# MLM (lemons) never contain banana.
DATASETS_WITH_BANANA = {
    "kaggle_fruits_fresh_rotten", "kaggle_fresh_stale",
    "mendeley_fruitvision", "kaggle_fruits_quality",
}


def short_id(m):
    return (f"{DS_CODE.get(m['dataset'], m['dataset'][:3].upper())}-"
            f"{BB_CODE.get(m['backbone_name'], m['backbone_name'][:3].upper())}-"
            f"{COND_CODE.get(m['condition'], m['condition'])}-ST")


def fruit_of(fname):
    """FR_SB_001.jpg -> 'strawberry' via the 2-letter fruit code."""
    parts = fname.split("_")
    code = parts[1] if len(parts) >= 2 else "??"
    return FRUIT_CODE.get(code, code.lower())


def load_models():
    """Return list of {id, dataset, saw_banana, preds:list}."""
    models = []
    for run_dir in sorted(RUN_DIR.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        p = run_dir / "eval" / f"{EVAL_TAG}_preds.json"
        e = run_dir / "eval" / f"{EVAL_TAG}.json"
        m = run_dir / "metrics.json"
        if not (p.exists() and e.exists() and m.exists()):
            continue
        with open(e) as f:
            if json.load(f).get("model_classes") != CLASSES:
                continue
        with open(m) as f:
            meta = json.load(f)
        with open(p) as f:
            recs = json.load(f)
        models.append({
            "id"        : short_id(meta),
            "dataset"   : meta["dataset"],
            "saw_banana": meta["dataset"] in DATASETS_WITH_BANANA,
            "preds"     : recs,
        })
    return models


def f1_on(recs):
    """Macro F1 over a list of prediction records."""
    if not recs:
        return None
    y_true = [CLASSES.index(r["true_label"]) for r in recs]
    y_pred = [CLASSES.index(r["pred_label"]) for r in recs]
    return round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)


def analyze(models):
    fruits = sorted({fruit_of(r["file"]) for m in models for r in m["preds"]})
    rows = []
    for m in models:
        by_fruit = {fr: [] for fr in fruits}
        for r in m["preds"]:
            by_fruit[fruit_of(r["file"])].append(r)
        row = {"id": m["id"], "dataset": m["dataset"], "saw_banana": m["saw_banana"],
               "f1_overall": f1_on(m["preds"])}
        for fr in fruits:
            row[f"f1_{fr}"] = f1_on(by_fruit[fr])
            row[f"n_{fr}"]  = len(by_fruit[fr])
        rows.append(row)
    return rows, fruits


def make_plot(rows, out_path):
    xs = [r.get("f1_strawberry") for r in rows]
    ys = [r.get("f1_banana") for r in rows]
    pts = [(x, y, r) for x, y, r in zip(xs, ys, rows) if x is not None and y is not None]
    if not pts:
        return False
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="grey", zorder=0)
    for x, y, r in pts:
        c = "#C44E52" if r["saw_banana"] else "#4C72B0"
        ax.scatter(x, y, color=c, s=35, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.scatter([], [], color="#C44E52", label="trained WITH banana")
    ax.scatter([], [], color="#4C72B0", label="never saw banana")
    mx = float(np.mean([x for x, _, _ in pts])); my = float(np.mean([y for _, y, _ in pts]))
    ax.scatter(mx, my, marker="*", s=400, color="gold", edgecolor="black",
               zorder=5, label=f"fleet mean ({mx:.2f}, {my:.2f})")
    ax.set_xlabel("F1 on strawberry"); ax.set_ylabel("F1 on banana")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_title("Per-model F1: strawberry vs banana\n"
                 "(below diagonal = better on strawberry than banana)")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    print("\n── Per-fruit evaluation breakdown — own_dataset (state) ──\n")
    models = load_models()
    if not models:
        print("No evaluated models found. Run 03_01_evaluate.py first.")
        return
    rows, fruits = analyze(models)
    print(f"Models: {len(rows)} | Fruits: {fruits}\n")

    # fleet means per fruit
    print("  Fleet mean F1 per fruit (across all models):")
    for fr in fruits + ["overall"]:
        key = f"f1_{fr}" if fr != "overall" else "f1_overall"
        vals = [r[key] for r in rows if r.get(key) is not None]
        n = next((r.get(f"n_{fr}") for r in rows if fr != "overall"), None)
        ntag = f"  (n={n} imgs/model)" if n else ""
        if vals:
            print(f"    {fr:<12} {np.mean(vals)*100:5.1f}%{ntag}")

    # models with the biggest strawberry→banana gap
    if "f1_strawberry" in rows[0] and "f1_banana" in rows[0]:
        for r in rows:
            sb, bn = r.get("f1_strawberry"), r.get("f1_banana")
            r["gap"] = (sb - bn) if (sb is not None and bn is not None) else None
        gapped = [r for r in rows if r["gap"] is not None]

        print(f"\n  Most lopsided (strawberry ≫ banana) — top {args.top}:")
        print(f"  {'model':<18}{'straw':>7}{'banana':>8}{'gap':>7}{'saw_BN':>8}")
        for r in sorted(gapped, key=lambda x: -x["gap"])[:args.top]:
            print(f"  {r['id']:<18}{r['f1_strawberry']*100:>6.1f}%{r['f1_banana']*100:>7.1f}%"
                  f"{r['gap']*100:>+6.1f}{'yes' if r['saw_banana'] else 'no':>8}")

        # does training-with-banana help banana F1?
        with_bn    = [r["f1_banana"] for r in gapped if r["saw_banana"]]
        without_bn = [r["f1_banana"] for r in gapped if not r["saw_banana"]]
        if with_bn and without_bn:
            print(f"\n  Banana F1 by training exposure:")
            print(f"    models trained WITH banana : {np.mean(with_bn)*100:5.1f}%  (n={len(with_bn)})")
            print(f"    models that never saw banana: {np.mean(without_bn)*100:5.1f}%  (n={len(without_bn)})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "per_fruit.json", "w") as f:
        json.dump({"fruits": fruits, "models": rows}, f, indent=2)
    print(f"\n  Saved: {OUT_DIR/'per_fruit.json'}")
    if not args.no_plot and make_plot(rows, OUT_DIR / "per_fruit.png"):
        print(f"  Saved: {OUT_DIR/'per_fruit.png'}")


if __name__ == "__main__":
    main()
