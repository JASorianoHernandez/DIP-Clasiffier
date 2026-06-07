"""
02_02_finetune.py — Domain-adaptation fine-tuning on own_dataset.

Takes a model already trained on a public dataset (its best_model.pt) and
continues training it on own_dataset (the real photos), so it learns *our*
definition of early rot and *our* backgrounds — the domain gap surfaced by
03_04_error_analysis.py.

Evaluation uses stratified k-fold cross-validation: with only ~121 photos a
single train/test split gives a noisy number, so every photo is held out
exactly once (across folds) and the test predictions are pooled into one
aggregate score over all images. For each fold we measure the SAME model
before fine-tuning (baseline) and after, so the delta is honest.

Augmentation is geometry-heavy but color-light: the rot signal (small mold
spots, subtle hue) is fragile, so we keep ColorJitter soft and drop blur.

Outputs: run_outputs/_finetune/finetune_{source_id}.json  (+ .png)
Optionally (--save-final) trains one model on all 121 photos for deployment.

Usage:
    python 02_02_finetune.py
    python 02_02_finetune.py --run kaggle_fruits_quality_all_head_frozen_efficientnet_b2 --epochs 25
    python 02_02_finetune.py --folds 5 --save-final
"""

import sys
import json
import time
import argparse
import importlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    matthews_corrcoef, confusion_matrix,
)

from _backbone import get_backbone
from _dataset import NestedFruitDataset, IMAGENET_MEAN, IMAGENET_STD, VAL_TRANSFORMS

_train = importlib.import_module("02_01_train")   # ProjectionHead, CONFIG
ProjectionHead = _train.ProjectionHead

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Config ────────────────────────────────────────────────────
RUN_DIR     = Path("run_outputs")
OUT_DIR     = RUN_DIR / "_finetune"
OWN_DATASET = "./data/own_dataset"
CLASSES     = ["fresh", "rotten"]
ROTTEN_IDX  = 1

FT = {
    "epochs"       : 25,
    "batch_size"   : 16,
    "lr_head"      : 1e-4,   # 10x lower than from-scratch training (1e-3)
    "lr_backbone"  : 1e-5,
    "unfreeze"     : 1,      # unfreeze layer4 for domain adaptation
    "weight_decay" : 1e-4,
    "folds"        : 5,
    "seed"         : 42,
}

# Geometry-heavy, color-light augmentation (preserve the fragile rot signal)
FT_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

DS_CODE = {"kaggle_fruits_quality":"KFQ","mendeley_fruits":"MFR",
           "mendeley_lemon_varieties":"MLM","mendeley_fruitvision":"MFV",
           "kaggle_fruits_fresh_rotten":"KFR","kaggle_fresh_stale":"KFS"}
BB_CODE = {"resnet18":"R18","mobilenet_v3_small":"MN3",
           "efficientnet_b0":"EB0","efficientnet_b2":"EB2"}
COND_CODE = {"frozen":"C1","layer4":"C2","head_frozen":"C3","head_layer4":"C4"}


def short_id(m):
    return (f"{DS_CODE.get(m['dataset'], m['dataset'][:3].upper())}-"
            f"{BB_CODE.get(m['backbone_name'], m['backbone_name'][:3].upper())}-"
            f"{COND_CODE.get(m['condition'], m['condition'])}-ST")


# ── Source-model discovery ────────────────────────────────────

def find_source_models():
    """Binary fresh/rotten state models that can adapt to own_dataset."""
    out = []
    for run_dir in sorted(RUN_DIR.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        mpath = run_dir / "metrics.json"
        bpath = run_dir / "best_model.pt"
        if not (mpath.exists() and bpath.exists()):
            continue
        with open(mpath) as f:
            m = json.load(f)
        if sorted(m.get("class_names", [])) != sorted(CLASSES):
            continue
        # own_dataset eval F1 (if available) — best starting point
        own_f1 = None
        epath = run_dir / "eval" / "own_dataset_state.json"
        if epath.exists():
            with open(epath) as f:
                own_f1 = json.load(f).get("f1_macro")
        out.append({"run_dir": run_dir, "metrics": m,
                    "id": short_id(m), "own_f1": own_f1})
    return out


# ── Model build / load ────────────────────────────────────────

def build_and_load(source, device):
    """Reconstruct the source model with layer4 unfrozen for fine-tuning."""
    m        = source["metrics"]
    bb_name  = m["backbone_name"]
    use_head = m["use_head"]

    backbone = get_backbone(device, backbone_name=bb_name,
                            pretrained=False, unfreeze_layers=FT["unfreeze"])
    if use_head:
        proj   = ProjectionHead(in_dim=backbone.out_dim,
                                out_dim=_train.CONFIG["embedding_dim"]).to(device)
        linear = nn.Linear(_train.CONFIG["embedding_dim"], len(CLASSES)).to(device)
        head_params = list(proj.parameters()) + list(linear.parameters())
    else:
        proj   = None
        linear = nn.Linear(backbone.out_dim, len(CLASSES)).to(device)
        head_params = list(linear.parameters())

    ckpt = torch.load(source["run_dir"] / "best_model.pt",
                      map_location=device, weights_only=False)
    backbone.load_state_dict(ckpt["backbone"])
    linear.load_state_dict(ckpt["linear"])
    if proj is not None and ckpt.get("proj") is not None:
        proj.load_state_dict(ckpt["proj"])
    return backbone, proj, linear, head_params


@torch.no_grad()
def predict(backbone, proj, linear, loader, device):
    backbone.eval(); linear.eval()
    if proj is not None: proj.eval()
    labels, preds, probs = [], [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        feats  = backbone(imgs)
        emb    = proj(feats) if proj is not None else feats
        p      = torch.softmax(linear(emb), dim=1)
        preds.extend(p.argmax(1).cpu().numpy())
        probs.extend(p[:, ROTTEN_IDX].cpu().numpy())
        labels.extend(lbls.numpy())
    return np.array(labels), np.array(preds), np.array(probs)


def fine_tune_one(backbone, proj, linear, head_params, train_loader, device):
    if backbone.trainable_params():
        optimizer = optim.Adam([
            {"params": backbone.trainable_params(), "lr": FT["lr_backbone"]},
            {"params": head_params,                 "lr": FT["lr_head"]},
        ], weight_decay=FT["weight_decay"])
    else:
        optimizer = optim.Adam(head_params, lr=FT["lr_head"],
                               weight_decay=FT["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FT["epochs"])
    criterion = nn.CrossEntropyLoss()

    for _ in range(FT["epochs"]):
        backbone.train(); linear.train()
        if proj is not None: proj.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            feats  = backbone(imgs)
            emb    = proj(feats) if proj is not None else feats
            loss   = criterion(linear(emb), lbls)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()


def metrics_of(labels, preds, probs):
    return {
        "acc"          : round(float(accuracy_score(labels, preds)), 4),
        "f1_macro"     : round(float(f1_score(labels, preds, average="macro", zero_division=0)), 4),
        "mcc"          : round(float(matthews_corrcoef(labels, preds)), 4),
        "recall_rotten": round(float(recall_score(labels, preds, labels=[ROTTEN_IDX],
                                                   average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
    }


# ── Plot ──────────────────────────────────────────────────────

def make_plot(base, ft, source_id, out_path, n_photos):
    keys   = ["f1_macro", "acc", "mcc", "recall_rotten"]
    names  = ["F1 macro", "Accuracy", "MCC", "Recall (rotten)"]
    bvals  = [base[k] for k in keys]
    fvals  = [ft[k]   for k in keys]
    x = np.arange(len(keys)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, bvals, w, label="Baseline (no fine-tune)", color="#4C72B0")
    b2 = ax.bar(x + w/2, fvals, w, label="Fine-tuned on own_dataset", color="#C44E52")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05); ax.set_ylabel("score")
    ax.set_title(f"Fine-tuning on own_dataset — {source_id}\n"
                 f"({FT['folds']}-fold CV, pooled over {n_photos} photos)")
    ax.legend()
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


# ── Main ──────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=None,
                    help="Source run dir name. Omit for interactive selection.")
    ap.add_argument("--folds", type=int, default=FT["folds"])
    ap.add_argument("--epochs", type=int, default=FT["epochs"])
    ap.add_argument("--freeze", action="store_true",
                    help="Freeze backbone (linear-probe fine-tune) — safer on tiny data.")
    ap.add_argument("--lr-head", type=float, default=FT["lr_head"])
    ap.add_argument("--save-final", action="store_true",
                    help="After CV, train one model on all 121 photos for deployment.")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    FT["folds"]  = args.folds
    FT["epochs"] = args.epochs
    FT["lr_head"] = args.lr_head
    if args.freeze:
        FT["unfreeze"] = 0

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n── Fine-tuning on own_dataset ──   device: {device}\n")

    # ── source model ──
    sources = find_source_models()
    if not sources:
        print("No compatible fresh/rotten models found.")
        return
    if args.run:
        sel = next((s for s in sources if s["run_dir"].name == args.run), None)
        if sel is None:
            print(f"Run not found among compatible models: {args.run}")
            return
    else:
        ranked = sorted(sources, key=lambda s: (s["own_f1"] is not None, s["own_f1"] or 0),
                        reverse=True)
        print("Compatible source models (best own_dataset F1 first):")
        for i, s in enumerate(ranked[:15], 1):
            of = f"{s['own_f1']*100:.1f}%" if s["own_f1"] is not None else "n/a"
            star = "  ★ recommended" if i == 1 else ""
            print(f"  {i:2d}. {s['id']:<18} own_F1={of:>6}  {s['run_dir'].name}{star}")
        raw = input("\nSelect source model [1]: ").strip() or "1"
        sel = ranked[int(raw) - 1]
    print(f"\n  Source: {sel['id']}  ({sel['run_dir'].name})")
    base_own = f"{sel['own_f1']*100:.1f}%" if sel["own_f1"] is not None else "n/a"
    print(f"  Baseline own_dataset F1 (full eval): {base_own}\n")

    # ── data + folds ──
    full_train = NestedFruitDataset(Path(OWN_DATASET), transform=FT_TRANSFORMS, label_mode="state")
    full_eval  = NestedFruitDataset(Path(OWN_DATASET), transform=VAL_TRANSFORMS, label_mode="state")
    targets = np.array(full_train.targets)
    if full_train.classes != CLASSES:
        print(f"  [WARN] class order is {full_train.classes}, expected {CLASSES}")
    skf = StratifiedKFold(n_splits=FT["folds"], shuffle=True, random_state=FT["seed"])

    # pooled predictions across folds
    base_lab, base_prd, base_prb = [], [], []
    ft_lab,   ft_prd,   ft_prb   = [], [], []
    per_fold = []
    t0 = time.time()

    for k, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(targets)), targets), 1):
        torch.manual_seed(FT["seed"] + k)
        train_loader = DataLoader(Subset(full_train, tr_idx), batch_size=FT["batch_size"],
                                  shuffle=True, num_workers=0)
        test_loader  = DataLoader(Subset(full_eval, te_idx), batch_size=FT["batch_size"],
                                  shuffle=False, num_workers=0)

        backbone, proj, linear, head_params = build_and_load(sel, device)

        # baseline (source weights, before any fine-tuning) on this fold's test
        bl, bp, bpr = predict(backbone, proj, linear, test_loader, device)
        # fine-tune, then re-evaluate the SAME held-out test
        fine_tune_one(backbone, proj, linear, head_params, train_loader, device)
        fl, fp, fpr = predict(backbone, proj, linear, test_loader, device)

        base_lab += bl.tolist(); base_prd += bp.tolist(); base_prb += bpr.tolist()
        ft_lab   += fl.tolist(); ft_prd   += fp.tolist(); ft_prb   += fpr.tolist()
        f1b = f1_score(bl, bp, average="macro", zero_division=0)
        f1f = f1_score(fl, fp, average="macro", zero_division=0)
        per_fold.append({"fold": k, "n_test": len(te_idx),
                         "f1_base": round(float(f1b), 4), "f1_ft": round(float(f1f), 4)})
        print(f"  Fold {k}/{FT['folds']}  n_test={len(te_idx):3d}  "
              f"F1 base={f1b*100:5.1f}%  →  ft={f1f*100:5.1f}%  ({(f1f-f1b)*100:+.1f})")

    base_m = metrics_of(np.array(base_lab), np.array(base_prd), np.array(base_prb))
    ft_m   = metrics_of(np.array(ft_lab),   np.array(ft_prd),   np.array(ft_prb))
    elapsed = round(time.time() - t0, 1)

    # ── report ──
    print(f"\n{'='*60}")
    print(f"  POOLED over {len(base_lab)} photos ({FT['folds']}-fold CV, {elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"  {'metric':<16}{'baseline':>10}{'fine-tuned':>12}{'delta':>9}")
    for k, name in [("f1_macro","F1 macro"),("acc","Accuracy"),
                    ("mcc","MCC"),("recall_rotten","Recall rotten")]:
        d = ft_m[k] - base_m[k]
        print(f"  {name:<16}{base_m[k]:>10.4f}{ft_m[k]:>12.4f}{d:>+9.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "source_id": sel["id"], "source_run": sel["run_dir"].name,
        "n_photos": len(base_lab), "folds": FT["folds"], "epochs": FT["epochs"],
        "ft_config": FT, "per_fold": per_fold,
        "baseline": base_m, "finetuned": ft_m,
        "delta_f1_pts": round((ft_m["f1_macro"] - base_m["f1_macro"]) * 100, 2),
    }
    jpath = OUT_DIR / f"finetune_{sel['id']}.json"
    with open(jpath, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {jpath}")
    if not args.no_plot:
        ppath = OUT_DIR / f"finetune_{sel['id']}.png"
        make_plot(base_m, ft_m, sel["id"], ppath, len(base_lab))
        print(f"  Saved: {ppath}")

    # ── optional: deployable model trained on ALL photos ──
    if args.save_final:
        print(f"\n  Training final model on all {len(targets)} photos for deployment...")
        torch.manual_seed(FT["seed"])
        loader = DataLoader(full_train, batch_size=FT["batch_size"], shuffle=True, num_workers=0)
        backbone, proj, linear, head_params = build_and_load(sel, device)
        fine_tune_one(backbone, proj, linear, head_params, loader, device)
        final_dir = RUN_DIR / f"own_dataset_finetune_from_{sel['id']}"
        final_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"backbone": backbone.state_dict(),
                    "linear": linear.state_dict(),
                    "proj": proj.state_dict() if proj is not None else None,
                    "class_names": CLASSES}, final_dir / "best_model.pt")
        print(f"  Saved deployable model: {final_dir/'best_model.pt'}")


if __name__ == "__main__":
    main()
