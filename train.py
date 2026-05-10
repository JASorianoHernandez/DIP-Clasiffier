"""
train.py — Food/Fruit Freshness Classifier
==========================================
Transfer learning with CrossEntropy over a pretrained backbone.
Adapted from the NMFC baseline for ImageFolder datasets.

Four conditions (ablation study):
  frozen      — backbone frozen + Linear head
  layer4      — last backbone block unfrozen + Linear head
  head_frozen — backbone frozen + ProjectionHead(→128) + Linear
  head_layer4 — last backbone block unfrozen + ProjectionHead + Linear

Checkpointing:
  - checkpoint.pt            rolling save every epoch (resume from last point)
  - checkpoint_epoch_N.pt    snapshot every save_every epochs (permanent, never overwritten)
  - best_model.pt            saved whenever val accuracy improves
  - metrics.json             written at the end of training

  If a checkpoint.pt already exists in the run dir, training resumes automatically.

Usage:
    python train.py
    python train.py --dataset_path ./AnusDraft/korean_fruit_dataset

Output: run_outputs/<dataset>_<condition>_<backbone>/
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix,
)

from dataset import get_loaders, get_loaders_nested
from backbone import get_backbone, BACKBONE_NAMES, BACKBONE_REGISTRY

# ─────────────────────────────────────────────────────────────
# Conditions: (unfreeze_layers, use_projection_head)
# ─────────────────────────────────────────────────────────────

CONDITIONS = ["frozen", "layer4", "head_frozen", "head_layer4"]

COND_CFG = {
    "frozen"     : (0, False),
    "layer4"     : (1, False),
    "head_frozen": (0, True),
    "head_layer4": (1, True),
}

CONFIG = {
    "epochs"        : 50,
    "batch_size"    : 32,
    "num_workers"   : 4,
    "lr"            : 1e-3,
    "lr_backbone"   : 1e-5,
    "embedding_dim" : 128,
    "eval_every"    : 5,   # evaluate val accuracy every N epochs
    "save_every"    : 10,  # save a named snapshot every N epochs
    "weight_decay"  : 1e-4,
    "val_split"     : 0.2,
    "seed"          : 42,
}

BASE_OUT = os.path.join("run_outputs")

CHECKPOINT_FILE = "checkpoint.pt"
BEST_MODEL_FILE = "best_model.pt"


# ─────────────────────────────────────────────────────────────
# Projection head — same as NMFC baseline
# ─────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(backbone, proj, linear, loader, device, unfreeze, criterion):
    """
    Returns a dict with all validation metrics:
      acc, val_loss, f1_macro, precision_macro, recall_macro,
      f1_per_class, precision_per_class, recall_per_class, confusion_matrix
    """
    backbone.eval()
    if proj is not None:
        proj.eval()
    linear.eval()

    all_preds, all_labels = [], []
    total_loss, steps = 0.0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        feats  = backbone(images)
        emb    = proj(feats) if proj is not None else feats
        logits = linear(emb)
        total_loss += criterion(logits, labels).item()
        steps      += 1
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    if unfreeze > 0:
        backbone.train()
    if proj is not None:
        proj.train()
    linear.train()

    return {
        "acc"               : round(float(accuracy_score(all_labels, all_preds)), 6),
        "val_loss"          : round(total_loss / steps, 6),
        "f1_macro"          : round(float(f1_score(all_labels, all_preds, average="macro",    zero_division=0)), 6),
        "precision_macro"   : round(float(precision_score(all_labels, all_preds, average="macro", zero_division=0)), 6),
        "recall_macro"      : round(float(recall_score(all_labels, all_preds, average="macro",    zero_division=0)), 6),
        "f1_per_class"      : [round(v, 6) for v in f1_score(all_labels, all_preds,        average=None, zero_division=0).tolist()],
        "precision_per_class": [round(v, 6) for v in precision_score(all_labels, all_preds, average=None, zero_division=0).tolist()],
        "recall_per_class"  : [round(v, 6) for v in recall_score(all_labels, all_preds,    average=None, zero_division=0).tolist()],
        "confusion_matrix"  : confusion_matrix(all_labels, all_preds).tolist(),
    }


# ─────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────

def _save_checkpoint(run_dir, epoch, backbone, proj, linear, optimizer, scheduler,
                     best_acc, acc_history, class_names,
                     filename=CHECKPOINT_FILE, best_f1=0.0):
    ckpt = {
        "epoch"       : epoch,
        "backbone"    : backbone.state_dict(),
        "linear"      : linear.state_dict(),
        "proj"        : proj.state_dict() if proj is not None else None,
        "optimizer"   : optimizer.state_dict(),
        "scheduler"   : scheduler.state_dict(),
        "best_acc"    : best_acc,
        "best_f1"     : best_f1,
        "acc_history" : acc_history,
        "class_names" : class_names,
    }
    torch.save(ckpt, os.path.join(run_dir, filename))


def train(dataset_path, condition, backbone_name="resnet18", fruits=None, label_mode="state"):
    unfreeze, use_head = COND_CFG[condition]

    device = (
        "cuda"  if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # Deterministic run_dir — no timestamp — so resume always finds the same folder
    dataset_label = Path(dataset_path).name
    fruits_tag    = "_".join(fruits) if fruits else "all"
    mode_suffix   = "_fs" if label_mode == "fruit_state" else ""
    run_dir       = os.path.join(BASE_OUT, f"{dataset_label}_{fruits_tag}_{condition}_{backbone_name}{mode_suffix}")
    os.makedirs(run_dir, exist_ok=True)

    ckpt_path = os.path.join(run_dir, CHECKPOINT_FILE)
    resuming  = os.path.exists(ckpt_path)

    # Auto-detect layout:
    #   Layout B — has train/ + test/ subfolders (get_loaders)
    #   Layout C — fruit/state nested structure   (get_loaders_nested)
    p = Path(dataset_path)
    has_split = (p / "train").exists() and (p / "test").exists()
    use_nested = not has_split
    if use_nested:
        train_loader, val_loader, num_classes, class_names = get_loaders_nested(
            dataset_path,
            fruits=fruits,
            batch_size=CONFIG["batch_size"],
            num_workers=CONFIG["num_workers"],
            val_split=CONFIG["val_split"],
            seed=CONFIG["seed"],
            label_mode=label_mode,
        )
    else:
        train_loader, val_loader, num_classes, class_names = get_loaders(
            dataset_path,
            batch_size=CONFIG["batch_size"],
            num_workers=CONFIG["num_workers"],
            val_split=CONFIG["val_split"],
            seed=CONFIG["seed"],
        )

    print(f"\n{'='*60}")
    print(f"  Dataset  : {dataset_label}")
    print(f"  Fruits   : {fruits if fruits else 'all'}")
    print(f"  Classes  : {class_names}  ({num_classes})")
    print(f"  Condition: {condition} | Backbone: {backbone_name} | Device: {device}")
    print(f"  Run dir  : {run_dir}")
    if resuming:
        print(f"  Resuming from checkpoint...")
    print(f"{'='*60}")

    backbone = get_backbone(device, backbone_name=backbone_name, pretrained=True, unfreeze_layers=unfreeze)

    if use_head:
        proj   = ProjectionHead(in_dim=backbone.out_dim, out_dim=CONFIG["embedding_dim"]).to(device)
        linear = nn.Linear(CONFIG["embedding_dim"], num_classes).to(device)
        head_params = list(proj.parameters()) + list(linear.parameters())
    else:
        proj   = None
        linear = nn.Linear(backbone.out_dim, num_classes).to(device)
        head_params = list(linear.parameters())

    if unfreeze > 0 and backbone.trainable_params():
        optimizer = optim.Adam([
            {"params": backbone.trainable_params(), "lr": CONFIG["lr_backbone"]},
            {"params": head_params,                 "lr": CONFIG["lr"]},
        ], weight_decay=CONFIG["weight_decay"])
    else:
        optimizer = optim.Adam(
            head_params, lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
        )

    scheduler    = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    criterion    = nn.CrossEntropyLoss()
    best_acc     = 0.0
    best_f1      = 0.0
    acc_history  = []
    start_epoch  = 1

    # ── Resume from checkpoint if it exists ──
    if resuming:
        ckpt        = torch.load(ckpt_path, map_location=device)
        start_epoch = ckpt["epoch"] + 1
        best_acc    = ckpt["best_acc"]
        best_f1     = ckpt.get("best_f1", 0.0)
        acc_history = ckpt["acc_history"]
        backbone.load_state_dict(ckpt["backbone"])
        linear.load_state_dict(ckpt["linear"])
        if proj is not None and ckpt["proj"] is not None:
            proj.load_state_dict(ckpt["proj"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        print(f"  Resumed at epoch {start_epoch} | best acc: {best_acc*100:.2f}% | best F1: {best_f1*100:.2f}%")

    if start_epoch > CONFIG["epochs"]:
        print("  Training already complete.")
        return best_acc

    for epoch in range(start_epoch, CONFIG["epochs"] + 1):
        linear.train()
        if proj is not None:
            proj.train()
        if unfreeze > 0:
            backbone.train()

        total_loss, steps = 0.0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            if unfreeze > 0:
                feats = backbone(images)
            else:
                with torch.no_grad():
                    feats = backbone(images)

            emb    = proj(feats) if proj is not None else feats
            logits = linear(emb)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps      += 1

        scheduler.step()
        avg_loss = total_loss / steps

        if epoch % CONFIG["eval_every"] == 0 or epoch == CONFIG["epochs"]:
            m = evaluate(backbone, proj, linear, val_loader, device, unfreeze, criterion)

            entry = {"epoch": epoch, "train_loss": round(avg_loss, 6), **m}
            acc_history.append(entry)

            if m["acc"] > best_acc:
                best_acc = m["acc"]
            if m["f1_macro"] > best_f1:
                best_f1 = m["f1_macro"]
                torch.save({
                    "epoch"      : epoch,
                    "backbone"   : backbone.state_dict(),
                    "linear"     : linear.state_dict(),
                    "proj"       : proj.state_dict() if proj else None,
                    "class_names": class_names,
                }, os.path.join(run_dir, BEST_MODEL_FILE))

            print(f"  Epoch {epoch:3d}/{CONFIG['epochs']} | "
                  f"train_loss={avg_loss:.4f} | val_loss={m['val_loss']:.4f} | "
                  f"acc={m['acc']*100:.2f}% | f1={m['f1_macro']*100:.2f}% | "
                  f"best_f1={best_f1*100:.2f}%")
        else:
            print(f"  Epoch {epoch:3d}/{CONFIG['epochs']} | train_loss={avg_loss:.4f}")

        # Rolling checkpoint every epoch
        _save_checkpoint(run_dir, epoch, backbone, proj, linear,
                         optimizer, scheduler, best_acc, acc_history, class_names,
                         best_f1=best_f1)

        # Named snapshot every save_every epochs
        if epoch % CONFIG["save_every"] == 0:
            _save_checkpoint(run_dir, epoch, backbone, proj, linear,
                             optimizer, scheduler, best_acc, acc_history, class_names,
                             filename=f"checkpoint_epoch_{epoch}.pt", best_f1=best_f1)

    # Final confusion matrix from last eval
    final_cm    = acc_history[-1].get("confusion_matrix", []) if acc_history else []
    best_f1_entry = max(acc_history, key=lambda h: h.get("f1_macro", 0)) if acc_history else {}

    metrics = {
        "dataset"          : dataset_label,
        "condition"        : condition,
        "backbone_name"    : backbone_name,
        "label_mode"       : label_mode,
        "num_classes"      : num_classes,
        "class_names"      : class_names,
        "unfreeze_layers"  : unfreeze,
        "use_head"         : use_head,
        "best_acc"         : round(best_acc, 6),
        "best_f1"          : round(best_f1, 6),
        "best_precision"   : round(best_f1_entry.get("precision_macro", 0), 6),
        "best_recall"      : round(best_f1_entry.get("recall_macro", 0), 6),
        "final_confusion_matrix": final_cm,
        "acc_history"      : acc_history,
        "config"           : CONFIG,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Best accuracy : {best_acc*100:.2f}%")
    print(f"  Best F1 macro : {best_f1*100:.2f}%")
    print(f"  Saved         : {run_dir}/")
    return best_acc


# ─────────────────────────────────────────────────────────────
# CLI helpers
# ─────────────────────────────────────────────────────────────

def parse_selection(user_input, n):
    s = user_input.strip().lower()
    if s == "all":
        return list(range(1, n + 1))
    selected = set()
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            selected.update(range(int(a), int(b) + 1))
        else:
            selected.add(int(part))
    invalid = [x for x in selected if x < 1 or x > n]
    if invalid:
        raise ValueError(f"Out of range: {invalid}")
    return sorted(selected)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

DATASETS = {
    1: ("mendeley_fruitvision",       "./data/mendeley_fruitvision",       "5 fruits × fresh/formalin/rotten   — 10,154 imgs  [C]"),
    2: ("mendeley_fruits",            "./data/mendeley_fruits",            "3 fruits × fresh/rotten            —  1,655 imgs  [C]"),
    3: ("mendeley_lemon_varieties",   "./data/mendeley_lemon_varieties",   "lemon    × fresh/rotten            —  1,956 imgs  [C]"),
    4: ("kaggle_fruits_fresh_rotten", "./data/kaggle_fruits_fresh_rotten", "3 fruits × fresh/rotten            — 13,599 imgs  [C]"),
    5: ("kaggle_fresh_stale",         "./data/kaggle_fresh_stale",         "9 items  × fresh/rotten            — 27,317 imgs  [C]"),
    6: ("kaggle_fruits_quality",      "./data/kaggle_fruits_quality",      "12 fruits mixed × fresh/rotten     —    359 imgs  [B]"),
}
# [C] = Layout C: fruit/state nested  →  get_loaders_nested()
# [B] = Layout B: train/test split    →  get_loaders()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--fruits",       type=str, default=None,
                        help="Comma-separated fruit names, e.g. apple,banana")
    args = parser.parse_args()

    print("\n── Freshness Classifier — Transfer Learning ──\n")

    # ── Dataset selection ──
    if args.dataset_path:
        dataset_path    = args.dataset_path
        selected_fruits = [f.strip() for f in args.fruits.split(",")] if args.fruits else None
    else:
        print("Dataset:")
        for i, (name, path, desc) in DATASETS.items():
            print(f"  {i}. {name:<30} {desc}")
        raw = input("Select dataset [1]: ").strip() or "1"
        _, dataset_path, _ = DATASETS[int(raw)]

        if not Path(dataset_path).exists():
            print(f"\n  [ERROR] Dataset not found: {dataset_path}")
            print(f"  Run first:  python prepare_datasets.py")
            exit(1)

        # Fruit selection only available via --fruits flag
        selected_fruits = None

    # ── Label mode ──
    print("\nLabel mode:")
    print("  1. state        — classify by state only  (fresh / rotten / formalin)")
    print("  2. fruit_state  — classify by fruit+state (peach_fresh / peach_rotten / ...)")
    print("  3. all          — run both modes sequentially")
    raw = input("Select mode [1]: ").strip() or "1"
    if raw == "2":
        selected_label_modes = ["fruit_state"]
    elif raw == "3":
        selected_label_modes = ["state", "fruit_state"]
    else:
        selected_label_modes = ["state"]

    # ── Backbone ──
    BACKBONES = {i+1: name for i, name in enumerate(BACKBONE_NAMES)}
    print("\nBackbone:")
    for i, name in BACKBONES.items():
        _, _, out_dim, _ = BACKBONE_REGISTRY[name]
        tag = " ← default" if name == "resnet18" else ""
        print(f"  {i}. {name:<22} ({out_dim} dim){tag}")
    while True:
        try:
            raw = input("\nBackbone [1]: ").strip() or "1"
            b_idx = parse_selection(raw, len(BACKBONES))
            if len(b_idx) != 1:
                print("  Select exactly one backbone.")
                continue
            selected_backbone = BACKBONES[b_idx[0]]
            break
        except (ValueError, TypeError):
            print("  Invalid input.")

    # ── Conditions ──
    print("\nConditions:")
    print("  1. frozen      — frozen backbone + Linear")
    print("  2. layer4      — layer4 unfrozen + Linear")
    print("  3. head_frozen — frozen backbone + ProjectionHead + Linear")
    print("  4. head_layer4 — layer4 unfrozen + ProjectionHead + Linear")
    cond_input        = input("Select conditions (e.g. 1, 1-2, all) [1]: ").strip() or "1"
    cond_indices      = parse_selection(cond_input, len(CONDITIONS))
    selected_conditions = [CONDITIONS[i - 1] for i in cond_indices]

    print(f"\n  Dataset    : {dataset_path}")
    print(f"  Fruits     : {selected_fruits if selected_fruits else 'all'}")
    print(f"  Backbone   : {selected_backbone}")
    print(f"  Conditions : {selected_conditions}\n")

    results = {}
    for label_mode in selected_label_modes:
        for cond in selected_conditions:
            acc = train(dataset_path, cond,
                        backbone_name=selected_backbone,
                        fruits=selected_fruits,
                        label_mode=label_mode)
            results[f"{cond} [{label_mode}]"] = f"{acc*100:.2f}%"

    print(f"\n{'─'*50}")
    print("  Summary")
    print(f"{'─'*50}")
    for cond, acc in results.items():
        print(f"  {cond:<15} {acc}")
    print(f"{'─'*50}")
