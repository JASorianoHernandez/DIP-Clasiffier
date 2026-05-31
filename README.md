# DIP Classifier — Fruit and Vegetable Freshness Classification

Transfer learning pipeline for freshness classification (fresh / rotten) using publicly
available fruit and vegetable datasets. The project explores three backbone architectures
(ResNet-18, EfficientNet-B0, EfficientNet-B2), four training conditions (C1--C4),
and measures domain shift through cross-dataset evaluation on real photos.

---

## Project Pipeline

The project is divided into four numbered stages. Internal libraries (Stage 00) are
imported automatically and never run directly.

```
Stage 00 — Internal libraries
  _backbone.py      Backbone registry and weight loading
  _dataset.py       Data loaders, transforms, stratified split

Stage 01 — Data preparation
  01_01 → 01_03     Organize datasets and own photos

Stage 02 — Training
  02_01             Train models with interactive menu

Stage 03 — Evaluation & Analysis
  03_01             Cross-dataset evaluation on real photos
  03_02             Training curves and comparison plots

Stage 04 — Reporting
  04_01             Excel experiment tracker
  04_02             Excel evaluation report with rankings
```

---

## Scripts

### Stage 00 — Internal Libraries

| Script | Description |
|--------|-------------|
| `_backbone.py` | Registry of all backbone architectures (ResNet, MobileNet, EfficientNet). Maps string names to factory functions that return ImageNet pre-trained models. Exposes `get_backbone()` and `unfreeze_layers()`. Imported by `02_01_train.py`. |
| `_dataset.py` | Data loaders for three folder layouts (flat, pre-split, nested fruit/state). Applies training augmentation (crop, flip, rotation, color jitter, blur) and validation transforms. Performs stratified 80/20 split. Imported by `02_01_train.py`. |

### Stage 01 — Data Preparation

| Script | Description |
|--------|-------------|
| `01_01_prepare_datasets.py` | Reads raw datasets from `datasets/` and reorganizes them into a unified `data/{dataset}/{fruit}/{state}/` structure compatible with the training pipeline. Handles flat, pre-split, and nested source layouts. |
| `01_02_rename_own_dataset.py` | Renames own photos to a standard convention `{STATE}_{FRUIT}_{###}.jpg` (e.g. `FR_SB_001.jpg`). Interactive menu to select which fruits to rename. |
| `01_03_prepare_own_dataset.py` | Resizes own photos so the shortest side is 256 px and copies them to `data/own_dataset/{fruit}/{state}/`. Skips already processed images. |

### Stage 02 — Training

| Script | Description |
|--------|-------------|
| `02_01_train.py` | Interactive training loop. Menu selects dataset, label mode, backbone and condition. Runs 60 epochs with Adam + cosine annealing. Saves `metrics.json` (all metrics per 5 epochs), `best_model.pt` (best F1), and rolling checkpoints for resume. |

### Stage 03 — Evaluation & Analysis

| Script | Description |
|--------|-------------|
| `03_01_evaluate.py` | Loads any `best_model.pt` and evaluates it on any dataset. Computes F1, accuracy, precision, recall, MCC, AUC-ROC, per-image confidence, and domain shift (Train F1 − Eval F1). Generates confusion matrix, ROC curve, confidence histogram, and per-class bar chart. |
| `03_02_analyze.py` | Reads all `metrics.json` files and generates training-curve comparison plots (accuracy, F1, precision, recall, loss, heatmap, confusion matrices, per-class bars). |

### Stage 04 — Reporting

| Script | Description |
|--------|-------------|
| `04_01_generate_tracker.py` | Generates `experiments_tracker.xlsx`. One sheet per backbone with color-coded run status (complete / running / pending / n/a) and metrics (F1, ACC, MCC, AUC, time). |
| `04_02_generate_eval_report.py` | Generates `eval_report.xlsx`. One sheet per evaluated dataset with metrics and embedded plots. Four ranking sheets: by F1, by domain shift (robustness), by recall (food safety), by accuracy. |

---

## Experiment ID Format

Every run is identified by a 4-part code:

```
{DATASET}-{BACKBONE}-{CONDITION}-{LABELMODE}

Example:  KFR-EB0-C3-ST
          │    │    │   └─ ST = state (fresh / rotten)
          │    │    └───── C3 = head_frozen condition
          │    └────────── EB0 = EfficientNet-B0
          └─────────────── KFR = kaggle_fruits_fresh_rotten
```

| Code | Dataset | Code | Backbone | Code | Condition | Code | Mode |
|------|---------|------|----------|------|-----------|------|------|
| KFQ | kaggle_fruits_quality | R18 | ResNet-18 | C1 | frozen | ST | state |
| MFR | mendeley_fruits | R34 | ResNet-34 | C2 | layer4 | FS | fruit_state |
| MLM | mendeley_lemon_varieties | R50 | ResNet-50 | C3 | head_frozen | | |
| MFV | mendeley_fruitvision | MN3 | MobileNetV3-S | C4 | head_layer4 | | |
| KFR | kaggle_fruits_fresh_rotten | EB0 | EfficientNet-B0 | | | | |
| KFS | kaggle_fresh_stale | EB2 | EfficientNet-B2 | | | | |
| OWN | own_dataset | | | | | | |

---

## Training Conditions

Each condition controls how much of the pre-trained backbone is allowed to adapt.

|  | Backbone frozen | layer4 free |
|--|----------------|-------------|
| **Linear head only** | C1 | C2 |
| **Projection head** | C3 | C4 |

**C1 — frozen + linear:**
Backbone completely locked. Only a linear layer `d_out → C` is trained.
Classical linear probe — fastest, weakest.

**C2 — layer4 + linear:**
Last backbone block (~2M params) unfrozen. Linear classifier only.
Allows high-level feature adaptation without a projection head.

**C3 — frozen + projection head:**
Backbone frozen. A two-layer non-linear head `d_out → 256 → 128 → C`
is trained from scratch. Optimal condition for EfficientNet backbones.

**C4 — layer4 + projection head:**
Layer4 unfrozen + projection head. Trained with dual learning rates
(`lr=1e-3` head, `lr_backbone=1e-5` backbone) to prevent catastrophic forgetting.
Optimal condition for ResNet-18.

---

## Experimental Procedure

**1. Data preparation**
Raw datasets from Kaggle and Mendeley are reorganized into a unified
`fruit/state/` folder structure. Own photos are renamed to a standard convention
and resized to 256 px.

**2. Training**
Each experiment selects a dataset, label mode (`state` or `fruit_state`),
backbone, and condition. A stratified 80/20 train/val split is applied with
fixed seed 42. Training runs for 60 epochs. Metrics recorded every 5 epochs:
accuracy, macro F1, precision, recall, MCC, AUC-ROC, per-class metrics,
confusion matrix, learning rate, and confidence statistics.

**3. Cross-dataset evaluation**
`best_model.pt` from any run is evaluated on any other dataset to measure
domain shift. Generates per-image predictions with confidence scores,
ROC curves, confusion matrices, and Train F1 vs Eval F1 comparison.

**4. Reporting**
Excel trackers summarize experiment status and results with color coding.
The eval report ranks models by performance, generalization, food-safety
recall, and accuracy across all evaluated datasets.

---

## Results Summary

Best macro F1 per dataset and backbone (state mode, 60 epochs).

### ResNet-18

| Dataset | Images | Best Cond | F1 | MCC | AUC |
|---|---|---|---|---|---|
| kaggle_fruits_fresh_rotten | 13,599 | C4 / C2 | **99.8%** | 0.996 | 1.000 |
| kaggle_fresh_stale | 27,317 | C4 | **99.0%** | 0.979 | 1.000 |
| mendeley_lemon_varieties | 1,956 | C4 | **98.5%** | 0.970 | 1.000 |
| mendeley_fruits | 1,655 | C4 | **96.4%** | 0.929 | 0.995 |
| mendeley_fruitvision | 10,154 | C2 | **91.0%** | 0.863 | 0.981 |
| kaggle_fruits_quality | 359 | C2 | **91.6%** | 0.839 | 0.947 |

### EfficientNet-B0

| Dataset | Images | Best Cond | F1 | MCC | AUC |
|---|---|---|---|---|---|
| kaggle_fruits_fresh_rotten | 13,599 | C3 | **100.0%** | 1.000 | 1.000 |
| kaggle_fresh_stale | 27,317 | C3 | **99.4%** | 0.988 | 1.000 |
| mendeley_lemon_varieties | 1,956 | C3 | **98.7%** | 0.975 | 0.999 |
| mendeley_fruits | 1,655 | C3 | **97.0%** | 0.940 | 0.996 |
| mendeley_fruitvision | 10,154 | C3 | **91.6%** | 0.872 | 0.988 |
| kaggle_fruits_quality | 359 | C4 | 86.1% | 0.727 | 0.883 |

### EfficientNet-B2 (in progress)

| Dataset | Images | Best Cond | F1 | MCC | AUC |
|---|---|---|---|---|---|
| kaggle_fruits_fresh_rotten | 13,599 | C3 | **99.9%** | 0.999 | 1.000 |
| mendeley_fruitvision | 10,154 | C3 | **92.3%** | 0.881 | 0.987 |

---

## Key Findings (preliminary)

- **C3 is optimal for EfficientNet** (head\_frozen — backbone frozen + projection head). In 5 of 6 completed datasets, EB0 C3 outperforms EB0 C4.
- **C4 is optimal for ResNet-18** (head\_layer4). Consistent across all datasets.
- **Dataset size determines which backbone wins.** Below ~500 images ResNet-18 outperforms EfficientNet-B0. Above ~1,500 images EfficientNet-B0 consistently wins.
- **EfficientNet-B0 C3 reached 100% F1** on kaggle\_fruits\_fresh\_rotten — the only perfect result across all experiments.
- **Formalin is the hardest class.** FruitVision is the only dataset where no backbone exceeds 92.3%. Directly relevant to Korean fermented foods.
- **Domain shift is real and quantifiable.** A model with 99.8% F1 on training data drops to 70.8% on real phone camera photos of the same category.
- **Generalization ≠ training performance.** The best generalizing model (lemon C4, 91.7% on own photos) is not the one with the highest training F1 (fruits\_fresh\_rotten EB0 C3, 100%).

---

## Cross-Dataset Evaluation on Own Photos

Models trained on public datasets evaluated on 121 real strawberry photos (`own_dataset`):

| ID | Train F1 | Eval F1 | Drop |
|---|---|---|---|
| MLM-R18-C4-ST | 98.5% | **91.7%** | 6.8% |
| MLM-EB0-C4-ST | 95.4% | **91.7%** | 3.7% |
| MLM-R18-C2-ST | 98.0% | 87.5% | 10.5% |
| MFR-EB0-C3-ST | 97.0% | 87.3% | 9.7% |
| KFS-R18-C4-ST | 99.0% | 87.3% | 11.7% |
| KFR-R18-C4-ST | 99.8% | 70.8% | 29.0% |
| KFR-EB0-C3-ST | 100.0% | 65.7% | 34.3% |

> Lemon Varieties models generalize best — more diverse photographic conditions produce more transferable features.

---

## Datasets

| Dataset | ID | Categories | Classes | Images | Source |
|---------|----|-----------|---------|--------|--------|
| FruitVision | MFV | Apple, Banana, Grape, Mango, Orange | Fresh / Formalin / Rotten | 10,154 | Mendeley |
| Lemon Varieties | MLM | 7 lemon varieties | Fresh / Rotten | 1,956 | Mendeley |
| Fruits Classification | MFR | Peach, Pomegranate, Strawberry | Fresh / Rotten | 1,655 | Mendeley |
| Fruits Quality | KFQ | 12 mixed fruits | Fresh / Rotten | 359 | Kaggle |
| Fruits Fresh/Rotten | KFR | Apple, Banana, Orange | Fresh / Rotten | 13,599 | Kaggle |
| Fresh & Stale | KFS | 9 fruits/vegetables | Fresh / Rotten | 27,317 | Kaggle |

---

## Requirements

```bash
conda create -n DIP_env python=3.10
conda activate DIP_env
pip install torch torchvision scikit-learn matplotlib numpy openpyxl
```

---

## Repository Structure

```
DIP-Classfier/
├── _backbone.py                   # Stage 00 — internal library
├── _dataset.py                    # Stage 00 — internal library
├── 01_01_prepare_datasets.py
├── 01_02_rename_own_dataset.py
├── 01_03_prepare_own_dataset.py
├── 02_01_train.py
├── 03_01_evaluate.py
├── 03_02_analyze.py
├── 04_01_generate_tracker.py
├── 04_02_generate_eval_report.py
├── data/                          # Prepared datasets (images excluded from git)
├── datasets/                      # Raw downloaded datasets (images excluded from git)
├── run_outputs/                   # Training results, metrics, plots
│   └── {run_name}/
│       ├── metrics.json
│       ├── best_model.pt
│       ├── checkpoint.pt
│       └── eval/
├── Other/
│   ├── LatexReport/
│   ├── Literature/
│   ├── AnusDraft/
│   └── TanzinaDraft/
├── experiments_tracker.xlsx
├── eval_report.xlsx
└── README.md
```

---

## Changelog

| # | Date | Description |
|---|------|-------------|
| 11 | 2026-05-31 | Restructured README. EB0 fresh_stale complete (99.4%). EB2 fruits_fresh_rotten complete (99.9%). Updated LaTeX report with backbone comparison and cross-dataset sections. |
| 10 | 2026-05-31 | Updated README with full results summary, key findings and cross-dataset evaluation. |
| 9 | 2026-05-31 | EfficientNet-B2 Phase 2 started. EB0 C3 reaches 100% F1 on fruits\_fresh\_rotten. |
| 8 | 2026-05-30 | EfficientNet-B0 Phase 2 experiments. Script reorganization with stage numbering. eval\_report.xlsx added. |
| 7 | 2026-05-29 | All ResNet-18 baseline runs complete (C1-C4, state+fruit\_state, all 6 datasets). |
| 6 | 2026-05-17 | train.py updated with MCC, AUC-ROC, learning rate, train/val split tracking. |
| 5 | 2026-05-11 | Added evaluate.py, generate\_tracker.py, own dataset pipeline. |
| 4 | 2026-05-08 | LaTeX report updated with methodology, formulas, software description. |
| 3 | 2026-05-07 | Reorganized project into Other/ subfolder. |
| 2 | 2026-05-07 | New run results, updated plots, report tables. |
| 1 | 2026-05-06 | Initial commit: transfer learning pipeline, 6 datasets, ResNet-18 baseline. |
