# DIP Classifier — Freshness Classification of Korean Food

Transfer learning pipeline for fruit and vegetable freshness classification (fresh / rotten).
Preliminary experiments use publicly available datasets as a proxy before building a Korean food dataset.

---

## Repository Structure

```
DIP-Classfier/
│
│  ── Internal libraries (not run directly) ──
├── _backbone.py               # Backbone registry (ResNet, MobileNet, EfficientNet)
├── _dataset.py                # Data loaders and stratified split
│
│  ── Stage 01: Data Preparation ──
├── 01_01_prepare_datasets.py  # Reorganize public datasets into data/
├── 01_02_rename_own_dataset.py# Rename own photos to standard convention
├── 01_03_prepare_own_dataset.py# Resize and organize own photos into data/
│
│  ── Stage 02: Training ──
├── 02_01_train.py             # Training loop with interactive menu
│
│  ── Stage 03: Evaluation & Analysis ──
├── 03_01_evaluate.py          # Cross-dataset evaluation with plots
├── 03_02_analyze.py           # Training curves and comparison plots
│
│  ── Stage 04: Reporting ──
├── 04_01_generate_tracker.py  # Excel experiment tracker
├── 04_02_generate_eval_report.py # Excel evaluation report
│
├── data/                      # Prepared datasets (images excluded from git)
├── datasets/                  # Raw downloaded datasets (images excluded from git)
├── run_outputs/               # Training results, metrics, plots
│   └── {run_name}/
│       ├── metrics.json       # All metrics per epoch
│       ├── best_model.pt      # Best model weights
│       ├── checkpoint.pt      # Resume checkpoint
│       └── eval/              # Cross-dataset evaluation results
│
├── Other/
│   ├── LatexReport/           # LaTeX report source and figures
│   ├── Literature/            # Reference papers
│   ├── AnusDraft/             # Korean fruit scraper scripts
│   └── TanzinaDraft/          # Korean food scraper scripts
│
├── experiments_tracker.xlsx   # Experiment status per backbone/dataset/condition
├── eval_report.xlsx           # Cross-dataset evaluation rankings
└── README.md
```

---

## Requirements

Create and activate the environment:

```bash
conda create -n DIP_env python=3.10
conda activate DIP_env
pip install torch torchvision scikit-learn matplotlib numpy openpyxl
```

---

## Usage

Scripts are run in stage order:

### Stage 01 — Data Preparation

```bash
# Public datasets (run once per dataset)
python 01_01_prepare_datasets.py

# Own photos (run once)
python 01_02_rename_own_dataset.py    # rename to standard convention
python 01_03_prepare_own_dataset.py   # resize and copy to data/
```

### Stage 02 — Training

```bash
python 02_01_train.py
```
Interactive menu: select dataset, label mode, backbone and condition (C1–C4).
Results saved to `run_outputs/{run_name}/metrics.json`.

### Stage 03 — Evaluation & Analysis

```bash
# Cross-dataset evaluation (own photos vs trained models)
python 03_01_evaluate.py

# Training curves and comparison plots
python 03_02_analyze.py
```

### Stage 04 — Reporting

```bash
# Update experiment tracker Excel
python 04_01_generate_tracker.py

# Generate evaluation report Excel
python 04_02_generate_eval_report.py
```

---

## Experiment ID Format

Every training run is uniquely identified by a 4-part code:

```
{DATASET}-{BACKBONE}-{CONDITION}-{LABELMODE}

Example:  KFR-EB0-C3-ST
          │    │    │   └─ ST = state mode (fresh / rotten)
          │    │    └───── C3 = head_frozen condition
          │    └────────── EB0 = EfficientNet-B0
          └─────────────── KFR = kaggle_fruits_fresh_rotten
```

### Dataset codes

| Code | Dataset | Images | Classes |
|------|---------|--------|---------|
| KFQ | kaggle_fruits_quality | 359 | fresh / rotten |
| MFR | mendeley_fruits | 1,655 | fresh / rotten |
| MLM | mendeley_lemon_varieties | 1,956 | fresh / rotten |
| MFV | mendeley_fruitvision | 10,154 | fresh / formalin / rotten |
| KFR | kaggle_fruits_fresh_rotten | 13,599 | fresh / rotten |
| KFS | kaggle_fresh_stale | 27,317 | fresh / rotten |
| OWN | own_dataset | custom | fresh / rotten |

### Backbone codes

| Code | Model | Params | Output dim |
|------|-------|--------|------------|
| R18 | ResNet-18 | 11.2M | 512 |
| R34 | ResNet-34 | 21.3M | 512 |
| R50 | ResNet-50 | 25.6M | 2048 |
| MN3 | MobileNetV3-Small | 2.5M | 576 |
| EB0 | EfficientNet-B0 | 5.3M | 1280 |
| EB2 | EfficientNet-B2 | 9.1M | 1408 |

### Label mode codes

| Code | Mode | Description |
|------|------|-------------|
| ST | state | Classify by freshness only: `fresh`, `rotten`, `formalin` |
| FS | fruit_state | Classify by fruit + freshness: `apple_fresh`, `apple_rotten`, etc. |

---

## Training Conditions

Each experiment uses one of four training conditions that differ in how much of the pre-trained backbone is allowed to adapt to the new domain.

All backbones are initialized with ImageNet pre-trained weights. The backbone produces a feature vector (e.g. 512 dimensions for ResNet-18) that is passed through a head to produce the final classification.

### C1 — frozen + linear head
The entire backbone is locked — no weights are updated during training. Only a single linear layer (`d_out → C classes`) is trained from scratch. This is the classical *linear probe*: the pre-trained backbone acts as a fixed feature extractor. Fastest to train, weakest performance.

```
ImageNet backbone (frozen) → Linear(C)
```

### C2 — layer4 + linear head
The last residual block of the backbone (`layer4`, ~2M parameters) is unfrozen and fine-tuned alongside the linear head. All earlier layers remain frozen. No projection head — just a direct linear classifier. Allows the backbone to adapt its highest-level features to the target domain.

```
ImageNet backbone (layers 1-3 frozen, layer4 free) → Linear(C)
```

### C3 — frozen + projection head
The backbone is fully frozen (like C1), but a two-layer non-linear projection head is added before the classifier:

```
ImageNet backbone (frozen) → Linear(256) → ReLU → Linear(128) → ReLU → Linear(C)
```

The projection head can warp the feature space to find better decision boundaries without modifying the backbone. This is the **optimal condition for EfficientNet** — its 1280-dim features are already rich enough without fine-tuning.

### C4 — layer4 + projection head
Combines C2 and C3: `layer4` is unfrozen **and** the projection head is used. The backbone and head are trained jointly, but with different learning rates (`lr=1e-3` for head, `lr_backbone=1e-5` for layer4) to prevent catastrophic forgetting. This is the **optimal condition for ResNet-18**.

```
ImageNet backbone (layer4 free) → Linear(256) → ReLU → Linear(128) → ReLU → Linear(C)
```

### Summary

|  | Backbone frozen | layer4 free |
|--|----------------|-------------|
| **Linear head only** | C1 | C2 |
| **Projection head** | C3 | C4 |

---

## Experimental Procedure

The full workflow followed in this project:

### 1. Data preparation
Raw datasets downloaded from Kaggle and Mendeley are reorganized into a unified `fruit/state/` folder structure using `01_01_prepare_datasets.py`. Own photos are renamed to a standard convention (`KFS-R18-C3-ST` style) and resized to 256px using `01_02` and `01_03`.

### 2. Training
`02_01_train.py` presents an interactive menu to select dataset, label mode, backbone and condition. A stratified 80/20 train/validation split is applied at runtime with a fixed seed (42) for reproducibility. Training runs for 60 epochs with Adam optimizer and cosine annealing learning rate schedule. Metrics (accuracy, F1, precision, recall, MCC, AUC-ROC, confusion matrix) are recorded every 5 epochs and saved to `metrics.json`. The best model by validation F1 is saved as `best_model.pt`.

### 3. Cross-dataset evaluation
`03_01_evaluate.py` loads a `best_model.pt` from any completed run and evaluates it on any other dataset — including own photos never seen during training. This measures domain shift: how much performance degrades when moving from internet photos to real-world photos. Generates per-image predictions with confidence scores, ROC curves, and confusion matrices.

### 4. Analysis and reporting
`03_02_analyze.py` generates training curves across all runs. `04_01_generate_tracker.py` and `04_02_generate_eval_report.py` produce Excel files with color-coded results tables and multiple ranking views (by F1, recall, robustness/domain shift).

---

## Results Summary

Experiments in progress. Best macro F1 per dataset and backbone (state mode, 60 epochs).

### ResNet-18

| Dataset | Images | Best Cond | Best F1 |
|---|---|---|---|
| kaggle_fruits_fresh_rotten | 13,599 | C4 / C2 | **99.8%** |
| kaggle_fresh_stale | 27,317 | C4 | **99.0%** |
| mendeley_lemon_varieties | 1,956 | C4 | **98.5%** |
| mendeley_fruits | 1,655 | C4 | **96.4%** |
| mendeley_fruitvision | 10,154 | C2 | **91.0%** |
| kaggle_fruits_quality | 359 | C2 | **91.6%** |

### EfficientNet-B0

| Dataset | Images | Best Cond | Best F1 |
|---|---|---|---|
| kaggle_fruits_fresh_rotten | 13,599 | C3 | **100.0%** |
| mendeley_lemon_varieties | 1,956 | C3 | **98.7%** |
| mendeley_fruits | 1,655 | C3 | **97.0%** |
| kaggle_fresh_stale | 27,317 | C3 | pending |
| mendeley_fruitvision | 10,154 | C3 | **91.6%** |
| kaggle_fruits_quality | 359 | C4 / C2 | **86.1%** |

### EfficientNet-B2 (in progress)

| Dataset | Images | Best Cond | Best F1 |
|---|---|---|---|
| mendeley_fruitvision | 10,154 | C3 | **92.3%** |

---

## Key Findings (preliminary)

- **C3 is optimal for EfficientNet** (head_frozen). In 4 of 5 completed datasets, EB0 C3 outperforms EB0 C4. The richer pre-trained features of EfficientNet do not benefit from backbone fine-tuning.
- **C4 is optimal for ResNet-18** (head_layer4). Consistent across all datasets.
- **Dataset size determines which backbone wins.** Below ~500 images ResNet-18 outperforms EfficientNet-B0. Above ~1,500 images EfficientNet-B0 consistently wins.
- **EfficientNet-B0 C3 reached 100% F1** on kaggle_fruits_fresh_rotten — the only perfect result across all experiments.
- **Formalin is the hardest class.** FruitVision is the only dataset where no backbone exceeds 92.3% — directly relevant to Korean fermented foods.
- **Domain shift is real and quantifiable.** A model with 99.8% F1 on training data drops to 70.8% on real phone camera photos of the same category.
- **Generalization ≠ training performance.** The best generalizing model (lemon C4, 91.7% on own photos) is not the one with the highest training F1 (fruits_fresh_rotten EB0 C3, 100%).

---

## Cross-Dataset Evaluation on Own Photos

Models trained on public datasets evaluated on 121 real strawberry photos:

| ID | Train F1 | Eval F1 | Drop |
|---|---|---|---|
| MLM-R18-C4-ST | 98.5% | **91.7%** | 6.8% |
| MLM-EB0-C4-ST | 95.4% | **91.7%** | 3.7% |
| MLM-R18-C2-ST | 98.0% | 87.5% | 10.5% |
| MFR-EB0-C3-ST | 97.0% | 87.3% | 9.7% |
| KFR-R18-C4-ST | 99.8% | 70.8% | 29.0% |

> Lemon Varieties models generalize best to real photos — likely due to more diverse photographic conditions in that dataset.

---

## Datasets

| Dataset | Categories | Classes | Images | Source |
|---------|-----------|---------|--------|--------|
| FruitVision | Apple, Banana, Grape, Mango, Orange | Fresh / Formalin / Rotten | 10,154 | Mendeley |
| Lemon Varieties | 7 lemon varieties | Fresh / Rotten | 1,956 | Mendeley |
| Fruits Classification | Peach, Pomegranate, Strawberry | Fresh / Rotten | 1,655 | Mendeley |
| Fruits Quality | 12 mixed fruits | Fresh / Rotten | 359 | Kaggle |
| Fruits Fresh/Rotten | Apple, Banana, Orange | Fresh / Rotten | 13,599 | Kaggle |
| Fresh & Stale | 9 fruits/vegetables | Fresh / Rotten | 27,317 | Kaggle |

---

## Changelog

| # | Date | Description |
|---|------|-------------|
| 10 | 2026-05-31 | Updated README with full results summary, key findings and cross-dataset evaluation. |
| 9 | 2026-05-31 | EfficientNet-B2 Phase 2 started. EB0 C3 reaches 100% F1 on fruits_fresh_rotten. EB2 outperforms EB0/R18 on fruitvision. |
| 8 | 2026-05-30 | EfficientNet-B0 Phase 2 experiments started. Script reorganization with stage numbering (00–04). eval_report.xlsx and experiments_tracker.xlsx added. |
| 7 | 2026-05-29 | All ResNet-18 baseline runs complete (C1-C4, state+fruit_state, all 6 datasets). |
| 6 | 2026-05-17 | train.py updated with MCC, AUC-ROC, learning rate tracking, train/val split info. |
| 5 | 2026-05-11 | Added evaluate.py for cross-dataset evaluation. generate_tracker.py. Own dataset pipeline. |
| 4 | 2026-05-08 | LaTeX report updated with methodology, formulas, software description. |
| 3 | 2026-05-07 | Reorganized project into Other/ subfolder. |
| 2 | 2026-05-07 | New run results, updated plots, report tables. |
| 1 | 2026-05-06 | Initial commit: transfer learning pipeline, 6 datasets, ResNet-18 baseline. |
