# DIP Classifier — Fruit and Vegetable Freshness Classification

Transfer-learning pipeline for produce freshness classification (fresh / rotten). Models are
trained on six public fruit/vegetable datasets across four backbones (ResNet-18,
MobileNetV3-Small, EfficientNet-B0/B2) and four training conditions (C1–C4), then tested on
our own photos of real produce to measure domain shift. A per-fruit analysis shows that
generalization tracks **training exposure** (a model collapses on fruits its training set
never contained), and a domain-adaptation fine-tuning step lifts a balanced base model from
86.3% to **94.5% F1** on the real photos — the project's best general fresh/rotten classifier.

On the two 6-class (fruit×state) datasets that have published baselines, our models **exceed
the state of the art**: a perfect **100%** accuracy single model and a **99.96%** 16-model
ensemble on apple/banana/orange (vs 99.61% published), and **96.68%** on
peach/pomegranate/strawberry (vs 95.0% published). See [Benchmark vs Published Work](#benchmark-vs-published-work).

---

## Visual Overview

![Project Pipeline](Other/Animation/drawio/DIPv2.gif#gh-light-mode-only)
![Project Pipeline](Other/Animation/drawio/DIPv2.gif#gh-dark-mode-only)

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
  02_02             Fine-tune a base model on own photos (k-fold CV)

Stage 03 — Evaluation & Analysis
  03_01             Cross-dataset evaluation on real photos
  03_02             Training curves and comparison plots
  03_03             Model ensemble on own photos
  03_04             Per-image error analysis on own photos
  03_05             Per-fruit evaluation breakdown

Stage 04 — Reporting
  04_01             Excel experiment tracker
  04_02             Excel evaluation report with rankings
  04_03             Consolidated own_dataset domain-adaptation report
  04_04             Benchmark report vs published papers (KFR / MFR)
```

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
| MFR | mendeley_fruits | MN3 | MobileNetV3-S | C2 | layer4 | FS | fruit_state |
| MLM | mendeley_lemon_varieties | EB0 | EfficientNet-B0 | C3 | head_frozen | | |
| MFV | mendeley_fruitvision | EB2 | EfficientNet-B2 | C4 | head_layer4 | | |
| KFR | kaggle_fruits_fresh_rotten | | | | | | |
| KFS | kaggle_fresh_stale | | | | | | |
| OWN | own_dataset | | | | | | |

---

## Datasets

| Dataset | ID | Categories | Classes | Images | Source |
|---|---|---|---|---|---|
| FruitVision | MFV | Apple, Banana, Grape, Mango, Orange | Fresh / Formalin / Rotten | 10,154 | Mendeley |
| Lemon Varieties | MLM | 7 lemon varieties | Fresh / Rotten | 1,956 | Mendeley |
| Fruits Classification | MFR | Peach, Pomegranate, Strawberry | Fresh / Rotten | 1,655 | Mendeley |
| Fruits Quality | KFQ | 12 mixed fruits | Fresh / Rotten | 359 | Kaggle |
| Fruits Fresh/Rotten | KFR | Apple, Banana, Orange | Fresh / Rotten | 13,599 | Kaggle |
| Fresh & Stale | KFS | 9 fruits/vegetables | Fresh / Rotten | 27,317 | Kaggle |
| **Own Dataset** | **OWN** | **Strawberry, Banana** | **Fresh / Rotten** | **219** | **Own photos** |

`own_dataset` holds our own photos of real produce (121 strawberry + 98 banana, balanced
fresh/rotten). It is **never used for training** — only as a held-out, real-world test set
to measure domain shift and, in Stage 02, as the fine-tuning target.

---

## Training Conditions

Four conditions form an ablation study over two dimensions: whether the backbone is partially fine-tuned, and whether a non-linear projection head is used.

|  | Backbone frozen | Layer4 free |
|--|----------------|-------------|
| **Linear head only** | C1 | C2 |
| **Projection head** | C3 | C4 |

---

**C1 — Linear probe (frozen backbone + linear head)**

All backbone weights are locked. Only a single linear layer `d_out → C` is optimized. This is the classical *linear probe* — evaluates the quality of ImageNet features without any domain adaptation. Fastest to train, but cannot compensate for mismatches between ImageNet and food freshness features.

---

**C2 — Partial fine-tuning (layer4 free + linear head)**

The last residual block of the backbone (`layer4`, ~2M params for ResNet-18) is unfrozen and trained alongside the linear classifier. All earlier layers remain frozen. This allows the highest-level feature detectors to specialize toward freshness-relevant patterns while preserving low and mid-level ImageNet features. Same classifier as C1 — no extra non-linear capacity.

---

**C3 — Projection head + frozen backbone**

The backbone is fully frozen, but instead of a single linear layer, a two-layer non-linear projection head is used:
```
d_out → Linear(256) → ReLU → Linear(128) → ReLU → Linear(C)
```
The non-linear head reshapes the feature space to form more discriminative decision boundaries without touching the backbone. This is the **optimal condition for EfficientNet** variants — their 1280/1408-dim features are already expressive enough without fine-tuning.

---

**C4 — Projection head + partial fine-tuning (layer4 free)**

Combines the projection head of C3 with the backbone fine-tuning of C2. Uses a **dual learning rate** strategy to prevent catastrophic forgetting:
- Head: `lr = 1e-3`
- Backbone layer4: `lr_backbone = 1e-5` (100× smaller)

The drastically lower backbone LR ensures gradual adaptation — the backbone specializes toward freshness features without destroying the hierarchical representations inherited from ImageNet pre-training. This is the **optimal condition for ResNet-18**.

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
| `02_02_finetune.py` | Domain-adaptation fine-tuning: continues training a public-trained `best_model.pt` on own_dataset to learn the local domain (real backgrounds, early rot). Stratified k-fold CV pools held-out predictions over all photos for a stable before/after comparison. Geometry-heavy, color-light augmentation preserves the fragile rot signal. `--freeze` (linear-probe) avoids catastrophic forgetting on small data; `--save-final` trains one deployable model on all photos. |

### Stage 03 — Evaluation & Analysis

| Script | Description |
|--------|-------------|
| `03_01_evaluate.py` | Loads any `best_model.pt` and evaluates it on any dataset. Computes F1, accuracy, precision, recall, MCC, AUC-ROC, per-image confidence, and domain shift (Train F1 − Eval F1). Generates confusion matrix, ROC curve, confidence histogram, and per-class bar chart. |
| `03_02_analyze.py` | Reads all `metrics.json` files and generates training-curve comparison plots (accuracy, F1, precision, recall, loss, heatmap, confusion matrices, per-class bars). |
| `03_03_ensemble.py` | Combines several trained models into an ensemble and evaluates it on **any dataset / label mode** (2..N classes) via `--eval-dataset` / `--eval-mode` — used both for own_dataset (state) and the 6-class KFR/MFR benchmark (fruit_state). Builds the ensemble directly from saved per-image predictions (no GPU). Supports selecting members (all / top-K / by dataset+condition / by backbone+condition / manual) and combination methods (mean softmax, F1-weighted, confidence-adaptive). Reports ensemble F1/MCC/AUC vs the best single model, and saves a plot pairing the per-model F1 bars with the **ensemble confusion matrix** (percentages + counts). |
| `03_04_error_analysis.py` | Flips the table from per-model to per-image: across all evaluated models, how many miss each photo? Images missed by most models are intrinsically hard (correlated errors no ensemble can fix) and define the accuracy ceiling. Reports difficulty buckets, the hardest images, and the confidence of the wrong predictions. |
| `03_05_per_fruit.py` | Splits each model's own_dataset predictions by fruit (parsed from the filename code, e.g. `FR_SB`→strawberry, `RT_BN`→banana) and reports F1 per fruit. Reveals whether a model generalizes across fruits or only handles the fruit family it trained on, and contrasts banana F1 for models that did vs did not see banana in training. Scatter plot of strawberry-vs-banana F1 per model. |

### Stage 04 — Reporting

| Script | Description |
|--------|-------------|
| `04_01_generate_tracker.py` | Generates `04_01_experiments_tracker.xlsx`. One sheet per backbone with color-coded run status (complete / running / pending / n/a) and metrics (F1, ACC, MCC, AUC, time). |
| `04_02_generate_eval_report.py` | Generates `04_02_eval_report.xlsx`. One sheet per evaluated dataset with metrics and embedded plots. Four ranking sheets: by F1, by domain shift (robustness), by recall (food safety), by accuracy. |
| `04_03_generate_own_report.py` | Generates `04_03_own_dataset_report.xlsx`: a visual summary of the domain-adaptation work that consolidates the Stage 02/03 analyses (fine-tuning, per-fruit, error analysis, ensemble) by reading their saved JSON and embedding their plots. No GPU; regenerates instantly. Sheets: Summary, Fine-tuning, Per-fruit, Hardest images, Ensemble. |
| `04_04_generate_benchmark_report.py` | Generates `04_04_benchmark_report.xlsx`: compares our best single models and 16-model ensembles against the published papers on the same datasets (KFR, MFR) using accuracy. Auto-discovers the fruit_state eval results and ensembles; paper figures are hardcoded from the PDFs. Sheets: Summary, KFR vs Papers, MFR vs Papers. |

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

**4. Model ensemble**
The per-image predictions of several models are combined (averaged softmax,
optionally weighted by validation F1 or per-image confidence) into a single
ensemble decision on any dataset / label mode. On the 6-class KFR/MFR benchmark,
ensembling the 16 models (4 backbones × 4 conditions) helps where headroom
remains (+0.85 pts F1 on MFR) but saturates where the best single model is
already perfect (KFR). On own_dataset, combining a few strong, diverse models
gives only a small, sample-sensitive gain — the bottleneck there is data
diversity, not how predictions are pooled.

**5. Error and per-fruit analysis**
Per-image error analysis shows the failures are concentrated and confidently
wrong: a handful of borderline / early-rot photos are missed by most models at
~85–90% confidence, which is why ensembling cannot fix them. Splitting the
predictions by fruit reveals the real driver of error: models whose training
data never contained a given fruit collapse on it. On own_dataset, models
trained without banana average ~42% F1 on banana vs ~71% for models that saw it
— and the apparent single-fruit "champion" (93% on strawberry) drops to ~32% on
banana, ranking near last overall. Generalization tracks training exposure, not
the leaderboard of any one fruit.

**6. Domain-adaptation fine-tuning**
A balanced, fruit-aware base model is fine-tuned on the full multi-fruit
own_dataset, evaluated with stratified k-fold CV. With the backbone **frozen**
(linear-probe) the model adapts without catastrophic forgetting: F1 improves
from 86.3% to **94.5%** (+8.2 pts, MCC 0.729 → 0.892) with every fold improving.
Unfreezing the backbone, or starting from a single-fruit base, instead overfits
the small dataset and degrades performance. The fine-tuned model is the project's
best general fresh/rotten classifier on real photos.

**7. Reporting**
Excel trackers summarize experiment status and results with color coding.
The eval report ranks models by performance, generalization, food-safety
recall, and accuracy across all evaluated datasets.

---

## Results Summary

Best macro F1 per dataset and backbone (**state mode**, 60 epochs). All 96 state-mode
experiments complete (6 datasets × 4 backbones × 4 conditions). For the 6-class
fruit_state results on KFR/MFR, see [Benchmark vs Published Work](#benchmark-vs-published-work).
Values read from each run's `metrics.json`.

### ResNet-18

| Dataset | Images | Best Cond | F1 | MCC | AUC |
|---|---|---|---|---|---|
| kaggle_fruits_fresh_rotten | 13,599 | C4 | **99.8%** | 0.996 | 1.000 |
| kaggle_fresh_stale | 27,317 | C4 | **99.0%** | 0.979 | 1.000 |
| mendeley_lemon_varieties | 1,956 | C4 | **98.5%** | 0.970 | 1.000 |
| mendeley_fruits | 1,655 | C4 | **96.4%** | 0.929 | 0.995 |
| mendeley_fruitvision | 10,154 | C2 | **91.0%** | 0.863 | 0.981 |
| kaggle_fruits_quality | 359 | C2 | **91.6%** | 0.839 | 0.947 |

### MobileNetV3-Small

| Dataset | Images | Best Cond | F1 | MCC | AUC |
|---|---|---|---|---|---|
| kaggle_fruits_fresh_rotten | 13,599 | C3 | **100.0%** | 1.000 | 1.000 |
| kaggle_fresh_stale | 27,317 | C3 | **99.2%** | 0.984 | 1.000 |
| mendeley_lemon_varieties | 1,956 | C3 | **98.2%** | 0.964 | 0.998 |
| mendeley_fruits | 1,655 | C3 | **96.7%** | 0.934 | 0.997 |
| mendeley_fruitvision | 10,154 | C3 | **92.4%** | 0.884 | 0.989 |
| kaggle_fruits_quality | 359 | C1 | **91.6%** | 0.839 | 0.959 |

### EfficientNet-B0

| Dataset | Images | Best Cond | F1 | MCC | AUC |
|---|---|---|---|---|---|
| kaggle_fruits_fresh_rotten | 13,599 | C3 | **100.0%** | 1.000 | 1.000 |
| kaggle_fresh_stale | 27,317 | C3 | **99.4%** | 0.988 | 1.000 |
| mendeley_lemon_varieties | 1,956 | C3 | **98.7%** | 0.975 | 0.999 |
| mendeley_fruits | 1,655 | C3 | **97.0%** | 0.940 | 0.996 |
| mendeley_fruitvision | 10,154 | C3 | **91.6%** | 0.872 | 0.988 |
| kaggle_fruits_quality | 359 | C4 | 86.1% | 0.727 | 0.883 |

### EfficientNet-B2

| Dataset | Images | Best Cond | F1 | MCC | AUC |
|---|---|---|---|---|---|
| kaggle_fruits_fresh_rotten | 13,599 | C3 | **99.9%** | 0.999 | 1.000 |
| kaggle_fresh_stale | 27,317 | C3 | **99.3%** | 0.987 | 1.000 |
| mendeley_lemon_varieties | 1,956 | C3 | **98.0%** | 0.959 | 0.999 |
| mendeley_fruits | 1,655 | C3 | **96.4%** | 0.928 | 0.995 |
| mendeley_fruitvision | 10,154 | C3 | **92.3%** | 0.881 | 0.987 |
| kaggle_fruits_quality | 359 | C1 | **88.9%** | 0.779 | 0.941 |

---

## Benchmark vs Published Work

The two datasets with published baselines (KFR: apple/banana/orange; MFR:
peach/pomegranate/strawberry) were trained in **fruit_state** mode (6 classes = fruit × state,
matching the papers' protocol) across all 4 backbones × 4 conditions (32 models), then
combined into a 16-model ensemble per dataset (`03_03`). Comparison uses **accuracy** — the
metric all three papers report.

| Dataset | Source / Model | Split | Accuracy |
|---|---|---|---|
| **KFR** | Palakodati et al. (2020), CNN | 60-10-30 | 97.82% |
| **KFR** | Chakraborty et al. (2021), MobileNetV2 | 80-20 | 99.61% |
| **KFR** | **Ours — best single (EB0/MN3-C3)** | 80-20 | **100.0%** |
| **KFR** | **Ours — ensemble (16 models)** | 80-20 | **99.96%** |
| **MFR** | Sharma & Kumar (2025), ResNet50 | 70-15-15 | 95.00% |
| **MFR** | Ours — best single (EB0-C3) | 80-20 | 95.77% |
| **MFR** | **Ours — ensemble (16 models)** | 80-20 | **96.68%** |

We exceed the prior state of the art on both datasets (+0.35 pts on KFR, +1.68 pts on MFR).

**Best fruit_state F1 per backbone (C3 = best condition for all but ResNet-18):**

| Backbone | KFR (C3) | MFR (C3) |
|---|---|---|
| ResNet-18 | 99.40% (C4: 99.65) | 94.09% |
| MobileNetV3-S | **100.0%** | 95.25% |
| EfficientNet-B0 | **100.0%** | **95.72%** |
| EfficientNet-B2 | 99.97% | 93.98% |

**Ensemble behavior** mirrors the headroom: on KFR the best single model is already perfect, so
the ensemble cannot improve (−0.03 pts); on MFR, where margin remains, the four diverse backbone
families produce complementary errors and the ensemble adds **+0.85 pts F1** (96.57% vs 95.72%).
The result is invariant across the three weighting schemes (mean / weighted / adaptive).

---

## Own Photos: Domain Shift, Per-Fruit Analysis & Fine-Tuning

All 80 state-mode models are evaluated on the full `own_dataset` (219 real photos: 121
strawberry + 98 banana). Using the complete multi-fruit set exposes two findings that a
single-fruit test would hide.

### Generalization tracks training exposure

Splitting each model's predictions by fruit shows that a model collapses on a fruit its
training set never contained:

| Model group | F1 on banana |
|---|---|
| Trained **with** banana (KFR, KFS, MFV, KFQ) | **71.4%** |
| Never saw banana (MFR, MLM) | **42.0%** |

The apparent single-fruit "champion" is an illusion of the test set: **MFR-EB2-C3** scores
93.3% on strawberry but only **32.4% on banana**, ranking 77th of 80 overall. The best
*all-round* base models are EfficientNet-B0 trained on banana-containing data
(e.g. **KFQ-EB0-C1**: 88.3% strawberry / 82.8% banana).

The hardest images are all **ripe-but-fresh bananas** with brown speckles, which ~79/80
models confidently (≈94%) call rotten — public datasets label such browning as "rotten", so
the domain gap is largest exactly there.

### Fine-tuning closes the gap

Fine-tuning the balanced, banana-aware base (KFQ-EB0-C1) on the full multi-fruit own_dataset
with the backbone **frozen** (linear-probe), validated by 5-fold cross-validation:

| Metric | Baseline | Fine-tuned | Δ |
|---|---|---|---|
| F1 macro | 86.3% | **94.5%** | +8.2 |
| Accuracy | 86.3% | **94.5%** | +8.2 |
| MCC | 0.729 | **0.892** | +0.163 |
| Recall (rotten) | 90.8% | 91.7% | +0.9 |

Every fold improves (`[97.7, 95.5, 88.5, 93.2, 97.7]`). Unfreezing the backbone, or starting
from a single-fruit base, instead overfits the small dataset and degrades performance.
Ensembling gives only a small, sample-sensitive gain — the bottleneck is **data diversity**,
not how predictions are pooled. The fine-tuned model is the project's best general
fresh/rotten classifier on real photos.

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
├── 02_02_finetune.py              # domain-adaptation fine-tuning (k-fold)
├── 03_01_evaluate.py
├── 03_02_analyze.py
├── 03_03_ensemble.py
├── 03_04_error_analysis.py
├── 03_05_per_fruit.py
├── 04_01_generate_tracker.py
├── 04_02_generate_eval_report.py
├── 04_03_generate_own_report.py
├── 04_04_generate_benchmark_report.py
├── data/                          # Prepared datasets (images excluded from git)
├── datasets/                      # Raw downloaded datasets (images excluded from git)
├── run_outputs/                   # Training results, metrics, plots
│   ├── {run_name}/
│   │   ├── metrics.json
│   │   ├── best_model.pt
│   │   ├── checkpoint.pt
│   │   └── eval/                  # own_dataset predictions per model
│   ├── own_dataset_finetune_from_KFQ-EB0-C1-ST/   # deployable fine-tuned model
│   ├── _finetune/                 # fine-tuning results (json + plots)
│   ├── _per_fruit/                # per-fruit breakdown
│   ├── _error_analysis/           # per-image difficulty
│   ├── _ensemble/                 # ensemble variants
│   └── _plots/                    # training / eval comparison plots
├── Other/
│   ├── Animation/drawio/          # pipeline diagram (DIPv2.gif)
│   ├── LatexReport/               # full project report (report_en.tex)
│   └── IEEEDraft/                 # IEEE conference paper (paper.tex)
├── 04_01_experiments_tracker.xlsx
├── 04_02_eval_report.xlsx
├── 04_03_own_dataset_report.xlsx
├── 04_04_benchmark_report.xlsx
└── README.md
```

---

## Changelog

| # | Date | Description |
|---|------|-------------|
| 15 | 2026-06-10 | Completed KFR + MFR fruit_state across all 4 backbones (32 models). Generalized `03_03` ensemble to any dataset / N classes and added ensemble confusion-matrix plots. New `04_04_generate_benchmark_report.py`: we **beat published SOTA** on both benchmarks (KFR 99.96% ensemble / 100% single vs 99.61%; MFR 96.68% vs 95.0%). Excel outputs renamed with `04_0x_` prefixes. Updated LaTeX report and IEEE paper with the benchmark, ensemble, MobileNetV3, and per-fruit/fine-tuning findings. |
| 14 | 2026-06-07 | Added banana to own_dataset (now 219 multi-fruit photos). Per-fruit analysis (`03_05`) revealed generalization tracks training exposure. Domain-adaptation fine-tuning (`02_02`, frozen backbone) reaches **94.5% F1**. New analyses: ensemble (`03_03`), error analysis (`03_04`). Consolidated report (`04_03`). Fixed `03_01` to score pure-eval sets on all images and `04_02` ID bug. |
| 13 | 2026-06-06 | Pipeline diagram (DIPv2.gif) added to README with stage-by-stage inputs/outputs. |
| 12 | 2026-06-01 | MobileNetV3 experiments running. Detailed C1-C4 explanations in README and IEEE paper. analyze.py plots restructured with per-backbone grids and numbered --plots flag. |
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
