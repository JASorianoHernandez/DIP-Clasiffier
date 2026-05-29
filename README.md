# DIP Classifier — Freshness Classification of Korean Food

Transfer learning pipeline for fruit and vegetable freshness classification (fresh / rotten).
Preliminary experiments use publicly available datasets as a proxy before building a Korean food dataset.

---

## Repository Structure

```
DIP-Classfier/
│
├── train.py                  # Training loop with interactive menu
├── evaluate.py               # Cross-dataset evaluation
├── analyze.py                # Generate comparison plots from all runs
├── generate_tracker.py       # Generate Excel experiment tracker
├── prepare_datasets.py       # Reorganize public datasets into data/
├── prepare_own_dataset.py    # Resize and organize own photos into data/
├── rename_own_dataset.py     # Rename own photos to standard convention
│
├── backbone.py               # Backbone registry (ResNet, MobileNet, EfficientNet)
├── dataset.py                # Data loaders and stratified split
│
├── data/                     # Prepared datasets (images excluded from git)
├── datasets/                 # Raw downloaded datasets (images excluded from git)
├── run_outputs/              # Training results, metrics, plots
│   └── {run_name}/
│       ├── metrics.json      # All metrics per epoch
│       ├── best_model.pt     # Best model weights
│       ├── checkpoint.pt     # Resume checkpoint
│       └── eval/             # Cross-dataset evaluation results
│
├── Other/
│   ├── LatexReport/          # LaTeX report source and figures
│   ├── Literature/           # Reference papers
│   ├── AnusDraft/            # Korean fruit scraper scripts
│   └── TanzinaDraft/         # Korean food scraper scripts
│
└── experiments_tracker.xlsx  # Experiment status and results tracker
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

Scripts are run in this order:

### 1. Prepare datasets
```bash
python prepare_datasets.py
```
Reorganizes raw datasets from `datasets/` into unified `data/{dataset}/{fruit}/{state}/` structure.

### 2. Train
```bash
python train.py
```
Interactive menu to select dataset, label mode, backbone and condition (C1–C4).
Results saved to `run_outputs/{run_name}/metrics.json`.

### 3. Evaluate (cross-dataset)
```bash
python evaluate.py
```
Loads a trained model and evaluates it on any dataset, including your own photos.
Generates per-image predictions and plots.

### 4. Analyze
```bash
python analyze.py
```
Reads all `metrics.json` files and generates comparison plots across all runs.

### 5. Update tracker
```bash
python generate_tracker.py
```
Generates `experiments_tracker.xlsx` with current run status and metrics.

---

## Training Conditions

| ID | Name | Backbone | Head | Description |
|----|------|----------|------|-------------|
| C1 | `frozen` | Frozen | Linear only | No backbone adaptation |
| C2 | `layer4` | Layer4 free | Linear only | Partial fine-tuning |
| C3 | `head_frozen` | Frozen | Projection + Linear | Full head, frozen backbone |
| C4 | `head_layer4` | Layer4 free | Projection + Linear | Full head + fine-tuning |

---

## Preliminary Results (ResNet-18)

> Experiments in progress. Results shown are best validation metrics across 60 epochs.

### kaggle_fruits_quality (359 images — Fresh / Rotten)

| Condition | Label Mode | Acc | F1 | Precision | Recall |
|-----------|-----------|-----|-----|-----------|--------|
| C1 frozen | state | 86.1% | 86.1% | 86.2% | 86.1% |
| C2 layer4 | state | 91.7% | 91.6% | 92.2% | 91.7% |
| C3 head_frozen | state | 86.1% | 86.0% | 87.1% | 86.1% |
| C4 head_layer4 | state | 88.9% | 88.9% | 89.0% | 88.9% |

### mendeley_fruits (1,655 images — Peach / Pomegranate / Strawberry)

| Condition | Label Mode | Acc | F1 | Precision | Recall |
|-----------|-----------|-----|-----|-----------|--------|
| C1 frozen | state | 94.3% | 94.3% | 94.5% | 94.3% |
| C1 frozen | fruit_state | 90.9% | 91.0% | 91.1% | 91.3% |
| C2 layer4 | state | 95.5% | 95.5% | 95.7% | 95.5% |
| C2 layer4 | fruit_state | 92.7% | 92.7% | 92.8% | 92.9% |
| C3 head_frozen | state | 95.2% | 95.2% | 95.2% | 95.2% |
| C3 head_frozen | fruit_state | 94.3% | 94.1% | 94.1% | 94.2% |
| **C4 head_layer4** | **state** | **96.4%** | **96.4%** | **96.5%** | **96.4%** |
| C4 head_layer4 | fruit_state | 93.4% | 93.4% | 93.4% | 93.5% |

### mendeley_lemon_varieties (1,956 images — 7 lemon varieties)

| Condition | Label Mode | Acc | F1 | Precision | Recall |
|-----------|-----------|-----|-----|-----------|--------|
| C1 frozen | state | 96.2% | 96.2% | 96.3% | 96.2% |
| C2 layer4 | state | 98.0% | 98.0% | 98.0% | 98.0% |
| C3 head_frozen | state | 96.2% | 96.2% | 96.3% | 96.2% |
| **C4 head_layer4** | **state** | **98.5%** | **98.5%** | **98.5%** | **98.5%** |

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

## Experiment Status

See `experiments_tracker.xlsx` for full status across all datasets, backbones and conditions.
