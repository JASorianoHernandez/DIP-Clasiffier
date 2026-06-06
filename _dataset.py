"""
dataset.py — Image loaders for the freshness dataset.

Supported layouts:

  Layout A — flat (AnusDraft style):
    dataset/
      apple_fresh/  apple_rotten/  banana_fresh/ ...

  Layout B — pre-split (TanzinaDraft style):
    dataset/
      train/  freshkimchi/  rottenkimchi/ ...
      test/   freshkimchi/  rottenkimchi/ ...

  Layout C — nested fruit/state (our main datasets after prepare_datasets.py):
    data/fruits_original/
      apple/
        fresh/    formalin/    rotten/
      banana/
        fresh/    formalin/    rotten/

    get_loaders_nested() handles this layout.
    Classification target is the STATE (fresh / formalin / rotten).
    Optionally filter by specific fruits.

Usage:
    # Layouts A / B
    train_loader, val_loader, num_classes, class_names = get_loaders("./dataset")

    # Layout C
    train_loader, val_loader, num_classes, class_names = get_loaders_nested(
        "./data/fruits_original",
        fruits=["apple", "banana"],   # None = all fruits
    )
"""

import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# ─────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─────────────────────────────────────────────────────────────
# Layout C — custom Dataset for fruit/state nested structure
# ─────────────────────────────────────────────────────────────

class NestedFruitDataset(Dataset):
    """
    Reads a fruit/state nested directory.

      root/apple/fresh/img.jpg
      root/apple/rotten/img.jpg

    label_mode:
      "state"       → class = state only  (fresh / rotten / formalin)
      "fruit_state" → class = fruit_state (apple_fresh / apple_rotten / ...)

    Optionally restrict to a subset of fruits.
    """

    def __init__(self, root: Path, fruits: list = None,
                 transform=None, label_mode: str = "state"):
        self.root       = root
        self.transform  = transform
        self.label_mode = label_mode
        self.samples    = []

        # Discover class names based on mode
        if label_mode == "state":
            all_classes = sorted({
                state_dir.name
                for fruit_dir in root.iterdir() if fruit_dir.is_dir()
                for state_dir in fruit_dir.iterdir() if state_dir.is_dir()
            })
        else:  # fruit_state
            all_classes = sorted({
                f"{fruit_dir.name}_{state_dir.name}"
                for fruit_dir in root.iterdir() if fruit_dir.is_dir()
                for state_dir in fruit_dir.iterdir() if state_dir.is_dir()
            })

        self.classes      = all_classes
        self.class_to_idx = {s: i for i, s in enumerate(all_classes)}

        # Collect samples
        for fruit_dir in sorted(root.iterdir()):
            if not fruit_dir.is_dir():
                continue
            if fruits and fruit_dir.name not in fruits:
                continue
            for state_dir in sorted(fruit_dir.iterdir()):
                if not state_dir.is_dir():
                    continue
                key   = state_dir.name if label_mode == "state" else f"{fruit_dir.name}_{state_dir.name}"
                label = self.class_to_idx.get(key)
                if label is None:
                    continue
                for img_path in sorted(state_dir.iterdir()):
                    if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTS:
                        self.samples.append((img_path, label))

        self.targets = [s[1] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def _stratified_split(targets: list, val_split: float, seed: int):
    """Returns (train_indices, val_indices) stratified by class."""
    generator = torch.Generator().manual_seed(seed)
    n = len(targets)
    perm = torch.randperm(n, generator=generator).tolist()

    class_indices = {}
    for idx in perm:
        class_indices.setdefault(targets[idx], []).append(idx)

    train_idx, val_idx = [], []
    for indices in class_indices.values():
        n_val = max(1, round(len(indices) * val_split))
        val_idx   += indices[:n_val]
        train_idx += indices[n_val:]

    return train_idx, val_idx


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def get_loaders(
    dataset_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.2,
    seed: int = 42,
):
    """
    Layouts A and B (flat or pre-split ImageFolder).
    Returns (train_loader, val_loader, num_classes, class_names).
    """
    root = Path(dataset_path)

    if (root / "train").exists() and (root / "test").exists():
        train_ds    = datasets.ImageFolder(root=str(root / "train"), transform=TRAIN_TRANSFORMS)
        val_ds      = datasets.ImageFolder(root=str(root / "test"),  transform=VAL_TRANSFORMS)
        num_classes = len(train_ds.classes)
        class_names = train_ds.classes

    else:
        full_train = datasets.ImageFolder(root=str(root), transform=TRAIN_TRANSFORMS)
        full_val   = datasets.ImageFolder(root=str(root), transform=VAL_TRANSFORMS)

        train_idx, val_idx = _stratified_split(full_train.targets, val_split, seed)

        train_ds    = Subset(full_train, train_idx)
        val_ds      = Subset(full_val,   val_idx)
        num_classes = len(full_train.classes)
        class_names = full_train.classes

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, num_classes, class_names


def get_loaders_nested(
    dataset_path: str,
    fruits: list = None,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.2,
    seed: int = 42,
    label_mode: str = "state",
):
    """
    Layout C — nested fruit/state structure.
    Returns (train_loader, val_loader, num_classes, class_names).

    fruits     : list of fruit names to include. None = all fruits.
    label_mode : "state"       → classify by state (fresh/rotten/formalin)
                 "fruit_state" → classify by fruit+state (apple_fresh/apple_rotten/...)
    """
    root = Path(dataset_path)

    full_train = NestedFruitDataset(root, fruits=fruits, transform=TRAIN_TRANSFORMS, label_mode=label_mode)
    full_val   = NestedFruitDataset(root, fruits=fruits, transform=VAL_TRANSFORMS,   label_mode=label_mode)

    train_idx, val_idx = _stratified_split(full_train.targets, val_split, seed)

    train_ds = Subset(full_train, train_idx)
    val_ds   = Subset(full_val,   val_idx)

    # val_split=1.0 (pure evaluation) leaves no training samples — a DataLoader
    # with shuffle over 0 samples would crash, so return None in that case.
    train_loader = (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=True)
        if len(train_ds) > 0 else None
    )
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, len(full_train.classes), full_train.classes


def get_available_fruits(dataset_path: str) -> list:
    """Return sorted list of fruit names found in a nested dataset."""
    root = Path(dataset_path)
    return sorted([d.name for d in root.iterdir() if d.is_dir()])
