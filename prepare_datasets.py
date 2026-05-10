"""
prepare_datasets.py — Reorganize all compatible datasets into a consistent structure.

Source datasets (in datasets/):                Output (in data/):
─────────────────────────────────────────────────────────────────────────────────
mendeley_fruitvision/                          data/mendeley_fruitvision/
  Apple/Fresh|Formalin-mixed|Rotten/             apple/fresh|formalin|rotten/

mendeley_fruits_classification/                data/mendeley_fruits/
  fresh_peaches_done/ rotten_peaches_done/ ...   peach|pomegranate|strawberry/fresh|rotten/

kaggle_fruits_fresh_rotten/                    data/kaggle_fruits_fresh_rotten/
  dataset/train|test/freshapples|rottenapples/   apple|banana|orange/fresh|rotten/

kaggle_fruits_quality/                         data/kaggle_fruits_quality/
  Quality Dataset/train|test|valid/fresh|rotten/ train|test/fresh|rotten/  (Layout B)

mendeley_lemon_varieties/                      data/mendeley_lemon_varieties/
  Fresh Lemon/{7 varieties}/  Rotten Lemon/      lemon/fresh|rotten/

kaggle_fresh_stale_classification/             data/kaggle_fresh_stale/
  Train|Test/fresh{fruit}|rotten{fruit}/         apple|banana|...|tomato/fresh|rotten/

Files are COPIED — originals are never touched.
Idempotent: safe to run multiple times.

Usage:
    python prepare_datasets.py                  ← all datasets
    python prepare_datasets.py --dry_run        ← preview without copying
    python prepare_datasets.py --only fruits    ← one dataset only

--only choices: fruitvision, fruits, kaggle_fresh_rotten, kaggle_quality,
                lemons, kaggle_fresh_stale
"""

import os
import shutil
import argparse
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

HERE         = Path(__file__).parent
DATASETS_DIR = HERE / "datasets"
DST_ROOT     = HERE / "data"
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ─────────────────────────────────────────────────────────────
# Mappings
# ─────────────────────────────────────────────────────────────

FRUITVISION_STATE_MAP = {
    "Fresh"          : "fresh",
    "Formalin-mixed" : "formalin",
    "Rotten"         : "rotten",
}

MENDELEY_FRUITS_MAP = {
    "fresh_peaches_done"       : ("peach",       "fresh"),
    "rotten_peaches_done"      : ("peach",       "rotten"),
    "fresh_pomegranates_done"  : ("pomegranate", "fresh"),
    "rotten_pomegranates_done" : ("pomegranate", "rotten"),
    "fresh_strawberries_done"  : ("strawberry",  "fresh"),
    "rotten_strawberries_done" : ("strawberry",  "rotten"),
}

KAGGLE_FRESH_ROTTEN_MAP = {
    "freshapples"   : ("apple",  "fresh"),
    "rottenapples"  : ("apple",  "rotten"),
    "freshbanana"   : ("banana", "fresh"),
    "rottenbanana"  : ("banana", "rotten"),
    "freshoranges"  : ("orange", "fresh"),
    "rottenoranges" : ("orange", "rotten"),
}

# Handles typos in test set (patato, tamto)
KAGGLE_FRESH_STALE_MAP = {
    "freshapples"      : ("apple",       "fresh"),
    "rottenapples"     : ("apple",       "rotten"),
    "freshbanana"      : ("banana",      "fresh"),
    "rottenbanana"     : ("banana",      "rotten"),
    "freshbittergroud" : ("bittergourd", "fresh"),
    "rottenbittergroud": ("bittergourd", "rotten"),
    "freshcapsicum"    : ("capsicum",    "fresh"),
    "rottencapsicum"   : ("capsicum",    "rotten"),
    "freshcucumber"    : ("cucumber",    "fresh"),
    "rottencucumber"   : ("cucumber",    "rotten"),
    "freshokra"        : ("okra",        "fresh"),
    "rottenokra"       : ("okra",        "rotten"),
    "freshoranges"     : ("orange",      "fresh"),
    "rottenoranges"    : ("orange",      "rotten"),
    "freshpotato"      : ("potato",      "fresh"),
    "rottenpotato"     : ("potato",      "rotten"),
    "freshpatato"      : ("potato",      "fresh"),   # typo in test set
    "rottenpatato"     : ("potato",      "rotten"),  # typo in test set
    "freshtomato"      : ("tomato",      "fresh"),
    "rottentomato"     : ("tomato",      "rotten"),
    "freshtamto"       : ("tomato",      "fresh"),   # typo in test set
    "rottentamto"      : ("tomato",      "rotten"),  # typo in test set
}


# ─────────────────────────────────────────────────────────────
# Core helper
# ─────────────────────────────────────────────────────────────

def copy_images(src_dir: Path, dst_dir: Path, dry_run: bool) -> int:
    if not src_dir.exists():
        return 0
    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in src_dir.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            if not dry_run and not (dst_dir / f.name).exists():
                shutil.copy2(f, dst_dir / f.name)
            count += 1
    return count


def _header(name, src, dst):
    print(f"\n{'='*58}")
    print(f"  {name}")
    print(f"  Source : {src}")
    print(f"  Output : {dst}")
    print(f"{'='*58}")


# ─────────────────────────────────────────────────────────────
# Dataset 1 — Mendeley FruitVision
# ─────────────────────────────────────────────────────────────

def prepare_fruitvision(dry_run: bool):
    src = DATASETS_DIR / "mendeley_fruitvision"
    dst = DST_ROOT / "mendeley_fruitvision"
    _header("Mendeley FruitVision  (fresh / formalin / rotten)", src, dst)
    if not src.exists():
        print(f"  [ERROR] Not found: {src}"); return

    total = 0
    for fruit_dir in sorted(src.iterdir()):
        if not fruit_dir.is_dir(): continue
        fruit = fruit_dir.name.lower().replace(" ", "_")
        for state_dir in sorted(fruit_dir.iterdir()):
            if not state_dir.is_dir(): continue
            state = FRUITVISION_STATE_MAP.get(state_dir.name)
            if not state:
                print(f"  [WARN] Unknown state: {state_dir.name}"); continue
            n = copy_images(state_dir, dst / fruit / state, dry_run)
            total += n
            print(f"  {fruit:<12} / {state:<10}  {n:>4} imgs")
    print(f"\n  Total: {total} images")


# ─────────────────────────────────────────────────────────────
# Dataset 2 — Mendeley Fruits Classification
# ─────────────────────────────────────────────────────────────

def prepare_mendeley_fruits(dry_run: bool):
    src = DATASETS_DIR / "mendeley_fruits_classification"
    dst = DST_ROOT / "mendeley_fruits"
    _header("Mendeley Fruits Classification  (fresh / rotten)", src, dst)
    if not src.exists():
        print(f"  [ERROR] Not found: {src}"); return

    total = 0
    for folder in sorted(src.iterdir()):
        if not folder.is_dir(): continue
        mapping = MENDELEY_FRUITS_MAP.get(folder.name)
        if not mapping:
            print(f"  [WARN] Unknown folder: {folder.name}"); continue
        fruit, state = mapping
        n = copy_images(folder, dst / fruit / state, dry_run)
        total += n
        print(f"  {fruit:<12} / {state:<10}  {n:>4} imgs")
    print(f"\n  Total: {total} images")


# ─────────────────────────────────────────────────────────────
# Dataset 3 — Kaggle Fruits Fresh and Rotten
# ─────────────────────────────────────────────────────────────

def prepare_kaggle_fresh_rotten(dry_run: bool):
    src = DATASETS_DIR / "kaggle_fruits_fresh_rotten" / "dataset"
    dst = DST_ROOT / "kaggle_fruits_fresh_rotten"
    _header("Kaggle Fruits Fresh and Rotten  (apple/banana/orange × fresh/rotten)", src, dst)
    if not src.exists():
        print(f"  [ERROR] Not found: {src}"); return

    total = 0
    for split in ["train", "test"]:
        split_dir = src / split
        if not split_dir.exists(): continue
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir(): continue
            mapping = KAGGLE_FRESH_ROTTEN_MAP.get(cls_dir.name.lower())
            if not mapping:
                print(f"  [WARN] Unknown class: {cls_dir.name}"); continue
            fruit, state = mapping
            n = copy_images(cls_dir, dst / fruit / state, dry_run)
            total += n
            print(f"  [{split}] {fruit:<8} / {state:<8}  {n:>4} imgs")
    print(f"\n  Total: {total} images")


# ─────────────────────────────────────────────────────────────
# Dataset 4 — Kaggle Fruits Quality (Layout B: train/test)
# ─────────────────────────────────────────────────────────────

def prepare_kaggle_quality(dry_run: bool):
    src = DATASETS_DIR / "kaggle_fruits_quality" / "Quality Dataset"
    dst = DST_ROOT / "kaggle_fruits_quality"
    _header("Kaggle Fruits Quality  (fresh / rotten — 12 fruits mixed)", src, dst)
    if not src.exists():
        print(f"  [ERROR] Not found: {src}"); return

    total = 0
    # valid/ is merged into test/
    split_map = {"train": "train", "test": "test", "valid": "test"}
    for src_split, dst_split in split_map.items():
        split_dir = src / src_split
        if not split_dir.exists(): continue
        for state_dir in sorted(split_dir.iterdir()):
            if not state_dir.is_dir(): continue
            state = state_dir.name.lower()
            if state not in ("fresh", "rotten"):
                print(f"  [WARN] Unknown state: {state_dir.name}"); continue
            n = copy_images(state_dir, dst / dst_split / state, dry_run)
            total += n
            print(f"  [{src_split}→{dst_split}] {state:<8}  {n:>4} imgs")
    print(f"\n  Total: {total} images")


# ─────────────────────────────────────────────────────────────
# Dataset 5 — Mendeley Lemon Varieties
# ─────────────────────────────────────────────────────────────

def prepare_lemons(dry_run: bool):
    src = DATASETS_DIR / "mendeley_lemon_varieties"
    dst = DST_ROOT / "mendeley_lemon_varieties"
    _header("Mendeley Lemon Varieties  (fresh=7 varieties merged / rotten)", src, dst)
    if not src.exists():
        print(f"  [ERROR] Not found: {src}"); return

    total = 0
    fresh_dir  = src / "Fresh Lemon"
    rotten_dir = src / "Rotten Lemon"

    # Fresh: merge all 7 lemon varieties into lemon/fresh/
    if fresh_dir.exists():
        for variety_dir in sorted(fresh_dir.iterdir()):
            if not variety_dir.is_dir(): continue
            n = copy_images(variety_dir, dst / "lemon" / "fresh", dry_run)
            total += n
            print(f"  {variety_dir.name:<25} → lemon/fresh   {n:>4} imgs")

    # Rotten: single folder → lemon/rotten/
    if rotten_dir.exists():
        n = copy_images(rotten_dir, dst / "lemon" / "rotten", dry_run)
        total += n
        print(f"  {'Rotten Lemon':<25} → lemon/rotten  {n:>4} imgs")

    print(f"\n  Total: {total} images")


# ─────────────────────────────────────────────────────────────
# Dataset 6 — Kaggle Fresh and Stale Classification
# ─────────────────────────────────────────────────────────────

def prepare_kaggle_fresh_stale(dry_run: bool):
    src = DATASETS_DIR / "kaggle_fresh_stale_classification"
    dst = DST_ROOT / "kaggle_fresh_stale"
    _header("Kaggle Fresh and Stale  (9 fruits+veg × fresh/rotten)", src, dst)
    if not src.exists():
        print(f"  [ERROR] Not found: {src}"); return

    total = 0
    for split in ["Train", "Test"]:
        split_dir = src / split
        if not split_dir.exists(): continue
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir(): continue
            mapping = KAGGLE_FRESH_STALE_MAP.get(cls_dir.name.lower())
            if not mapping:
                print(f"  [WARN] Unknown class: {cls_dir.name}"); continue
            fruit, state = mapping
            n = copy_images(cls_dir, dst / fruit / state, dry_run)
            total += n
            print(f"  [{split}] {fruit:<12} / {state:<8}  {n:>4} imgs")
    print(f"\n  Total: {total} images")


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────

def print_summary():
    if not DST_ROOT.exists():
        return
    print(f"\n{'='*58}")
    print(f"  Final structure: data/")
    print(f"{'='*58}")
    for ds_dir in sorted(DST_ROOT.iterdir()):
        if not ds_dir.is_dir(): continue
        print(f"\n  {ds_dir.name}/")
        for sub in sorted(ds_dir.iterdir()):
            if not sub.is_dir(): continue
            # Layout B (train/test) or Layout C (fruit/state)
            children = [d for d in sub.iterdir() if d.is_dir()]
            if children:
                parts = []
                for child in sorted(children):
                    n = sum(1 for f in child.iterdir()
                            if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
                    parts.append(f"{child.name}({n})")
                print(f"    {sub.name:<18} {' | '.join(parts)}")
            else:
                n = sum(1 for f in sub.iterdir()
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
                print(f"    {sub.name:<18} {n} imgs")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

CHOICES = ["fruitvision", "fruits", "kaggle_fresh_rotten",
           "kaggle_quality", "lemons", "kaggle_fresh_stale"]

RUNNERS = {
    "fruitvision"       : prepare_fruitvision,
    "fruits"            : prepare_mendeley_fruits,
    "kaggle_fresh_rotten": prepare_kaggle_fresh_rotten,
    "kaggle_quality"    : prepare_kaggle_quality,
    "lemons"            : prepare_lemons,
    "kaggle_fresh_stale": prepare_kaggle_fresh_stale,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry_run", action="store_true",
                        help="Preview without copying any files")
    parser.add_argument("--only", choices=CHOICES,
                        help="Process only one dataset")
    args = parser.parse_args()

    if args.dry_run:
        print("\n  [DRY RUN] No files will be copied.\n")

    if args.only:
        RUNNERS[args.only](args.dry_run)
    else:
        for fn in RUNNERS.values():
            fn(args.dry_run)

    if not args.dry_run:
        print_summary()
    else:
        print("\n  Run without --dry_run to actually copy the files.")
