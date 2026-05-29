"""
Korean Food Freshness Image Scraper
=====================================
Downloads images of Korean foods (fresh + spoiled/rotten states)
from DuckDuckGo image search and organizes them into the EXACT
same folder structure as the Kaggle fresh/rotten dataset.

Target structure:
dataset/
├── train/
│   ├── freshkimchi/
│   ├── rottenkimchi/
│   ├── freshrice/
│   ├── rottenrice/
│   ├── freshtteok/
│   ├── rottentteok/
│   ├── freshbanchan/        (Korean side dish)
│   └── rottenbanchan/
└── test/
    ├── freshkimchi/
    ├── rottenkimchi/
    ... (same structure)

Usage:
    pip install requests duckduckgo_search Pillow tqdm
    python korean_food_scraper.py

    Optional flags:
    --output_dir   : where to save (default: ./korean_food_dataset)
    --per_class    : images per class (default: 80)
    --test_split   : fraction for test set (default: 0.2)
    --min_size     : minimum image dimension in pixels (default: 100)
"""

import os
import time
import uuid
import argparse
import shutil
from pathlib import Path
from PIL import Image

try:
    from bing_image_downloader import downloader
except ImportError:
    print("Installing bing-image-downloader...")
    os.system("pip install bing-image-downloader --quiet")
    from bing_image_downloader import downloader


# ─────────────────────────────────────────────
# SEARCH QUERIES
# Each food has FRESH queries and ROTTEN queries.
# Multiple queries per class = more diverse images.
# ─────────────────────────────────────────────
FOOD_QUERIES = {
    "kimchi": {
        "fresh": [
            "fresh kimchi bright red color",
            "new kimchi cabbage napa",
            "homemade kimchi jar",
            "Korean kimchi side dish restaurant",
            "kimchi bowl closeup fresh",
        ],
        "rotten": [
            "moldy kimchi white fungus",
            "spoiled fermented cabbage mold",
            "rotten kimchi bacteria",
            "expired kimchi fuzzy mold",
            "kimchi food waste spoiled",
            "moldy fermented vegetables",
        ],
    },
    "rice": {
        "fresh": [
            "fresh cooked white rice bowl",
            "steamed rice Korean meal",
            "fluffy white rice closeup",
            "fresh rice bowl Asian",
            "white rice grain texture",
        ],
        "rotten": [
            "moldy rice food waste",
            "spoiled cooked rice green mold",
            "rotten rice bacteria fungus",
            "rice mold fuzzy white",
            "expired rice leftovers mold",
            "moldy grain rice",
        ],
    },
    "tteok": {
        "fresh": [
            "fresh Korean rice cake white",
            "tteok rice cake fresh made",
            "Korean mochi tteok",
            "rice cake garaetteok fresh",
            "white rice cake Korean cuisine",
        ],
        "rotten": [
            "moldy rice cake white fungus",
            "spoiled mochi mold",
            "rotten rice cake bacteria",
            "expired rice cake fuzzy",
            "moldy rice cake food waste",
            "rice cake mold green",
        ],
    },
    "banchan": {
        "fresh": [
            "fresh Korean banchan colorful",
            "Korean side dishes vegetables",
            "namul banchan fresh greens",
            "Korean vegetable side dish",
            "fresh Korean spinach banchan",
        ],
        "rotten": [
            "moldy vegetables food waste",
            "spoiled greens bacteria mold",
            "rotten vegetable side dish",
            "wilted moldy vegetables",
            "expired vegetable mold fuzzy",
            "moldy leafy greens",
        ],
    },
}


def validate_and_process_image(source_path: Path, dest_path: Path, min_size: int = 100) -> bool:
    """
    Validate downloaded image and process it for the dataset.
    Returns True if successful, False otherwise.
    Validates: is a real image, meets minimum size, not corrupt.
    """
    try:
        # Open and validate
        img = Image.open(source_path)
        img.verify()  # catches corrupt files

        # Re-open after verify (verify closes the file)
        img = Image.open(source_path)
        w, h = img.size

        # Check minimum size
        if w < min_size or h < min_size:
            return False

        # Convert to RGB JPEG for consistency
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize to 256x256 (model will crop to 224x224 during training)
        img = img.resize((256, 256), Image.LANCZOS)
        img.save(dest_path, "JPEG", quality=90)
        return True

    except Exception:
        # Silently fail but clean up any partial file
        if dest_path.exists():
            try:
                dest_path.unlink()
            except:
                pass
        return False


def scrape_class(
    queries: list,
    save_dir: Path,
    target_count: int,
    min_size: int,
    delay: float = 0.3,
) -> int:
    """
    Download images using Bing Image Downloader, validate, and organize them.
    Stops when target_count images are collected.
    Returns actual count saved.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path("./temp_downloads")
    temp_dir.mkdir(exist_ok=True)
    saved = 0

    for query_idx, query in enumerate(queries):
        if saved >= target_count:
            break

        remaining = target_count - saved
        print(f"    Query {query_idx+1}/{len(queries)}: '{query}' (need {remaining} more)")

        try:
            # Download to temp directory using Bing
            query_temp_dir = temp_dir / f"query_{query_idx}_{uuid.uuid4().hex[:6]}"

            downloader.download(
                query,
                limit=remaining * 2,  # Download extra to account for invalid images
                output_dir=str(query_temp_dir),
                adult_filter_off=True,  # Allow unappetizing images (for rotten food)
                force_replace=False,
                timeout=60,
                verbose=False
            )

            # Validate and move images from temp to final location
            # bing-image-downloader creates a subfolder with the query name
            downloaded_folder = query_temp_dir / query

            if downloaded_folder.exists():
                downloaded_this_query = 0
                for img_file in downloaded_folder.iterdir():
                    if saved >= target_count:
                        break

                    if img_file.is_file():
                        # Validate and process image
                        dest_path = save_dir / f"{uuid.uuid4().hex[:12]}.jpg"
                        if validate_and_process_image(img_file, dest_path, min_size):
                            saved += 1
                            downloaded_this_query += 1

                            if downloaded_this_query % 5 == 0:
                                print(f"      + Processed {downloaded_this_query} from this query ({saved}/{target_count} total)")

                print(f"      + Got {downloaded_this_query} valid images from this query")
            else:
                print(f"      x No images downloaded for this query")

            # Cleanup temp directory for this query
            if query_temp_dir.exists():
                shutil.rmtree(query_temp_dir, ignore_errors=True)

        except Exception as e:
            print(f"      x Error: {e}")
            # Clean up on error
            if 'query_temp_dir' in locals() and query_temp_dir.exists():
                shutil.rmtree(query_temp_dir, ignore_errors=True)
            continue

        # Small delay between queries to be polite
        if saved < target_count:
            time.sleep(delay)

    # Final cleanup of temp directory
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

    return saved


def build_dataset(
    output_dir: str = "./korean_food_dataset",
    per_class: int = 80,
    test_split: float = 0.2,
    min_size: int = 100,
):
    """
    Main function: scrape all classes, split into train/test,
    save in Kaggle-compatible folder structure.
    """
    output_path = Path(output_dir)
    train_count = int(per_class * (1 - test_split))
    test_count = per_class - train_count

    print("\n" + "=" * 55)
    print("  Korean Food Freshness Dataset Builder")
    print("=" * 55)
    print(f"  Output dir  : {output_path.resolve()}")
    print(f"  Per class   : {per_class} images ({train_count} train / {test_count} test)")
    print(f"  Foods       : {list(FOOD_QUERIES.keys())}")
    print(f"  Classes     : fresh + rotten per food")
    print(f"  Total target: {per_class * 2 * len(FOOD_QUERIES)} images")
    print("=" * 55 + "\n")

    summary = {}

    for food_name, states in FOOD_QUERIES.items():
        for state, queries in states.items():

            class_name = f"{state}{food_name}"  # e.g. "freshkimchi"
            print(f"\n{'='*55}")
            print(f"[{class_name.upper()}]")
            print(f"{'='*55}")

            # ── TRAIN ──
            train_dir = output_path / "train" / class_name
            print(f"  TRAIN SET (target: {train_count} images)...")
            # Use slower delay for rotten food to be more polite and get better results
            delay = 0.5 if state == "rotten" else 0.3
            train_saved = scrape_class(
                queries=queries,
                save_dir=train_dir,
                target_count=train_count,
                min_size=min_size,
                delay=delay,
            )

            if train_saved < train_count:
                print(f"  ! WARNING: Only got {train_saved}/{train_count} train images")
            else:
                print(f"  + SUCCESS: Got {train_saved}/{train_count} train images")

            # ── TEST ──
            test_dir = output_path / "test" / class_name
            print(f"\n  TEST SET (target: {test_count} images)...")
            test_saved = scrape_class(
                queries=queries,
                save_dir=test_dir,
                target_count=test_count,
                min_size=min_size,
                delay=delay,
            )

            if test_saved < test_count:
                print(f"  ! WARNING: Only got {test_saved}/{test_count} test images")
            else:
                print(f"  + SUCCESS: Got {test_saved}/{test_count} test images")

            summary[class_name] = {
                "train": train_saved,
                "test": test_saved,
            }

            total_for_class = train_saved + test_saved
            expected_for_class = per_class
            if total_for_class == 0:
                print(f"\n  xxx ERROR: NO IMAGES downloaded for {class_name}!")
            elif total_for_class < expected_for_class * 0.5:
                print(f"\n  !!! WARNING: Very few images ({total_for_class}/{expected_for_class}) for {class_name}")

            print(f"  Summary: {train_saved} train + {test_saved} test = {total_for_class} total")

    # ── SUMMARY ──
    print("\n" + "=" * 55)
    print("  FINAL SUMMARY")
    print("=" * 55)
    total_train = total_test = 0
    for cls, counts in summary.items():
        t = counts["train"]
        v = counts["test"]
        total_train += t
        total_test += v
        bar = "#" * (t // 5) + "-" * ((train_count - t) // 5)
        print(f"  {cls:<20} train={t:>3}  test={v:>3}  [{bar}]")
    print("-" * 55)
    print(f"  TOTAL                train={total_train}  test={total_test}")
    print("=" * 55)

    # ── WRITE LABELS FILE ──
    labels_path = output_path / "class_labels.txt"
    with open(labels_path, "w") as f:
        f.write("# Korean Food Freshness Dataset\n")
        f.write("# Class label -> folder name mapping\n")
        f.write("# Matches Kaggle sriramr/fruits-fresh-and-rotten format\n\n")
        for i, cls in enumerate(sorted(summary.keys())):
            f.write(f"{i}: {cls}\n")

    print(f"\n  Labels file saved: {labels_path}")
    print(f"  Dataset saved:     {output_path.resolve()}")
    print("\n  Ready to merge with Kaggle dataset or use standalone.")
    print("  Load in PyTorch with: torchvision.datasets.ImageFolder(root)")


# ─────────────────────────────────────────────
# MERGING UTILITY
# Run this after downloading both datasets to
# combine Kaggle + Korean into one master dataset
# ─────────────────────────────────────────────
def merge_with_kaggle(kaggle_dir: str, korean_dir: str, merged_dir: str):
    """
    Merges the Korean dataset into an existing Kaggle dataset folder.
    Korean folders are simply added alongside Kaggle folders.
    Both stay separate classes — no mixing of labels.

    Example after merge:
    merged/train/
        freshapples/      ← from Kaggle
        rottenapples/     ← from Kaggle
        freshbanana/      ← from Kaggle
        ...
        freshkimchi/      ← from Korean scraper  ← NEW
        rottenkimchi/     ← from Korean scraper  ← NEW
        freshrice/        ← from Korean scraper  ← NEW
        ...
    """
    import shutil

    kaggle_path = Path(kaggle_dir)
    korean_path = Path(korean_dir)
    merged_path = Path(merged_dir)

    for split in ["train", "test"]:
        # Copy Kaggle classes
        kaggle_split = kaggle_path / split
        if kaggle_split.exists():
            for class_folder in kaggle_split.iterdir():
                if class_folder.is_dir():
                    dest = merged_path / split / class_folder.name
                    shutil.copytree(class_folder, dest, dirs_exist_ok=True)

        # Copy Korean classes
        korean_split = korean_path / split
        if korean_split.exists():
            for class_folder in korean_split.iterdir():
                if class_folder.is_dir():
                    dest = merged_path / split / class_folder.name
                    shutil.copytree(class_folder, dest, dirs_exist_ok=True)

    print(f"Merged dataset saved to: {Path(merged_dir).resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape Korean food images into Kaggle-compatible structure"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./korean_food_dataset",
        help="Where to save the dataset",
    )
    parser.add_argument(
        "--per_class",
        type=int,
        default=80,
        help="Target images per class (default: 80)",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.2,
        help="Fraction for test set (default: 0.2)",
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=100,
        help="Minimum image dimension in pixels (default: 100)",
    )
    parser.add_argument(
        "--merge_kaggle",
        type=str,
        default=None,
        help="Path to existing Kaggle dataset to merge with",
    )
    parser.add_argument(
        "--merged_output",
        type=str,
        default="./merged_dataset",
        help="Output path for merged dataset",
    )

    args = parser.parse_args()

    # Run scraper
    build_dataset(
        output_dir=args.output_dir,
        per_class=args.per_class,
        test_split=args.test_split,
        min_size=args.min_size,
    )

    # Optionally merge with Kaggle
    if args.merge_kaggle:
        print("\nMerging with Kaggle dataset...")
        merge_with_kaggle(
            kaggle_dir=args.merge_kaggle,
            korean_dir=args.output_dir,
            merged_dir=args.merged_output,
        )
