"""
Korean Fruit Freshness Image Scraper
=====================================
Downloads images of 5 Korean fruits in 3 states (Fresh / Semi-fresh / Rotten)
using Korean-language search queries to bias toward Korea-specific results.

Target structure:
korean_fruit_dataset/
├── apple_fresh/         (100 images)
├── apple_semifresh/     (100 images)
├── apple_rotten/        (100 images)
├── banana_fresh/        (100 images)
├── banana_semifresh/    (100 images)
├── banana_rotten/       (100 images)
├── strawberry_fresh/    (100 images)
├── strawberry_semifresh/(100 images)
├── strawberry_rotten/   (100 images)
├── mandarin_fresh/      (100 images)   <- Jeju gyul (귤)
├── mandarin_semifresh/  (100 images)
├── mandarin_rotten/     (100 images)
├── shinemuscat_fresh/   (100 images)
├── shinemuscat_semifresh/(100 images)
└── shinemuscat_rotten/  (100 images)

Total: 15 folders × 100 images = 1,500 images

Usage:
    pip install bing-image-downloader Pillow
    python korean_fruit_scraper.py

    Optional flags:
      --output_dir   : where to save (default: ./korean_fruit_dataset)
      --per_class    : images per class (default: 100)
      --min_size     : minimum image dimension in pixels (default: 100)
      --only         : comma-separated list to scrape a subset, e.g.
                       --only apple_fresh,banana_rotten
"""

import os
import sys
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
    os.system(f"{sys.executable} -m pip install bing-image-downloader --quiet")
    from bing_image_downloader import downloader


# ─────────────────────────────────────────────
# KOREAN-ONLY SEARCH QUERIES
# Each fruit has 3 states. Multiple queries per
# state increase diversity & raise the chance of
# hitting the 100-image target.
# ─────────────────────────────────────────────
FRUIT_QUERIES = {
    "apple": {
        "fresh": [
            "fresh apple photo",
            "red apple close up",
            "green apple",
            "shiny apple fruit",
            "crisp apple photography",
            "apple fruit white background",
            "fresh picked apple",
            "apple macro photo",
        ],
        "semifresh": [
            "bruised apple",
            "blemished apple photo",
            "wrinkled apple",
            "old apple photo",
            "damaged apple close up",
            "imperfect apple",
            "apple with bruise",
            "slightly overripe apple",
        ],
        "rotten": [
            "rotten apple photo",
            "moldy apple close up",
            "decayed apple",
            "rotting apple",
            "apple mold",
            "spoiled apple",
            "apple decay close up",
            "decomposed apple",
        ],
    },
    "banana": {
        "fresh": [
            "fresh banana photo",
            "yellow banana close up",
            "banana bunch",
            "ripe banana",
            "banana fruit photography",
            "banana white background",
            "fresh yellow banana",
        ],
        "semifresh": [
            "banana brown spots close up",
            "aging banana",
            "yellow banana dark spots",
            "ripe banana spots",
            "banana ripening",
            "banana turning brown",
            "banana with spots",
            "banana age spots fruit",
        ],
        "rotten": [
            "rotten banana photo",
            "black banana",
            "moldy banana close up",
            "decayed banana",
            "rotting banana",
            "spoiled banana",
            "banana mold",
            "banana decay",
        ],
    },
    "strawberry": {
        "fresh": [
            "fresh strawberry photo",
            "red strawberry close up",
            "strawberry fruit",
            "ripe strawberry",
            "strawberry macro photography",
            "fresh strawberries white background",
            "strawberry single",
            "perfect strawberry",
        ],
        "semifresh": [
            "overripe strawberry",
            "old strawberry photo",
            "wilted strawberry",
            "soft strawberry",
            "bruised strawberry close up",
            "mushy strawberry",
            "strawberry going bad",
            "strawberry overripe",
        ],
        "rotten": [
            "rotten strawberry photo",
            "moldy strawberry close up",
            "decayed strawberry",
            "rotting strawberry",
            "spoiled strawberry",
            "strawberry mold",
            "strawberry decay",
            "strawberry fungus",
        ],
    },
    "mandarin": {
        "fresh": [
            "fresh mandarin orange photo",
            "mandarin orange close up",
            "clementine fruit",
            "fresh tangerine",
            "mandarin fruit photography",
            "bright orange tangerine",
            "mandarin citrus white background",
            "fresh mandarin macro",
        ],
        "semifresh": [
            "wrinkled mandarin orange",
            "old tangerine photo",
            "shriveled mandarin",
            "overripe mandarin",
            "soft mandarin orange",
            "old clementine",
        ],
        "rotten": [
            "rotten mandarin orange photo",
            "moldy tangerine close up",
            "decayed mandarin",
            "rotting orange mold",
            "spoiled mandarin",
            "mandarin mold close up",
        ],
    },
    "shinemuscat": {
        "fresh": [
            "shine muscat grapes",
            "green grapes close up photo",
            "muscat grapes fresh",
            "green seedless grapes",
            "fresh green grapes bunch",
            "muscat grape photography",
            "green grape cluster",
            "shine muscat grape bunch",
        ],
        "semifresh": [
            "shriveled green grapes",
            "old muscat grapes",
            "aging green grapes",
            "muscat grapes old",
            "green grape overripe",
            "wrinkled muscat grape",
            "green grapes drying out",
            "grape cluster wilting",
        ],
        "rotten": [
            "rotten grapes photo",
            "moldy grapes close up",
            "decayed grapes",
            "rotting grapes",
            "spoiled grapes",
            "grape mold",
            "grape decay close up",
            "fungus grapes",
        ],
    },
}


def validate_and_process_image(source_path: Path, dest_path: Path, min_size: int = 100) -> bool:
    """
    Validate downloaded image and process it for the dataset.
    Returns True if successful, False otherwise.
    """
    try:
        # Verify the file is a valid image
        img = Image.open(source_path)
        img.verify()

        # Re-open after verify (verify closes the file)
        img = Image.open(source_path)
        w, h = img.size

        if w < min_size or h < min_size:
            return False

        # Convert to RGB and resize to 256x256 for ML training consistency
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((256, 256), Image.LANCZOS)
        img.save(dest_path, "JPEG", quality=90)
        return True
    except Exception:
        if dest_path.exists():
            try:
                dest_path.unlink()
            except Exception:
                pass
        return False


def scrape_class(
    queries: list,
    save_dir: Path,
    target_count: int,
    min_size: int,
    delay: float = 0.4,
    temp_root: Path = Path("./_temp_downloads"),
) -> int:
    """
    Download images using Bing Image Downloader, validate, and organize them.
    Stops when target_count valid images are collected.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(exist_ok=True)
    saved = 0

    # Skip queries we've already satisfied (resume support)
    saved = sum(1 for _ in save_dir.glob("*.jpg"))
    if saved >= target_count:
        print(f"    [skip] already have {saved}/{target_count} images")
        return saved

    for query_idx, query in enumerate(queries):
        if saved >= target_count:
            break

        remaining = target_count - saved
        print(f"    Query {query_idx+1}/{len(queries)}: '{query}' (need {remaining} more)")

        query_temp_dir = temp_root / f"q_{uuid.uuid4().hex[:8]}"

        try:
            # Download a healthy buffer — many images fail validation
            downloader.download(
                query,
                limit=max(remaining * 3, 25),
                output_dir=str(query_temp_dir),
                adult_filter_off=True,   # we need rotten/moldy imagery
                force_replace=False,
                timeout=60,
                verbose=False,
            )

            # bing-image-downloader nests results inside a folder named after the query
            downloaded_folder = query_temp_dir / query
            if not downloaded_folder.exists():
                # Some versions sanitize the folder name — fall back to the only subfolder
                subfolders = [p for p in query_temp_dir.iterdir() if p.is_dir()]
                if subfolders:
                    downloaded_folder = subfolders[0]

            if downloaded_folder.exists():
                got_this_query = 0
                for img_file in sorted(downloaded_folder.iterdir()):
                    if saved >= target_count:
                        break
                    if not img_file.is_file():
                        continue

                    dest_path = save_dir / f"{uuid.uuid4().hex[:12]}.jpg"
                    if validate_and_process_image(img_file, dest_path, min_size):
                        saved += 1
                        got_this_query += 1
                        if got_this_query % 10 == 0:
                            print(f"      + {got_this_query} valid so far ({saved}/{target_count})")

                print(f"      = {got_this_query} valid images from this query")
            else:
                print(f"      x No images downloaded for this query")

        except Exception as e:
            print(f"      x Error: {e}")
        finally:
            if query_temp_dir.exists():
                shutil.rmtree(query_temp_dir, ignore_errors=True)

        if saved < target_count:
            time.sleep(delay)

    return saved


def build_dataset(
    output_dir: str = "./korean_fruit_dataset",
    per_class: int = 100,
    min_size: int = 100,
    only: list = None,
):
    output_path = Path(output_dir)

    # Build the list of (fruit, state) jobs to run
    jobs = []
    for fruit, states in FRUIT_QUERIES.items():
        for state, queries in states.items():
            class_name = f"{fruit}_{state}"
            if only and class_name not in only:
                continue
            jobs.append((class_name, queries))

    print("\n" + "=" * 60)
    print("  Korean Fruit Freshness Dataset Builder")
    print("=" * 60)
    print(f"  Output dir   : {output_path.resolve()}")
    print(f"  Per class    : {per_class} images")
    print(f"  Classes      : {len(jobs)}")
    print(f"  Total target : {per_class * len(jobs)} images")
    print("=" * 60 + "\n")

    summary = {}
    for class_name, queries in jobs:
        print(f"\n{'='*60}")
        print(f"[{class_name.upper()}]  target: {per_class} images")
        print(f"{'='*60}")

        save_dir = output_path / class_name
        # Polite delay — longer for spoiled-fruit queries (smaller index, more strain)
        delay = 0.6 if class_name.endswith(("_rotten", "_semifresh")) else 0.4
        got = scrape_class(
            queries=queries,
            save_dir=save_dir,
            target_count=per_class,
            min_size=min_size,
            delay=delay,
        )
        summary[class_name] = got

        if got < per_class:
            print(f"  ! Got {got}/{per_class}  (under target)")
        else:
            print(f"  + Got {got}/{per_class}")

    # ── SUMMARY ──
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    total = 0
    for cls in sorted(summary.keys()):
        n = summary[cls]
        total += n
        bar_len = max(0, n // 4)
        bar = "#" * bar_len + "-" * max(0, (per_class // 4) - bar_len)
        print(f"  {cls:<25} {n:>4}/{per_class}  [{bar}]")
    print("-" * 60)
    print(f"  TOTAL: {total} images across {len(summary)} classes")
    print("=" * 60)

    # Cleanup any leftover temp folder
    temp = Path("./_temp_downloads")
    if temp.exists():
        shutil.rmtree(temp, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=str, default="./korean_fruit_dataset")
    parser.add_argument("--per_class", type=int, default=100)
    parser.add_argument("--min_size", type=int, default=100)
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="comma-separated list of class names to scrape, e.g. apple_fresh,banana_rotten",
    )
    args = parser.parse_args()

    only = [s.strip() for s in args.only.split(",") if s.strip()] or None

    build_dataset(
        output_dir=args.output_dir,
        per_class=args.per_class,
        min_size=args.min_size,
        only=only,
    )


if __name__ == "__main__":
    main()
