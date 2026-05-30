"""
prepare_own_dataset.py — Resize and organize your own fruit photos into data/.

Source layout  (datasets/OWN_DATSET/):
  STRAWBERRY/
    Fresh/    img1.jpg  img2.jpg ...
    Rotten/   img1.jpg  img2.jpg ...
  APPLE/
    Fresh/    ...
    Rotten/   ...

Output layout  (data/own_dataset/):
  strawberry/
    fresh/    img1.jpg  img2.jpg ...
    rotten/   img1.jpg  img2.jpg ...

Each image is resized so its shortest side = 256 px (aspect ratio preserved),
matching the input expected by the training pipeline (Resize(256) → crop 224).

Usage:
    python prepare_own_dataset.py
"""

from pathlib import Path
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────
SRC_ROOT  = Path("datasets/OWN_DATSET")
DST_ROOT  = Path("data/own_dataset")
TARGET_PX = 256

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Image resize ──────────────────────────────────────────────

def resize_image(src: Path, dst: Path):
    """Resize so shortest side = TARGET_PX, save as JPEG quality 95."""
    img = Image.open(src).convert("RGB")
    w, h = img.size
    if w <= h:
        new_size = (TARGET_PX, int(h * TARGET_PX / w))
    else:
        new_size = (int(w * TARGET_PX / h), TARGET_PX)
    img = img.resize(new_size, Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, "JPEG", quality=95)


# ── Fruit scanner ─────────────────────────────────────────────

def scan_fruits():
    """Return list of (display_name, src_path, image_counts_per_state)."""
    if not SRC_ROOT.exists():
        return []
    results = []
    for fruit_dir in sorted(SRC_ROOT.iterdir()):
        if not fruit_dir.is_dir():
            continue
        counts = {}
        for state_dir in fruit_dir.iterdir():
            if not state_dir.is_dir():
                continue
            n = sum(1 for f in state_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
            counts[state_dir.name.lower()] = n
        results.append((fruit_dir.name, fruit_dir, counts))
    return results


# ── Preparation ───────────────────────────────────────────────

def prepare_fruit(fruit_name: str, src_fruit: Path):
    dst_fruit = DST_ROOT / fruit_name.lower()
    copied = skipped = errors = 0

    for state_dir in sorted(src_fruit.iterdir()):
        if not state_dir.is_dir():
            continue
        state_name = state_dir.name.lower()
        dst_state  = dst_fruit / state_name
        dst_state.mkdir(parents=True, exist_ok=True)

        images = [f for f in sorted(state_dir.iterdir())
                  if f.is_file() and f.suffix.lower() in IMAGE_EXTS]

        state_copied = state_skipped = 0
        for img_path in images:
            dst_img = dst_state / img_path.name
            if dst_img.exists():
                state_skipped += 1
                continue
            try:
                resize_image(img_path, dst_img)
                state_copied += 1
            except Exception as e:
                print(f"    WARN {img_path.name}: {e}")
                errors += 1

        print(f"    {state_name:<12} {state_copied:3d} resized  "
              f"{state_skipped:3d} already exist")
        copied  += state_copied
        skipped += state_skipped

    return copied, skipped, errors


# ── Menu ──────────────────────────────────────────────────────

def main():
    fruits = scan_fruits()

    if not fruits:
        print(f"No fruit folders found in {SRC_ROOT}/")
        print("Create a folder for each fruit, e.g.:")
        print("  datasets/OWN_DATSET/STRAWBERRY/Fresh/")
        print("  datasets/OWN_DATSET/STRAWBERRY/Rotten/")
        return

    print("\nOwn Dataset Preparation")
    print("=" * 50)
    print(f"Source  : {SRC_ROOT}")
    print(f"Output  : {DST_ROOT}")
    print(f"Resize  : short side → {TARGET_PX} px\n")

    print("Available fruits:")
    for i, (name, _, counts) in enumerate(fruits, 1):
        summary = "  ".join(f"{s}: {n}" for s, n in counts.items())
        print(f"  {i}. {name:<20} {summary}")
    print(f"  {len(fruits)+1}. All fruits\n")

    raw = input(f"Select [1-{len(fruits)+1}] or comma-separated (e.g. 1,3): ").strip()

    if raw == str(len(fruits) + 1) or raw == "":
        selected = fruits
    else:
        try:
            indices  = [int(x.strip()) - 1 for x in raw.split(",")]
            selected = [fruits[i] for i in indices if 0 <= i < len(fruits)]
        except (ValueError, IndexError):
            print("Invalid selection.")
            return

    if not selected:
        print("No fruits selected.")
        return

    print(f"\nPreparing: {', '.join(n for n, _, _ in selected)}\n")

    for name, src_path, _ in selected:
        print(f"[{name}]")
        copied, skipped, errors = prepare_fruit(name, src_path)
        total = copied + skipped
        print(f"  Total   : {total} images  "
              f"({copied} resized, {skipped} skipped, {errors} errors)\n")

    print(f"Done. Data ready at: {DST_ROOT}")
    print("\nNext step — add 'own_dataset' to train.py DATASETS dict, then:")
    print("  python train.py")


if __name__ == "__main__":
    main()
