"""
rename_own_dataset.py — Rename photos in datasets/OWN_DATSET/ to a clean convention.

Convention:  {STATE}_{FRUIT}_{number:03d}.jpg
  FR_SB_001.jpg  ← Fresh Strawberry #1
  RT_SB_001.jpg  ← Rotten Strawberry #1

State codes:
  FR = Fresh
  RT = Rotten
  SF = Semifresh  (future)

Fruit codes (extend this dict as you add more fruits):
  STRAWBERRY → SB
  APPLE      → AP
  BANANA     → BN
  ORANGE     → OR
  MANDARIN   → MD

Usage:
    python rename_own_dataset.py
    python rename_own_dataset.py --dry-run   (preview without renaming)
"""

import argparse
from pathlib import Path

SRC_ROOT   = Path("datasets/OWN_DATSET")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Code mappings ─────────────────────────────────────────────

STATE_CODES = {
    "fresh"     : "FR",
    "rotten"    : "RT",
    "semifresh" : "SF",
    "formalin"  : "FM",
}

FRUIT_CODES = {
    "strawberry" : "SB",
    "apple"      : "AP",
    "banana"     : "BN",
    "orange"     : "OR",
    "mandarin"   : "MD",
    "lemon"      : "LM",
    "grape"      : "GP",
    "mango"      : "MG",
    "peach"      : "PC",
}


def get_fruit_code(folder_name: str) -> str:
    """Return code for fruit, or first 2 uppercase chars if not in dict."""
    key = folder_name.lower()
    if key in FRUIT_CODES:
        return FRUIT_CODES[key]
    # fallback: first 2 letters uppercased
    code = folder_name[:2].upper()
    print(f"  WARN: '{folder_name}' not in FRUIT_CODES dict — using '{code}'")
    print(f"        Add it to FRUIT_CODES in rename_own_dataset.py for consistency.")
    return code


def rename_fruit(fruit_dir: Path, fruit_code: str, dry_run: bool):
    total_renamed = total_skipped = 0

    for state_dir in sorted(fruit_dir.iterdir()):
        if not state_dir.is_dir():
            continue

        state_key  = state_dir.name.lower()
        state_code = STATE_CODES.get(state_key)

        if state_code is None:
            print(f"  WARN: unknown state '{state_dir.name}' — skipping")
            continue

        images = sorted([
            f for f in state_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        ])

        if not images:
            print(f"  {state_dir.name}: no images found")
            continue

        print(f"  {state_dir.name} ({len(images)} images):")

        renamed = skipped = 0
        for i, img_path in enumerate(images, start=1):
            new_name = f"{state_code}_{fruit_code}_{i:03d}.jpg"
            new_path = img_path.parent / new_name

            if img_path.name == new_name:
                skipped += 1
                continue

            if new_path.exists():
                print(f"    SKIP  {new_name}  (target already exists)")
                skipped += 1
                continue

            if dry_run:
                print(f"    DRY   {img_path.name}  →  {new_name}")
            else:
                img_path.rename(new_path)
                print(f"    OK    {img_path.name}  →  {new_name}")
            renamed += 1

        action = "would rename" if dry_run else "renamed"
        print(f"    {action}: {renamed}   already correct: {skipped}\n")
        total_renamed += renamed
        total_skipped += skipped

    return total_renamed, total_skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview renames without touching any files")
    args = parser.parse_args()

    if not SRC_ROOT.exists():
        print(f"Source folder not found: {SRC_ROOT}")
        return

    fruit_dirs = sorted([d for d in SRC_ROOT.iterdir() if d.is_dir()])

    if not fruit_dirs:
        print(f"No fruit folders found in {SRC_ROOT}/")
        return

    print("\nOwn Dataset — Rename Tool")
    print("=" * 50)
    if args.dry_run:
        print("DRY RUN — no files will be changed\n")

    print("Available fruits:")
    for i, d in enumerate(fruit_dirs, 1):
        code = get_fruit_code(d.name)
        count = sum(1 for s in d.iterdir() if s.is_dir()
                    for f in s.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        print(f"  {i}. {d.name:<20} code={code}  ({count} images)")
    print(f"  {len(fruit_dirs)+1}. All fruits\n")

    raw = input(f"Select [1-{len(fruit_dirs)+1}]: ").strip()

    if raw == str(len(fruit_dirs) + 1) or raw == "":
        selected = fruit_dirs
    else:
        try:
            indices  = [int(x.strip()) - 1 for x in raw.split(",")]
            selected = [fruit_dirs[i] for i in indices if 0 <= i < len(fruit_dirs)]
        except (ValueError, IndexError):
            print("Invalid selection.")
            return

    print()
    total_r = total_s = 0
    for fruit_dir in selected:
        fruit_code = get_fruit_code(fruit_dir.name)
        print(f"[{fruit_dir.name}]  →  code: {fruit_code}")
        r, s = rename_fruit(fruit_dir, fruit_code, dry_run=args.dry_run)
        total_r += r
        total_s += s

    action = "Would rename" if args.dry_run else "Renamed"
    print(f"{action}: {total_r} files   Already correct: {total_s}")
    if args.dry_run:
        print("\nRun without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
