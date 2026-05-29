# Korean Fruit Freshness Scraper

Scrapes ~100 Korea-specific images for each of these classes:

| Fruit              | Korean         | Fresh                   | Semi-fresh              | Rotten                    |
|--------------------|----------------|-------------------------|-------------------------|---------------------------|
| Apple              | 사과           | smooth skin, bright     | small bruises / spots   | dark patches / mold       |
| Banana             | 바나나         | yellow                  | brown spots             | dark brown / black        |
| Strawberry         | 딸기           | firm, red               | softened, slightly dark | mold / collapse           |
| Mandarin (Jeju gyul) | 귤 / 감귤    | smooth peel             | dull / soft spots       | mold / discoloration      |
| Shine Muscat       | 샤인머스캣     | plump, green, glossy    | softened / shrunken     | mold / collapse           |

5 fruits × 3 states × 100 images = **1,500 images** total.

## Install

```bash
pip install bing-image-downloader Pillow
```

## Run

```bash
# Full scrape (default 100 per class, all 15 classes)
python korean_fruit_scraper.py

# Custom output folder
python korean_fruit_scraper.py --output_dir ./my_dataset

# Only some classes (handy for re-runs / fixing under-quota classes)
python korean_fruit_scraper.py --only apple_rotten,shinemuscat_semifresh

# Smaller per-class count for a quick test
python korean_fruit_scraper.py --per_class 20
```

## Output structure

```
korean_fruit_dataset/
├── apple_fresh/         (100 .jpg, 256×256)
├── apple_semifresh/
├── apple_rotten/
├── banana_fresh/
├── banana_semifresh/
├── banana_rotten/
├── strawberry_fresh/
├── strawberry_semifresh/
├── strawberry_rotten/
├── mandarin_fresh/
├── mandarin_semifresh/
├── mandarin_rotten/
├── shinemuscat_fresh/
├── shinemuscat_semifresh/
└── shinemuscat_rotten/
```

Every image is validated (corrupt files dropped), converted to RGB JPEG, and resized to 256×256 so it loads cleanly into PyTorch's `ImageFolder` or any standard image classifier.

## Notes

- Search queries are Korean-only (사과, 딸기, 샤인머스캣, …) to bias results toward Korean produce.
- The "semi-fresh" and "rotten" classes are the hardest to fill — Bing's index is thinner for spoiled-fruit imagery. If you fall short of 100, re-run with `--only <class>` and the script will resume (it counts existing files in each folder before downloading more).
- After scraping, **manually skim each folder** — image search is noisy, and a 5-minute pass to delete obvious mismatches will dramatically improve a downstream classifier.
- If Bing rate-limits you, wait a few minutes and re-run; finished folders are skipped.
