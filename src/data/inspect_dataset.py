from pathlib import Path
import pandas as pd


DATA_PATH = list(Path("data/raw").glob("*.csv"))[0]

print(f"Dataset: {DATA_PATH.name}")
print("\nProfiling dataset...")


total_rows = 0
category_counts = {}
min_date = None
max_date = None

for chunk in pd.read_csv(
    DATA_PATH,
    usecols=["category", "published_date"],
    chunksize=50_000
):
    total_rows += len(chunk)

    # Count categories
    counts = chunk["category"].value_counts()

    for category, count in counts.items():
        category_counts[category] = (
            category_counts.get(category, 0) + count
        )

    # Parse publication dates
    dates = pd.to_datetime(
    chunk["published_date"],
    format="%m/%d/%y",
    errors="coerce"
)

    chunk_min = dates.min()
    chunk_max = dates.max()

    if pd.notna(chunk_min):
        min_date = chunk_min if min_date is None else min(min_date, chunk_min)

    if pd.notna(chunk_max):
        max_date = chunk_max if max_date is None else max(max_date, chunk_max)


print(f"\nTotal papers: {total_rows:,}")
print(f"Earliest publication date: {min_date}")
print(f"Latest publication date: {max_date}")

print("\nTop 15 research categories:")
for category, count in sorted(
    category_counts.items(),
    key=lambda x: x[1],
    reverse=True
)[:15]:
    print(f"{category}: {count:,}")