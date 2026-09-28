from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = Path("data/processed/research_corpus.csv")
OUTPUT_DIR = Path("data/processed/eda")


def main():

    print("Loading research corpus...")

    df = pd.read_csv(DATA_PATH)

    # Convert dates
    df["published_date"] = pd.to_datetime(
        df["published_date"],
        errors="coerce"
    )

    # Extract year
    df["year"] = df["published_date"].dt.year

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # 1. Papers per year
    # --------------------------------------------------

    yearly_counts = (
        df.groupby("year")
        .size()
        .reset_index(name="paper_count")
    )

    print("\nPapers per year:")
    print(yearly_counts.to_string(index=False))

    yearly_counts.to_csv(
        OUTPUT_DIR / "papers_per_year.csv",
        index=False
    )

    # Plot
    plt.figure(figsize=(10, 6))

    plt.plot(
        yearly_counts["year"],
        yearly_counts["paper_count"],
        marker="o"
    )

    plt.title("AI/ML Research Papers Published per Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Papers")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "papers_per_year.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------
    # 2. Category distribution by year
    # --------------------------------------------------

    category_year = (
        df.groupby(["year", "category"])
        .size()
        .reset_index(name="paper_count")
    )

    category_year.to_csv(
        OUTPUT_DIR / "category_year_counts.csv",
        index=False
    )

    # --------------------------------------------------
    # 3. Category share by year
    # --------------------------------------------------

    yearly_totals = (
        category_year.groupby("year")["paper_count"]
        .sum()
        .reset_index(name="total")
    )

    category_share = category_year.merge(
        yearly_totals,
        on="year"
    )

    category_share["share"] = (
        category_share["paper_count"]
        / category_share["total"]
    )

    category_share.to_csv(
        OUTPUT_DIR / "category_share_by_year.csv",
        index=False
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\nYearly range:")
    print(
        f"{yearly_counts['year'].min()} → "
        f"{yearly_counts['year'].max()}"
    )

    print("\nEDA files saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()