from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

RAW_PATH = list(Path("data/raw").glob("*.csv"))[0]
OUTPUT_PATH = Path("data/processed/research_corpus.csv")


# --------------------------------------------------
# Configuration
# --------------------------------------------------

START_YEAR = 2015

TARGET_CATEGORIES = [
    "Machine Learning",
    "Computer Vision and Pattern Recognition",
    "Computation and Language (Natural Language Processing)",
    "Artificial Intelligence",
    "Machine Learning (Statistics)",
    "Neural and Evolutionary Computing",
]


# --------------------------------------------------
# Text cleaning
# --------------------------------------------------

def clean_text(text):
    """Basic cleaning for research abstracts."""

    if pd.isna(text):
        return ""

    text = str(text)

    # Normalize whitespace
    text = " ".join(text.split())

    return text.strip()


# --------------------------------------------------
# Main processing
# --------------------------------------------------

def main():

    print(f"Reading: {RAW_PATH.name}")
    print("Preparing research corpus...\n")

    processed_chunks = []

    for chunk in pd.read_csv(
        RAW_PATH,
        usecols=[
            "id",
            "title",
            "category",
            "category_code",
            "published_date",
            "authors",
            "first_author",
            "summary",
            "summary_word_count",
        ],
        chunksize=50_000,
    ):

        # Parse publication date
        chunk["published_date"] = pd.to_datetime(
            chunk["published_date"],
            format="%m/%d/%y",
            errors="coerce",
        )

        # Filter by publication year
        chunk = chunk[
            chunk["published_date"].dt.year >= START_YEAR
        ]

        # Keep target research categories
        chunk = chunk[
            chunk["category"].isin(TARGET_CATEGORIES)
        ]

        # Remove missing title/abstract
        chunk = chunk.dropna(
            subset=["title", "summary"]
        )

        # Clean text
        chunk["title"] = chunk["title"].apply(clean_text)
        chunk["summary"] = chunk["summary"].apply(clean_text)

        # Remove empty abstracts
        chunk = chunk[
            chunk["summary"].str.len() > 0
        ]

        processed_chunks.append(chunk)

    # Combine processed chunks
    df = pd.concat(
        processed_chunks,
        ignore_index=True,
    )

    # Remove duplicate papers
    df = df.drop_duplicates(
        subset="id"
    )

    # Sort chronologically
    df = df.sort_values(
        "published_date"
    ).reset_index(drop=True)

    # Save
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    print("Dataset preparation complete!\n")

    print(f"Total papers: {len(df):,}")

    print(
        f"Date range: "
        f"{df['published_date'].min().date()} → "
        f"{df['published_date'].max().date()}"
    )

    print("\nPapers by category:")

    print(
        df["category"]
        .value_counts()
        .to_string()
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()