from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


DATA_PATH = Path("data/processed/research_corpus.csv")


def main():

    print("Loading research corpus...")

    df = pd.read_csv(
        DATA_PATH,
        usecols=["id", "title", "summary"]
    )

    # Combine title and abstract
    df["text"] = (
        df["title"].fillna("")
        + " "
        + df["summary"].fillna("")
    )

    print(f"Papers loaded: {len(df):,}")

    # --------------------------------------------------
    # TF-IDF representation
    # --------------------------------------------------

    vectorizer = TfidfVectorizer(
    stop_words="english",
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b",
    min_df=10,
    max_df=0.90,
    ngram_range=(1, 2),
    max_features=20_000,
    sublinear_tf=True,
)

    print("\nBuilding TF-IDF matrix...")

    X = vectorizer.fit_transform(df["text"])

    print(f"Matrix shape: {X.shape}")
    print(f"Non-zero values: {X.nnz:,}")

    # --------------------------------------------------
    # Inspect vocabulary
    # --------------------------------------------------

    terms = vectorizer.get_feature_names_out()

    print(f"\nVocabulary size: {len(terms):,}")

    print("\nFirst 50 vocabulary terms:")

    print(terms[:50])


if __name__ == "__main__":
    main()