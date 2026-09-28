from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


DATA_PATH = Path("data/processed/research_corpus.csv")
OUTPUT_DIR = Path("data/processed/svd")

N_COMPONENTS = 100

TFIDF_CONFIG = {
    "stop_words": "english",
    "token_pattern": r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b",
    "min_df": 10,
    "max_df": 0.90,
    "ngram_range": (1, 2),
    "max_features": 20_000,
    "sublinear_tf": True,
}

def main():
    print("Loading research corpus...")

    df = pd.read_csv(
    DATA_PATH,
    usecols=["id", "title", "summary", "published_date", "category"]
)

    df["text"] = (
        df["title"].fillna("")
        + " "
        + df["summary"].fillna("")
    )

    print(f"Papers loaded: {len(df):,}")

    metadata = df[
        ["id", "title", "published_date", "category"]
    ].copy()

    print("\nBuilding TF-IDF matrix...")

    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    X = vectorizer.fit_transform(df["text"])

    print(f"TF-IDF matrix shape: {X.shape}")

    print("\nApplying Truncated SVD...")

    svd = TruncatedSVD(
        n_components=N_COMPONENTS,
        random_state=42
    )

    X_svd = svd.fit_transform(X)

    print(f"SVD matrix shape: {X_svd.shape}")

    print(
        f"Explained variance: "
        f"{svd.explained_variance_ratio_.sum():.2%}"
    )

    print("\nSaving SVD representation...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(
        OUTPUT_DIR / "paper_embeddings.npy",
        X_svd
    )

    print(
        f"Saved: {OUTPUT_DIR / 'paper_embeddings.npy'}"
    )

    metadata.to_csv(
        OUTPUT_DIR / "paper_metadata.csv",
        index=False
    )

    print(
        f"Saved: {OUTPUT_DIR / 'paper_metadata.csv'}"
    )


if __name__ == "__main__":
    main()