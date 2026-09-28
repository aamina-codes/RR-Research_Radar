from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


EMBEDDINGS_PATH = Path(
    "data/processed/svd/paper_embeddings.npy"
)

LABELS_PATH = Path(
    "data/processed/clusters/kmeans_30_labels.npy"
)

METADATA_PATH = Path(
    "data/processed/svd/paper_metadata.csv"
)

CORPUS_PATH = Path(
    "data/processed/research_corpus.csv"
)

OUTPUT_DIR = Path(
    "data/processed/topics"
)

TOP_TERMS = 15
TOP_PAPERS = 5


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

    print("Loading cluster information...")

    embeddings = np.load(EMBEDDINGS_PATH)
    labels = np.load(LABELS_PATH)

    metadata = pd.read_csv(METADATA_PATH)

    corpus = pd.read_csv(
        CORPUS_PATH,
        usecols=["id", "title", "summary"]
    )

    corpus["text"] = (
        corpus["title"].fillna("")
        + " "
        + corpus["summary"].fillna("")
    )

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Metadata shape: {metadata.shape}")
    print(f"Corpus shape: {corpus.shape}")

    if len(labels) != len(corpus):
        raise ValueError(
            "Number of cluster labels does not match "
            "number of corpus rows."
        )

    # ---------------------------------------------------------
    # TF-IDF representation for topic interpretation
    # ---------------------------------------------------------

    print("\nBuilding TF-IDF representation...")

    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)

    X_tfidf = vectorizer.fit_transform(
        corpus["text"]
    )

    terms = vectorizer.get_feature_names_out()

    print(
        f"TF-IDF matrix shape: {X_tfidf.shape}"
    )

    print(
        f"Vocabulary size: {len(terms):,}"
    )

    # ---------------------------------------------------------
    # Topic interpretation
    # ---------------------------------------------------------

    print(
        "\nCalculating characteristic terms "
        "and representative papers..."
    )

    topic_results = []
    paper_results = []

    unique_clusters = sorted(
        np.unique(labels)
    )

    for cluster_id in unique_clusters:

        cluster_mask = labels == cluster_id

        cluster_indices = np.where(
            cluster_mask
        )[0]

        cluster_size = len(
            cluster_indices
        )

        # -----------------------------------------------------
        # Characteristic terms
        # -----------------------------------------------------

        cluster_tfidf = X_tfidf[
            cluster_mask
        ].mean(axis=0)

        cluster_tfidf = np.asarray(
            cluster_tfidf
        ).ravel()

        top_term_indices = np.argsort(
            cluster_tfidf
        )[::-1][:TOP_TERMS]

        top_terms = [
            terms[index]
            for index in top_term_indices
        ]

        # -----------------------------------------------------
        # Representative papers
        # -----------------------------------------------------

        cluster_embeddings = embeddings[
            cluster_indices
        ]

        cluster_centroid = (
            cluster_embeddings.mean(axis=0)
        ).reshape(1, -1)

        similarities = cosine_similarity(
            cluster_embeddings,
            cluster_centroid
        ).ravel()

        top_paper_positions = np.argsort(
            similarities
        )[::-1][:TOP_PAPERS]

        top_paper_indices = (
            cluster_indices[
                top_paper_positions
            ]
        )

        # -----------------------------------------------------
        # Store topic information
        # -----------------------------------------------------

        topic_results.append({
            "cluster": cluster_id,
            "paper_count": cluster_size,
            "top_terms": ", ".join(
                top_terms
            )
        })

        # -----------------------------------------------------
        # Store representative papers
        # -----------------------------------------------------

        for rank, paper_index in enumerate(
            top_paper_indices,
            start=1
        ):

            paper = corpus.iloc[
                paper_index
            ]

            paper_results.append({
                "cluster": cluster_id,
                "rank": rank,
                "similarity_to_centroid": (
                    similarities[
                        top_paper_positions[
                            rank - 1
                        ]
                    ]
                ),
                "paper_id": paper["id"],
                "title": paper["title"]
            })

        # -----------------------------------------------------
        # Print results
        # -----------------------------------------------------

        print(
            f"\nCluster {cluster_id} "
            f"({cluster_size:,} papers)"
        )

        print(
            "  Terms:"
        )

        print(
            "  "
            + ", ".join(top_terms)
        )

        print(
            "  Representative papers:"
        )

        for rank, paper_index in enumerate(
            top_paper_indices,
            start=1
        ):

            title = corpus.iloc[
                paper_index
            ]["title"]

            print(
                f"    {rank}. {title}"
            )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    topic_summary = pd.DataFrame(
        topic_results
    )

    topic_output_path = (
        OUTPUT_DIR / "cluster_topics.csv"
    )

    topic_summary.to_csv(
        topic_output_path,
        index=False
    )

    representative_papers = pd.DataFrame(
        paper_results
    )

    papers_output_path = (
        OUTPUT_DIR / "representative_papers.csv"
    )

    representative_papers.to_csv(
        papers_output_path,
        index=False
    )

    print(
        "\nSaved topic summary to:"
        f"\n{topic_output_path}"
    )

    print(
        "\nSaved representative papers to:"
        f"\n{papers_output_path}"
    )


if __name__ == "__main__":
    main()