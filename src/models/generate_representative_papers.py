from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


EMBEDDINGS_PATH = Path(
    "data/processed/svd/paper_embeddings.npy"
)

METADATA_PATH = Path(
    "data/processed/svd/paper_metadata.csv"
)

LABELS_PATH = Path(
    "data/processed/clusters/kmeans_30_labels.npy"
)

TOPIC_LABELS_PATH = Path(
    "data/processed/topics/topic_labels.csv"
)

OUTPUT_PATH = Path(
    "data/processed/topics/representative_papers_25.csv"
)

N_REPRESENTATIVES = 25


def main():

    print("Loading existing SVD embeddings...")

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    labels = np.load(
        LABELS_PATH
    )

    topic_labels = pd.read_csv(
        TOPIC_LABELS_PATH
    )

    print(
        f"Embeddings shape: {embeddings.shape}"
    )

    print(
        f"Metadata rows: {len(metadata):,}"
    )

    print(
        f"Cluster labels: {len(labels):,}"
    )

    # ---------------------------------------------------------
    # Basic consistency checks
    # ---------------------------------------------------------

    if len(embeddings) != len(metadata):
        raise ValueError(
            "Number of embeddings does not match metadata rows."
        )

    if len(embeddings) != len(labels):
        raise ValueError(
            "Number of embeddings does not match cluster labels."
        )

    # ---------------------------------------------------------
    # Add cluster labels to metadata
    # ---------------------------------------------------------

    metadata = metadata.copy()

    metadata["cluster"] = labels

    # ---------------------------------------------------------
    # Generate representative papers
    # ---------------------------------------------------------

    results = []

    clusters = sorted(
        metadata["cluster"].unique()
    )

    print(
        f"\nClusters found: {len(clusters)}"
    )

    print(
        f"Selecting top {N_REPRESENTATIVES} "
        "representative papers per cluster..."
    )

    for cluster in clusters:

        cluster_mask = (
            metadata["cluster"] == cluster
        )

        cluster_indices = np.where(
            cluster_mask
        )[0]

        cluster_embeddings = embeddings[
            cluster_indices
        ]

        # -----------------------------------------------------
        # Calculate cluster centroid
        # -----------------------------------------------------

        centroid = cluster_embeddings.mean(
            axis=0,
            keepdims=True
        )

        # -----------------------------------------------------
        # Measure similarity of every paper
        # to its cluster centroid
        # -----------------------------------------------------

        similarities = cosine_similarity(
            cluster_embeddings,
            centroid
        ).ravel()

        # -----------------------------------------------------
        # Sort papers by similarity
        # -----------------------------------------------------

        ranked_order = np.argsort(
            similarities
        )[::-1]

        top_n = min(
            N_REPRESENTATIVES,
            len(ranked_order)
        )

        selected_positions = ranked_order[
            :top_n
        ]

        # -----------------------------------------------------
        # Store representative papers
        # -----------------------------------------------------

        for rank, position in enumerate(
            selected_positions,
            start=1
        ):

            original_index = cluster_indices[
                position
            ]

            row = metadata.iloc[
                original_index
            ]

            results.append(
                {
                    "cluster": int(cluster),
                    "rank": rank,
                    "similarity_to_centroid": float(
                        similarities[position]
                    ),
                    "paper_id": row.get(
                        "paper_id",
                        row.get("id", "")
                    ),
                    "title": row.get(
                        "title",
                        ""
                    ),
                }
            )

    # ---------------------------------------------------------
    # Create output dataframe
    # ---------------------------------------------------------

    result_df = pd.DataFrame(
        results
    )

    # Add human-readable topic labels

    result_df = result_df.merge(
        topic_labels[
            ["cluster", "topic_label"]
        ],
        on="cluster",
        how="left"
    )

    result_df = result_df[
        [
            "cluster",
            "topic_label",
            "rank",
            "similarity_to_centroid",
            "paper_id",
            "title",
        ]
    ]

    result_df = result_df.sort_values(
        [
            "cluster",
            "rank",
        ]
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print(
        f"\nSaved representative papers to:"
        f"\n{OUTPUT_PATH}"
    )

    print(
        f"\nTotal representative papers: "
        f"{len(result_df):,}"
    )

    print(
        f"Expected maximum: "
        f"{len(clusters) * N_REPRESENTATIVES:,}"
    )

    print(
        "\nRepresentatives per cluster:"
    )

    print(
        result_df.groupby(
            "cluster"
        ).size().to_string()
    )

    print(
        "\nRepresentative-paper generation complete."
    )


if __name__ == "__main__":
    main()