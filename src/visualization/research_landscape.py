from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "svd"
    / "paper_embeddings.npy"
)

CLUSTER_LABELS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clusters"
    / "kmeans_30_labels.npy"
)

TOPIC_LABELS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "topics"
    / "topic_labels.csv"
)

RADAR_RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "trends"
    / "research_radar_results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "visualizations"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "research_landscape.csv"
)


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)
    print("RESEARCH LANDSCAPE")
    print("=" * 70)

    # ------------------------------------------------------------
    # Load semantic embeddings
    # ------------------------------------------------------------

    print("\nLoading SVD embeddings...")

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    print(
        f"Embeddings shape: {embeddings.shape}"
    )

    # ------------------------------------------------------------
    # Load cluster assignments
    # ------------------------------------------------------------

    print("\nLoading cluster labels...")

    cluster_labels = np.load(
        CLUSTER_LABELS_PATH
    )

    print(
        f"Cluster labels: {len(cluster_labels):,}"
    )

    # ------------------------------------------------------------
    # Load topic labels
    # ------------------------------------------------------------

    print("\nLoading topic labels...")

    topic_labels = pd.read_csv(
        TOPIC_LABELS_PATH
    )

    print(
        f"Topics available: {len(topic_labels)}"
    )

    # ------------------------------------------------------------
    # Load Radar results
    # ------------------------------------------------------------

    print("\nLoading Research Radar results...")

    radar = pd.read_csv(
        RADAR_RESULTS_PATH
    )

    print(
        f"Radar records: {len(radar)}"
    )

    # ------------------------------------------------------------
    # Validate dimensions
    # ------------------------------------------------------------

    if len(embeddings) != len(cluster_labels):

        raise ValueError(
            "The number of embeddings does not match "
            "the number of cluster labels."
        )

    # ------------------------------------------------------------
    # Create topic centroids
    # ------------------------------------------------------------

    print(
        "\nCalculating topic centroids..."
    )

    unique_clusters = np.sort(
        np.unique(cluster_labels)
    )

    centroids = []
    cluster_records = []

    for cluster in unique_clusters:

        mask = (
            cluster_labels == cluster
        )

        cluster_embeddings = embeddings[
            mask
        ]

        centroid = cluster_embeddings.mean(
            axis=0
        )

        centroids.append(
            centroid
        )

        cluster_records.append(
            {
                "cluster": int(cluster),
                "paper_count": int(
                    mask.sum()
                ),
            }
        )

    centroids = np.vstack(
        centroids
    )

    print(
        f"Topic centroids: {centroids.shape}"
    )

    # ------------------------------------------------------------
    # PCA → 2D
    # ------------------------------------------------------------

    print(
        "\nReducing topic centroids to 2D with PCA..."
    )

    pca = PCA(
        n_components=2,
        random_state=42,
    )

    coordinates = pca.fit_transform(
        centroids
    )

    explained_variance = (
        pca.explained_variance_ratio_
    )

    print(
        "PCA explained variance:"
    )

    print(
        f"  PC1: {explained_variance[0]:.4f}"
    )

    print(
        f"  PC2: {explained_variance[1]:.4f}"
    )

    print(
        f"  Combined: "
        f"{explained_variance.sum():.4f}"
    )

    # ------------------------------------------------------------
    # Build landscape dataframe
    # ------------------------------------------------------------

    landscape = pd.DataFrame(
        {
            "cluster": unique_clusters,
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
        }
    )

    # ------------------------------------------------------------
    # Merge topic labels
    # ------------------------------------------------------------

    landscape = landscape.merge(
        topic_labels[
            [
                "cluster",
                "topic_label",
            ]
        ],
        on="cluster",
        how="left",
    )

    # ------------------------------------------------------------
    # Merge paper counts
    # ------------------------------------------------------------

    landscape = landscape.merge(
        pd.DataFrame(
            cluster_records
        ),
        on="cluster",
        how="left",
    )

    # ------------------------------------------------------------
    # Merge Radar information
    # ------------------------------------------------------------

    radar_columns = [
        "topic_label",
        "emerging_candidate",
        "radar_status",
        "trend_slope_display",
        "recent_momentum_display",
        "trend_p_value_fdr_display",
        "trend_r_squared_display",
        "coherence_class",
        "semantic_similarity_display",
    ]

    available_radar_columns = [
        column
        for column in radar_columns
        if column in radar.columns
    ]

    landscape = landscape.merge(
        radar[
            available_radar_columns
        ],
        on="topic_label",
        how="left",
    )

    # ------------------------------------------------------------
    # Add PCA metadata
    # ------------------------------------------------------------

    landscape["pca_pc1_variance"] = (
        explained_variance[0]
    )

    landscape["pca_pc2_variance"] = (
        explained_variance[1]
    )

    landscape["pca_combined_variance"] = (
        explained_variance.sum()
    )

    # ------------------------------------------------------------
    # Sort by cluster
    # ------------------------------------------------------------

    landscape = landscape.sort_values(
        "cluster"
    ).reset_index(
        drop=True
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    landscape.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "RESEARCH LANDSCAPE COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTopics mapped: {len(landscape)}"
    )

    print(
        f"Emerging candidates: "
        f"{int(landscape['emerging_candidate'].fillna(False).sum())}"
    )

    print(
        "\nTopic coordinates:"
    )

    print(
        landscape[
            [
                "cluster",
                "topic_label",
                "paper_count",
                "x",
                "y",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nSaved landscape data to:"
    )

    print(
        OUTPUT_PATH
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()