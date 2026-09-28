from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


EMBEDDINGS_PATH = Path(
    "data/processed/svd/paper_embeddings.npy"
)
OUTPUT_DIR = Path("data/processed/clusters")

CLUSTER_RANGE = [5, 10, 15, 20, 25, 30]
SELECTED_K = 30


def main():
    print("Loading SVD embeddings...")

    X = np.load(EMBEDDINGS_PATH)

    print(f"Embedding matrix shape: {X.shape}")

    results = []

    for k in CLUSTER_RANGE:
        print(f"\nRunning K-Means with {k} clusters...")

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(X)

        if k == SELECTED_K:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            np.save(
                OUTPUT_DIR / f"kmeans_{k}_labels.npy",
                labels
            )

            print(
                f"Saved cluster labels: "
                f"{OUTPUT_DIR / f'kmeans_{k}_labels.npy'}"
            )

        print("Clustering complete.")

        sample_size = min(10_000, len(X))

        score = silhouette_score(
            X,
            labels,
            sample_size=sample_size,
            random_state=42
        )

        print(f"Silhouette score: {score:.4f}")

        results.append({
            "k": k,
            "silhouette_score": score
        })

    print("\nCluster evaluation results:")

    for result in results:
        print(
            f"k={result['k']:>2} "
            f"→ silhouette={result['silhouette_score']:.4f}"
        )


if __name__ == "__main__":
    main()