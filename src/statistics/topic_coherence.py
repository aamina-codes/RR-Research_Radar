from pathlib import Path

import pandas as pd


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

REPRESENTATIVE_PAPERS_PATH = Path(
    "data/processed/topics/representative_papers_25.csv"
)

RADAR_SIGNALS_PATH = Path(
    "data/processed/trends/research_radar_signals.csv"
)

OUTPUT_DIR = Path("data/processed/trends")
OUTPUT_PATH = OUTPUT_DIR / "topic_coherence.csv"


# -------------------------------------------------------------------
# Coherence thresholds
# -------------------------------------------------------------------

# These are descriptive interpretation thresholds.
# They are NOT statistically validated cutoffs.

HIGH_COHERENCE_THRESHOLD = 0.90
MODERATE_COHERENCE_THRESHOLD = 0.80


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def classify_coherence(mean_similarity: float) -> str:
    """
    Assign a descriptive semantic-coherence category based on
    representative-paper similarity to the cluster centroid.
    """

    if mean_similarity >= HIGH_COHERENCE_THRESHOLD:
        return "high_semantic_coherence"

    if mean_similarity >= MODERATE_COHERENCE_THRESHOLD:
        return "moderate_semantic_coherence"

    return "broad_or_mixed_cluster"


def build_interpretation(row: pd.Series) -> str:
    """
    Combine Research Radar emergence status with semantic coherence.
    """

    is_emerging = bool(row["emerging_candidate"])
    coherence_class = row["coherence_class"]

    if is_emerging:
        if coherence_class == "high_semantic_coherence":
            return "emerging_focused_research_area"

        if coherence_class == "moderate_semantic_coherence":
            return "emerging_area_with_moderate_focus"

        return "emerging_but_broad_or_mixed"

    if coherence_class == "high_semantic_coherence":
        return "established_focused_research_area"

    if coherence_class == "moderate_semantic_coherence":
        return "established_area_with_moderate_focus"

    return "established_but_broad_or_mixed"


# -------------------------------------------------------------------
# Main analysis
# -------------------------------------------------------------------

def main():

    # ================================================================
    # 1. Load representative papers
    # ================================================================

    print("Loading representative papers...")

    if not REPRESENTATIVE_PAPERS_PATH.exists():
        raise FileNotFoundError(
            f"Representative-paper file not found:\n"
            f"{REPRESENTATIVE_PAPERS_PATH}"
        )

    representatives = pd.read_csv(
        REPRESENTATIVE_PAPERS_PATH
    )

    print(
        f"Representative papers loaded: "
        f"{len(representatives):,}"
    )

    required_columns = {
        "cluster",
        "topic_label",
        "rank",
        "similarity_to_centroid",
        "paper_id",
        "title",
    }

    missing_columns = (
        required_columns
        - set(representatives.columns)
    )

    if missing_columns:
        raise ValueError(
            "Representative-paper file is missing required "
            f"columns: {sorted(missing_columns)}"
        )

    representatives["similarity_to_centroid"] = pd.to_numeric(
        representatives["similarity_to_centroid"],
        errors="coerce",
    )

    representatives = representatives.dropna(
        subset=["similarity_to_centroid"]
    )

    print(
        f"Valid representative papers: "
        f"{len(representatives):,}"
    )

    # ================================================================
    # 2. Calculate semantic coherence
    # ================================================================

    print("\nCalculating semantic coherence...")

    coherence = (
        representatives
        .groupby(
            ["cluster", "topic_label"]
        )["similarity_to_centroid"]
        .agg(
            representative_papers="count",
            mean_similarity="mean",
            median_similarity="median",
            min_similarity="min",
            max_similarity="max",
            similarity_std="std",
        )
        .reset_index()
    )

    # A standard deviation is undefined for a single observation.
    coherence["similarity_std"] = (
        coherence["similarity_std"]
        .fillna(0)
    )

    # ================================================================
    # 3. Classify semantic coherence
    # ================================================================

    coherence["coherence_class"] = (
        coherence["mean_similarity"]
        .apply(classify_coherence)
    )

    # ================================================================
    # 4. Load Research Radar signals
    # ================================================================

    print("\nLoading Research Radar signals...")

    if not RADAR_SIGNALS_PATH.exists():
        raise FileNotFoundError(
            f"Research Radar signals file not found:\n"
            f"{RADAR_SIGNALS_PATH}"
        )

    radar = pd.read_csv(
        RADAR_SIGNALS_PATH
    )

    required_radar_columns = {
        "cluster",
        "topic_label",
        "emerging_candidate",
    }

    missing_radar_columns = (
        required_radar_columns
        - set(radar.columns)
    )

    if missing_radar_columns:
        raise ValueError(
            "Research Radar signals file is missing required "
            f"columns: {sorted(missing_radar_columns)}"
        )

    radar_status = radar[
        [
            "cluster",
            "topic_label",
            "emerging_candidate",
        ]
    ].copy()

    # ---------------------------------------------------------------
    # Convert emerging_candidate safely to boolean
    # ---------------------------------------------------------------

    if radar_status["emerging_candidate"].dtype == object:

        radar_status["emerging_candidate"] = (
            radar_status["emerging_candidate"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                    "yes": True,
                    "no": False,
                }
            )
        )

    radar_status["emerging_candidate"] = (
        radar_status["emerging_candidate"]
        .fillna(False)
        .astype(bool)
    )

    # ================================================================
    # 5. Merge Radar status with coherence
    # ================================================================

    results = coherence.merge(
        radar_status,
        on=[
            "cluster",
            "topic_label",
        ],
        how="left",
    )

    results["emerging_candidate"] = (
        results["emerging_candidate"]
        .fillna(False)
        .astype(bool)
    )

    # ================================================================
    # 6. Add combined interpretation
    # ================================================================

    results["interpretation"] = (
        results.apply(
            build_interpretation,
            axis=1,
        )
    )

    # ================================================================
    # 7. Additional descriptive metrics
    # ================================================================

    results["similarity_range"] = (
        results["max_similarity"]
        - results["min_similarity"]
    )

    results["coherence_threshold_high"] = (
        HIGH_COHERENCE_THRESHOLD
    )

    results["coherence_threshold_moderate"] = (
        MODERATE_COHERENCE_THRESHOLD
    )

    # Emerging candidates first, then highest coherence.
    results = results.sort_values(
        by=[
            "emerging_candidate",
            "mean_similarity",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)

    # ================================================================
    # 8. Save results
    # ================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ================================================================
    # 9. Console summary
    # ================================================================

    print("\n" + "=" * 70)
    print("TOPIC SEMANTIC COHERENCE")
    print("=" * 70)

    print(
        f"\nTopics analyzed: "
        f"{len(results)}"
    )

    papers_per_topic = (
        representatives
        .groupby("cluster")
        .size()
    )

    print(
        f"Representative papers per topic: "
        f"{papers_per_topic.min()}–"
        f"{papers_per_topic.max()}"
    )

    # ---------------------------------------------------------------
    # Emerging candidates
    # ---------------------------------------------------------------

    emerging = results[
        results["emerging_candidate"]
    ]

    print("\nEmerging Radar candidates:")

    if len(emerging) == 0:

        print("No emerging candidates found.")

    else:

        for _, row in emerging.iterrows():

            print(
                f"\n{row['topic_label']}"
                f"\n  Representative papers: "
                f"{int(row['representative_papers'])}"
                f"\n  Mean similarity: "
                f"{row['mean_similarity']:.4f}"
                f"\n  Median similarity: "
                f"{row['median_similarity']:.4f}"
                f"\n  Min similarity: "
                f"{row['min_similarity']:.4f}"
                f"\n  Max similarity: "
                f"{row['max_similarity']:.4f}"
                f"\n  Similarity std: "
                f"{row['similarity_std']:.4f}"
                f"\n  Coherence: "
                f"{row['coherence_class']}"
                f"\n  Interpretation: "
                f"{row['interpretation']}"
            )

    # ---------------------------------------------------------------
    # Coherence class counts
    # ---------------------------------------------------------------

    print("\n" + "-" * 70)
    print("COHERENCE CLASS COUNTS")
    print("-" * 70)

    print(
        results["coherence_class"]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------------
    # Emerging topic summary
    # ---------------------------------------------------------------

    print("\n" + "-" * 70)
    print("EMERGING TOPICS BY SEMANTIC COHERENCE")
    print("-" * 70)

    if len(emerging) > 0:

        display_columns = [
            "topic_label",
            "mean_similarity",
            "median_similarity",
            "min_similarity",
            "max_similarity",
            "coherence_class",
            "interpretation",
        ]

        print(
            emerging[display_columns]
            .to_string(index=False)
        )

    # ---------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        f"Saved coherence analysis to:\n"
        f"{OUTPUT_PATH}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()