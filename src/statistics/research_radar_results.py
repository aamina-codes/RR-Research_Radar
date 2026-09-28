from pathlib import Path

import pandas as pd


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

SIGNALS_PATH = Path(
    "data/processed/trends/research_radar_signals.csv"
)

COHERENCE_PATH = Path(
    "data/processed/trends/topic_coherence.csv"
)

OUTPUT_DIR = Path(
    "data/processed/trends"
)

OUTPUT_PATH = OUTPUT_DIR / "research_radar_results.csv"


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def format_percentage(value):
    """Format a numeric value as a percentage."""

    if pd.isna(value):
        return "N/A"

    return f"{value:.2f}%"


def format_pp(value):
    """Format a percentage-point change."""

    if pd.isna(value):
        return "N/A"

    return f"{value:+.3f} pp"


def build_signal_explanation(row):
    """
    Create a human-readable explanation of why a topic
    appears in the Research Radar.
    """

    signals = []

    if row["statistically_supported_trend"]:
        signals.append("statistically supported long-term growth")

    if row["meaningful_long_term_growth"]:
        signals.append("meaningful long-term growth")

    if row["positive_recent_momentum"]:
        signals.append("positive recent momentum")

    if row["sufficient_research_volume"]:
        signals.append("sufficient research volume")

    if row["strong_trend_fit"]:
        signals.append("strong trend fit")

    if not signals:
        return "No strong Radar signals detected."

    if len(signals) == 1:
        return signals[0].capitalize() + "."

    return (
        ", ".join(signals[:-1]).capitalize()
        + ", and "
        + signals[-1]
        + "."
    )


def build_research_interpretation(row):
    """
    Combine statistical Radar evidence with semantic coherence.
    """

    if row["emerging_candidate"]:

        if row["coherence_class"] == "high_semantic_coherence":
            return (
                "Emerging research area with statistically supported "
                "growth, recent momentum, and high semantic focus."
            )

        if row["coherence_class"] == "moderate_semantic_coherence":
            return (
                "Emerging research area with statistically supported "
                "growth and moderate semantic focus."
            )

        return (
            "Statistically emerging signal, but the discovered "
            "cluster is broad or semantically mixed."
        )

    if row["coherence_class"] == "high_semantic_coherence":
        return (
            "Established research area with strong semantic focus "
            "but without the full Radar emergence signal."
        )

    if row["coherence_class"] == "moderate_semantic_coherence":
        return (
            "Established research area with moderate semantic focus."
        )

    return (
        "Research cluster with broad or mixed semantic focus."
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():

    print("=" * 70)
    print("RESEARCH RADAR RESULTS")
    print("=" * 70)

    # ================================================================
    # 1. Load Research Radar signals
    # ================================================================

    print("\nLoading Research Radar signals...")

    if not SIGNALS_PATH.exists():
        raise FileNotFoundError(
            f"Research Radar signals file not found:\n"
            f"{SIGNALS_PATH}"
        )

    signals = pd.read_csv(SIGNALS_PATH)

    print(
        f"Signal records loaded: "
        f"{len(signals):,}"
    )

    # ================================================================
    # 2. Validate signal columns
    # ================================================================

    required_signal_columns = {
        "cluster",
        "topic_label",
        "latest_complete_year",
        "latest_complete_year_papers",
        "trend_slope",
        "trend_p_value_fdr",
        "trend_r_squared",
        "recent_share_change_percentage_points",
        "supporting_signal_count",
        "statistically_supported_trend",
        "meaningful_long_term_growth",
        "strong_trend_fit",
        "positive_recent_momentum",
        "sufficient_research_volume",
        "emerging_candidate",
    }

    missing_signal_columns = (
        required_signal_columns
        - set(signals.columns)
    )

    if missing_signal_columns:
        raise ValueError(
            "Research Radar signals file is missing required "
            f"columns: {sorted(missing_signal_columns)}"
        )

    # ================================================================
    # 3. Load semantic coherence
    # ================================================================

    print("\nLoading semantic coherence...")

    if not COHERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Topic coherence file not found:\n"
            f"{COHERENCE_PATH}"
        )

    coherence = pd.read_csv(
        COHERENCE_PATH
    )

    print(
        f"Coherence records loaded: "
        f"{len(coherence):,}"
    )

    # ================================================================
    # 4. Validate coherence columns
    # ================================================================

    required_coherence_columns = {
        "cluster",
        "topic_label",
        "representative_papers",
        "mean_similarity",
        "median_similarity",
        "min_similarity",
        "max_similarity",
        "similarity_std",
        "coherence_class",
        "interpretation",
    }

    missing_coherence_columns = (
        required_coherence_columns
        - set(coherence.columns)
    )

    if missing_coherence_columns:
        raise ValueError(
            "Topic coherence file is missing required "
            f"columns: {sorted(missing_coherence_columns)}"
        )

    # ================================================================
    # 5. Keep relevant coherence fields
    # ================================================================

    coherence_fields = [
        "cluster",
        "topic_label",
        "representative_papers",
        "mean_similarity",
        "median_similarity",
        "min_similarity",
        "max_similarity",
        "similarity_std",
        "coherence_class",
    ]

    coherence = coherence[
        coherence_fields
    ].copy()

    # ================================================================
    # 6. Merge statistical + semantic evidence
    # ================================================================

    print("\nCombining statistical and semantic evidence...")

    results = signals.merge(
        coherence,
        on=[
            "cluster",
            "topic_label",
        ],
        how="left",
        suffixes=("", "_coherence"),
    )

    # ================================================================
    # 7. Validate merge
    # ================================================================

    unmatched = results[
        results["mean_similarity"].isna()
    ]

    if len(unmatched) > 0:

        print(
            f"\nWARNING: "
            f"{len(unmatched)} topics have no semantic "
            f"coherence record."
        )

        if len(unmatched) <= 10:
            print(
                unmatched[
                    [
                        "cluster",
                        "topic_label",
                    ]
                ].to_string(index=False)
            )

    else:

        print(
            "All Radar topics matched with semantic "
            "coherence results."
        )

    # ================================================================
    # 8. Add human-readable evidence explanation
    # ================================================================

    results["signal_explanation"] = (
        results.apply(
            build_signal_explanation,
            axis=1,
        )
    )

    results["research_interpretation"] = (
        results.apply(
            build_research_interpretation,
            axis=1,
        )
    )

    # ================================================================
    # 9. Add presentation-friendly fields
    # ================================================================

    results["trend_slope_display"] = (
        results["trend_slope"]
        .apply(format_pp)
    )

    results["recent_momentum_display"] = (
        results[
            "recent_share_change_percentage_points"
        ]
        .apply(format_pp)
    )

    results["trend_p_value_fdr_display"] = (
        results["trend_p_value_fdr"]
        .apply(
            lambda x: (
                f"{x:.4g}"
                if not pd.isna(x)
                else "N/A"
            )
        )
    )

    results["trend_r_squared_display"] = (
        results["trend_r_squared"]
        .apply(
            lambda x: (
                f"{x:.3f}"
                if not pd.isna(x)
                else "N/A"
            )
        )
    )

    results["semantic_similarity_display"] = (
        results["mean_similarity"]
        .apply(
            lambda x: (
                f"{x:.4f}"
                if not pd.isna(x)
                else "N/A"
            )
        )
    )

    # ================================================================
    # 10. Add Radar status
    # ================================================================

    results["radar_status"] = results[
        "emerging_candidate"
    ].map(
        {
            True: "Emerging candidate",
            False: "Monitoring signal",
        }
    )

    # ================================================================
    # 11. Add evidence profile
    # ================================================================

    def build_evidence_profile(row):

        evidence = []

        if row["statistically_supported_trend"]:
            evidence.append("trend")

        if row["positive_recent_momentum"]:
            evidence.append("momentum")

        if row["sufficient_research_volume"]:
            evidence.append("volume")

        if row["strong_trend_fit"]:
            evidence.append("fit")

        if row["coherence_class"] == "high_semantic_coherence":
            evidence.append("semantic_focus")

        elif row["coherence_class"] == "moderate_semantic_coherence":
            evidence.append("moderate_semantic_focus")

        return " + ".join(evidence)

    results["evidence_profile"] = (
        results.apply(
            build_evidence_profile,
            axis=1,
        )
    )

    # ================================================================
    # 12. Create clean final column order
    # ================================================================

    final_columns = [
        # Identification
        "cluster",
        "topic_label",

        # Radar status
        "radar_status",
        "emerging_candidate",

        # Statistical evidence
        "latest_complete_year",
        "latest_complete_year_papers",
        "trend_slope",
        "trend_slope_display",
        "trend_p_value_fdr",
        "trend_p_value_fdr_display",
        "trend_r_squared",
        "trend_r_squared_display",
        "recent_share_change_percentage_points",
        "recent_momentum_display",

        # Signal flags
        "statistically_supported_trend",
        "meaningful_long_term_growth",
        "strong_trend_fit",
        "positive_recent_momentum",
        "sufficient_research_volume",
        "supporting_signal_count",

        # Semantic validation
        "representative_papers",
        "mean_similarity",
        "semantic_similarity_display",
        "median_similarity",
        "min_similarity",
        "max_similarity",
        "similarity_std",
        "coherence_class",

        # Explainability
        "evidence_profile",
        "signal_explanation",
        "research_interpretation",
    ]

    # Keep only columns that actually exist.
    final_columns = [
        column
        for column in final_columns
        if column in results.columns
    ]

    results = results[
        final_columns
    ].copy()

    # ================================================================
    # 13. Sort results
    # ================================================================

    # Emerging candidates first.
    # Within each group, use supporting signal count and semantic
    # similarity for readability. This is NOT a quality ranking.

    results = results.sort_values(
        by=[
            "emerging_candidate",
            "supporting_signal_count",
            "mean_similarity",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    # ================================================================
    # 14. Save
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
    # 15. Console summary
    # ================================================================

    print("\n" + "=" * 70)
    print("FINAL RESEARCH RADAR")
    print("=" * 70)

    print(
        f"\nTotal topics: "
        f"{len(results)}"
    )

    print(
        f"Emerging candidates: "
        f"{results['emerging_candidate'].sum()}"
    )

    print(
        f"Monitoring signals: "
        f"{(~results['emerging_candidate']).sum()}"
    )

    print("\nEmerging candidates:")

    emerging = results[
        results["emerging_candidate"]
    ]

    if len(emerging) > 0:

        for _, row in emerging.iterrows():

            print(
                f"\n{row['topic_label']}"
                f"\n  Status: "
                f"{row['radar_status']}"
                f"\n  2024 papers: "
                f"{int(row['latest_complete_year_papers']):,}"
                f"\n  Long-term trend: "
                f"{row['trend_slope_display']}"
                f"\n  FDR p-value: "
                f"{row['trend_p_value_fdr_display']}"
                f"\n  Trend R²: "
                f"{row['trend_r_squared_display']}"
                f"\n  Recent momentum: "
                f"{row['recent_momentum_display']}"
                f"\n  Supporting signals: "
                f"{int(row['supporting_signal_count'])}/5"
                f"\n  Semantic coherence: "
                f"{row['coherence_class']}"
                f"\n  Mean similarity: "
                f"{row['semantic_similarity_display']}"
                f"\n  Evidence profile: "
                f"{row['evidence_profile']}"
            )

    print("\n" + "-" * 70)
    print("COHERENCE BREAKDOWN OF EMERGING CANDIDATES")
    print("-" * 70)

    if len(emerging) > 0:

        print(
            emerging[
                [
                    "topic_label",
                    "mean_similarity",
                    "coherence_class",
                ]
            ].to_string(index=False)
        )

    print("\n" + "-" * 70)
    print("RADAR STATUS COUNTS")
    print("-" * 70)

    print(
        results["radar_status"]
        .value_counts()
        .to_string()
    )

    print("\n" + "=" * 70)
    print(
        f"Saved final Radar results to:\n"
        f"{OUTPUT_PATH}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()