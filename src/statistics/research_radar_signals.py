from pathlib import Path

import pandas as pd


TREND_PATH = Path(
    "data/processed/trends/topic_trend_analysis.csv"
)

EMERGING_PATH = Path(
    "data/processed/trends/emerging_topics.csv"
)

OUTPUT_DIR = Path(
    "data/processed/trends"
)

OUTPUT_PATH = (
    OUTPUT_DIR / "research_radar_signals.csv"
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

LATEST_COMPLETE_YEAR = 2024

MIN_VOLUME = 50

FDR_ALPHA = 0.05

MIN_SLOPE = 0.05

MIN_R2 = 0.30

MIN_RECENT_MOMENTUM = 0.05


def main():

    # --------------------------------------------------
    # Load existing analyses
    # --------------------------------------------------

    print("Loading trend analysis...")

    trends = pd.read_csv(
        TREND_PATH
    )

    print(
        f"Trend rows: {len(trends):,}"
    )

    print(
        "\nLoading emerging-topic analysis..."
    )

    emerging = pd.read_csv(
        EMERGING_PATH
    )

    print(
        f"Emerging-topic rows: "
        f"{len(emerging):,}"
    )

    # --------------------------------------------------
    # Keep the exact columns produced by the
    # existing trend_analysis.py
    # --------------------------------------------------

    trend_columns = [
        "cluster",
        "topic_label",
        "years_used",
        "trend_slope",
        "trend_p_value",
        "trend_r_squared",
        "trend_direction",
        "latest_complete_year",
        "latest_share_percent",
        "previous_average_share_percent",
        "recent_share_change_percentage_points",
        "recent_momentum_direction",
        "trend_p_value_fdr",
        "statistically_significant",
        "emerging_candidate",
    ]

    missing_trend_columns = [
        column
        for column in trend_columns
        if column not in trends.columns
    ]

    if missing_trend_columns:

        raise ValueError(
            "Missing columns in topic_trend_analysis.csv: "
            f"{missing_trend_columns}"
        )

    trend_data = trends[
        trend_columns
    ].copy()

    # --------------------------------------------------
    # Keep the exact columns produced by the
    # existing emerging_topics.py
    # --------------------------------------------------

    emerging_columns = [
        "cluster",
        "topic_label",
        "latest_complete_year",
        "latest_complete_year_papers",
        "trend_slope",
        "trend_p_value_fdr",
        "trend_r_squared",
        "trend_direction",
        "latest_share_percent",
        "previous_average_share_percent",
        "recent_share_change_percentage_points",
        "recent_momentum_direction",
        "long_term_signal",
        "recent_momentum_signal",
        "volume_signal",
        "signals_supported",
        "radar_signal_strength",
        "evidence_class",
        "emerging_topic",
    ]

    missing_emerging_columns = [
        column
        for column in emerging_columns
        if column not in emerging.columns
    ]

    if missing_emerging_columns:

        raise ValueError(
            "Missing columns in emerging_topics.csv: "
            f"{missing_emerging_columns}"
        )

    emerging_data = emerging[
        emerging_columns
    ].copy()

    # --------------------------------------------------
    # Only use the latest complete year
    # --------------------------------------------------

    emerging_data = emerging_data[
        emerging_data[
            "latest_complete_year"
        ]
        == LATEST_COMPLETE_YEAR
    ].copy()

    # --------------------------------------------------
    # Avoid duplicate topic records
    # --------------------------------------------------

    trend_data = trend_data.drop_duplicates(
        subset=["cluster"]
    )

    emerging_data = emerging_data.drop_duplicates(
        subset=["cluster"]
    )

    # --------------------------------------------------
    # Merge both analytical layers
    # --------------------------------------------------

    radar = trend_data.merge(
        emerging_data[
            [
                "cluster",
                "latest_complete_year_papers",
                "long_term_signal",
                "recent_momentum_signal",
                "volume_signal",
                "signals_supported",
                "radar_signal_strength",
                "evidence_class",
                "emerging_topic",
            ]
        ],
        on="cluster",
        how="left",
        suffixes=(
            "",
            "_emerging"
        ),
        validate="one_to_one",
    )

    # --------------------------------------------------
    # Create transparent statistical signals
    # --------------------------------------------------

    radar[
        "statistically_supported_trend"
    ] = (
        (
            radar[
                "trend_direction"
            ]
            == "increasing"
        )
        &
        (
            radar[
                "trend_p_value_fdr"
            ]
            < FDR_ALPHA
        )
    )

    radar[
        "meaningful_long_term_growth"
    ] = (
        radar[
            "trend_slope"
        ]
        >= MIN_SLOPE
    )

    radar[
        "strong_trend_fit"
    ] = (
        radar[
            "trend_r_squared"
        ]
        >= MIN_R2
    )

    radar[
        "positive_recent_momentum"
    ] = (
        radar[
            "recent_share_change_percentage_points"
        ]
        >= MIN_RECENT_MOMENTUM
    )

    radar[
        "sufficient_research_volume"
    ] = (
        radar[
            "latest_complete_year_papers"
        ]
        >= MIN_VOLUME
    )

    # --------------------------------------------------
    # Final Research Radar candidate rule
    #
    # A topic must have:
    #
    # 1. Increasing statistically supported trend
    # 2. Meaningful long-term growth
    # 3. Positive recent momentum
    # 4. Sufficient research volume
    #
    # R² is supporting evidence, not a hard gate.
    # --------------------------------------------------

    radar[
        "radar_emerging_candidate"
    ] = (
        radar[
            "statistically_supported_trend"
        ]
        &
        radar[
            "meaningful_long_term_growth"
        ]
        &
        radar[
            "positive_recent_momentum"
        ]
        &
        radar[
            "sufficient_research_volume"
        ]
    )

    # --------------------------------------------------
    # Count supporting signals
    # --------------------------------------------------

    signal_columns = [
        "statistically_supported_trend",
        "meaningful_long_term_growth",
        "strong_trend_fit",
        "positive_recent_momentum",
        "sufficient_research_volume",
    ]

    radar[
        "supporting_signal_count"
    ] = radar[
        signal_columns
    ].sum(
        axis=1
    )

    # --------------------------------------------------
    # Evidence classification
    # --------------------------------------------------

    def classify_evidence(row):

        if row[
            "radar_emerging_candidate"
        ]:

            return "emerging_candidate"

        if (
            row[
                "statistically_supported_trend"
            ]
            and row[
                "positive_recent_momentum"
            ]
        ):

            return "growth_with_recent_momentum"

        if row[
            "statistically_supported_trend"
        ]:

            return "statistically_supported_growth"

        if row[
            "positive_recent_momentum"
        ]:

            return "recent_momentum"

        if row[
            "sufficient_research_volume"
        ]:

            return "established_volume"

        return "limited_signal"

    radar[
        "radar_evidence_class"
    ] = radar.apply(
        classify_evidence,
        axis=1
    )

    # --------------------------------------------------
    # Human-readable interpretation
    # --------------------------------------------------

    def create_interpretation(row):

        if row[
            "radar_emerging_candidate"
        ]:

            return (
                "Increasing long-term prevalence, "
                "positive recent momentum, and "
                "sufficient research volume."
            )

        if (
            row[
                "statistically_supported_trend"
            ]
            and not row[
                "positive_recent_momentum"
            ]
        ):

            return (
                "Long-term increasing trend is "
                "statistically supported, but "
                "recent momentum is not positive."
            )

        if (
            row[
                "positive_recent_momentum"
            ]
            and not row[
                "statistically_supported_trend"
            ]
        ):

            return (
                "Recent prevalence increased, "
                "but long-term statistical evidence "
                "does not meet the trend threshold."
            )

        if row[
            "sufficient_research_volume"
        ]:

            return (
                "Substantial research volume, "
                "without the full emerging-topic "
                "signal."
            )

        return (
            "No complete emerging-topic signal "
            "under the current rules."
        )

    radar[
        "interpretation"
    ] = radar.apply(
        create_interpretation,
        axis=1
    )

    # --------------------------------------------------
    # Final organizational ordering
    #
    # This is NOT a quality ranking.
    # Candidates are simply placed first.
    # --------------------------------------------------

    radar = radar.sort_values(
        [
            "radar_emerging_candidate",
            "supporting_signal_count",
        ],
        ascending=[
            False,
            False,
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    radar.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------
    # Console output
    # --------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "RESEARCH RADAR SIGNALS"
    )

    print(
        "========================================"
    )

    print(
        f"\nSaved to:"
    )

    print(
        OUTPUT_PATH
    )

    candidates = radar[
        radar[
            "radar_emerging_candidate"
        ]
    ]

    print(
        f"\nEmerging candidates: "
        f"{len(candidates)} / "
        f"{len(radar)}"
    )

    # --------------------------------------------------
    # Candidate details
    # --------------------------------------------------

    print(
        "\n----------------------------------------"
    )

    print(
        "EMERGING CANDIDATES"
    )

    print(
        "----------------------------------------"
    )

    if candidates.empty:

        print(
            "No topics currently satisfy "
            "all Research Radar criteria."
        )

    else:

        for _, row in candidates.iterrows():

            print(
                f"\n{row['topic_label']}"
            )

            print(
                f"  2024 papers: "
                f"{int(row['latest_complete_year_papers']):,}"
            )

            print(
                f"  Trend slope: "
                f"{row['trend_slope']:.3f} pp/year"
            )

            print(
                f"  FDR p-value: "
                f"{row['trend_p_value_fdr']:.6f}"
            )

            print(
                f"  R²: "
                f"{row['trend_r_squared']:.3f}"
            )

            print(
                f"  Recent momentum: "
                f"{row['recent_share_change_percentage_points']:.3f} pp"
            )

            print(
                f"  Supporting signals: "
                f"{int(row['supporting_signal_count'])}/5"
            )

    # --------------------------------------------------
    # Evidence class summary
    # --------------------------------------------------

    print(
        "\n----------------------------------------"
    )

    print(
        "EVIDENCE CLASSES"
    )

    print(
        "----------------------------------------"
    )

    class_counts = (
        radar[
            "radar_evidence_class"
        ]
        .value_counts()
    )

    for evidence_class, count in (
        class_counts.items()
    ):

        print(
            f"{evidence_class}: {count}"
        )


if __name__ == "__main__":
    main()