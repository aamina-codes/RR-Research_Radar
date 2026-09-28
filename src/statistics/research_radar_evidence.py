from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT_PATH = Path(
    "data/processed/trends/research_radar_signals.csv"
)

OUTPUT_DIR = Path(
    "data/processed/trends"
)

OUTPUT_PATH = (
    OUTPUT_DIR / "research_radar_evidence.csv"
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

LATEST_COMPLETE_YEAR = 2024


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("Loading Research Radar signals...")

    radar = pd.read_csv(
        INPUT_PATH
    )

    print(
        f"Topics loaded: {len(radar):,}"
    )

    # --------------------------------------------------
    # Validate required columns
    # --------------------------------------------------

    required_columns = [
        "cluster",
        "topic_label",
        "latest_complete_year_papers",
        "trend_slope",
        "trend_p_value_fdr",
        "trend_r_squared",
        "trend_direction",
        "recent_share_change_percentage_points",
        "recent_momentum_direction",
        "statistically_supported_trend",
        "meaningful_long_term_growth",
        "strong_trend_fit",
        "positive_recent_momentum",
        "sufficient_research_volume",
        "radar_emerging_candidate",
        "supporting_signal_count",
        "radar_evidence_class",
        "interpretation",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in radar.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing columns in "
            "research_radar_signals.csv: "
            f"{missing_columns}"
        )

    # --------------------------------------------------
    # Create evidence statements
    # --------------------------------------------------

    def trend_evidence(row):

        if row[
            "statistically_supported_trend"
        ]:

            return (
                f"Long-term prevalence is increasing "
                f"at approximately "
                f"{row['trend_slope']:.3f} "
                f"percentage points per year, "
                f"with an FDR-adjusted p-value of "
                f"{row['trend_p_value_fdr']:.6f}."
            )

        if row[
            "trend_direction"
        ] == "increasing":

            return (
                f"Long-term prevalence is increasing "
                f"at approximately "
                f"{row['trend_slope']:.3f} "
                f"percentage points per year, "
                f"but the trend does not meet the "
                f"current statistical significance threshold."
            )

        if row[
            "trend_direction"
        ] == "decreasing":

            return (
                f"Long-term prevalence is decreasing "
                f"at approximately "
                f"{abs(row['trend_slope']):.3f} "
                f"percentage points per year."
            )

        return (
            "No clear long-term increasing or "
            "decreasing trend was detected."
        )

    radar[
        "long_term_trend_evidence"
    ] = radar.apply(
        trend_evidence,
        axis=1
    )

    # --------------------------------------------------
    # Recent momentum evidence
    # --------------------------------------------------

    def momentum_evidence(row):

        momentum = row[
            "recent_share_change_percentage_points"
        ]

        direction = row[
            "recent_momentum_direction"
        ]

        if direction == "increasing":

            return (
                f"Recent prevalence increased by "
                f"{momentum:.3f} percentage points "
                f"relative to the previous-period baseline."
            )

        if direction == "decreasing":

            return (
                f"Recent prevalence decreased by "
                f"{abs(momentum):.3f} percentage points "
                f"relative to the previous-period baseline."
            )

        return (
            "Recent prevalence shows little "
            "directional change."
        )

    radar[
        "recent_momentum_evidence"
    ] = radar.apply(
        momentum_evidence,
        axis=1
    )

    # --------------------------------------------------
    # Research volume evidence
    # --------------------------------------------------

    def volume_evidence(row):

        papers = int(
            row[
                "latest_complete_year_papers"
            ]
        )

        return (
            f"The topic contains {papers:,} papers "
            f"in {LATEST_COMPLETE_YEAR}, providing "
            f"a measurable research volume."
        )

    radar[
        "research_volume_evidence"
    ] = radar.apply(
        volume_evidence,
        axis=1
    )

    # --------------------------------------------------
    # Trend fit evidence
    # --------------------------------------------------

    def fit_evidence(row):

        r_squared = row[
            "trend_r_squared"
        ]

        if row[
            "strong_trend_fit"
        ]:

            return (
                f"The linear trend explains approximately "
                f"{r_squared:.1%} of the variation in "
                f"yearly topic prevalence."
            )

        return (
            f"The linear trend explains approximately "
            f"{r_squared:.1%} of the variation in "
            f"yearly topic prevalence."
        )

    radar[
        "trend_fit_evidence"
    ] = radar.apply(
        fit_evidence,
        axis=1
    )

    # --------------------------------------------------
    # Overall evidence summary
    # --------------------------------------------------

    def build_summary(row):

        topic = row[
            "topic_label"
        ]

        if row[
            "radar_emerging_candidate"
        ]:

            return (
                f"{topic} shows an emerging research signal "
                f"because its long-term prevalence is increasing "
                f"with statistical support, recent prevalence "
                f"is positive, and the topic has sufficient "
                f"research volume."
            )

        evidence_class = row[
            "radar_evidence_class"
        ]

        if evidence_class == (
            "statistically_supported_growth"
        ):

            return (
                f"{topic} shows statistically supported "
                f"long-term growth, but does not currently "
                f"show the full combination of recent momentum "
                f"and volume required by the Research Radar."
            )

        if evidence_class == (
            "recent_momentum"
        ):

            return (
                f"{topic} shows recent positive momentum, "
                f"but the available long-term evidence does "
                f"not currently satisfy the Research Radar "
                f"growth criteria."
            )

        if evidence_class == (
            "established_volume"
        ):

            return (
                f"{topic} has substantial research volume, "
                f"but its current trend evidence does not "
                f"produce the full emerging-topic signal."
            )

        return (
            f"{topic} does not currently show enough "
            f"evidence to satisfy the Research Radar "
            f"emerging-topic criteria."
        )

    radar[
        "evidence_summary"
    ] = radar.apply(
        build_summary,
        axis=1
    )

    # --------------------------------------------------
    # Signal explanation
    # --------------------------------------------------

    def signal_explanation(row):

        signals = []

        if row[
            "statistically_supported_trend"
        ]:

            signals.append(
                "statistically supported growth"
            )

        if row[
            "meaningful_long_term_growth"
        ]:

            signals.append(
                "meaningful long-term slope"
            )

        if row[
            "strong_trend_fit"
        ]:

            signals.append(
                "strong trend fit"
            )

        if row[
            "positive_recent_momentum"
        ]:

            signals.append(
                "positive recent momentum"
            )

        if row[
            "sufficient_research_volume"
        ]:

            signals.append(
                "sufficient research volume"
            )

        if not signals:

            return "No supporting signals met the current thresholds."

        return (
            "Supporting signals: "
            + "; ".join(signals)
            + "."
        )

    radar[
        "signal_explanation"
    ] = radar.apply(
        signal_explanation,
        axis=1
    )

    # --------------------------------------------------
    # Final evidence dataset
    # --------------------------------------------------

    evidence_columns = [
        "cluster",
        "topic_label",

        "latest_complete_year_papers",

        "trend_slope",
        "trend_p_value_fdr",
        "trend_r_squared",
        "trend_direction",

        "recent_share_change_percentage_points",
        "recent_momentum_direction",

        "statistically_supported_trend",
        "meaningful_long_term_growth",
        "strong_trend_fit",
        "positive_recent_momentum",
        "sufficient_research_volume",

        "supporting_signal_count",

        "radar_emerging_candidate",
        "radar_evidence_class",

        "long_term_trend_evidence",
        "recent_momentum_evidence",
        "research_volume_evidence",
        "trend_fit_evidence",

        "signal_explanation",
        "evidence_summary",
        "interpretation",
    ]

    evidence = radar[
        evidence_columns
    ].copy()

    # --------------------------------------------------
    # Sort candidates first
    # --------------------------------------------------

    evidence = evidence.sort_values(
        [
            "radar_emerging_candidate",
            "supporting_signal_count",
            "trend_slope",
        ],
        ascending=[
            False,
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

    evidence.to_csv(
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
        "RESEARCH RADAR EVIDENCE"
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

    print(
        f"\nTotal topics: "
        f"{len(evidence):,}"
    )

    candidates = evidence[
        evidence[
            "radar_emerging_candidate"
        ]
    ]

    print(
        f"Emerging candidates: "
        f"{len(candidates):,}"
    )

    print(
        "\n----------------------------------------"
    )

    print(
        "EMERGING TOPIC EVIDENCE"
    )

    print(
        "----------------------------------------"
    )

    if candidates.empty:

        print(
            "No emerging candidates found."
        )

    else:

        for _, row in candidates.iterrows():

            print(
                f"\n{row['topic_label']}"
            )

            print(
                f"  Papers: "
                f"{int(row['latest_complete_year_papers']):,}"
            )

            print(
                f"  Trend: "
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
                f"  Signals: "
                f"{int(row['supporting_signal_count'])}/5"
            )

            print(
                f"  Evidence:"
            )

            print(
                f"    {row['evidence_summary']}"
            )

    print(
        "\n========================================"
    )


if __name__ == "__main__":
    main()