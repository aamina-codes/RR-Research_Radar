from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/processed/trends/topic_trend_analysis.csv"
)

OUTPUT_DIR = Path(
    "data/processed/trends"
)

OUTPUT_PATH = (
    OUTPUT_DIR / "emerging_topics.csv"
)


# --------------------------------------------------
# Detection thresholds
# --------------------------------------------------

# Minimum number of papers a topic must have in the
# latest complete year to avoid flagging extremely
# small research areas.
MIN_RECENT_PAPERS = 50

# Statistical significance threshold after
# Benjamini-Hochberg FDR correction.
FDR_THRESHOLD = 0.05

# Minimum positive long-term slope.
#
# The slope is measured in percentage points of
# research-share change per year.
MIN_TREND_SLOPE = 0.05

# Minimum recent share increase.
MIN_RECENT_MOMENTUM = 0.05

# Minimum R-squared for a reasonably consistent
# linear trend.
MIN_R_SQUARED = 0.30


def main():

    print("Loading statistical trend analysis...")

    df = pd.read_csv(
        INPUT_PATH
    )

    print(
        f"Topics loaded: {len(df)}"
    )

    required_columns = {
        "cluster",
        "topic_label",
        "trend_slope",
        "trend_p_value_fdr",
        "trend_r_squared",
        "trend_direction",
        "latest_complete_year",
        "latest_share_percent",
        "previous_average_share_percent",
        "recent_share_change_percentage_points",
        "recent_momentum_direction",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------
    # Load prevalence data so we can obtain the actual
    # paper count in the latest complete year.
    # --------------------------------------------------

    prevalence_path = Path(
        "data/processed/trends/"
        "topic_prevalence_by_year.csv"
    )

    prevalence = pd.read_csv(
        prevalence_path
    )

    latest_complete_year = int(
        prevalence.loc[
            prevalence["year_status"] == "complete",
            "year"
        ].max()
    )

    print(
        f"\nLatest complete year: "
        f"{latest_complete_year}"
    )

    latest_counts = prevalence[
        (
            prevalence["year"]
            == latest_complete_year
        )
        &
        (
            prevalence["year_status"]
            == "complete"
        )
    ][
        [
            "cluster",
            "paper_count",
        ]
    ].copy()

    latest_counts = latest_counts.rename(
        columns={
            "paper_count":
            "latest_complete_year_papers"
        }
    )

    # --------------------------------------------------
    # Attach latest paper volume
    # --------------------------------------------------

    df = df.merge(
        latest_counts,
        on="cluster",
        how="left",
        validate="one_to_one"
    )

    # --------------------------------------------------
    # Signal 1:
    # Long-term statistically significant increase
    # --------------------------------------------------

    df["long_term_signal"] = (
        (df["trend_direction"] == "increasing")
        &
        (
            df["trend_p_value_fdr"]
            < FDR_THRESHOLD
        )
        &
        (
            df["trend_slope"]
            >= MIN_TREND_SLOPE
        )
        &
        (
            df["trend_r_squared"]
            >= MIN_R_SQUARED
        )
    )

    # --------------------------------------------------
    # Signal 2:
    # Recent positive momentum
    # --------------------------------------------------

    df["recent_momentum_signal"] = (
        (
            df["recent_momentum_direction"]
            == "increasing"
        )
        &
        (
            df[
                "recent_share_change_percentage_points"
            ]
            >= MIN_RECENT_MOMENTUM
        )
    )

    # --------------------------------------------------
    # Signal 3:
    # Meaningful research volume
    # --------------------------------------------------

    df["volume_signal"] = (
        df[
            "latest_complete_year_papers"
        ]
        >= MIN_RECENT_PAPERS
    )

    # --------------------------------------------------
    # Build an interpretable evidence count
    # --------------------------------------------------

    df["signals_supported"] = (
        df[
            [
                "long_term_signal",
                "recent_momentum_signal",
                "volume_signal",
            ]
        ]
        .sum(axis=1)
    )

    # --------------------------------------------------
    # Emerging topic classification
    # --------------------------------------------------
    #
    # A topic must satisfy all three signals:
    #
    # 1. statistically supported long-term growth
    # 2. positive recent momentum
    # 3. sufficient recent research volume
    #
    # This keeps the detector conservative and
    # interpretable.

    df["emerging_topic"] = (
        df["signals_supported"] == 3
    )

    # --------------------------------------------------
    # Evidence classification
    # --------------------------------------------------

    def classify_evidence(row):

        if row["emerging_topic"]:
            return "strong_emerging_signal"

        if (
            row["long_term_signal"]
            and row["volume_signal"]
        ):
            return "long_term_growth"

        if (
            row["recent_momentum_signal"]
            and row["volume_signal"]
        ):
            return "recent_momentum"

        if row["long_term_signal"]:
            return "statistically_supported_growth"

        if row["recent_momentum_signal"]:
            return "recent_momentum_only"

        return "no_emerging_signal"

    df["evidence_class"] = (
        df.apply(
            classify_evidence,
            axis=1
        )
    )

    # --------------------------------------------------
    # Create an interpretable radar signal
    # --------------------------------------------------
    #
    # This is NOT a subjective quality score.
    # It is simply a count of independent evidence
    # signals supported by the data.
    #
    # 0 = no signal
    # 1 = one signal
    # 2 = two signals
    # 3 = all three signals
    #
    # Keeping this transparent makes the methodology
    # easy to explain in the README/dashboard.

    df["radar_signal_strength"] = (
        df["signals_supported"]
    )

    # --------------------------------------------------
    # Sort results
    # --------------------------------------------------

    df = df.sort_values(
        [
            "emerging_topic",
            "radar_signal_strength",
            "trend_slope",
            "recent_share_change_percentage_points",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------
    # Select output columns
    # --------------------------------------------------

    output_columns = [
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

    results = df[
        output_columns
    ].copy()

    # --------------------------------------------------
    # Save complete analysis
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "\nSaved emerging-topic analysis:"
    )

    print(
        OUTPUT_PATH
    )

    # --------------------------------------------------
    # Display summary
    # --------------------------------------------------

    print(
        "\nEmerging-topic results:"
    )

    print(
        results[
            [
                "topic_label",
                "latest_complete_year_papers",
                "trend_slope",
                "trend_p_value_fdr",
                "trend_r_squared",
                "recent_share_change_percentage_points",
                "signals_supported",
                "evidence_class",
                "emerging_topic",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # Display only emerging topics
    # --------------------------------------------------

    emerging = results[
        results["emerging_topic"]
    ]

    print(
        "\n========================================"
    )

    print(
        "EMERGING TOPIC CANDIDATES"
    )

    print(
        "========================================"
    )

    if emerging.empty:

        print(
            "No topics currently satisfy all "
            "three emerging-topic signals."
        )

    else:

        for _, row in emerging.iterrows():

            print(
                f"\n{row['topic_label']}"
            )

            print(
                f"  Papers in "
                f"{int(row['latest_complete_year'])}: "
                f"{int(row['latest_complete_year_papers']):,}"
            )

            print(
                f"  Long-term slope: "
                f"{row['trend_slope']:.3f} "
                f"percentage points/year"
            )

            print(
                f"  FDR-adjusted p-value: "
                f"{row['trend_p_value_fdr']:.6f}"
            )

            print(
                f"  R²: "
                f"{row['trend_r_squared']:.3f}"
            )

            print(
                f"  Recent share change: "
                f"{row['recent_share_change_percentage_points']:.3f} "
                f"percentage points"
            )

            print(
                "  Signals: "
                f"{int(row['signals_supported'])}/3"
            )


if __name__ == "__main__":
    main()