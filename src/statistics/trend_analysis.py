from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress


INPUT_PATH = Path(
    "data/processed/trends/topic_prevalence_by_year.csv"
)

OUTPUT_DIR = Path(
    "data/processed/trends"
)

OUTPUT_PATH = (
    OUTPUT_DIR / "topic_trend_analysis.csv"
)


# We use complete years only for the main trend analysis.
# 2025 is a partial year in this dataset.
MIN_YEARS = 5

# Recent period used for measuring recent momentum.
RECENT_YEARS = 3


def calculate_trend(group):
    """
    Calculate a linear trend in topic share over time.

    The regression estimates whether the topic's share
    of the research corpus has generally increased or
    decreased across complete years.
    """

    group = group[
        group["year_status"] == "complete"
    ].copy()

    group = group.sort_values("year")

    if len(group) < MIN_YEARS:
        return pd.Series({
            "years_used": len(group),
            "trend_slope": np.nan,
            "trend_p_value": np.nan,
            "trend_r_squared": np.nan,
            "trend_direction": "insufficient_data",
        })

    x = group["year"].to_numpy()

    y = group[
        "topic_share_percent"
    ].to_numpy()

    result = linregress(x, y)

    slope = result.slope
    p_value = result.pvalue
    r_squared = result.rvalue ** 2

    if p_value < 0.05 and slope > 0:
        direction = "increasing"

    elif p_value < 0.05 and slope < 0:
        direction = "decreasing"

    else:
        direction = "no_clear_trend"

    return pd.Series({
        "years_used": len(group),
        "trend_slope": slope,
        "trend_p_value": p_value,
        "trend_r_squared": r_squared,
        "trend_direction": direction,
    })


def calculate_recent_momentum(group):
    """
    Compare the average topic share in the latest
    complete year against the average of the previous
    complete years.

    This provides a recent-momentum signal that is
    separate from the long-term regression trend.
    """

    group = group[
        group["year_status"] == "complete"
    ].copy()

    group = group.sort_values("year")

    if len(group) < RECENT_YEARS + 1:
        return pd.Series({
            "latest_complete_year": np.nan,
            "latest_share_percent": np.nan,
            "previous_average_share_percent": np.nan,
            "recent_share_change_percentage_points": np.nan,
            "recent_momentum_direction": "insufficient_data",
        })

    latest = group.iloc[-1]

    previous = group.iloc[
        -(RECENT_YEARS + 1):-1
    ]

    latest_share = latest[
        "topic_share_percent"
    ]

    previous_average = previous[
        "topic_share_percent"
    ].mean()

    change = (
        latest_share
        - previous_average
    )

    if change > 0:
        direction = "increasing"

    elif change < 0:
        direction = "decreasing"

    else:
        direction = "stable"

    return pd.Series({
        "latest_complete_year": int(
            latest["year"]
        ),
        "latest_share_percent": latest_share,
        "previous_average_share_percent": (
            previous_average
        ),
        "recent_share_change_percentage_points": (
            change
        ),
        "recent_momentum_direction": direction,
    })


def main():

    print("Loading topic prevalence data...")

    df = pd.read_csv(
        INPUT_PATH
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    required_columns = {
        "year",
        "cluster",
        "topic_label",
        "paper_count",
        "topic_share_percent",
        "year_status",
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

    print(
        "\nCalculating long-term trends..."
    )

    trend_results = (
        df.groupby(
            [
                "cluster",
                "topic_label",
            ],
            group_keys=False
        )
        .apply(
            calculate_trend,
            include_groups=False
        )
        .reset_index()
    )

    print(
        "\nCalculating recent momentum..."
    )

    momentum_results = (
        df.groupby(
            [
                "cluster",
                "topic_label",
            ],
            group_keys=False
        )
        .apply(
            calculate_recent_momentum,
            include_groups=False
        )
        .reset_index()
    )

    # --------------------------------------------------
    # Combine statistical signals
    # --------------------------------------------------

    results = trend_results.merge(
        momentum_results,
        on=[
            "cluster",
            "topic_label",
        ],
        how="left"
    )

    # --------------------------------------------------
    # Multiple-testing correction
    # --------------------------------------------------
    #
    # We are testing 30 topics simultaneously.
    # A raw p-value of 0.05 can therefore produce
    # false positives simply because many tests are
    # being performed.
    #
    # Benjamini-Hochberg FDR correction controls the
    # expected false discovery rate.

    valid_pvalues = (
        results["trend_p_value"]
        .notna()
    )

    pvalues = results.loc[
        valid_pvalues,
        "trend_p_value"
    ]

    if len(pvalues) > 0:

        sorted_indices = (
            pvalues
            .sort_values()
            .index
        )

        m = len(pvalues)

        adjusted = pd.Series(
            index=pvalues.index,
            dtype=float
        )

        previous_value = 1.0

        for rank, index in reversed(
            list(
                enumerate(
                    sorted_indices,
                    start=1
                )
            )
        ):

            p = pvalues.loc[index]

            adjusted_value = min(
                previous_value,
                p * m / rank
            )

            adjusted.loc[index] = (
                adjusted_value
            )

            previous_value = adjusted_value

        results[
            "trend_p_value_fdr"
        ] = np.nan

        results.loc[
            valid_pvalues,
            "trend_p_value_fdr"
        ] = adjusted

    else:

        results[
            "trend_p_value_fdr"
        ] = np.nan

    # --------------------------------------------------
    # Statistical significance
    # --------------------------------------------------

    results["statistically_significant"] = (
        results["trend_p_value_fdr"] < 0.05
    )

    # --------------------------------------------------
    # Emerging trend candidate
    # --------------------------------------------------
    #
    # This is intentionally conservative.
    #
    # A topic is considered a candidate only when:
    #
    # 1. Long-term trend is statistically significant
    # 2. Long-term trend is increasing
    # 3. Recent momentum is also increasing
    #
    # This is NOT the final emerging-topic detector.
    # It is an initial statistical candidate signal.

    results["emerging_candidate"] = (
        (results["trend_direction"] == "increasing")
        &
        (
            results["trend_p_value_fdr"] < 0.05
        )
        &
        (
            results["recent_momentum_direction"]
            == "increasing"
        )
    )

    # --------------------------------------------------
    # Sort by statistical evidence
    # --------------------------------------------------

    results = results.sort_values(
        [
            "emerging_candidate",
            "trend_p_value_fdr",
            "trend_slope",
        ],
        ascending=[
            False,
            True,
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

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "\nSaved trend analysis:"
    )

    print(
        OUTPUT_PATH
    )

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    display_columns = [
        "cluster",
        "topic_label",
        "trend_slope",
        "trend_p_value",
        "trend_p_value_fdr",
        "trend_r_squared",
        "trend_direction",
        "recent_share_change_percentage_points",
        "recent_momentum_direction",
        "emerging_candidate",
    ]

    print(
        "\nTrend analysis results:"
    )

    print(
        results[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nPotential emerging candidates:"
    )

    candidates = results[
        results["emerging_candidate"]
    ]

    if candidates.empty:

        print(
            "No topics currently meet all "
            "candidate criteria."
        )

    else:

        for _, row in candidates.iterrows():

            print(
                f"- {row['topic_label']}"
            )


if __name__ == "__main__":
    main()