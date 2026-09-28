from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/processed/trends/topic_prevalence_by_year.csv"
)

OUTPUT_DIR = Path(
    "data/processed/trends"
)

OUTPUT_PATH = (
    OUTPUT_DIR / "validated_change_points.csv"
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MIN_YEARS_PER_SEGMENT = 3

# Minimum absolute difference in mean topic share
# required before we consider a split interesting.
MIN_MEAN_SHIFT = 0.10

# Number of random permutations used for each topic.
#
# 2,000 gives us a reasonably stable exploratory
# significance estimate without making the script
# unnecessarily slow.
N_PERMUTATIONS = 2000

RANDOM_STATE = 42

FDR_ALPHA = 0.05


# --------------------------------------------------
# Benjamini-Hochberg FDR correction
# --------------------------------------------------

def benjamini_hochberg(
    p_values,
    alpha=0.05
):
    """
    Apply the Benjamini-Hochberg procedure.

    Returns:
        adjusted_p_values
        significant_flags
    """

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    n = len(p_values)

    adjusted = np.full(
        n,
        np.nan
    )

    valid_mask = np.isfinite(
        p_values
    )

    valid_p = p_values[
        valid_mask
    ]

    if len(valid_p) == 0:

        return adjusted, np.zeros(
            n,
            dtype=bool
        )

    order = np.argsort(
        valid_p
    )

    sorted_p = valid_p[
        order
    ]

    m = len(sorted_p)

    adjusted_sorted = (
        sorted_p
        * m
        / np.arange(
            1,
            m + 1
        )
    )

    # Enforce monotonicity from right to left.
    adjusted_sorted = np.minimum.accumulate(
        adjusted_sorted[::-1]
    )[::-1]

    adjusted_sorted = np.clip(
        adjusted_sorted,
        0,
        1
    )

    valid_indices = np.where(
        valid_mask
    )[0]

    adjusted_valid = np.empty(
        len(valid_p)
    )

    adjusted_valid[
        order
    ] = adjusted_sorted

    adjusted[
        valid_indices
    ] = adjusted_valid

    significant = (
        adjusted < alpha
    )

    significant[
        ~valid_mask
    ] = False

    return (
        adjusted,
        significant
    )


# --------------------------------------------------
# Calculate the strongest change point
# --------------------------------------------------

def find_best_change_point(
    years,
    shares
):
    """
    Find the split producing the largest absolute
    difference between the mean share before and
    after the split.
    """

    candidates = []

    for split_index in range(
        MIN_YEARS_PER_SEGMENT,
        len(years) - MIN_YEARS_PER_SEGMENT + 1
    ):

        before = shares[
            :split_index
        ]

        after = shares[
            split_index:
        ]

        mean_before = (
            before.mean()
        )

        mean_after = (
            after.mean()
        )

        shift = (
            mean_after
            - mean_before
        )

        candidates.append({
            "change_point_year": int(
                years[split_index]
            ),
            "mean_before": mean_before,
            "mean_after": mean_after,
            "mean_shift": shift,
            "absolute_shift": abs(
                shift
            )
        })

    if not candidates:

        return None

    return max(
        candidates,
        key=lambda x: x[
            "absolute_shift"
        ]
    )


# --------------------------------------------------
# Permutation test
# --------------------------------------------------

def permutation_test(
    shares,
    observed_shift,
    rng
):
    """
    Test whether the observed maximum split is
    larger than would be expected if the yearly
    topic shares were randomly ordered.

    For each permutation we find the maximum
    absolute split difference.

    This creates a null distribution of the
    strongest possible change point under
    random temporal ordering.
    """

    observed_abs = abs(
        observed_shift
    )

    permutation_statistics = []

    n = len(shares)

    for _ in range(
        N_PERMUTATIONS
    ):

        shuffled = rng.permutation(
            shares
        )

        best_shift = 0.0

        for split_index in range(
            MIN_YEARS_PER_SEGMENT,
            n - MIN_YEARS_PER_SEGMENT + 1
        ):

            before = shuffled[
                :split_index
            ]

            after = shuffled[
                split_index:
            ]

            shift = abs(
                after.mean()
                - before.mean()
            )

            if shift > best_shift:

                best_shift = shift

        permutation_statistics.append(
            best_shift
        )

    permutation_statistics = np.asarray(
        permutation_statistics
    )

    # +1 correction prevents zero p-values.
    p_value = (
        (
            permutation_statistics
            >= observed_abs
        ).sum()
        + 1
    ) / (
        N_PERMUTATIONS + 1
    )

    return p_value


# --------------------------------------------------
# Analyze one topic
# --------------------------------------------------

def analyze_topic(
    group,
    rng
):
    """
    Find and test the strongest structural shift
    for one topic.
    """

    group = group[
        group["year_status"] == "complete"
    ].copy()

    group = group.sort_values(
        "year"
    )

    years = group[
        "year"
    ].to_numpy()

    shares = group[
        "topic_share_percent"
    ].to_numpy()

    if len(years) < (
        MIN_YEARS_PER_SEGMENT * 2
    ):

        return {
            "change_point_year": np.nan,
            "mean_before": np.nan,
            "mean_after": np.nan,
            "mean_shift": np.nan,
            "absolute_shift": np.nan,
            "raw_p_value": np.nan,
        }

    best = find_best_change_point(
        years,
        shares
    )

    if best is None:

        return {
            "change_point_year": np.nan,
            "mean_before": np.nan,
            "mean_after": np.nan,
            "mean_shift": np.nan,
            "absolute_shift": np.nan,
            "raw_p_value": np.nan,
        }

    # If the observed effect is too small, we still
    # calculate the statistic but flag it later.
    p_value = permutation_test(
        shares,
        best["mean_shift"],
        rng
    )

    return {
        "change_point_year": best[
            "change_point_year"
        ],
        "mean_before": best[
            "mean_before"
        ],
        "mean_after": best[
            "mean_after"
        ],
        "mean_shift": best[
            "mean_shift"
        ],
        "absolute_shift": best[
            "absolute_shift"
        ],
        "raw_p_value": p_value,
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print(
        "Loading topic prevalence data..."
    )

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

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    results = []

    grouped = df.groupby(
        [
            "cluster",
            "topic_label",
        ]
    )

    total_topics = len(
        grouped
    )

    print(
        f"\nTopics to analyze: "
        f"{total_topics}"
    )

    print(
        f"Permutations per topic: "
        f"{N_PERMUTATIONS:,}"
    )

    print(
        "\nRunning permutation tests..."
    )

    for topic_number, (
        (cluster, topic_label),
        group
    ) in enumerate(
        grouped,
        start=1
    ):

        print(
            f"[{topic_number}/{total_topics}] "
            f"{topic_label}"
        )

        result = analyze_topic(
            group,
            rng
        )

        result.update({
            "cluster": cluster,
            "topic_label": topic_label,
        })

        results.append(
            result
        )

    results = pd.DataFrame(
        results
    )

    # --------------------------------------------------
    # Multiple-testing correction
    # --------------------------------------------------

    (
        results["fdr_adjusted_p_value"],
        results["fdr_significant"]
    ) = benjamini_hochberg(
        results[
            "raw_p_value"
        ].to_numpy(),
        alpha=FDR_ALPHA
    )

    # --------------------------------------------------
    # Effect-size threshold
    # --------------------------------------------------

    results[
        "meaningful_shift"
    ] = (
        results[
            "absolute_shift"
        ]
        >= MIN_MEAN_SHIFT
    )

    # --------------------------------------------------
    # Final validated change point
    # --------------------------------------------------

    results[
        "validated_change_point"
    ] = (
        results[
            "fdr_significant"
        ]
        & results[
            "meaningful_shift"
        ]
    )

    # --------------------------------------------------
    # Direction
    # --------------------------------------------------

    def direction(row):

        if not row[
            "validated_change_point"
        ]:

            return "not_validated"

        if row[
            "mean_shift"
        ] > 0:

            return "upward_shift"

        if row[
            "mean_shift"
        ] < 0:

            return "downward_shift"

        return "no_shift"

    results[
        "change_point_direction"
    ] = results.apply(
        direction,
        axis=1
    )

    # --------------------------------------------------
    # Relative shift
    # --------------------------------------------------

    results[
        "relative_shift_percent"
    ] = np.where(
        results[
            "mean_before"
        ].abs() > 1e-12,
        (
            results[
                "mean_shift"
            ]
            / results[
                "mean_before"
            ].abs()
        ) * 100,
        np.nan
    )

    # --------------------------------------------------
    # Sort
    # --------------------------------------------------

    results = results.sort_values(
        [
            "validated_change_point",
            "fdr_adjusted_p_value",
            "absolute_shift",
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
        "\n========================================"
    )

    print(
        "VALIDATED CHANGE-POINT RESULTS"
    )

    print(
        "========================================"
    )

    print(
        f"\nSaved to:"
        f"\n{OUTPUT_PATH}"
    )

    validated = results[
        results[
            "validated_change_point"
        ]
    ]

    print(
        f"\nValidated change points: "
        f"{len(validated)} / "
        f"{len(results)}"
    )

    if validated.empty:

        print(
            "\nNo change points passed the "
            "FDR and effect-size thresholds."
        )

    else:

        for _, row in (
            validated.iterrows()
        ):

            print(
                f"\n{row['topic_label']}"
            )

            print(
                f"  Change point: "
                f"{int(row['change_point_year'])}"
            )

            print(
                f"  Direction: "
                f"{row['change_point_direction']}"
            )

            print(
                f"  Mean before: "
                f"{row['mean_before']:.3f}%"
            )

            print(
                f"  Mean after: "
                f"{row['mean_after']:.3f}%"
            )

            print(
                f"  Mean shift: "
                f"{row['mean_shift']:.3f} pp"
            )

            print(
                f"  Raw p-value: "
                f"{row['raw_p_value']:.5f}"
            )

            print(
                f"  FDR-adjusted p-value: "
                f"{row['fdr_adjusted_p_value']:.5f}"
            )

            print(
                f"  Relative shift: "
                f"{row['relative_shift_percent']:.2f}%"
            )


if __name__ == "__main__":
    main()