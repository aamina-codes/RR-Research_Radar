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
    OUTPUT_DIR / "topic_change_points.csv"
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

# We need enough observations on both sides of a
# candidate change point to make the comparison useful.
MIN_YEARS_PER_SEGMENT = 3

# A change point must produce a meaningful difference
# between the average topic share before and after it.
MIN_MEAN_SHIFT = 0.10


def detect_change_point(group):
    """
    Find the year that produces the largest difference
    between the average topic share before and after it.

    Only complete years are considered.

    This is an exploratory structural-shift detector,
    not a claim of causal change.
    """

    group = group[
        group["year_status"] == "complete"
    ].copy()

    group = group.sort_values("year")

    years = group["year"].to_numpy()

    shares = (
        group["topic_share_percent"]
        .to_numpy()
    )

    if len(years) < (
        MIN_YEARS_PER_SEGMENT * 2
    ):

        return pd.Series({
            "change_point_year": np.nan,
            "change_point_mean_before": np.nan,
            "change_point_mean_after": np.nan,
            "change_point_mean_shift": np.nan,
            "change_point_relative_shift": np.nan,
            "change_point_detected": False,
        })

    candidate_results = []

    # --------------------------------------------------
    # Test every possible split.
    #
    # Example:
    #
    # 2015 2016 2017 | 2018 2019 2020 ...
    #
    # The vertical line represents a candidate
    # structural change.
    # --------------------------------------------------

    for split_index in range(
        MIN_YEARS_PER_SEGMENT,
        len(years) - MIN_YEARS_PER_SEGMENT + 1,
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

        mean_shift = (
            mean_after
            - mean_before
        )

        candidate_results.append({
            "year": int(
                years[split_index]
            ),
            "mean_before": mean_before,
            "mean_after": mean_after,
            "mean_shift": mean_shift,
        })

    candidates = pd.DataFrame(
        candidate_results
    )

    # Find the split with the largest absolute shift.
    best_index = (
        candidates["mean_shift"]
        .abs()
        .idxmax()
    )

    best = candidates.loc[
        best_index
    ]

    mean_before = (
        best["mean_before"]
    )

    mean_after = (
        best["mean_after"]
    )

    mean_shift = (
        best["mean_shift"]
    )

    # Relative change compared with the magnitude
    # of the pre-change mean.
    if abs(mean_before) > 1e-12:

        relative_shift = (
            mean_shift
            / abs(mean_before)
        ) * 100

    else:

        relative_shift = np.nan

    detected = (
        abs(mean_shift)
        >= MIN_MEAN_SHIFT
    )

    return pd.Series({
        "change_point_year": int(
            best["year"]
        ),
        "change_point_mean_before": (
            mean_before
        ),
        "change_point_mean_after": (
            mean_after
        ),
        "change_point_mean_shift": (
            mean_shift
        ),
        "change_point_relative_shift": (
            relative_shift
        ),
        "change_point_detected": detected,
    })


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

    print(
        "\nDetecting structural shifts..."
    )

    results = (
        df.groupby(
            [
                "cluster",
                "topic_label",
            ],
            group_keys=False
        )
        .apply(
            detect_change_point,
            include_groups=False
        )
        .reset_index()
    )

    # --------------------------------------------------
    # Classify direction
    # --------------------------------------------------

    def classify_direction(row):

        if not row["change_point_detected"]:
            return "no_clear_shift"

        if (
            row["change_point_mean_shift"]
            > 0
        ):
            return "upward_shift"

        if (
            row["change_point_mean_shift"]
            < 0
        ):
            return "downward_shift"

        return "no_clear_shift"

    results["change_point_direction"] = (
        results.apply(
            classify_direction,
            axis=1
        )
    )

    # --------------------------------------------------
    # Sort strongest shifts first
    # --------------------------------------------------

    results = results.sort_values(
        [
            "change_point_detected",
            "change_point_mean_shift",
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

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "\nSaved change-point analysis:"
    )

    print(
        OUTPUT_PATH
    )

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print(
        "\nChange-point results:"
    )

    print(
        results[
            [
                "cluster",
                "topic_label",
                "change_point_year",
                "change_point_mean_before",
                "change_point_mean_after",
                "change_point_mean_shift",
                "change_point_relative_shift",
                "change_point_direction",
                "change_point_detected",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # Display detected shifts
    # --------------------------------------------------

    detected = results[
        results["change_point_detected"]
    ]

    print(
        "\n========================================"
    )

    print(
        "DETECTED STRUCTURAL SHIFTS"
    )

    print(
        "========================================"
    )

    if detected.empty:

        print(
            "No topics currently meet the "
            "change-point threshold."
        )

    else:

        for _, row in detected.iterrows():

            print(
                f"\n{row['topic_label']}"
            )

            print(
                f"  Change point: "
                f"{int(row['change_point_year'])}"
            )

            print(
                f"  Mean share before: "
                f"{row['change_point_mean_before']:.3f}%"
            )

            print(
                f"  Mean share after: "
                f"{row['change_point_mean_after']:.3f}%"
            )

            print(
                f"  Mean shift: "
                f"{row['change_point_mean_shift']:.3f} "
                f"percentage points"
            )

            print(
                f"  Relative shift: "
                f"{row['change_point_relative_shift']:.2f}%"
            )

            print(
                f"  Direction: "
                f"{row['change_point_direction']}"
            )


if __name__ == "__main__":
    main()