from pathlib import Path

import numpy as np
import pandas as pd


CORPUS_PATH = Path(
    "data/processed/research_corpus.csv"
)

LABELS_PATH = Path(
    "data/processed/clusters/kmeans_30_labels.npy"
)

TOPIC_LABELS_PATH = Path(
    "data/processed/topics/topic_labels.csv"
)

OUTPUT_DIR = Path(
    "data/processed/trends"
)


def main():

    print("Loading research corpus...")

    df = pd.read_csv(
        CORPUS_PATH,
        usecols=[
            "id",
            "published_date",
        ]
    )

    print(
        f"Papers loaded: {len(df):,}"
    )

    print("\nLoading cluster assignments...")

    labels = np.load(
        LABELS_PATH
    )

    print(
        f"Cluster labels loaded: "
        f"{len(labels):,}"
    )

    if len(df) != len(labels):

        raise ValueError(
            "Number of papers and cluster "
            "labels do not match."
        )

    print("\nLoading topic labels...")

    topic_labels = pd.read_csv(
        TOPIC_LABELS_PATH,
        usecols=[
            "cluster",
            "topic_label",
        ]
    )

    # --------------------------------------------------
    # Attach cluster + topic information
    # --------------------------------------------------

    df["cluster"] = labels

    df = df.merge(
        topic_labels,
        on="cluster",
        how="left",
        validate="many_to_one"
    )

    if df["topic_label"].isna().any():

        missing_clusters = (
            df.loc[
                df["topic_label"].isna(),
                "cluster"
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Missing topic labels for clusters: "
            f"{missing_clusters}"
        )

    # --------------------------------------------------
    # Parse publication dates
    # --------------------------------------------------

    df["published_date"] = pd.to_datetime(
        df["published_date"],
        errors="coerce"
    )

    invalid_dates = (
        df["published_date"].isna().sum()
    )

    if invalid_dates > 0:

        print(
            f"\nWarning: {invalid_dates:,} "
            "papers have invalid dates."
        )

        df = df.dropna(
            subset=["published_date"]
        )

    df["year"] = (
        df["published_date"]
        .dt.year
        .astype(int)
    )

    print(
        "\nYear range:"
    )

    print(
        f"{df['year'].min()} → "
        f"{df['year'].max()}"
    )

    # --------------------------------------------------
    # Identify complete vs partial years
    # --------------------------------------------------

    # The dataset only contains papers through
    # January 30, 2025.
    #
    # Therefore 2025 is a partial year and should
    # not be directly compared with full years.

    max_date = df["published_date"].max()

    latest_year = max_date.year

    year_status = (
        df.groupby("year")["published_date"]
        .agg(
            first_date="min",
            last_date="max",
            paper_count="count"
        )
        .reset_index()
    )

    year_status["is_partial_year"] = (
        year_status["year"] == latest_year
    )

    year_status["year_status"] = np.where(
        year_status["is_partial_year"],
        "partial",
        "complete"
    )

    # --------------------------------------------------
    # Total papers per year
    # --------------------------------------------------

    yearly_totals = (
        df.groupby("year")
        .size()
        .reset_index(
            name="total_papers"
        )
    )

    # --------------------------------------------------
    # Topic counts by year
    # --------------------------------------------------

    topic_year = (
        df.groupby(
            [
                "year",
                "cluster",
                "topic_label",
            ]
        )
        .size()
        .reset_index(
            name="paper_count"
        )
    )

    # --------------------------------------------------
    # Add yearly totals
    # --------------------------------------------------

    topic_year = topic_year.merge(
        yearly_totals,
        on="year",
        how="left"
    )

    # --------------------------------------------------
    # Calculate topic share
    # --------------------------------------------------

    topic_year["topic_share"] = (
        topic_year["paper_count"]
        / topic_year["total_papers"]
    )

    topic_year["topic_share_percent"] = (
        topic_year["topic_share"]
        * 100
    )

    # --------------------------------------------------
    # Sort data
    # --------------------------------------------------

    topic_year = topic_year.sort_values(
        [
            "cluster",
            "year",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------
    # Year-over-year growth
    # --------------------------------------------------

    topic_year["previous_year_count"] = (
        topic_year
        .groupby("cluster")["paper_count"]
        .shift(1)
    )

    topic_year["previous_year_share"] = (
        topic_year
        .groupby("cluster")["topic_share"]
        .shift(1)
    )

    # Growth in absolute paper count
    topic_year["count_growth"] = (
        topic_year["paper_count"]
        - topic_year["previous_year_count"]
    )

    # Percentage growth in paper count
    topic_year["count_growth_percent"] = (
        (
            topic_year["paper_count"]
            - topic_year["previous_year_count"]
        )
        / topic_year["previous_year_count"]
    ) * 100

    # Change in share of research output
    topic_year["share_change"] = (
        topic_year["topic_share"]
        - topic_year["previous_year_share"]
    )

    topic_year["share_change_percentage_points"] = (
        topic_year["share_change"]
        * 100
    )

    # --------------------------------------------------
    # Add year completeness information
    # --------------------------------------------------

    topic_year = topic_year.merge(
        year_status[
            [
                "year",
                "first_date",
                "last_date",
                "year_status",
                "is_partial_year",
            ]
        ],
        on="year",
        how="left"
    )

    # --------------------------------------------------
    # Clean infinite values
    # --------------------------------------------------

    topic_year = topic_year.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    topic_output = (
        OUTPUT_DIR
        / "topic_prevalence_by_year.csv"
    )

    yearly_output = (
        OUTPUT_DIR
        / "year_summary.csv"
    )

    topic_year.to_csv(
        topic_output,
        index=False
    )

    year_status.to_csv(
        yearly_output,
        index=False
    )

    print(
        "\nSaved topic prevalence:"
    )

    print(
        topic_output
    )

    print(
        "\nSaved year summary:"
    )

    print(
        yearly_output
    )

    # --------------------------------------------------
    # Display summary
    # --------------------------------------------------

    print(
        "\nTopic prevalence preview:"
    )

    preview = topic_year[
        [
            "year",
            "topic_label",
            "paper_count",
            "topic_share_percent",
            "year_status",
        ]
    ].head(20)

    print(
        preview.to_string(
            index=False
        )
    )

    print(
        "\nYear summary:"
    )

    print(
        year_status.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()