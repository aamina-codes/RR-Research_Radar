from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ------------------------------------------
# Paths
# ------------------------------------------

PREVALENCE_PATH = Path(
    "data/processed/trends/topic_prevalence_by_year.csv"
)

OUTPUT_DIR = Path(
    "data/processed/visualizations/topic_trend_plots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def clean_filename(text):

    text = text.lower()

    replacements = [
        (" ", "_"),
        ("-", "_"),
        ("/", "_"),
        ("(", ""),
        (")", ""),
    ]

    for old, new in replacements:
        text = text.replace(old, new)

    return text


def main():

    print("Loading topic prevalence data...")

    df = pd.read_csv(
        PREVALENCE_PATH
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    topics = sorted(
        df["topic_label"].unique()
    )

    print(
        f"Topics found: {len(topics)}"
    )

    print(
        "\nCreating plots..."
    )

    for topic in topics:

        topic_df = df[
            df["topic_label"] == topic
        ].sort_values("year")

        years = topic_df["year"]
        shares = topic_df[
            "topic_share_percent"
        ]

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            years,
            shares,
            marker="o"
        )

        plt.title(
            topic
        )

        plt.xlabel(
            "Year"
        )

        plt.ylabel(
            "Topic Share (%)"
        )

        plt.grid(True)

        plt.tight_layout()

        filename = (
            clean_filename(topic)
            + "_trend.png"
        )

        output_path = (
            OUTPUT_DIR / filename
        )

        plt.savefig(
            output_path,
            dpi=300
        )

        plt.close()

    print(
        "\nSaved plots:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        f"\nTotal plots created: "
        f"{len(topics)}"
    )


if __name__ == "__main__":
    main()