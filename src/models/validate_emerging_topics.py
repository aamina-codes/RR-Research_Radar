from pathlib import Path
import pandas as pd


TOPICS_PATH = Path(
    "data/processed/topics/cluster_topics.csv"
)

TOPIC_LABELS_PATH = Path(
    "data/processed/topics/topic_labels.csv"
)

REPRESENTATIVE_PAPERS_PATH = Path(
    "data/processed/topics/representative_papers.csv"
)

RADAR_PATH = Path(
    "data/processed/trends/research_radar_signals.csv"
)

OUTPUT_DIR = Path(
    "data/processed/topics/emerging_topic_validation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("Loading Research Radar outputs...")

    topics_df = pd.read_csv(TOPICS_PATH)
    labels_df = pd.read_csv(TOPIC_LABELS_PATH)
    papers_df = pd.read_csv(REPRESENTATIVE_PAPERS_PATH)
    radar_df = pd.read_csv(RADAR_PATH)

    print(f"Topic records loaded: {len(topics_df):,}")
    print(f"Topic labels loaded: {len(labels_df):,}")
    print(f"Representative papers loaded: {len(papers_df):,}")
    print(f"Radar records loaded: {len(radar_df):,}")

    # ---------------------------------------------------------
    # Identify Research Radar candidates
    # ---------------------------------------------------------

    candidates = radar_df[
        radar_df["radar_emerging_candidate"] == True
    ].copy()

    candidate_names = candidates["topic_label"].tolist()

    print("\nEmerging Research Radar candidates:")

    for topic in candidate_names:
        print(f"  - {topic}")

    # ---------------------------------------------------------
    # Attach human-readable topic labels to clusters
    # ---------------------------------------------------------

    labels_df = labels_df[
        ["cluster", "topic_label"]
    ].copy()

    topics_df = topics_df.merge(
        labels_df,
        on="cluster",
        how="left"
    )

    papers_df = papers_df.merge(
        labels_df,
        on="cluster",
        how="left"
    )

    # ---------------------------------------------------------
    # Validate label matching
    # ---------------------------------------------------------

    missing_topic_labels = topics_df[
        topics_df["topic_label"].isna()
    ]

    if not missing_topic_labels.empty:

        print(
            "\nWARNING:"
            f" {len(missing_topic_labels)} clusters "
            "could not be matched to topic labels."
        )

    # ---------------------------------------------------------
    # Extract candidate topic information
    # ---------------------------------------------------------

    candidate_topics = topics_df[
        topics_df["topic_label"].isin(candidate_names)
    ].copy()

    candidate_topics = candidate_topics[
        [
            "cluster",
            "topic_label",
            "paper_count",
            "top_terms",
        ]
    ]

    candidate_topics = candidate_topics.sort_values(
        "topic_label"
    )

    topic_output = (
        OUTPUT_DIR /
        "emerging_topic_keywords.csv"
    )

    candidate_topics.to_csv(
        topic_output,
        index=False
    )

    print(
        f"\nSaved topic keyword validation data to:\n"
        f"{topic_output}"
    )

    # ---------------------------------------------------------
    # Extract representative papers
    # ---------------------------------------------------------

    candidate_papers = papers_df[
        papers_df["topic_label"].isin(candidate_names)
    ].copy()

    candidate_papers = candidate_papers[
        [
            "cluster",
            "topic_label",
            "rank",
            "similarity_to_centroid",
            "paper_id",
            "title",
        ]
    ]

    candidate_papers = candidate_papers.sort_values(
        [
            "topic_label",
            "rank",
        ]
    )

    papers_output = (
        OUTPUT_DIR /
        "emerging_topic_representative_papers.csv"
    )

    candidate_papers.to_csv(
        papers_output,
        index=False
    )

    print(
        f"Saved representative papers to:\n"
        f"{papers_output}"
    )

    # ---------------------------------------------------------
    # Create a human-readable validation report
    # ---------------------------------------------------------

    report_lines = []

    report_lines.append(
        "RESEARCH RADAR — EMERGING TOPIC VALIDATION"
    )

    report_lines.append(
        "=" * 60
    )

    report_lines.append("")

    report_lines.append(
        "Purpose:"
    )

    report_lines.append(
        "Inspect the keywords and representative papers "
        "behind the statistically detected emerging-topic "
        "candidates."
    )

    report_lines.append("")

    report_lines.append(
        "The candidates were identified by the Research Radar "
        "signal layer. This report does not alter or rerank "
        "those signals."
    )

    # ---------------------------------------------------------
    # Report each candidate
    # ---------------------------------------------------------

    for topic in candidate_names:

        report_lines.append("")
        report_lines.append(
            "-" * 60
        )

        report_lines.append(
            f"TOPIC: {topic}"
        )

        report_lines.append(
            "-" * 60
        )

        # Topic metadata

        topic_info = candidate_topics[
            candidate_topics["topic_label"] == topic
        ]

        if not topic_info.empty:

            row = topic_info.iloc[0]

            report_lines.append(
                f"\nCluster: {row['cluster']}"
            )

            report_lines.append(
                f"Papers in cluster: {row['paper_count']}"
            )

            report_lines.append(
                f"\nTop terms:\n{row['top_terms']}"
            )

        # Representative papers

        topic_papers = candidate_papers[
            candidate_papers["topic_label"] == topic
        ].sort_values("rank")

        report_lines.append(
            f"\nRepresentative papers: "
            f"{len(topic_papers)}"
        )

        for _, paper in topic_papers.iterrows():

            title = str(
                paper["title"]
            ).strip()

            similarity = paper[
                "similarity_to_centroid"
            ]

            rank = int(
                paper["rank"]
            )

            report_lines.append(
                f"\n{rank}. {title}"
            )

            report_lines.append(
                f"   Similarity to centroid: "
                f"{float(similarity):.4f}"
            )

    # ---------------------------------------------------------
    # Save report
    # ---------------------------------------------------------

    report_output = (
        OUTPUT_DIR /
        "emerging_topic_validation_report.txt"
    )

    report_output.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    print(
        f"\nSaved validation report to:\n"
        f"{report_output}"
    )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print("\n" + "=" * 60)

    print(
        "EMERGING TOPIC VALIDATION PREPARATION COMPLETE"
    )

    print("=" * 60)

    print(
        f"\nCandidates inspected: "
        f"{len(candidate_names)}"
    )

    print(
        f"Representative papers available: "
        f"{len(candidate_papers)}"
    )

    print(
        "\nNext step:"
    )

    print(
        "Review the generated CSV and TXT files to determine "
        "whether the four statistically detected topics are "
        "semantically coherent."
    )


if __name__ == "__main__":
    main()