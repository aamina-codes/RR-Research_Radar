from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/topics/cluster_topics.csv"
)

OUTPUT_PATH = Path(
    "data/processed/topics/topic_labels.csv"
)


TOPIC_LABELS = {
    0: "Multi-Armed Bandits",
    1: "Text and Social Media Analysis",
    2: "Domain Adaptation",
    3: "Named Entity and Relation Extraction",
    4: "Neural Networks and Deep Learning",
    5: "Question Answering",
    6: "Computer Vision and Open-Source Methods",
    7: "Word Embeddings and Semantic Representations",
    8: "Natural Language Processing",
    9: "Time-Series Forecasting",
    10: "Image Segmentation and Image Analysis",
    11: "General Machine Learning and AI",
    12: "Federated Learning",
    13: "Object Detection and Tracking",
    14: "Bayesian and Variational Inference",
    15: "Optimization for Neural Network Training",
    16: "Neural Machine Translation",
    17: "3D Vision and Pose Estimation",
    18: "Clustering Methods",
    19: "Generative Adversarial Networks",
    20: "Evolutionary and Multi-Objective Optimization",
    21: "Reinforcement Learning",
    22: "Adversarial Robustness",
    23: "Convolutional Neural Networks",
    24: "Large Language Models",
    25: "General Learning and Classification Methods",
    26: "Graph Neural Networks",
    27: "Dialogue Systems",
    28: "Video Understanding and Action Recognition",
    29: "Multilingual and Cross-Lingual NLP",
}


def main():

    print("Loading cluster topics...")

    df = pd.read_csv(INPUT_PATH)

    print(
        f"Clusters loaded: {len(df)}"
    )

    print("\nAssigning topic labels...")

    df["topic_label"] = df["cluster"].map(
        TOPIC_LABELS
    )

    missing_labels = df[
        df["topic_label"].isna()
    ]

    if not missing_labels.empty:

        raise ValueError(
            "Some clusters do not have "
            "topic labels: "
            f"{missing_labels['cluster'].tolist()}"
        )

    # Reorder columns for readability
    df = df[
        [
            "cluster",
            "topic_label",
            "paper_count",
            "top_terms",
        ]
    ]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nSaved topic labels to:"
        f"\n{OUTPUT_PATH}"
    )

    print("\nTopic map:")

    for _, row in df.iterrows():

        print(
            f"{int(row['cluster']):>2}  "
            f"{row['topic_label']:<45} "
            f"{int(row['paper_count']):>6,} papers"
        )


if __name__ == "__main__":
    main()