# RR-Research_Radar

> An NLP and statistical ML system for discovering emerging trends in scientific research.

Research Radar analyzes scientific literature to identify research areas showing **sustained growth, recent momentum, and statistical evidence of emergence**.

Instead of relying only on raw publication counts or existing research categories, the system discovers latent research topics from paper abstracts and combines **NLP, unsupervised machine learning, statistical trend analysis, and semantic validation** to build an evidence-based research radar.

---

## Overview

Scientific research grows rapidly, making it difficult to distinguish between:

- topics with large publication volumes,
- topics experiencing temporary increases in attention,
- and research areas showing sustained evidence of growth.

Research Radar addresses this by building a multi-stage pipeline that transforms scientific abstracts into interpretable research trends.

The project currently analyzes:

- **112,363 scientific papers**
- **2015–2025** publication coverage
- **30 discovered research topics**
- **4 emerging research candidates**
- statistical trend testing with **Benjamini–Hochberg FDR correction**
- semantic validation using representative papers
- a 2D research landscape built from topic embeddings

The analysis uses complete publication years through **2024** for the main trend comparisons. The dataset only contains papers through **January 30, 2025**, so 2025 is treated as a partial year.

---

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rr-research-radar.streamlit.app/)

Explore the interactive Research Radar dashboard:

- **Research Overview** — emerging-topic summary and radar signals
- **Research Trend Explorer** — explore topic prevalence over time
- **Topic Deep Dive** — inspect individual topics, evidence, semantic coherence, and representative papers
- **Research Landscape** — explore the discovered research topics in a 2D semantic map

---

## Why This Project?

A simple question such as:

> "Which areas of AI research are emerging?"

is harder to answer than it initially appears.

A topic can have:

- a large number of papers but declining share,
- a recent spike without sustained long-term growth,
- statistically significant growth but very low research volume,
- or strong statistical signals while representing a broad or mixed semantic cluster.

Research Radar therefore does not use a single popularity metric.

Instead, it combines multiple pieces of evidence:

```text
Research Volume
      +
Long-Term Trend
      +
Statistical Significance
      +
Recent Momentum
      +
Trend Fit
      +
Semantic Coherence
      ↓
Research Radar Signal

```

---

## What Research Radar Does

The pipeline follows six major stages:

```text
Scientific Papers
       │
       ▼
Text Preprocessing
       │
       ▼
TF-IDF Representation
       │
       ▼
SVD + Topic Discovery
       │
       ▼
Statistical Trend Analysis
       │
       ▼
Semantic Validation
       │
       ▼
Research Radar
```

### 1. Corpus Preparation

The original dataset contains scientific papers from multiple research areas.

The project focuses the analysis on six broad AI/ML-related categories:

- Machine Learning
- Computer Vision and Pattern Recognition
- Computation and Language
- Artificial Intelligence
- Machine Learning (Statistics)
- Neural and Evolutionary Computing

These categories are used to define the broad corpus.

They are not used as the discovered research topics.


### 2. NLP Representation

Paper titles and abstracts are combined and transformed using TF-IDF.

The baseline representation uses:

- English stop-word removal
- unigram and bigram features
- minimum document frequency filtering
- maximum document frequency filtering
- sublinear TF scaling
- up to 20,000 vocabulary features

The resulting TF-IDF representation contains:

```bash
112,363 papers
20,000 features
10,102,546 non-zero values
```

### 3. Dimensionality Reduction

The sparse TF-IDF matrix is reduced using Truncated SVD.

```text
TF-IDF
  ↓
Truncated SVD
  ↓
100-dimensional paper representations
```

The 100-dimensional representation explains approximately 10.20% of the variance.

The reduced representation is used for clustering and semantic comparisons rather than as a claim that all textual information is preserved.


### 4. Topic Discovery

Research topics are discovered using K-Means clustering over the reduced paper representations.

Several candidate values of k were evaluated:

| Topics | Silhouette Score  |
|---|---|
| **5** | 0.0347 |
| **10** | 0.0412 |
| **15** | 0.0371 |
| **20** | 0.0543 |
| **25** | 0.0484 |
| **30** | 0.0679 |

```bash
k = 30 produced the strongest silhouette score among the tested configurations and was used for the current research radar.
```

The resulting topics include areas such as:

- Graph Neural Networks
- Federated Learning
- Time-Series Forecasting
- Large Language Models
- Reinforcement Learning
- Neural Machine Translation
- Question Answering
- Object Detection and Tracking
- Multilingual and Cross-Lingual NLP
- Generative Adversarial Networks
- Bayesian and Variational Inference
- Adversarial Robustness

The topic labels are assigned after inspecting cluster-level terms and representative papers.

---

## Research Radar Methodology

The core of the project is the Research Radar signal layer.

For each discovered topic, the system evaluates several dimensions.

### Long-Term Trend

A linear regression is fitted to the topic's share of publications across complete years.

The analysis records:

- trend slope
- p-value
- R²
- trend direction

A positive slope indicates that the topic's share of the research corpus has increased over time.


### Multiple-Testing Correction

Because 30 topics are tested simultaneously, raw p-values are adjusted using the Benjamini–Hochberg False Discovery Rate (FDR) procedure.

The current statistical significance threshold is:

```bash
FDR-adjusted p-value < 0.05
```

This reduces the risk of treating random fluctuations across many topic-level tests as statistically meaningful discoveries.

### Recent Momentum

Long-term growth alone is not enough.

The system compares the latest complete year with the average topic share across the preceding three complete years.

This provides a separate signal for recent acceleration or deceleration.

### Research Volume

A topic also needs enough recent research activity to make its trend interpretable.

The current radar uses a minimum threshold of:

```bash
50 papers in the latest complete year
```

### Trend Fit

R² is used as a descriptive measure of how well the linear trend explains the observed topic-share trajectory.

It is treated as supporting evidence rather than as a standalone definition of an emerging topic.


### Semantic Coherence

Statistical growth can sometimes occur in broad or mixed clusters.

To validate the semantic structure of each topic, the system:

- calculates cluster centroids,
- measures paper-to-centroid similarity,
- selects representative papers,
- inspects their titles and cluster-level terms.

The current descriptive interpretation is:

| Mean Similarity | Interpretation  |
|---|---|
| **≥ 0.90** | High semantic coherence |
| **≥ 0.80** | Moderate semantic coherence |
| **< 0.80** | Broad or mixed cluster |

These thresholds are descriptive rather than formally validated topic-coherence thresholds.

---

## Emerging Research Candidates

Under the current Research Radar signal rules, four topics satisfy the emerging-candidate criteria:

| Topic | 2024 Papers | Trend Slope | FDR p-value | R² | Recent Momentum |
|---|---:|---:|---:|---:|---:|
| Graph Neural Networks | 505 | +0.474 pp/year | 0.000631 | 0.892 | +0.127 pp |
| Federated Learning | 181 | +0.211 pp/year | 0.000631 | 0.873 | +0.090 pp |
| Time-Series Forecasting | 294 | +0.148 pp/year | 0.001765 | 0.807 | +0.855 pp |
| General Machine Learning and AI | 1,541 | +0.285 pp/year | 0.022207 | 0.568 | +1.389 pp |

These should be interpreted as emerging candidates within this dataset and methodology, rather than universal predictions about the future of research.

The semantic validation also shows that the clusters are not equally focused:

- Federated Learning — high semantic coherence
- Time-Series Forecasting — high semantic coherence
- Graph Neural Networks — moderate semantic coherence
- General Machine Learning and AI — broad/mixed cluster

This distinction is important because a statistically strong trend does not automatically imply a narrowly defined research topic.

---

## A Useful Example: Popularity vs Evidence

The system intentionally separates recent momentum from statistically supported long-term growth.

For example, Large Language Models show strong recent momentum in the dataset, but their FDR-adjusted trend p-value is approximately **0.05246**, just above the current **0.05** threshold.

Therefore, the topic is not classified as an emerging candidate under the current statistical rules.

This illustrates the central idea behind Research Radar:

```bash
Recent attention and statistically supported sustained growth are not necessarily the same thing.
```
---

## Change-Point Analysis

The project also explored whether research-topic trajectories contained statistically supported structural breaks.

An initial exploratory change-point analysis was followed by a permutation-based validation procedure with multiple-testing correction.

The validated analysis did not identify statistically supported structural breaks among the 30 topics under the current thresholds.

Rather than weakening the statistical criteria simply to produce change points, the project retains this result as part of the analysis.

---

## Research Landscape

Research Radar includes a 2D visualization of the discovered topic space.

The visualization:

```text
100D SVD representations
        ↓
Topic centroids
        ↓
PCA
        ↓
2D Research Landscape
```

The first two principal components explain approximately:

```text
PC1: 10.69%
PC2:  7.63%
Combined: 18.31%
```
The landscape is therefore intended as a relative visualization of topic structure, not a complete representation of the underlying research space.

Emerging candidates are highlighted separately from the remaining topics.

---

## Dashboard

The project includes an interactive Streamlit dashboard with four main views:

### Research Overview

Provides the high-level Research Radar summary and current emerging candidates.

### Research Trend Explorer

Explore how research-topic prevalence changes over time.

### Topic Deep Dive

Inspect an individual topic's:

- research volume
- trend statistics
- semantic coherence
- top terms
- representative papers
- evidence signals
- historical trajectory

### Research Landscape

Explore the 2D PCA representation of the discovered research-topic space.

---

## Tech Stack

### Data & Analysis
- Python
- Pandas
- NumPy
- SciPy
  
### NLP
- scikit-learn
- TF-IDF
- text preprocessing
- semantic similarity
  
### Machine Learning
- Truncated SVD
- K-Means clustering
- PCA
  
### Statistics
- Linear regression
- p-values
- R²
- Benjamini–Hochberg FDR correction
- permutation-based validation
  
### Visualization
- Matplotlib
-Plotly
- Streamlit
  
### Development
- VS Code
- Git
- GitHub

---

## Project Structure

```text
RR-Research_Radar/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── external/
│   ├── processed/
│   └── raw/
│
├── notebooks/
│
├── sql/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── nlp/
│   ├── statistics/
│   └── visualization/
│
├── tests/
│
├── config.py
├── requirements.txt
├── .gitignore
└── README.md
```

Generated datasets and model artifacts are intentionally separated from the raw research corpus.

The raw dataset and large local artifacts are excluded from Git using **.gitignore**.

----

## Dataset

The project uses the following publicly available dataset:

arXiv Scientific Research Paper Dataset — Mendeley Data

### Dataset:
https://data.mendeley.com/datasets/mm6kst3krj/1

### DOI:
https://doi.org/10.17632/mm6kst3krj.1

The dataset contains structured information including:

- paper ID
- title
- abstract
- research category
- publication date
- update date
- authors
- first author

License: **CC BY 4.0**

The current processed corpus contains 112,363 papers after filtering the original dataset to the project's target research categories and publication period.

---

## Project Philosophy

Research Radar is built around a simple principle:

```bash
Don't confuse popularity with evidence.
```

Publication volume can tell us what is large.

Recent momentum can tell us what is changing.

Statistical testing can tell us whether a trend is distinguishable from random variation under the chosen model.

Semantic validation can tell us whether the discovered cluster actually represents a coherent research area.

Combining these signals produces a more transparent way to explore how scientific research evolves.

---

## Author

Aamina

`Data Science` . `NLP` . `Machine Learning` . `Statistical Analysis`
