from pathlib import Path

import pandas as pd
import streamlit as st


# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RADAR_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "trends"
    / "research_radar_results.csv"
)

PREVALENCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "trends"
    / "topic_prevalence_by_year.csv"
)

CLUSTER_TOPICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "topics"
    / "cluster_topics.csv"
)

REPRESENTATIVE_PAPERS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "topics"
    / "representative_papers_25.csv"
)

LANDSCAPE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "visualizations"
    / "research_landscape.csv"
)


# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="Research Radar",
    page_icon="🔭",
    layout="wide",
)


# ================================================================
# CUSTOM CSS
# ================================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0e0f14;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #f3f4f6;
    }

    p, li, label {
        color: #c9cbd3;
    }

    .metric-card {
        background-color: #171922;
        border: 1px solid #30333f;
        border-radius: 12px;
        padding: 1.1rem;
        text-align: center;
        margin-bottom: 1rem;
    }

    .metric-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #a8abb7;
        margin-top: 0.25rem;
    }

    .topic-card {
        background-color: #171922;
        border: 1px solid #30333f;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    .topic-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.7rem;
    }

    .info-box {
        padding: 1rem 1.2rem;
        border-radius: 10px;
        background-color: #171922;
        border: 1px solid #30333f;
        color: #f3f4f6;
        margin-bottom: 1.2rem;
    }

    .section-note {
        color: #9fa3b2;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ================================================================
# DATA LOADING
# ================================================================

@st.cache_data
def load_data():

    radar = pd.read_csv(
        RADAR_PATH
    )

    prevalence = pd.read_csv(
        PREVALENCE_PATH
    )

    cluster_topics = pd.read_csv(
        CLUSTER_TOPICS_PATH
    )

    representative_papers = pd.read_csv(
        REPRESENTATIVE_PAPERS_PATH
    )

    landscape = pd.read_csv(
        LANDSCAPE_PATH
    )

    return (
        radar,
        prevalence,
        cluster_topics,
        representative_papers,
        landscape,
    )


(
    radar,
    prevalence,
    cluster_topics,
    representative_papers,
    landscape,
) = load_data()


# ================================================================
# TOPIC NAME CLEANUP
# ================================================================

TOPIC_DISPLAY_FIXES = {
    "Object Detectionand Tracking":
        "Object Detection and Tracking",

    "Evolutionary and Multi-ObjectiveOptimization":
        "Evolutionary and Multi-Objective Optimization",
}


def clean_topic_name(topic):

    return TOPIC_DISPLAY_FIXES.get(
        topic,
        topic,
    )


# Apply display-only cleanup.
# The underlying CSV files remain unchanged.

radar["display_topic_label"] = (
    radar["topic_label"]
    .apply(clean_topic_name)
)

landscape["display_topic_label"] = (
    landscape["topic_label"]
    .apply(clean_topic_name)
)


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def find_column(
    df,
    candidates,
):

    for candidate in candidates:

        if candidate in df.columns:
            return candidate

    return None


def prepare_trend_data(
    prevalence_df,
    topic_label,
):

    topic_df = prevalence_df[
        prevalence_df["topic_label"]
        == topic_label
    ].copy()

    if topic_df.empty:
        return topic_df

    year_col = find_column(
        topic_df,
        [
            "year",
            "published_year",
        ],
    )

    share_col = find_column(
        topic_df,
        [
            "topic_share_percent",
            "share_percent",
            "topic_share",
        ],
    )

    if (
        year_col is None
        or share_col is None
    ):
        return pd.DataFrame()

    topic_df = topic_df[
        [
            year_col,
            share_col,
        ]
    ].copy()

    topic_df.columns = [
        "year",
        "share",
    ]

    topic_df["year"] = pd.to_numeric(
        topic_df["year"],
        errors="coerce",
    )

    topic_df["share"] = pd.to_numeric(
        topic_df["share"],
        errors="coerce",
    )

    topic_df = topic_df.dropna()

    # Exclude incomplete 2025 data.
    topic_df = topic_df[
        topic_df["year"] <= 2024
    ]

    return topic_df.sort_values(
        "year"
    )


def radar_value(
    row,
    candidates,
    default="—",
):

    for column in candidates:

        if column in row.index:

            value = row[column]

            if pd.notna(value):
                return value

    return default


def format_number(
    value,
):

    if pd.isna(value):
        return "—"

    return f"{value:,.0f}"


def format_signed(
    value,
    decimals=3,
):

    if pd.isna(value):
        return "—"

    return f"{value:+.{decimals}f}"


# ================================================================
# SIDEBAR
# ================================================================

st.sidebar.title(
    "🔭 Research Radar"
)

st.sidebar.markdown(
    "Explore emerging research areas through "
    "statistical and semantic evidence."
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Research Overview",
        "Research Trend Explorer",
        "Topic Deep Dive",
        "Research Landscape",
    ],
)


# ================================================================
# PAGE 1 — RESEARCH OVERVIEW
# ================================================================

if page == "Research Overview":

    st.title(
        "🔭 Research Radar"
    )

    st.markdown(
        """
        **An NLP and statistical ML system for discovering
        emerging trends in scientific research.**
        """
    )

    st.markdown(
        """
        <div class="info-box">
        Research Radar analyzes scientific abstracts to discover
        research topics, measure how their prevalence changes over
        time, and identify areas showing multiple independent
        signals of emergence.
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_topics = len(radar)

    emerging = int(
        radar[
            "emerging_candidate"
        ]
        .fillna(False)
        .sum()
    )

    monitoring = (
        total_topics - emerging
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">
                    {total_topics}
                </div>
                <div class="metric-label">
                    Discovered Topics
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">
                    {emerging}
                </div>
                <div class="metric-label">
                    Emerging Candidates
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">
                    {monitoring}
                </div>
                <div class="metric-label">
                    Monitoring Signals
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.header(
        "Emerging Research Candidates"
    )

    candidates = radar[
        radar["emerging_candidate"]
        == True
    ].copy()

    for _, row in candidates.iterrows():

        topic = row[
            "display_topic_label"
        ]

        st.markdown(
            f"""
            <div class="topic-card">

            <div class="topic-title">
                {topic}
            </div>

            <b>2024 papers:</b>
            {int(row["latest_complete_year_papers"]):,}

            &nbsp;&nbsp;|&nbsp;&nbsp;

            <b>Trend:</b>
            {row["trend_slope"]:+.3f} pp/year

            &nbsp;&nbsp;|&nbsp;&nbsp;

            <b>Recent momentum:</b>
            {row["recent_share_change_percentage_points"]:+.3f} pp

            <br><br>

            <b>FDR-adjusted p-value:</b>
            {row["trend_p_value_fdr"]:.5f}

            &nbsp;&nbsp;|&nbsp;&nbsp;

            <b>Trend R²:</b>
            {row["trend_r_squared"]:.3f}

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.header(
        "How to read the Radar"
    )

    st.markdown(
        """
        A topic becomes an **Emerging Candidate** when multiple
        signals align:

        - Statistically supported long-term growth
        - Positive recent momentum
        - Meaningful research volume
        - Reasonable trend fit
        - Semantic coherence evidence

        These signals describe patterns in this dataset. They do
        not represent a ranking of research fields or a prediction
        of future importance.
        """
    )


# ================================================================
# PAGE 2 — RESEARCH TREND EXPLORER
# ================================================================

elif page == "Research Trend Explorer":

    st.title(
        "📈 Research Trend Explorer"
    )

    st.markdown(
        """
        Explore how the prevalence of each discovered research
        topic changes across complete publication years.
        """
    )

    topic_options = sorted(
        radar[
            "topic_label"
        ]
        .dropna()
        .unique()
    )

    selected_topic = st.selectbox(
        "Select a research topic",
        topic_options,
        format_func=clean_topic_name,
    )

    trend_df = prepare_trend_data(
        prevalence,
        selected_topic,
    )

    if trend_df.empty:

        st.warning(
            "Trend data is unavailable for this topic."
        )

    else:

        st.line_chart(
            trend_df.set_index(
                "year"
            )[
                ["share"]
            ],
            height=450,
        )

        st.caption(
            "Topic share represents the percentage of papers "
            "assigned to this topic in each complete year. "
            "2025 is excluded because the dataset only covers "
            "January 2025."
        )

        selected_row = radar[
            radar["topic_label"]
            == selected_topic
        ]

        if not selected_row.empty:

            row = selected_row.iloc[0]

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Long-term slope",
                    f'{row["trend_slope"]:+.3f} pp/year',
                )

            with col2:

                st.metric(
                    "FDR p-value",
                    f'{row["trend_p_value_fdr"]:.5f}',
                )

            with col3:

                st.metric(
                    "Trend R²",
                    f'{row["trend_r_squared"]:.3f}',
                )


# ================================================================
# PAGE 3 — TOPIC DEEP DIVE
# ================================================================

elif page == "Topic Deep Dive":

    st.title(
        "🔬 Topic Deep Dive"
    )

    topic_options = sorted(
        radar[
            "topic_label"
        ]
        .dropna()
        .unique()
    )

    selected_topic = st.selectbox(
        "Select a research topic",
        topic_options,
        format_func=clean_topic_name,
    )

    radar_match = radar[
        radar["topic_label"]
        == selected_topic
    ]

    if radar_match.empty:

        st.warning(
            "No Radar data found for this topic."
        )

    else:

        row = radar_match.iloc[0]

        display_topic = clean_topic_name(
            selected_topic
        )

        st.header(
            display_topic
        )

        # --------------------------------------------------------
        # Profile
        # --------------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "2024 papers",
                f'{int(row["latest_complete_year_papers"]):,}',
            )

        with col2:

            st.metric(
                "Trend slope",
                f'{row["trend_slope"]:+.3f} pp/year',
            )

        with col3:

            st.metric(
                "Recent momentum",
                f'{row["recent_share_change_percentage_points"]:+.3f} pp',
            )

        with col4:

            status = (
                "Emerging candidate"
                if row["emerging_candidate"]
                else "Monitoring signal"
            )

            st.metric(
                "Radar status",
                status,
            )

        # --------------------------------------------------------
        # Cluster structure
        # --------------------------------------------------------

        st.header(
            "Topic Structure"
        )

        cluster_match = cluster_topics[
            cluster_topics["cluster"]
            == row["cluster"]
        ]

        if not cluster_match.empty:

            cluster_row = (
                cluster_match.iloc[0]
            )

            st.markdown(
                f"""
                <div class="info-box">

                <b>Cluster size:</b>
                {int(cluster_row["paper_count"]):,} papers

                <br><br>

                <b>Top semantic terms:</b>

                <br>

                {cluster_row["top_terms"]}

                </div>
                """,
                unsafe_allow_html=True,
            )

        # --------------------------------------------------------
        # Semantic validation
        # --------------------------------------------------------

        st.header(
            "Semantic Validation"
        )

        coherence_class = radar_value(
            row,
            [
                "coherence_class",
            ],
        )

        similarity = radar_value(
            row,
            [
                "mean_similarity",
                "semantic_similarity",
            ],
        )

        st.markdown(
            f"""
            <div class="info-box">

            <b>Coherence:</b>
            {coherence_class}

            <br><br>

            <b>Mean representative-paper similarity:</b>
            {similarity}

            <br><br>

            This measures how closely representative papers
            sit around the topic's semantic centroid in the
            reduced embedding space.

            </div>
            """,
            unsafe_allow_html=True,
        )

        # --------------------------------------------------------
        # Trend
        # --------------------------------------------------------

        st.header(
            "Topic Trend"
        )

        trend_df = prepare_trend_data(
            prevalence,
            selected_topic,
        )

        if not trend_df.empty:

            st.line_chart(
                trend_df.set_index(
                    "year"
                )[
                    ["share"]
                ],
                height=400,
            )

        # --------------------------------------------------------
        # Evidence
        # --------------------------------------------------------

        st.header(
            "Evidence Breakdown"
        )

        evidence_columns = [
            (
                "Long-term trend",
                "trend_slope",
                "pp/year",
            ),
            (
                "Recent momentum",
                "recent_share_change_percentage_points",
                "pp",
            ),
            (
                "Trend fit",
                "trend_r_squared",
                "",
            ),
            (
                "FDR-adjusted significance",
                "trend_p_value_fdr",
                "",
            ),
        ]

        for (
            label,
            column,
            suffix,
        ) in evidence_columns:

            if column in row.index:

                value = row[column]

                if pd.notna(value):

                    if column in [
                        "trend_slope",
                        "recent_share_change_percentage_points",
                    ]:

                        display_value = (
                            f"{value:+.3f} {suffix}"
                        )

                    else:

                        display_value = (
                            f"{value:.5f}"
                            if "p_value" in column
                            else f"{value:.3f}"
                        )

                    st.write(
                        f"**{label}:** "
                        f"{display_value}"
                    )

        # --------------------------------------------------------
        # Representative papers
        # --------------------------------------------------------

        st.header(
            "Representative Papers"
        )

        papers = representative_papers[
            representative_papers["topic_label"]
            == selected_topic
        ].copy()

        if not papers.empty:

            papers = papers.sort_values(
                "rank"
            ).head(10)

            for _, paper in papers.iterrows():

                st.markdown(
                    f"""
                    **{int(paper["rank"])}.**
                    {paper["title"]}
                    """
                )

        else:

            st.info(
                "No representative papers available."
            )

        # --------------------------------------------------------
        # Methodology
        # --------------------------------------------------------

        st.header(
            "Methodology"
        )

        st.markdown(
            """
            Topics were discovered using TF-IDF representations,
            SVD dimensionality reduction, and KMeans clustering.

            Topic prevalence was then measured across publication
            years. Long-term trends were evaluated using linear
            regression with Benjamini-Hochberg FDR correction.

            Semantic validation uses representative papers selected
            by similarity to each topic centroid in the SVD space.
            """
        )


# ================================================================
# PAGE 4 — RESEARCH LANDSCAPE
# ================================================================

elif page == "Research Landscape":

    st.title(
        "🌐 Research Landscape"
    )

    st.markdown(
        """
        A 2D semantic map of the **30 research topics discovered
        from the scientific-paper corpus**.
        """
    )

    st.markdown(
        """
        <div class="info-box">

        Each point represents a discovered research topic.

        <br><br>

        <b>Position</b> represents the topic's location after
        projecting its semantic centroid into two dimensions.

        <br><br>

        <b>Size</b> represents the number of papers assigned
        to that topic.

        <br><br>

        <b>Labels</b> identify the four Research Radar emerging
        candidates. Hover over any topic for additional evidence.

        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # Interactive semantic map
    # ------------------------------------------------------------

    try:

        import plotly.graph_objects as go

        plot_df = landscape.copy()

        # --------------------------------------------------------
        # Prepare display fields
        # --------------------------------------------------------

        plot_df["display_topic_label"] = (
            plot_df["topic_label"]
            .apply(clean_topic_name)
        )

        plot_df["emerging_candidate"] = (
            plot_df[
                "emerging_candidate"
            ]
            .fillna(False)
            .astype(bool)
        )

        plot_df["status"] = (
            plot_df[
                "emerging_candidate"
            ]
            .apply(
                lambda value:
                    "Emerging candidate"
                    if value
                    else "Monitoring signal"
            )
        )

        # --------------------------------------------------------
        # Find useful Radar columns
        # --------------------------------------------------------

        def get_numeric_column(
            row,
            candidates,
        ):

            for column in candidates:

                if column in row.index:

                    value = row[column]

                    if pd.notna(value):

                        return value

            return None

        plot_df["trend_display"] = plot_df.apply(
            lambda row:
                get_numeric_column(
                    row,
                    [
                        "trend_slope",
                    ],
                ),
            axis=1,
        )

        plot_df["momentum_display"] = plot_df.apply(
            lambda row:
                get_numeric_column(
                    row,
                    [
                        "recent_share_change_percentage_points",
                    ],
                ),
            axis=1,
        )

        plot_df["r2_display"] = plot_df.apply(
            lambda row:
                get_numeric_column(
                    row,
                    [
                        "trend_r_squared",
                    ],
                ),
            axis=1,
        )

        plot_df["fdr_display"] = plot_df.apply(
            lambda row:
                get_numeric_column(
                    row,
                    [
                        "trend_p_value_fdr",
                    ],
                ),
            axis=1,
        )

        plot_df["similarity_display"] = plot_df.apply(
            lambda row:
                get_numeric_column(
                    row,
                    [
                        "mean_similarity",
                        "semantic_similarity",
                    ],
                ),
            axis=1,
        )

        # --------------------------------------------------------
        # Split emerging / monitoring topics
        # --------------------------------------------------------

        emerging_df = plot_df[
            plot_df[
                "emerging_candidate"
            ]
        ].copy()

        monitoring_df = plot_df[
            ~plot_df[
                "emerging_candidate"
            ]
        ].copy()

        # --------------------------------------------------------
        # Create figure
        # --------------------------------------------------------

        fig = go.Figure()

        # --------------------------------------------------------
        # Monitoring signals
        # --------------------------------------------------------

        monitoring_customdata = (
            monitoring_df[
                [
                    "display_topic_label",
                    "paper_count",
                    "trend_display",
                    "momentum_display",
                    "r2_display",
                    "fdr_display",
                    "coherence_class",
                    "similarity_display",
                ]
            ]
            .fillna("—")
            .values
        )

        fig.add_trace(
            go.Scatter(
                x=monitoring_df["x"],
                y=monitoring_df["y"],
                mode="markers",
                name="Monitoring signal",
                marker={
                    "size": monitoring_df[
                        "paper_count"
                    ],
                    "sizemode": "area",
                    "sizeref": (
                        2.0
                        * monitoring_df[
                            "paper_count"
                        ].max()
                        / (42 ** 2)
                    ),
                    "sizemin": 7,
                    "opacity": 0.72,
                },
                customdata=monitoring_customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b>"
                    "<br><br>"
                    "Research volume: "
                    "%{customdata[1]:,} papers"
                    "<br>"
                    "Radar status: Monitoring signal"
                    "<br>"
                    "Long-term trend: "
                    "%{customdata[2]:+.3f} pp/year"
                    "<br>"
                    "Recent momentum: "
                    "%{customdata[3]:+.3f} pp"
                    "<br>"
                    "Trend R²: %{customdata[4]:.3f}"
                    "<br>"
                    "FDR-adjusted p-value: "
                    "%{customdata[5]:.5f}"
                    "<br>"
                    "Semantic coherence: "
                    "%{customdata[6]}"
                    "<br>"
                    "Mean similarity: "
                    "%{customdata[7]:.4f}"
                    "<extra></extra>"
                ),
            )
        )

        # --------------------------------------------------------
        # Emerging candidates
        # --------------------------------------------------------

        emerging_customdata = (
            emerging_df[
                [
                    "display_topic_label",
                    "paper_count",
                    "trend_display",
                    "momentum_display",
                    "r2_display",
                    "fdr_display",
                    "coherence_class",
                    "similarity_display",
                ]
            ]
            .fillna("—")
            .values
        )

        fig.add_trace(
            go.Scatter(
                x=emerging_df["x"],
                y=emerging_df["y"],
                mode="markers+text",
                name="Emerging candidate",
                text=emerging_df[
                    "display_topic_label"
                ],
                textposition="top center",
                textfont={
                    "size": 12,
                },
                marker={
                    "size": emerging_df[
                        "paper_count"
                    ],
                    "sizemode": "area",
                    "sizeref": (
                        2.0
                        * monitoring_df[
                            "paper_count"
                        ].max()
                        / (48 ** 2)
                    ),
                    "sizemin": 10,
                    "opacity": 0.95,
                    "line": {
                        "width": 2,
                    },
                },
                customdata=emerging_customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b>"
                    "<br><br>"
                    "Research volume: "
                    "%{customdata[1]:,} papers"
                    "<br>"
                    "Radar status: Emerging candidate"
                    "<br>"
                    "Long-term trend: "
                    "%{customdata[2]:+.3f} pp/year"
                    "<br>"
                    "Recent momentum: "
                    "%{customdata[3]:+.3f} pp"
                    "<br>"
                    "Trend R²: %{customdata[4]:.3f}"
                    "<br>"
                    "FDR-adjusted p-value: "
                    "%{customdata[5]:.5f}"
                    "<br>"
                    "Semantic coherence: "
                    "%{customdata[6]}"
                    "<br>"
                    "Mean similarity: "
                    "%{customdata[7]:.4f}"
                    "<extra></extra>"
                ),
            )
        )

        # --------------------------------------------------------
        # Layout
        # --------------------------------------------------------

        fig.update_layout(
            height=720,
            paper_bgcolor="#0e0f14",
            plot_bgcolor="#0e0f14",
            font={
                "color": "#d8dae2",
            },
            legend_title_text="",
            hoverlabel={
                "font_size": 13,
            },
            margin={
                "l": 40,
                "r": 40,
                "t": 40,
                "b": 50,
            },
            xaxis={
                "title": "Semantic dimension 1",
                "gridcolor": "#282b35",
                "zerolinecolor": "#363944",
            },
            yaxis={
                "title": "Semantic dimension 2",
                "gridcolor": "#282b35",
                "zerolinecolor": "#363944",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    except ImportError:

        st.error(
            "Plotly is required for the Research Landscape."
        )

        st.code(
            "python -m pip install plotly"
        )

    # ------------------------------------------------------------
    # Emerging candidate summary
    # ------------------------------------------------------------

    st.subheader(
        "Emerging candidates on the map"
    )

    emerging_topics = landscape[
        landscape["emerging_candidate"]
        == True
    ].copy()

    emerging_topics[
        "display_topic_label"
    ] = emerging_topics[
        "topic_label"
    ].apply(
        clean_topic_name
    )

    emerging_topics = emerging_topics.sort_values(
        "display_topic_label"
    )

    for _, row in emerging_topics.iterrows():

        st.markdown(
            f"""
            **{row["display_topic_label"]}**
            — {int(row["paper_count"]):,} papers
            """
        )

    # ------------------------------------------------------------
    # Map methodology
    # ------------------------------------------------------------

    st.subheader(
        "How the map was constructed"
    )

    combined_variance = (
        landscape[
            "pca_combined_variance"
        ].iloc[0]
    )

    st.markdown(
        f"""
        The original paper representations were reduced to
        100 dimensions using SVD. Papers belonging to each of
        the 30 discovered clusters were then averaged to create
        one semantic centroid per topic.

        PCA was applied to those 30 topic centroids to create
        the two coordinates shown above.

        The two displayed dimensions explain approximately
        **{combined_variance:.1%}** of the variance in the
        topic-centroid representation.

        Therefore, spatial proximity should be interpreted as
        a visual indication of semantic similarity rather than
        as a complete representation of the underlying
        high-dimensional research space.
        """
    )

    st.caption(
        "The landscape is a semantic visualization, not a ranking "
        "of research topics."
    )