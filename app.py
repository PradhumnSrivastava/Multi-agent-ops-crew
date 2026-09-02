import pandas as pd
import streamlit as st

from multi_agent_ops.graph import build_graph
from multi_agent_ops.state import OpsState
from multi_agent_ops.tools.data_tools import validate_support_data


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Multi-Agent Ops Crew",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(37, 99, 235, 0.12),
                transparent 32%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(14, 165, 233, 0.08),
                transparent 28%
            ),
            #050914;
        color: #e5e7eb;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* ---------- HIDE STREAMLIT UI ---------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    /* ---------- HERO ---------- */

    .hero {
        position: relative;
        overflow: hidden;
        padding: 3.5rem 3.2rem;
        margin-bottom: 2.8rem;
        border-radius: 24px;
        border: 1px solid rgba(59, 130, 246, 0.35);

        background:
            linear-gradient(
                135deg,
                rgba(15, 23, 42, 0.98),
                rgba(8, 18, 40, 0.96)
            );

        box-shadow:
            0 30px 80px rgba(0, 0, 0, 0.35),
            inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }

    .hero::before {
        content: "";
        position: absolute;
        width: 420px;
        height: 420px;
        right: -180px;
        top: -230px;
        border-radius: 50%;
        background: rgba(37, 99, 235, 0.15);
        filter: blur(10px);
    }

    .hero-badge {
        display: inline-block;
        padding: 0.45rem 0.9rem;
        margin-bottom: 1.1rem;

        border-radius: 999px;
        border: 1px solid rgba(59, 130, 246, 0.55);

        background: rgba(30, 64, 175, 0.18);
        color: #60a5fa;

        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .hero-title {
        margin: 0;
        color: #f8fafc;
        font-size: 3.4rem;
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -0.04em;
    }

    .hero-description {
        max-width: 780px;
        margin-top: 1.2rem;

        color: #94a3b8;
        font-size: 1.08rem;
        line-height: 1.75;
    }

    /* ---------- SECTION LABEL ---------- */

    .section-label {
        margin-top: 1.2rem;
        margin-bottom: 0.45rem;

        color: #60a5fa;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.15em;
        text-transform: uppercase;
    }

    .section-title {
        margin-bottom: 0.5rem;

        color: #f8fafc;
        font-size: 1.65rem;
        font-weight: 750;
    }

    .section-description {
        margin-bottom: 1.2rem;

        color: #64748b;
        line-height: 1.65;
    }

    /* ---------- INPUTS ---------- */

    .stTextArea textarea,
    .stTextInput input {
        background: #0b1220 !important;
        color: #e5e7eb !important;

        border: 1px solid #1e293b !important;
        border-radius: 12px !important;
    }

    .stTextArea textarea:focus,
    .stTextInput input:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 1px #2563eb !important;
    }

    [data-testid="stFileUploader"] {
        background: #0b1220;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 0.8rem;
    }

    /* ---------- BUTTON ---------- */

    .stButton > button {
        width: 100%;

        min-height: 3.2rem;

        border-radius: 12px;
        border: 1px solid #2563eb;

        background:
            linear-gradient(
                135deg,
                #2563eb,
                #1d4ed8
            );

        color: white;
        font-size: 0.95rem;
        font-weight: 700;

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);

        box-shadow:
            0 12px 30px rgba(37, 99, 235, 0.25);
    }

    /* ---------- METRICS ---------- */

    [data-testid="stMetric"] {
        padding: 1.1rem;

        background:
            linear-gradient(
                145deg,
                rgba(15, 23, 42, 0.95),
                rgba(8, 15, 28, 0.95)
            );

        border: 1px solid #1e293b;
        border-radius: 15px;

        min-height: 110px;
    }

    [data-testid="stMetricLabel"] {
        color: #64748b !important;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }

    /* ---------- CARDS ---------- */

    .info-card {
        padding: 1.35rem;

        border-radius: 16px;
        border: 1px solid #1e293b;

        background:
            linear-gradient(
                145deg,
                rgba(15, 23, 42, 0.96),
                rgba(8, 15, 28, 0.96)
            );
    }

    .card-title {
        margin-bottom: 0.45rem;

        color: #f8fafc;
        font-weight: 700;
        font-size: 1rem;
    }

    .card-text {
        color: #94a3b8;
        line-height: 1.65;
    }

    /* ---------- REPORT ---------- */

    .report-container {
        padding: 2rem;

        border-radius: 18px;
        border: 1px solid #1e293b;

        background:
            linear-gradient(
                145deg,
                rgba(15, 23, 42, 0.96),
                rgba(7, 13, 25, 0.98)
            );

        box-shadow:
            0 20px 60px rgba(0, 0, 0, 0.25);
    }

    .report-container h1,
    .report-container h2,
    .report-container h3 {
        color: #f8fafc;
    }

    .report-container p,
    .report-container li {
        color: #cbd5e1;
        line-height: 1.75;
    }

    .report-container strong {
        color: #f8fafc;
    }

    /* ---------- STATUS ---------- */

    .status-card {
        padding: 1.3rem 1.5rem;

        border-radius: 16px;
        border: 1px solid rgba(34, 197, 94, 0.25);

        background:
            linear-gradient(
                135deg,
                rgba(22, 101, 52, 0.12),
                rgba(15, 23, 42, 0.95)
            );
    }

    .status-label {
        color: #64748b;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .status-value {
        margin-top: 0.35rem;

        color: #86efac;
        font-size: 1.2rem;
        font-weight: 750;
    }

    /* ---------- DIVIDER ---------- */

    hr {
        border-color: #172033 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# HERO
# ---------------------------------------------------------




# ---------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------

col_problem, col_data = st.columns(
    [1.15, 1],
    gap="large",
)


with col_problem:

    st.markdown(
        '<div class="section-label">01 — Investigation</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Define Your Business Problem</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            Describe the operational problem, performance issue,
            or business question you want the agent crew to investigate.
        </div>
        """,
        unsafe_allow_html=True,
    )

    problem = st.text_area(
        "Business problem",
        placeholder=(
            "Example: Why has customer support resolution time "
            "increased over the last four months?"
        ),
        height=170,
        label_visibility="collapsed",
    )


with col_data:

    st.markdown(
        '<div class="section-label">02 — Data</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Upload Company Data</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            Upload a CSV containing the operational data required
            for the analysis. The system will validate and pass
            the data into the multi-agent workflow.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Company operational dataset",
        type=["csv"],
        label_visibility="collapsed",
    )


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

df = None

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        validate_support_data(df)

        st.success(
            f"Dataset loaded successfully — "
            f"{len(df):,} rows × {len(df.columns):,} columns"
        )

        with st.expander("Preview Company Data"):

            st.dataframe(
                df,
                use_container_width=True,
                height=280,
            )

    except Exception as exc:

        st.error(
            f"Invalid CSV file: {exc}"
        )

        df = None


# ---------------------------------------------------------
# ANALYZE BUTTON
# ---------------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

analyze = st.button(
    "Run Multi-Agent Analysis",
    type="primary",
)


if not analyze:
    st.stop()


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

if not problem.strip():

    st.error(
        "Please describe the business problem before starting the analysis."
    )

    st.stop()


if df is None:

    st.error(
        "Please upload a valid company CSV file before starting the analysis."
    )

    st.stop()


# ---------------------------------------------------------
# INITIAL STATE
# ---------------------------------------------------------

company_data = df.to_dict(
    orient="records"
)


initial_state: OpsState = {
    "problem": problem.strip(),
    "company_data": company_data,
    "plan": {},
    "research_findings": {},
    "data_findings": {},
    "business_analysis": {},
    "review": {},
    "revision": {},
    "revision_count": 0,
    "status": "STARTED",
    "final_report": {},
}


# ---------------------------------------------------------
# RUN WORKFLOW
# ---------------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-label">03 — Agent Workflow</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">Multi-Agent Analysis</div>',
    unsafe_allow_html=True,
)

with st.spinner(
    "Researching, analyzing data, reviewing findings, and generating the executive report..."
):

    try:

        graph = build_graph()

        result = graph.invoke(
            initial_state
        )

    except Exception as exc:

        st.error(
            f"Workflow failed: {exc}"
        )

        st.stop()


# ---------------------------------------------------------
# WORKFLOW STATUS
# ---------------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-label">04 — Workflow</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">Workflow Status</div>',
    unsafe_allow_html=True,
)

status = result.get(
    "status",
    "UNKNOWN",
)

revision_count = result.get(
    "revision_count",
    0,
)

status_col, revision_col = st.columns(
    2,
)

with status_col:

    st.markdown(
        f"""
        <div class="status-card">

            <div class="status-label">
                Workflow Status
            </div>

            <div class="status-value">
                {status}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with revision_col:

    st.metric(
        "Revision Attempts",
        revision_count,
    )


# ---------------------------------------------------------
# DATA FINDINGS
# ---------------------------------------------------------

data_findings = result.get(
    "data_findings",
    {},
)

metrics = data_findings.get(
    "metrics",
    {},
)


if metrics:

    st.divider()

    st.markdown(
        '<div class="section-label">05 — Analytics</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Key Performance Metrics</div>',
        unsafe_allow_html=True,
    )

    metric_items = list(
        metrics.items()
    )

    # Streamlit supports a maximum practical number of
    # columns in a single row, so split into rows of four.

    for start in range(
        0,
        len(metric_items),
        4,
    ):

        row = metric_items[
            start:start + 4
        ]

        columns = st.columns(
            len(row)
        )

        for column, (name, value) in zip(
            columns,
            row,
        ):

            with column:

                display_name = (
                    name
                    .replace("_", " ")
                    .title()
                )

                st.metric(
                    display_name,
                    value,
                )


# ---------------------------------------------------------
# DATA VISUALIZATION
# ---------------------------------------------------------

if len(df) > 1:

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_columns:

        st.markdown(
            '<div class="section-title">Data Overview</div>',
            unsafe_allow_html=True,
        )

        selected_columns = st.multiselect(
            "Select numeric columns to visualize",
            numeric_columns,
            default=numeric_columns[:4],
        )

        if selected_columns:

            chart_df = df[
                selected_columns
            ]

            st.line_chart(
                chart_df,
                use_container_width=True,
            )


# ---------------------------------------------------------
# QUALITY REVIEW
# ---------------------------------------------------------

review = result.get(
    "review",
    {},
)


if review:

    st.divider()

    st.markdown(
        '<div class="section-label">06 — Quality Control</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Quality Review</div>',
        unsafe_allow_html=True,
    )

    review_status = review.get(
        "review_status",
        "UNKNOWN",
    )

    st.metric(
        "Review Status",
        review_status,
    )

    review_analysis = review.get(
        "analysis",
        "",
    )

    if review_analysis:

        with st.expander(
            "View Review Details"
        ):

            st.markdown(
                review_analysis
            )


# ---------------------------------------------------------
# FINAL EXECUTIVE REPORT
# ---------------------------------------------------------

final_report = result.get(
    "final_report",
    {},
)

report = final_report.get(
    "report",
    "",
)


st.divider()

st.markdown(
    '<div class="section-label">07 — Executive Output</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">Final Executive Report</div>',
    unsafe_allow_html=True,
)


if report:

    st.markdown(
        '<div class="report-container">',
        unsafe_allow_html=True,
    )

    st.markdown(
        report
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

else:

    st.warning(
        "No final executive report was generated."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown(
    """
    <div style="
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #172033;
        text-align: center;
        color: #475569;
        font-size: 0.78rem;
    ">
        Multi-Agent Ops Crew · Research · Data Analysis · Review · Revision
    </div>
    """,
    unsafe_allow_html=True,
)