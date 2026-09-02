import pandas as pd
import streamlit as st

from multi_agent_ops.graph import build_graph
from multi_agent_ops.state import OpsState
from multi_agent_ops.tools.data_tools import validate_support_data


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Multi-Agent Ops Crew",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Global ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(15, 65, 130, 0.20),
                transparent 35%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(20, 45, 100, 0.15),
                transparent 35%
            ),
            #030712;
        color: #f8fafc;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    /* ---------- Header ---------- */

    .app-header {
        padding: 10px 0 28px 0;
        border-bottom: 1px solid rgba(59, 130, 246, 0.20);
        margin-bottom: 35px;
    }

    .app-title {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 900;
        letter-spacing: 1px;
        margin: 0;
    }

    .app-subtitle {
        color: #7f93b8;
        font-size: 0.98rem;
        margin-top: 8px;
    }

    /* ---------- Section Headers ---------- */

    .section-number {
        color: #3b82f6;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .section-title {
        color: #f8fafc;
        font-size: 1.35rem;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .section-description {
        color: #7183a3;
        font-size: 0.90rem;
        margin-bottom: 18px;
    }

    /* ---------- Cards ---------- */

    .metric-card {
        background:
            linear-gradient(
                145deg,
                rgba(15, 32, 65, 0.95),
                rgba(5, 12, 28, 0.95)
            );
        border: 1px solid rgba(59, 130, 246, 0.22);
        border-radius: 14px;
        padding: 18px 18px;
        min-height: 110px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.22);
    }

    .metric-label {
        color: #7183a3;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 1.65rem;
        font-weight: 850;
        margin-top: 8px;
    }

    .status-card {
        background:
            linear-gradient(
                145deg,
                rgba(10, 29, 58, 0.95),
                rgba(4, 12, 25, 0.95)
            );
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 10px;
    }

    /* ---------- Review ---------- */

    .review-pass {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.30);
        border-radius: 12px;
        padding: 16px 20px;
        color: #d1fae5;
    }

    .review-warning {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.30);
        border-radius: 12px;
        padding: 16px 20px;
        color: #fef3c7;
    }

    /* ---------- Report ---------- */

    .report-container {
        background:
            linear-gradient(
                145deg,
                rgba(12, 25, 50, 0.96),
                rgba(4, 10, 22, 0.98)
            );
        border: 1px solid rgba(59, 130, 246, 0.20);
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.25);
    }

    .report-container h1,
    .report-container h2,
    .report-container h3 {
        color: #ffffff;
    }

    .report-container p,
    .report-container li {
        color: #c1cce0;
        line-height: 1.75;
    }

    /* ---------- Inputs ---------- */

    textarea,
    input {
        background-color: #070f20 !important;
        color: #f8fafc !important;
    }

    [data-testid="stFileUploader"] {
        background: #070f20;
        border: 1px dashed rgba(59, 130, 246, 0.45);
        border-radius: 14px;
        padding: 12px;
    }

    /* ---------- Buttons ---------- */

    .stButton > button {
        border-radius: 10px;
        font-weight: 750;
        min-height: 48px;
        border: 1px solid rgba(59, 130, 246, 0.45);
        background: linear-gradient(
            135deg,
            #2563eb,
            #1d4ed8
        );
        color: white;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #60a5fa;
        box-shadow: 0 0 25px rgba(37, 99, 235, 0.25);
        transform: translateY(-1px);
    }

    /* ---------- Dataframe ---------- */

    [data-testid="stDataFrame"] {
        border: 1px solid rgba(59, 130, 246, 0.18);
        border-radius: 12px;
        overflow: hidden;
    }

    /* ---------- Divider ---------- */

    hr {
        border-color: rgba(59, 130, 246, 0.15);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="app-header">
        <div class="app-title">MULTI-AGENT OPS CREW</div>
        <div class="app-subtitle">
            AI-powered operational intelligence for data-driven business decisions
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INPUT SECTION
# ============================================================

col_problem, col_data = st.columns(
    [1.15, 0.85],
    gap="large",
)


with col_problem:

    st.markdown(
        '<div class="section-number">01 — Investigation</div>',
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
        "Business Problem",
        placeholder=(
            "Example: Why has customer support resolution time "
            "increased over the last four months?"
        ),
        height=150,
        label_visibility="collapsed",
    )


with col_data:

    st.markdown(
        '<div class="section-number">02 — Data</div>',
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
            for the analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Company operational dataset",
        type=["csv"],
    )


# ============================================================
# LOAD CSV
# ============================================================

df = None

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        validate_support_data(df)

        st.success(
            f"Dataset loaded successfully — "
            f"{len(df):,} rows × {len(df.columns)} columns"
        )

        with st.expander("Preview Company Dataset"):

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


st.divider()


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze = st.button(
    "Run Multi-Agent Analysis",
    type="primary",
    use_container_width=True,
)


if analyze:

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if not problem.strip():

        st.error(
            "Please describe the business problem before running the analysis."
        )

        st.stop()

    if df is None:

        st.error(
            "Please upload a valid company CSV file."
        )

        st.stop()

    # --------------------------------------------------------
    # Initial State
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Run Graph
    # --------------------------------------------------------

    with st.status(
        "Running Multi-Agent Ops Crew...",
        expanded=True,
    ) as workflow_status:

        try:

            st.write("Initializing agent workflow...")

            graph = build_graph()

            st.write("Running research, data analysis and review agents...")

            result = graph.invoke(
                initial_state
            )

            workflow_status.update(
                label="Analysis completed successfully",
                state="complete",
                expanded=False,
            )

        except Exception as exc:

            workflow_status.update(
                label="Workflow failed",
                state="error",
                expanded=True,
            )

            st.error(
                f"Workflow failed: {exc}"
            )

            st.stop()

    # ========================================================
    # WORKFLOW STATUS
    # ========================================================

    st.markdown(
        '<div class="section-number">03 — Workflow</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Workflow Status</div>',
        unsafe_allow_html=True,
    )

    status_col, revision_col = st.columns(2)

    with status_col:

        status = result.get(
            "status",
            "UNKNOWN",
        )

        st.markdown(
            f"""
            <div class="status-card">
                <div class="metric-label">Workflow Status</div>
                <div class="metric-value">{status}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with revision_col:

        revision_count = result.get(
            "revision_count",
            0,
        )

        st.markdown(
            f"""
            <div class="status-card">
                <div class="metric-label">Revision Attempts</div>
                <div class="metric-value">{revision_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


    # ========================================================
    # DATA FINDINGS
    # ========================================================

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
            '<div class="section-number">04 — Analytics</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">Key Metrics</div>',
            unsafe_allow_html=True,
        )

        metric_items = list(
            metrics.items()
        )

        columns = st.columns(
            min(len(metric_items), 4)
        )

        for index, (name, value) in enumerate(
            metric_items
        ):

            column = columns[
                index % len(columns)
            ]

            with column:

                formatted_name = (
                    str(name)
                    .replace("_", " ")
                    .title()
                )

                if isinstance(value, float):

                    formatted_value = (
                        f"{value:,.2f}"
                    )

                elif isinstance(value, int):

                    formatted_value = (
                        f"{value:,}"
                    )

                else:

                    formatted_value = str(
                        value
                    )

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">
                            {formatted_name}
                        </div>

                        <div class="metric-value">
                            {formatted_value}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


    # ========================================================
    # DATA VISUALIZATION
    # ========================================================

    if not df.empty:

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_columns:

            st.divider()

            st.markdown(
                '<div class="section-number">05 — Data Visualization</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="section-title">Operational Trends</div>',
                unsafe_allow_html=True,
            )

            selected_column = st.selectbox(
                "Select metric to visualize",
                numeric_columns,
            )

            chart_df = df[
                [selected_column]
            ].copy()

            chart_df.index = range(
                1,
                len(chart_df) + 1
            )

            st.line_chart(
                chart_df,
                use_container_width=True,
            )


    # ========================================================
    # QUALITY REVIEW
    # ========================================================

    review = result.get(
        "review",
        {},
    )

    if review:

        st.divider()

        st.markdown(
            '<div class="section-number">06 — Quality Control</div>',
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

        if "PASS" in str(review_status).upper():

            st.markdown(
                f"""
                <div class="review-pass">
                    <strong>Review Status:</strong>
                    {review_status}
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                f"""
                <div class="review-warning">
                    <strong>Review Status:</strong>
                    {review_status}
                </div>
                """,
                unsafe_allow_html=True,
            )

        review_analysis = review.get(
            "analysis",
            "",
        )

        if review_analysis:

            with st.expander(
                "View Detailed Review"
            ):

                st.markdown(
                    review_analysis
                )


    # ========================================================
    # FINAL REPORT
    # ========================================================

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
        '<div class="section-number">07 — Executive Intelligence</div>',
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
            '</div>',
            unsafe_allow_html=True,
        )

    else:

        st.warning(
            "No final report was generated."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        margin-top:50px;
        padding-top:20px;
        border-top:1px solid rgba(59,130,246,0.12);
        color:#536783;
        font-size:0.75rem;
    ">
        Multi-Agent Ops Crew · Research · Data · Review · Intelligence
    </div>
    """,
    unsafe_allow_html=True,
)