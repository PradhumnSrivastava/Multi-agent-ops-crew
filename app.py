import re

import pandas as pd
import plotly.express as px
import streamlit as st

from multi_agent_ops.graph import build_graph
from multi_agent_ops.state import OpsState
from multi_agent_ops.tools.data_tools import validate_support_data


st.set_page_config(
    page_title="Multi-Agent Ops Crew",
    page_icon="◈",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

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
                rgba(30, 64, 175, 0.10),
                transparent 30%
            ),
            #05070b;
        color: #f8fafc;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* ---------- REMOVE DEFAULT STREAMLIT HEADER SPACE ---------- */

    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* ---------- TEXT ---------- */

    h1,
    h2,
    h3,
    h4 {
        color: #f8fafc !important;
    }

    p,
    label,
    .stMarkdown {
        color: #cbd5e1;
    }

    /* ---------- HERO ---------- */

    .hero {
        position: relative;
        padding: 2.2rem 2.4rem;
        margin-bottom: 1.8rem;
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 22px;
        background:
            linear-gradient(
                135deg,
                rgba(15, 23, 42, 0.98),
                rgba(3, 7, 18, 0.98)
            );
        box-shadow:
            0 20px 60px rgba(0, 0, 0, 0.35),
            inset 0 1px 0 rgba(255, 255, 255, 0.03);
        overflow: hidden;
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -100px;
        top: -120px;
        border-radius: 50%;
        background: rgba(37, 99, 235, 0.16);
        filter: blur(30px);
    }

    .hero-brand {
        color: #ffffff;
        font-size: 1.05rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        color: #ffffff;
        font-size: 2.7rem;
        line-height: 1.05;
        font-weight: 850;
        letter-spacing: -0.04em;
        margin-bottom: 0.8rem;
    }

    .hero-description {
        max-width: 850px;
        color: #94a3b8;
        font-size: 1.02rem;
        line-height: 1.7;
    }

    /* ---------- SECTION LABEL ---------- */

    .section-label {
        color: #60a5fa;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .section-title {
        color: #f8fafc;
        font-size: 1.45rem;
        font-weight: 750;
        margin-bottom: 0.8rem;
    }

    .section-description {
        color: #94a3b8;
        margin-bottom: 1rem;
    }

    /* ---------- INPUTS ---------- */

    textarea,
    input {
        background-color: #0b1120 !important;
        color: #f8fafc !important;
        border: 1px solid #1e293b !important;
    }

    textarea:focus,
    input:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 1px #2563eb !important;
    }

    [data-testid="stFileUploader"] {
        background: #080d17;
        border: 1px dashed #334155;
        border-radius: 16px;
        padding: 0.6rem;
    }

    /* ---------- BUTTON ---------- */

    .stButton > button {
        width: 100%;
        min-height: 48px;
        border-radius: 12px;
        border: 1px solid #2563eb;
        background: linear-gradient(
            135deg,
            #2563eb,
            #1d4ed8
        );
        color: white;
        font-weight: 750;
        font-size: 0.95rem;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #60a5fa;
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.25);
        transform: translateY(-1px);
    }

    /* ---------- CARDS ---------- */

    .info-card {
        background: linear-gradient(
            145deg,
            rgba(15, 23, 42, 0.95),
            rgba(8, 12, 20, 0.98)
        );
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    .card-label {
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 750;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .card-value {
        color: #f8fafc;
        font-size: 1.4rem;
        font-weight: 800;
        margin-top: 0.35rem;
    }

    /* ---------- METRICS ---------- */

    .metric-card {
        background: linear-gradient(
            145deg,
            #0d1526,
            #080c14
        );
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 1rem;
        min-height: 105px;
        box-shadow:
            0 10px 25px rgba(0, 0, 0, 0.18);
    }

    .metric-label {
        color: #7dd3fc;
        font-size: 0.72rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-value {
        color: #ffffff;
        font-size: 1.65rem;
        font-weight: 850;
        margin-top: 0.35rem;
    }

    /* ---------- REVIEW ---------- */

    .review-pass {
        display: inline-block;
        padding: 0.45rem 0.75rem;
        border-radius: 999px;
        background: rgba(34, 197, 94, 0.10);
        border: 1px solid rgba(34, 197, 94, 0.35);
        color: #86efac;
        font-size: 0.78rem;
        font-weight: 800;
    }

    .review-warning {
        display: inline-block;
        padding: 0.45rem 0.75rem;
        border-radius: 999px;
        background: rgba(245, 158, 11, 0.10);
        border: 1px solid rgba(245, 158, 11, 0.35);
        color: #fcd34d;
        font-size: 0.78rem;
        font-weight: 800;
    }

    .review-failed {
        display: inline-block;
        padding: 0.45rem 0.75rem;
        border-radius: 999px;
        background: rgba(239, 68, 68, 0.10);
        border: 1px solid rgba(239, 68, 68, 0.35);
        color: #fca5a5;
        font-size: 0.78rem;
        font-weight: 800;
    }

    /* ---------- REPORT ---------- */

    .report-container {
        background:
            linear-gradient(
                145deg,
                rgba(15, 23, 42, 0.96),
                rgba(5, 9, 17, 0.98)
            );
        border: 1px solid #1e293b;
        border-radius: 20px;
        padding: 2rem 2.2rem;
        margin-top: 0.8rem;
        box-shadow:
            0 20px 50px rgba(0, 0, 0, 0.22);
    }

    .report-container h1,
    .report-container h2,
    .report-container h3,
    .report-container h4 {
        color: #f8fafc !important;
        margin-top: 1.6rem;
    }

    .report-container p {
        color: #cbd5e1 !important;
        line-height: 1.75;
    }

    .report-container li {
        color: #cbd5e1 !important;
        line-height: 1.7;
        margin-bottom: 0.4rem;
    }

    .report-container strong {
        color: #f8fafc !important;
    }

    .report-container hr {
        border-color: #1e293b;
    }

    /* ---------- DATAFRAME ---------- */

    [data-testid="stDataFrame"] {
        border: 1px solid #1e293b;
        border-radius: 12px;
        overflow: hidden;
    }

    /* ---------- DIVIDER ---------- */

    hr {
        border-color: #172033 !important;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #475569;
        font-size: 0.75rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #111827;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def clean_report_html(report: str) -> str:
    """
    Remove accidental HTML generated inside the final report
    while preserving Markdown formatting.
    """

    if not report:
        return ""

    cleaned = report

    # Remove complete metric-value blocks.
    cleaned = re.sub(
        r'<div\s+class=["\']metric-value["\'][^>]*>.*?</div>',
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Remove metric-label blocks.
    cleaned = re.sub(
        r'<div\s+class=["\']metric-label["\'][^>]*>.*?</div>',
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Remove generic div/span opening and closing tags.
    cleaned = re.sub(
        r"</?(?:div|span|section|article|p|strong|em)[^>]*>",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Remove remaining HTML tags.
    cleaned = re.sub(
        r"<[^>]+>",
        "",
        cleaned,
    )

    # Decode a few common escaped HTML entities.
    cleaned = (
        cleaned
        .replace("&nbsp;", " ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
    )

    # Remove excessive whitespace.
    cleaned = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned,
    )

    return cleaned.strip()


def format_metric_name(name: str) -> str:
    return name.replace("_", " ").title()


def review_badge(status: str) -> str:
    status_upper = status.upper()

    if "PASS_WITH_WARNINGS" in status_upper:
        return (
            '<span class="review-warning">'
            "PASS WITH WARNINGS"
            "</span>"
        )

    if "PASS" in status_upper:
        return (
            '<span class="review-pass">'
            "PASS"
            "</span>"
        )

    if "FAIL" in status_upper:
        return (
            '<span class="review-failed">'
            "FAILED"
            "</span>"
        )

    return (
        '<span class="review-warning">'
        f"{status}"
        "</span>"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # ========================================================
    # HERO
    # ========================================================

    st.markdown(
        """
        <div class="hero">
            <div class="hero-brand">
                MULTI-AGENT INTELLIGENCE PLATFORM
            </div>

            <div class="hero-title">
                Multi-Agent Ops Crew
            </div>

            <div class="hero-description">
                Turn operational data and business questions into
                structured, evidence-based executive insights using
                research, data analysis, review, and revision agents.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # BUSINESS PROBLEM
    # ========================================================

    st.markdown(
        '<div class="section-label">01 — Investigation</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">'
        "Define Your Business Problem"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        "Describe the operational problem, performance issue, "
        "or business question you want the agent crew to investigate."
        "</div>",
        unsafe_allow_html=True,
    )

    problem = st.text_area(
        "Business problem",
        placeholder=(
            "Example: Why did customer support "
            "resolution time increase?"
        ),
        height=120,
        label_visibility="collapsed",
    )

    # ========================================================
    # CSV UPLOAD
    # ========================================================

    st.markdown(
        '<div class="section-label">02 — Data</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">'
        "Upload Company Data"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        "Upload a CSV containing the operational data required "
        "for the analysis. The system will validate and pass "
        "the data into the multi-agent workflow."
        "</div>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Company operational dataset",
        type=["csv"],
    )

    df = None

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)

            validate_support_data(df)

            st.success(
                f"CSV loaded successfully — "
                f"{len(df):,} rows × {len(df.columns):,} columns."
            )

            with st.expander(
                "Preview uploaded dataset",
                expanded=False,
            ):
                st.dataframe(
                    df,
                    use_container_width=True,
                )

        except Exception as exc:
            st.error(
                f"Invalid CSV file: {exc}"
            )
            df = None

    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    st.markdown(
        '<div class="section-label">03 — Execute</div>',
        unsafe_allow_html=True,
    )

    analyze = st.button(
        "Analyze Business Problem",
        type="primary",
        use_container_width=True,
    )

    if not analyze:
        st.markdown(
            """
            <div class="footer">
                Multi-Agent Ops Crew · Research · Analytics ·
                Quality Control · Executive Intelligence
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if not problem.strip():
        st.error(
            "Please enter a business problem."
        )
        return

    if df is None:
        st.error(
            "Please upload a valid company CSV file."
        )
        return

    # ========================================================
    # INITIAL STATE
    # ========================================================

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

    # ========================================================
    # RUN WORKFLOW
    # ========================================================

    with st.spinner(
        "Running Multi-Agent Ops Crew..."
    ):

        try:
            graph = build_graph()

            result = graph.invoke(
                initial_state
            )

        except Exception as exc:

            st.error(
                "Workflow failed: "
                f"{exc}"
            )

            return

    # ========================================================
    # WORKFLOW STATUS
    # ========================================================

    st.markdown(
        '<div class="section-label">04 — Workflow</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">'
        "Workflow Status"
        "</div>",
        unsafe_allow_html=True,
    )

    status_col, revision_col = st.columns(2)

    with status_col:
        status_value = result.get(
            "status",
            "UNKNOWN",
        )

        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-label">
                    Current Status
                </div>
                <div class="card-value">
                    {status_value}
                </div>
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
            <div class="info-card">
                <div class="card-label">
                    Revision Attempts
                </div>
                <div class="card-value">
                    {revision_count}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ========================================================
    # KEY METRICS
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

        st.markdown(
            '<div class="section-label">05 — Analytics</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">'
            "Key Metrics"
            "</div>",
            unsafe_allow_html=True,
        )

        metric_items = list(
            metrics.items()
        )

        for start in range(
            0,
            len(metric_items),
            3,
        ):

            row = metric_items[
                start:start + 3
            ]

            columns = st.columns(
                len(row)
            )

            for column, (
                name,
                value,
            ) in zip(
                columns,
                row,
            ):

                with column:

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">
                                {format_metric_name(name)}
                            </div>

                            <div class="metric-value">
                                {value}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # ========================================================
    # DATA VISUALIZATION
    # ========================================================

    if not df.empty:

        numeric_columns = list(
            df.select_dtypes(
                include="number"
            ).columns
        )

        if numeric_columns:

            st.markdown(
                '<div class="section-label">'
                "06 — Data Visualization"
                "</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="section-title">'
                "Operational Trends"
                "</div>",
                unsafe_allow_html=True,
            )

            selected_metric = st.selectbox(
                "Select metric to visualize",
                numeric_columns,
            )

            chart_df = df.copy()

            fig = px.line(
                chart_df,
                y=selected_metric,
                markers=True,
                template="plotly_dark",
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10,
                ),
                font=dict(
                    color="#cbd5e1"
                ),
                xaxis=dict(
                    showgrid=True,
                    gridcolor="#172033",
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="#172033",
                ),
            )

            st.plotly_chart(
                fig,
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

        st.markdown(
            '<div class="section-label">'
            "07 — Quality Control"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">'
            "Quality Review"
            "</div>",
            unsafe_allow_html=True,
        )

        review_status = review.get(
            "review_status",
            "UNKNOWN",
        )

        st.markdown(
            f"""
            <div style="margin-bottom: 1rem;">
                <strong style="color:#cbd5e1;">
                    Review Status:
                </strong>
                &nbsp;
                {review_badge(review_status)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander(
            "View Detailed Review",
            expanded=False,
        ):

            review_analysis = review.get(
                "analysis",
                "No review available.",
            )

            st.markdown(
                review_analysis
            )

    # ========================================================
    # FINAL EXECUTIVE REPORT
    # ========================================================

    final_report = result.get(
        "final_report",
        {},
    )

    report = final_report.get(
        "report",
        "",
    )

    st.markdown(
        '<div class="section-label">'
        "08 — Executive Intelligence"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">'
        "Final Executive Report"
        "</div>",
        unsafe_allow_html=True,
    )

    if report:

        clean_report = clean_report_html(
            report
        )

        st.markdown(
            '<div class="report-container">',
            unsafe_allow_html=True,
        )

        st.markdown(
            clean_report
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    else:

        st.warning(
            "No final report was generated."
        )

    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown(
        """
        <div class="footer">
            Multi-Agent Ops Crew ·
            Evidence-Based Operational Intelligence
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()