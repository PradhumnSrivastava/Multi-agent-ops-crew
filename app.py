import pandas as pd
import streamlit as st

from multi_agent_ops.graph import build_graph
from multi_agent_ops.state import OpsState
from multi_agent_ops.tools.data_tools import validate_support_data


st.set_page_config(
    page_title="Multi-Agent Ops Crew",
    page_icon="📊",
    layout="wide",
)


def main() -> None:
    st.title("Multi-Agent Ops Crew")
    st.write(
        "AI-powered business operations analysis using "
        "research, data analysis, review, and revision agents."
    )

    st.divider()

    st.subheader("1. Describe the Business Problem")

    problem = st.text_area(
        "What business problem do you want to investigate?",
        placeholder=(
            "Example: Why did customer support "
            "resolution time increase?"
        ),
        height=120,
    )

    st.subheader("2. Upload Company Data")

    uploaded_file = st.file_uploader(
        "Upload your company CSV file",
        type=["csv"],
    )

    df = None

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            validate_support_data(df)

            st.success(
                f"CSV loaded successfully: "
                f"{len(df)} rows and {len(df.columns)} columns."
            )

            with st.expander("Preview uploaded data"):
                st.dataframe(
                    df,
                    use_container_width=True,
                )

        except Exception as exc:
            st.error(f"Invalid CSV file: {exc}")
            df = None

    st.divider()

    analyze = st.button(
        "Analyze Business Problem",
        type="primary",
        use_container_width=True,
    )

    if not analyze:
        return

    if not problem.strip():
        st.error("Please enter a business problem.")

        return

    if df is None:
        st.error("Please upload a valid company CSV file.")

        return

    company_data = df.to_dict(orient="records")

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

    with st.spinner(
        "Running Multi-Agent Ops Crew..."
    ):
        try:
            graph = build_graph()

            result = graph.invoke(initial_state)

        except Exception as exc:
            st.error(
                "Workflow failed. "
                f"Error: {exc}"
            )
            return

    st.divider()

    st.header("Workflow Status")

    status_col, revision_col = st.columns(2)

    with status_col:
        st.metric(
            "Status",
            result.get("status", "UNKNOWN"),
        )

    with revision_col:
        st.metric(
            "Revision Attempts",
            result.get("revision_count", 0),
        )

    st.divider()

    data_findings = result.get(
        "data_findings",
        {},
    )

    metrics = data_findings.get(
        "metrics",
        {},
    )

    if metrics:
        st.header("Key Metrics")

        metric_columns = st.columns(
            len(metrics)
        )

        for column, (name, value) in zip(
            metric_columns,
            metrics.items(),
        ):
            with column:
                st.metric(
                    name.replace("_", " ").title(),
                    value,
                )

    st.divider()

    review = result.get(
        "review",
        {},
    )

    if review:
        st.header("Quality Review")

        review_status = review.get(
            "review_status",
            "UNKNOWN",
        )

        st.write(
            f"**Review Status:** {review_status}"
        )

        with st.expander("View Review Details"):
            st.write(
                review.get(
                    "analysis",
                    "No review available.",
                )
            )

    st.divider()

    final_report = result.get(
        "final_report",
        {},
    )

    report = final_report.get(
        "report",
        "",
    )

    st.header("Final Executive Report")

    if report:
        st.markdown(report)
    else:
        st.warning(
            "No final report was generated."
        )


if __name__ == "__main__":
    main()