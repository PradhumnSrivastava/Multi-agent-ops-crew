from langgraph.graph import END, START, StateGraph

from multi_agent_ops.agents.business_analysis import business_analysis_agent
from multi_agent_ops.agents.data_analysis import data_analysis_agent
from multi_agent_ops.agents.final_report import final_report_agent
from multi_agent_ops.agents.research import research_agent
from multi_agent_ops.agents.review import review_agent
from multi_agent_ops.agents.revision import revision_agent
from multi_agent_ops.state import OpsState


MAX_REVISION_ATTEMPTS = 2


def quality_gate(state: OpsState) -> str:
    """Route the workflow based on review status and revision limit."""

    review = state.get("review", {})
    review_status = review.get("review_status", "")
    revision_count = state.get("revision_count", 0)

    if (
        review_status == "FAIL"
        and revision_count < MAX_REVISION_ATTEMPTS
    ):
        return "revision"

    return "final_report"


def build_graph():
    """Build and compile the Multi-Agent Ops Crew workflow."""

    graph = StateGraph(OpsState)

    # Agents
    graph.add_node("research", research_agent)
    graph.add_node("data_analysis", data_analysis_agent)
    graph.add_node("business_analysis", business_analysis_agent)
    graph.add_node("review", review_agent)
    graph.add_node("revision", revision_agent)
    graph.add_node("final_report", final_report_agent)

    # Main analysis pipeline
    graph.add_edge(START, "research")
    graph.add_edge("research", "data_analysis")
    graph.add_edge("data_analysis", "business_analysis")
    graph.add_edge("business_analysis", "review")

    # Quality gate
    graph.add_conditional_edges(
        "review",
        quality_gate,
        {
            "revision": "revision",
            "final_report": "final_report",
        },
    )

    # Revision loop
    graph.add_edge("revision", "review")

    # Final output
    graph.add_edge("final_report", END)

    return graph.compile()