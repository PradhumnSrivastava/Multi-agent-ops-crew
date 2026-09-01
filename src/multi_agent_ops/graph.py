from langgraph.graph import END, START, StateGraph

from multi_agent_ops.agents.business_analysis import business_analysis_agent
from multi_agent_ops.agents.data_analysis import data_analysis_agent
from multi_agent_ops.agents.final_report import final_report_agent
from multi_agent_ops.agents.research import research_agent
from multi_agent_ops.agents.review import review_agent
from multi_agent_ops.agents.revision import revision_agent
from multi_agent_ops.state import OpsState


def quality_gate(state: OpsState) -> str:
    """Route the workflow based on the review status."""

    review = state.get("review", {})

    review_status = review.get("review_status", "")

    if review_status == "FAIL":
        return "revision"

    return "final_report"


def build_graph():
    """Build the Multi-Agent Ops Crew workflow."""

    graph = StateGraph(OpsState)

    graph.add_node("research", research_agent)
    graph.add_node("data_analysis", data_analysis_agent)
    graph.add_node("business_analysis", business_analysis_agent)
    graph.add_node("review", review_agent)
    graph.add_node("revision", revision_agent)
    graph.add_node("final_report", final_report_agent)

    graph.add_edge(START, "research")

    graph.add_edge("research", "data_analysis")

    graph.add_edge(
        "data_analysis",
        "business_analysis",
    )

    graph.add_edge(
        "business_analysis",
        "review",
    )

    graph.add_conditional_edges(
        "review",
        quality_gate,
        {
            "revision": "revision",
            "final_report": "final_report",
        },
    )

    graph.add_edge(
        "revision",
        "review",
    )

    graph.add_edge(
        "final_report",
        END,
    )

    return graph.compile()