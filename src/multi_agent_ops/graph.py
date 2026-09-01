from langgraph.graph import END, START, StateGraph

from multi_agent_ops.agents.business_analysis import business_analysis_agent
from multi_agent_ops.agents.data_analysis import data_analysis_agent
from multi_agent_ops.agents.research import research_agent
from multi_agent_ops.agents.review import review_agent
from multi_agent_ops.state import OpsState


def build_graph():
    """Build the Multi-Agent Ops Crew workflow."""

    graph = StateGraph(OpsState)

    graph.add_node("research", research_agent)
    graph.add_node("data_analysis", data_analysis_agent)
    graph.add_node("business_analysis", business_analysis_agent)
    graph.add_node("review", review_agent)

    graph.add_edge(START, "research")
    graph.add_edge("research", "data_analysis")
    graph.add_edge("data_analysis", "business_analysis")
    graph.add_edge("business_analysis", "review")
    graph.add_edge("review", END)

    return graph.compile()