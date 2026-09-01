from langgraph.graph import StateGraph, START, END

from multi_agent_ops.state import OpsState
from multi_agent_ops.agents.research import research_agent


def supervisor(state: OpsState) -> dict:
    return {
        "status": "PLANNING",
        "plan": {
            "research_required": True,
            "data_analysis_required": True,
            "business_analysis_required": True,
        },
    }


builder = StateGraph(OpsState)

builder.add_node("supervisor", supervisor)
builder.add_node("research_agent", research_agent)

builder.add_edge(START, "supervisor")
builder.add_edge("supervisor", "research_agent")
builder.add_edge("research_agent", END)

graph = builder.compile()