from multi_agent_ops.graph import build_graph


def main():
    graph = build_graph()

    initial_state = {
        "problem": "Why did customer support resolution time increase?",
        "plan": {
            "research_required": True,
            "data_analysis_required": True,
            "business_analysis_required": True,
        },
        "research_findings": {},
        "data_findings": {},
        "business_analysis": {},
        "review": {},
        "status": "STARTED",
        "final_report": {},
    }

    result = graph.invoke(initial_state)

    print(result)


if __name__ == "__main__":
    main()