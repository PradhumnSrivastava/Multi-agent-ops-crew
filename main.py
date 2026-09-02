from multi_agent_ops.graph import build_graph
from multi_agent_ops.state import OpsState


def main() -> None:
    """Run the Multi-Agent Ops Crew."""

    problem = input("Enter the business problem: ").strip()

    if not problem:
        raise ValueError("Business problem cannot be empty.")

    graph = build_graph()

    initial_state: OpsState = {
        "problem": problem,
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

    print()
    print("=" * 70)
    print("MULTI-AGENT OPS CREW")
    print("=" * 70)
    print()

    try:
        result = graph.invoke(initial_state)

    except Exception as exc:
        print()
        print("=" * 70)
        print("WORKFLOW FAILED")
        print("=" * 70)
        print(f"Error: {exc}")
        raise

    final_report = result.get("final_report", {})
    report = final_report.get("report", "")

    print("=" * 70)
    print(f"Status: {result.get('status', 'UNKNOWN')}")
    print(f"Revision attempts: {result.get('revision_count', 0)}")
    print()
    print("FINAL EXECUTIVE REPORT")
    print("-" * 70)

    if report:
        print(report)
    else:
        print("No final report was generated.")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()