from dotenv import load_dotenv
from langsmith import Client, evaluate

from multi_agent_ops.graph import build_graph


load_dotenv()


DATASET_NAME = "multi-agent-ops-crew-evaluation"


def target_function(inputs: dict) -> dict:
    """Run the Multi-Agent Ops Crew for one evaluation example."""

    problem = inputs["problem"]

    initial_state = {
        "problem": problem,
        "plan": {},
        "research_findings": {},
        "data_findings": {},
        "business_analysis": {},
        "review": {},
        "status": "STARTED",
        "final_report": {},
    }

    graph = build_graph()

    result = graph.invoke(initial_state)

    return {
        "final_report": result.get("final_report", {}),
        "review": result.get("review", {}),
        "data_findings": result.get("data_findings", {}),
        "business_analysis": result.get("business_analysis", {}),
        "research_findings": result.get("research_findings", {}),
        "status": result.get("status"),
    }


def evaluate_report_quality(run, example) -> dict:
    """Evaluate basic quality of the generated final report."""

    output = run.outputs or {}
    final_report = output.get("final_report", {})

    if isinstance(final_report, dict):
        report = final_report.get("report", "")
    else:
        report = str(final_report)

    required_sections = [
        "Executive Summary",
        "Problem Statement",
        "Key Findings",
        "Data Evidence",
        "Business Interpretation",
        "Hypotheses",
        "Business Impact",
        "Recommended Actions",
        "Further Investigation",
    ]

    if not report:
        return {
            "key": "report_quality",
            "score": 0,
            "comment": "No final report was generated.",
        }

    missing_sections = [
        section
        for section in required_sections
        if section.lower() not in report.lower()
    ]

    score = (
        len(required_sections) - len(missing_sections)
    ) / len(required_sections)

    return {
        "key": "report_quality",
        "score": score,
        "comment": (
            "All required sections are present."
            if not missing_sections
            else f"Missing sections: {missing_sections}"
        ),
    }


def evaluate_report_generation(run, example) -> dict:
    """Check whether the workflow completed successfully."""

    output = run.outputs or {}

    final_report = output.get("final_report")
    status = output.get("status")

    if final_report and status == "FINAL_REPORT_COMPLETED":
        return {
            "key": "report_generation",
            "score": 1,
            "comment": "Final report generated successfully.",
        }

    return {
        "key": "report_generation",
        "score": 0,
        "comment": "Final report generation failed.",
    }


def run_evaluation_dataset():
    """Evaluate the Multi-Agent Ops Crew using LangSmith."""

    client = Client()

    datasets = list(
        client.list_datasets(
            dataset_name=DATASET_NAME
        )
    )

    if not datasets:
        raise ValueError(
            f"LangSmith dataset '{DATASET_NAME}' was not found."
        )

    print(f"Running LangSmith evaluation: {DATASET_NAME}")
    print()

    results = evaluate(
        target_function,
        data=DATASET_NAME,
        evaluators=[
            evaluate_report_quality,
            evaluate_report_generation,
        ],
        experiment_prefix="multi-agent-ops-crew",
        client=client,
    )

    print()
    print("Evaluation completed.")
    print()

    for result in results:
        print(result)


if __name__ == "__main__":
    run_evaluation_dataset()