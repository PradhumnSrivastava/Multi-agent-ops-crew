from typing import Any

from langsmith import Client
from langsmith.evaluation import evaluate

from multi_agent_ops.graph import build_graph


DATASET_NAME = "multi-agent-ops-crew-evaluation"


def target_function(inputs: dict[str, Any]) -> dict[str, Any]:
    """Run the complete Multi-Agent Ops Crew workflow."""

    problem = inputs["problem"]

    initial_state = {
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

    graph = build_graph()
    result = graph.invoke(initial_state)

    return result


def report_generation_evaluator(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check whether a final report was generated."""

    final_report = outputs.get("final_report", {})
    report = final_report.get("report", "")

    passed = isinstance(report, str) and len(report.strip()) > 100

    return {
        "key": "report_generation",
        "score": 1.0 if passed else 0.0,
        "comment": (
            "Final report generated successfully."
            if passed
            else "Final report generation failed."
        ),
    }


def report_quality_evaluator(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check whether the final report contains all required sections."""

    final_report = outputs.get("final_report", {})
    report = final_report.get("report", "")

    if not isinstance(report, str) or not report.strip():
        return {
            "key": "report_quality",
            "score": 0.0,
            "comment": "No final report was generated.",
        }

    required_sections = [
        "Executive Summary",
        "Problem Statement",
        "Key Findings",
        "Data Evidence",
        "Business Interpretation",
        "Evidence-Based Findings",
        "Hypotheses",
        "Business Impact",
        "Recommended Actions",
        "Further Investigation",
        "Analysis Limitations",
    ]

    report_lower = report.lower()

    missing_sections = [
        section
        for section in required_sections
        if section.lower() not in report_lower
    ]

    score = (
        (len(required_sections) - len(missing_sections))
        / len(required_sections)
    )

    return {
        "key": "report_quality",
        "score": score,
        "comment": (
            "All required report sections are present."
            if not missing_sections
            else f"Missing sections: {missing_sections}"
        ),
    }


def numerical_consistency_evaluator(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Check whether important numerical metrics from the workflow
    are preserved in the final report.
    """

    data_findings = outputs.get("data_findings", {})
    final_report = outputs.get("final_report", {})
    report = final_report.get("report", "")

    if not isinstance(report, str) or not report.strip():
        return {
            "key": "numerical_consistency",
            "score": 0.0,
            "comment": "Final report is missing.",
        }

    data_text = str(data_findings)

    metrics = [
        "35.71",
        "47.25",
        "7.69",
        "80.77",
        "118.75",
        "47.02",
    ]

    available_metrics = [
        metric
        for metric in metrics
        if metric in data_text
    ]

    preserved_metrics = [
        metric
        for metric in available_metrics
        if metric in report
    ]

    if not available_metrics:
        return {
            "key": "numerical_consistency",
            "score": 1.0,
            "comment": "No numerical metrics were available for comparison.",
        }

    score = len(preserved_metrics) / len(available_metrics)

    return {
        "key": "numerical_consistency",
        "score": score,
        "comment": (
            f"{len(preserved_metrics)}/{len(available_metrics)} "
            "important data metrics were preserved in the final report."
        ),
    }


def evidence_grounding_evaluator(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check whether the final report acknowledges review warnings."""

    final_report = outputs.get("final_report", {})
    report = final_report.get("report", "")

    review = outputs.get("review", {})
    review_text = str(review.get("analysis", ""))

    if not isinstance(report, str) or not report.strip():
        return {
            "key": "evidence_grounding",
            "score": 0.0,
            "comment": "Final report is missing.",
        }

    review_lower = review_text.lower()
    report_lower = report.lower()

    warning_terms = [
        "unsupported",
        "causation",
        "correlation",
        "hypothesis",
        "missing information",
        "limitation",
        "evidence gap",
    ]

    warnings = [
        term
        for term in warning_terms
        if term in review_lower
    ]

    acknowledged = [
        term
        for term in warnings
        if term in report_lower
    ]

    if not warnings:
        return {
            "key": "evidence_grounding",
            "score": 1.0,
            "comment": "No major review warnings were identified.",
        }

    score = len(acknowledged) / len(warnings)

    return {
        "key": "evidence_grounding",
        "score": score,
        "comment": (
            "Review warnings were acknowledged in the final report."
            if score == 1.0
            else "The final report does not clearly acknowledge all "
                 "relevant review warnings."
        ),
    }


def run_evaluation() -> None:
    """Run the complete LangSmith evaluation."""

    client = Client()

    print("=" * 70)
    print("MULTI-AGENT OPS CREW")
    print("LANGSMITH EVALUATION")
    print("=" * 70)
    print()
    print(f"Dataset: {DATASET_NAME}")
    print()

    evaluators = [
        report_generation_evaluator,
        report_quality_evaluator,
        numerical_consistency_evaluator,
        evidence_grounding_evaluator,
    ]

    results = evaluate(
        target_function,
        data=DATASET_NAME,
        evaluators=evaluators,
        experiment_prefix="multi-agent-ops-crew",
        client=client,
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)
    print()

    for result in results:
        print(result)


if __name__ == "__main__":
    run_evaluation()