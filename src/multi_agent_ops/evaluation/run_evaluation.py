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


def evaluate_report_generation(run, example) -> dict:
    """Check whether the complete workflow generated a final report."""

    output = run.outputs or {}

    final_report = output.get("final_report")
    status = output.get("status")

    if final_report and status == "FINAL_REPORT_COMPLETED":
        return {
            "key": "report_generation",
            "score": 1.0,
            "comment": "Final report generated successfully.",
        }

    return {
        "key": "report_generation",
        "score": 0.0,
        "comment": "Final report generation failed.",
    }


def evaluate_report_quality(run, example) -> dict:
    """Evaluate whether the final report contains the required sections."""

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
        "Analysis Limitations",
    ]

    if not report:
        return {
            "key": "report_quality",
            "score": 0.0,
            "comment": "No final report was generated.",
        }

    report_lower = report.lower()

    missing_sections = [
        section
        for section in required_sections
        if section.lower() not in report_lower
    ]

    score = (
        len(required_sections) - len(missing_sections)
    ) / len(required_sections)

    if not missing_sections:
        comment = "All required report sections are present."
    else:
        comment = f"Missing sections: {missing_sections}"

    return {
        "key": "report_quality",
        "score": score,
        "comment": comment,
    }


def evaluate_numerical_consistency(run, example) -> dict:
    """Check whether important data metrics are preserved in the final report."""

    output = run.outputs or {}

    final_report = output.get("final_report", {})
    data_findings = output.get("data_findings", {})

    if isinstance(final_report, dict):
        report = final_report.get("report", "")
    else:
        report = str(final_report)

    if not isinstance(data_findings, dict):
        data_findings = {}

    metrics = data_findings.get("metrics", {})

    if not report:
        return {
            "key": "numerical_consistency",
            "score": 0.0,
            "comment": "Final report is missing.",
        }

    if not isinstance(metrics, dict) or not metrics:
        return {
            "key": "numerical_consistency",
            "score": 0.0,
            "comment": "No data metrics were available for validation.",
        }

    important_metrics = [
        "ticket_change_pct",
        "resolution_time_change_pct",
        "staff_change_pct",
        "tickets_per_staff_first",
        "tickets_per_staff_last",
        "workload_change_pct",
    ]

    matched = 0
    checked = 0

    for metric_name in important_metrics:
        value = metrics.get(metric_name)

        if value is None:
            continue

        try:
            value_text = f"{float(value):.2f}"
        except (TypeError, ValueError):
            continue

        checked += 1

        if value_text in report:
            matched += 1

    if checked == 0:
        return {
            "key": "numerical_consistency",
            "score": 0.0,
            "comment": "No valid numerical metrics were available.",
        }

    score = matched / checked

    return {
        "key": "numerical_consistency",
        "score": score,
        "comment": (
            f"{matched}/{checked} important data metrics "
            "were preserved in the final report."
        ),
    }


def evaluate_evidence_grounding(run, example) -> dict:
    """Check whether the final report respects the available evidence."""

    output = run.outputs or {}

    final_report = output.get("final_report", {})
    review = output.get("review", {})

    if isinstance(final_report, dict):
        report = final_report.get("report", "")
    else:
        report = str(final_report)

    if isinstance(review, dict):
        review_analysis = review.get("analysis", "")
    else:
        review_analysis = str(review)

    if not report:
        return {
            "key": "evidence_grounding",
            "score": 0.0,
            "comment": "Final report is missing.",
        }

    report_lower = report.lower()
    review_lower = review_analysis.lower()

    grounding_terms = [
        "evidence",
        "inference",
        "hypothesis",
        "limitations",
        "further investigation",
    ]

    present_terms = [
        term
        for term in grounding_terms
        if term in report_lower
    ]

    structure_score = (
        len(present_terms) / len(grounding_terms)
    )

    warning_terms = [
        "unsupported",
        "not supported",
        "warning",
        "limitation",
        "insufficient evidence",
    ]

    review_has_warnings = any(
        term in review_lower
        for term in warning_terms
    )

    if review_has_warnings:

        warning_acknowledged = any(
            term in report_lower
            for term in warning_terms
        )

        if warning_acknowledged:
            score = min(
                1.0,
                structure_score + 0.2,
            )

            comment = (
                "The report contains evidence/inference structure "
                "and acknowledges review-related limitations or warnings."
            )

        else:
            score = max(
                0.0,
                structure_score - 0.2,
            )

            comment = (
                "The review contains warnings or unsupported-claim "
                "concerns that are not clearly acknowledged in the report."
            )

    else:
        score = structure_score

        comment = (
            "The report clearly distinguishes evidence, inference, "
            "hypothesis, and limitations."
        )

    return {
        "key": "evidence_grounding",
        "score": score,
        "comment": comment,
    }


def run_evaluation_dataset():
    """Evaluate the Multi-Agent Ops Crew using the LangSmith dataset."""

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

    print(
        f"Running LangSmith evaluation: {DATASET_NAME}"
    )
    print()

    results = evaluate(
        target_function,
        data=DATASET_NAME,
        evaluators=[
            evaluate_report_generation,
            evaluate_report_quality,
            evaluate_numerical_consistency,
            evaluate_evidence_grounding,
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

