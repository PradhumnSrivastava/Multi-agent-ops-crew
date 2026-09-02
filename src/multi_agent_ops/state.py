from typing import Any, TypedDict


class OpsState(TypedDict, total=False):
    """Shared state passed between all agents in the workflow."""

    problem: str

    # Company data uploaded by the user.
    company_data: list[dict[str, Any]]

    plan: dict[str, Any]

    research_findings: dict[str, Any]

    data_findings: dict[str, Any]

    business_analysis: dict[str, Any]

    review: dict[str, Any]

    revision: dict[str, Any]

    revision_count: int

    status: str

    final_report: dict[str, Any]