from typing import Any, TypedDict

from pydantic import BaseModel


class ResearchSource(BaseModel):
    """A web research source used by the research agent."""

    title: str
    url: str
    snippet: str


class OpsState(TypedDict, total=False):
    """Shared state passed between all agents in the workflow."""

    problem: str
    plan: dict[str, Any]
    research_findings: dict[str, Any]
    data_findings: dict[str, Any]
    business_analysis: dict[str, Any]
    review: dict[str, Any]
    revision: dict[str, Any]
    revision_count: int
    status: str
    final_report: dict[str, Any]