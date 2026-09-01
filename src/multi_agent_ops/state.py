from typing import TypedDict

from pydantic import BaseModel


class ResearchSource(BaseModel):
    title: str
    url: str
    snippet: str


class ResearchResult(BaseModel):
    analysis: str
    sources: list[ResearchSource]


class OpsState(TypedDict):
    problem: str
    plan: dict
    research_findings: dict
    data_findings: dict
    business_analysis: dict
    review: dict
    status: str
    final_report: dict