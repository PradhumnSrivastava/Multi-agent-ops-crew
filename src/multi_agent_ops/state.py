from typing import TypedDict


class OpsState(TypedDict):
    problem: str
    plan: dict
    research_findings: dict
    data_findings: dict
    business_analysis: dict
    review: dict
    status: str
    final_report: dict