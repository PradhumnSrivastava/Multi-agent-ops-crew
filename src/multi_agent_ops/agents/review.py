from langchain_huggingface import ChatHuggingFace
from pydantic import BaseModel, Field

from multi_agent_ops.state import OpsState
from multi_agent_ops.agents.research import create_llm


class ReviewResult(BaseModel):
    """Structured output produced by the review agent."""

    overall_assessment: str = Field(
        description="Overall assessment of the analysis quality."
    )
    strengths: list[str] = Field(
        description="Strong aspects of the analysis."
    )
    issues_identified: list[str] = Field(
        description="Issues found during review."
    )
    unsupported_claims: list[str] = Field(
        description="Claims that are not sufficiently supported."
    )
    missing_information: list[str] = Field(
        description="Information that is still required."
    )
    recommended_corrections: list[str] = Field(
        description="Corrections or improvements recommended."
    )
    review_status: str = Field(
        description="Final review status: PASS, PASS_WITH_WARNINGS, or FAIL."
    )


def create_review_llm() -> ChatHuggingFace:
    """Create the Hugging Face model used by the review agent."""

    return create_llm()


llm = create_review_llm()

structured_llm = llm.with_structured_output(ReviewResult)


def review_agent(state: OpsState) -> dict:
    """Review research, data analysis, and business analysis."""

    problem = state["problem"]
    research_findings = state["research_findings"]
    data_findings = state["data_findings"]
    business_analysis = state["business_analysis"]

    prompt = f"""
You are a senior quality reviewer in a multi-agent
business operations system.

Your job is to critically review the work produced by
the research, data analysis, and business analysis agents.

Business Problem:
{problem}

Research Findings:
{research_findings}

Data Findings:
{data_findings}

Business Analysis:
{business_analysis}

Review the analysis for:

1. Factual consistency
2. Numerical consistency
3. Whether conclusions are supported by the provided data
4. Whether research evidence is actually relevant
5. Unsupported assumptions or causal claims
6. Missing information
7. Overall quality

Important rules:

- Do not invent facts.
- Do not invent statistics.
- Do not introduce external evidence.
- Distinguish evidence, inference, and hypothesis.
- Flag unsupported causal claims.
- Verify numerical claims against the provided data.
- Be critical rather than automatically approving the analysis.
- Use only the information provided above.

Review status must be exactly one of:

PASS
PASS_WITH_WARNINGS
FAIL
"""

    result = structured_llm.invoke(prompt)

    return {
        "review": result.model_dump(),
        "status": "REVIEW_COMPLETED",
    }