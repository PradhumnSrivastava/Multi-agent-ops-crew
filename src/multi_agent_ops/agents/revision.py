from langchain_huggingface import ChatHuggingFace

from multi_agent_ops.agents.research import create_llm
from multi_agent_ops.state import OpsState


def create_revision_llm() -> ChatHuggingFace:
    """Create the Hugging Face model used by the revision agent."""

    return create_llm()


llm = create_revision_llm()


def revision_agent(state: OpsState) -> dict:
    """Revise the business analysis based on review feedback."""

    problem = state["problem"]
    research_findings = state["research_findings"]
    data_findings = state["data_findings"]
    business_analysis = state["business_analysis"]
    review = state["review"]

    prompt = f"""
You are a senior business analyst responsible for revising
an analysis after a critical quality review.

Business Problem:
{problem}

Research Findings:
{research_findings}

Data Findings:
{data_findings}

Previous Business Analysis:
{business_analysis}

Quality Review:
{review}

Your task is to produce a corrected business analysis.

Apply every valid correction identified by the reviewer.

Important rules:

- Do not invent facts.
- Do not invent statistics.
- Do not introduce external evidence.
- Use the Data Findings as the authoritative source for numbers.
- Do not change numerical values unless they are mathematically incorrect.
- Do not convert correlation into causation.
- Clearly distinguish evidence, inference, and hypothesis.
- Remove unsupported claims.
- Do not claim that research evidence proves a business cause.
- If information is missing, explicitly identify it as missing.
- Preserve useful parts of the previous analysis.
- Improve the analysis rather than simply rewriting it.

Structure the revised analysis as:

1. Business Interpretation
2. Evidence-Based Findings
3. Most Likely Contributing Factors
4. Hypotheses
5. Business Impact
6. Recommended Actions
7. Further Investigation
8. Limitations

Return only the revised business analysis.
"""

    response = llm.invoke(prompt)

    return {
        "business_analysis": {
            "analysis": response.content,
        },
        "revision": {
            "reason": "Business analysis revised based on quality review.",
        },
        "status": "REVISION_COMPLETED",
    }