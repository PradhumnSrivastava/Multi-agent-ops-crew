from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from multi_agent_ops.state import OpsState


def create_review_llm() -> ChatHuggingFace:
    from multi_agent_ops.agents.research import create_llm

    return create_llm()


llm = create_review_llm()


def review_agent(state: OpsState) -> dict:
    """Review research, data analysis, and business analysis for quality."""

    problem = state["problem"]
    research_findings = state["research_findings"]
    data_findings = state["data_findings"]
    business_analysis = state["business_analysis"]

    prompt = f"""
You are a senior quality reviewer in a multi-agent business
operations system.

Your job is to critically review the work produced by three agents.

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
6. Missing information or additional investigation required
7. Overall quality of the analysis

Important rules:

- Do not invent facts.
- Do not invent statistics.
- Do not introduce external evidence.
- Distinguish clearly between evidence, inference, and hypothesis.
- If a conclusion is not supported by the available data, explicitly flag it.
- Check whether the business analysis correctly interprets the data.
- Be critical rather than simply approving the previous agents.

Return a concise review containing:

- Overall assessment
- Strengths
- Issues identified
- Unsupported claims
- Missing information
- Recommended corrections
- Final review status: PASS, PASS_WITH_WARNINGS, or FAIL
"""

    response = llm.invoke(prompt)

    return {
        "review": {
            "analysis": response.content,
        },
        "status": "REVIEW_COMPLETED",
    }