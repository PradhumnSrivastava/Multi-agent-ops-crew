from langchain_huggingface import ChatHuggingFace

from multi_agent_ops.llm import create_llm
from multi_agent_ops.state import OpsState


def create_review_llm() -> ChatHuggingFace:
    """Create the LLM used by the review agent."""
    return create_llm(
        max_new_tokens=700,
        temperature=0.0,
    )


llm = create_review_llm()


def review_agent(state: OpsState) -> dict:
    """Review research, data analysis, and business analysis."""

    problem = state["problem"]
    research_findings = state["research_findings"]
    data_findings = state["data_findings"]
    business_analysis = state["business_analysis"]

    prompt = f"""
You are the senior quality reviewer of a business intelligence
multi-agent system.

Your job is to critically review the analysis before final reporting.

BUSINESS PROBLEM
{problem}

RESEARCH FINDINGS
{research_findings}

DATA FINDINGS
{data_findings}

BUSINESS ANALYSIS
{business_analysis}

Review the analysis for:

1. FACTUAL CONSISTENCY
Check whether claims agree with the supplied research and data.

2. NUMERICAL CONSISTENCY
Check important calculations using:

percentage change =
((new value - old value) / old value) * 100

Identify any incorrect numerical values.

3. EVIDENCE GROUNDING
Determine which claims are directly supported by the supplied
data or research.

4. CAUSALITY
Do not treat correlation or association as proof of causation.

5. UNSUPPORTED CLAIMS
Identify claims presented as facts when they are only hypotheses.

6. MISSING INFORMATION
Identify evidence required to establish stronger conclusions.

7. REPORT COMPLETENESS
Check whether the analysis covers:
- Business Interpretation
- Evidence-Based Findings
- Hypotheses
- Business Impact
- Recommended Actions
- Further Investigation
- Limitations

IMPORTANT RULES:

- Use only the supplied information.
- Do not invent facts.
- Do not invent statistics.
- Do not introduce external evidence.
- Be critical but concise.
- Separate evidence, inference, and hypothesis.
- Explicitly identify unsupported claims.
- Explicitly identify numerical errors.
- If there are no numerical errors, say so.

Return exactly these sections:

OVERALL ASSESSMENT:
[brief assessment]

STRENGTHS:
- ...
- ...

ISSUES IDENTIFIED:
- ...
- ...

UNSUPPORTED CLAIMS:
- ...
- ...

MISSING INFORMATION:
- ...
- ...

RECOMMENDED CORRECTIONS:
- ...
- ...

FINAL REVIEW STATUS:
PASS
or
PASS_WITH_WARNINGS
or
FAIL
"""

    response = llm.invoke(prompt)

    return {
        "review": {
            "analysis": response.content,
        },
        "status": "REVIEW_COMPLETED",
    }