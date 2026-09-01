from multi_agent_ops.llm import create_llm
from multi_agent_ops.state import OpsState


llm = create_llm(max_new_tokens=500)


def final_report_agent(state: OpsState) -> dict:
    """Generate a review-aware executive report."""

    problem = state["problem"]
    research_findings = state["research_findings"]
    data_findings = state["data_findings"]
    business_analysis = state["business_analysis"]
    review = state["review"]

    prompt = f"""
You are the final reporting analyst in a multi-agent business
operations system.

Your responsibility is to produce the final executive report
after research, data analysis, business analysis, and quality review.

Business Problem:

{problem}

Research Findings:

{research_findings}

Data Findings:

{data_findings}

Business Analysis:

{business_analysis}

Quality Review:

{review}

IMPORTANT:

The quality review is the final validation layer before reporting.

You MUST carefully apply every relevant issue identified by the
reviewer.

If the reviewer identifies an unsupported claim, do not present
that claim as an established fact.

If the reviewer identifies a numerical inconsistency, use the
original Data Findings as the authoritative source and verify
the calculation before including it.

If the reviewer identifies a causal claim that is not supported
by the available data, describe it as a possible explanation,
hypothesis, or association rather than a proven cause.

Do not hide important warnings identified by the reviewer.

Separate the final analysis into:

- Evidence: directly supported by the provided data or research.
- Inference: a reasonable interpretation of the available evidence.
- Hypothesis: a possible explanation that requires further validation.

Do not introduce information that is not present in the provided
inputs.

Create a concise, evidence-based executive report.

Structure:

1. Executive Summary
2. Problem Statement
3. Key Findings
4. Data Evidence
5. Business Interpretation
6. Evidence-Based Findings
7. Hypotheses / Potential Root Causes
8. Business Impact
9. Recommended Actions
10. Further Investigation
11. Analysis Limitations

Rules:

- Use only the information provided above.
- Do not invent facts.
- Do not invent statistics.
- Do not invent sources.
- Do not introduce external evidence.
- Preserve accurate numerical findings from Data Findings.
- Verify numerical claims before reporting them.
- Do not convert correlation into causation.
- Clearly distinguish evidence, inference, and hypothesis.
- Apply the reviewer's corrections.
- Mention important limitations identified by the reviewer.
- Keep the report concise and professional.
- Write for a business decision-maker.
"""

    response = llm.invoke(prompt)

    return {
        "final_report": {
            "report": response.content,
        },
        "status": "FINAL_REPORT_COMPLETED",
    }