from multi_agent_ops.llm import create_llm
from multi_agent_ops.state import OpsState


llm = create_llm(
    max_new_tokens=800,
    temperature=0.0,
)


def revision_agent(state: OpsState) -> dict:
    """Revise business analysis using the quality review."""

    problem = state["problem"]
    research_findings = state["research_findings"]
    data_findings = state["data_findings"]
    business_analysis = state["business_analysis"]
    review = state["review"]

    revision_count = state.get("revision_count", 0) + 1

    prompt = f"""
You are a senior business analyst responsible for correcting
an analysis after an independent quality review.

BUSINESS PROBLEM

{problem}


RESEARCH FINDINGS

{research_findings}


DATA FINDINGS

{data_findings}


CURRENT BUSINESS ANALYSIS

{business_analysis}


QUALITY REVIEW

{review}


YOUR TASK

Revise the business analysis according to the quality review.

The revised analysis must be more accurate and better grounded
than the original analysis.


STRICT RULES

1. Use only the information provided above.

2. Do not introduce external information.

3. Do not invent facts.

4. Do not invent statistics.

5. Do not invent sources.

6. Use Data Findings as the authoritative source for numerical
   values.

7. Recalculate important percentage changes when necessary.

8. If a numerical value cannot be verified from the provided data,
   remove it or explicitly identify it as unavailable.

9. Never present correlation as causation.

10. A possible cause must be described as a hypothesis unless
    the provided evidence directly establishes it.

11. Do not treat general research findings as proof of what caused
    the specific business problem.

12. Remove unsupported claims identified by the reviewer.

13. Preserve valid findings from the original analysis.

14. Clearly distinguish:
    - Evidence
    - Inference
    - Hypothesis

15. Address every important issue identified in the review.

16. Do not hide evidence gaps.

17. Do not mention this revision prompt or the internal workflow.


REVISED ANALYSIS STRUCTURE

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


IMPORTANT

The revised analysis must not claim that a factor caused the
business problem unless the provided evidence establishes
causation.

For example, prefer:

"The increase in workload coincided with the increase in
resolution time and may be a contributing factor."

instead of:

"The increase in workload caused resolution time to increase."

Return only the complete revised business analysis.
"""

    response = llm.invoke(prompt)

    return {
        "business_analysis": {
            "analysis": response.content.strip(),
        },
        "revision": {
            "analysis": response.content.strip(),
            "revision_number": revision_count,
        },
        "revision_count": revision_count,
        "status": "REVISION_COMPLETED",
    }