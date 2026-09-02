from multi_agent_ops.llm import create_llm
from multi_agent_ops.state import OpsState


llm = create_llm(
    max_new_tokens=900,
    temperature=0.0,
)


def final_report_agent(state: OpsState) -> dict:
    """Generate the final executive report."""

    problem = state["problem"]
    research_findings = state["research_findings"]
    data_findings = state["data_findings"]
    business_analysis = state["business_analysis"]
    review = state["review"]

    prompt = f"""
You are the final executive reporting analyst.

Create a concise, evidence-based business report using ONLY the
information supplied below.

BUSINESS PROBLEM
{problem}

RESEARCH FINDINGS
{research_findings}

DATA FINDINGS
{data_findings}

BUSINESS ANALYSIS
{business_analysis}

QUALITY REVIEW
{review}

CRITICAL RULES

1. The Data Findings are authoritative for numerical values.

2. Preserve important numerical metrics from Data Findings.

3. Do not invent facts, statistics, sources, causes, or impacts.

4. Do not introduce external information.

5. Do not present a hypothesis as a proven cause.

6. Do not convert correlation into causation.

7. If evidence is insufficient, explicitly state:
   "The available evidence is insufficient to establish this as a
   confirmed cause."

8. Apply valid corrections identified by the Quality Review.

9. Include important limitations identified by the reviewer.

10. Complete ALL eleven sections. Never stop before section 11.

11. Keep every section concise.

REPORT STRUCTURE

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

SECTION RULES

Executive Summary:
Summarize the problem, strongest evidence, and overall conclusion.

Problem Statement:
Clearly define what changed.

Key Findings:
List the most important findings.

Data Evidence:
Include the important numerical metrics from Data Findings.

Business Interpretation:
Explain what the evidence means without claiming unsupported causality.

Evidence-Based Findings:
Include only findings supported by the supplied evidence.

Hypotheses / Potential Root Causes:
List possible causes and clearly label them as hypotheses when they
are not directly proven.

Business Impact:
Describe only impacts reasonably supported by the available evidence.
Do not invent financial or customer metrics.

Recommended Actions:
Give practical actions that follow from the evidence and hypotheses.

Further Investigation:
State what additional data would be required to validate hypotheses.

Analysis Limitations:
Explicitly state evidence gaps and causal limitations.

FINAL VALIDATION BEFORE RESPONDING

Check that:
- all 11 headings are present;
- numerical values agree with Data Findings;
- unsupported claims are labelled as hypotheses;
- important review warnings are acknowledged;
- limitations are explicitly stated.

Return ONLY the final executive report.
"""

    response = llm.invoke(prompt)

    return {
        "final_report": {
            "report": response.content,
        },
        "status": "FINAL_REPORT_COMPLETED",
    }