from langchain_huggingface import ChatHuggingFace

from multi_agent_ops.llm import create_llm
from multi_agent_ops.state import OpsState


llm: ChatHuggingFace = create_llm(
    max_new_tokens=700,
    temperature=0.0,
)


def review_agent(state: OpsState) -> dict:
    """Review research, data analysis, and business analysis."""

    problem = state["problem"]
    research_findings = state["research_findings"]
    data_findings = state["data_findings"]
    business_analysis = state["business_analysis"]

    prompt = f"""
You are a senior quality reviewer in a multi-agent
business operations intelligence system.

Your responsibility is to critically review the work produced
by the research, data analysis, and business analysis agents.

BUSINESS PROBLEM
{problem}

RESEARCH FINDINGS
{research_findings}

DATA FINDINGS
{data_findings}

BUSINESS ANALYSIS
{business_analysis}


REVIEW OBJECTIVES

1. FACTUAL CONSISTENCY

Check whether the business analysis is consistent with the
research findings and data findings.

2. NUMERICAL VALIDATION

Independently verify important numerical calculations.

Use:

Percentage Change =
((New Value - Old Value) / Old Value) * 100

Check every important percentage reported by previous agents.

If a number is incorrect, explicitly state:

- Reported value
- Correct value
- Source values

3. DATA GROUNDING

Identify claims that are not directly supported by the
provided data.

4. RESEARCH GROUNDING

Check whether research findings actually support the claims
made in the business analysis.

Research about a general business relationship must NOT be
treated as proof that the same relationship caused the
specific problem in this business.

5. CAUSALITY

Do not allow correlation or association to be presented as
causation.

For example:

Incorrect:
"Staff reduction caused resolution time to increase."

Acceptable:
"Staff reduction coincided with an increase in resolution time
and may be a contributing factor."

6. UNSUPPORTED CLAIMS

Identify claims presented as facts when they should instead
be treated as:

- inference
- hypothesis
- possible explanation

7. MISSING INFORMATION

Identify the additional information required to establish
stronger causal conclusions.

8. REPORT COMPLETENESS

Check whether the analysis contains sufficient information
for:

- Key findings
- Evidence
- Inferences
- Hypotheses
- Business impact
- Recommended actions
- Further investigation
- Limitations


REVIEW RULES

- Do not invent facts.
- Do not invent statistics.
- Do not introduce external evidence.
- Use only the information provided above.
- Independently verify numerical calculations.
- Be critical.
- Do not convert correlation into causation.
- Numerical errors must be explicitly reported.
- Unsupported claims must be explicitly reported.
- Important evidence gaps must be explicitly reported.


FINAL REVIEW STATUS

Choose exactly ONE:

PASS
PASS_WITH_WARNINGS
FAIL

Use:

PASS
when the analysis is factually grounded, numerically consistent,
and does not contain important unsupported claims.

PASS_WITH_WARNINGS
when the analysis is mostly correct but contains minor issues,
hypotheses, or evidence gaps that do not require a full revision.

FAIL
when there are major numerical errors, unsupported conclusions,
serious grounding problems, or substantial missing information
that requires revision.


Return the review using exactly this format:

OVERALL ASSESSMENT:
<brief assessment>

STRENGTHS:
- <strength>
- <strength>

ISSUES IDENTIFIED:
- <issue>
- <issue>

UNSUPPORTED CLAIMS:
- <claim>
- <claim>

MISSING INFORMATION:
- <missing information>
- <missing information>

RECOMMENDED CORRECTIONS:
- <correction>
- <correction>

FINAL REVIEW STATUS:
<PASS or PASS_WITH_WARNINGS or FAIL>
"""

    response = llm.invoke(prompt)

    review_text = response.content.strip()

    # Extract the review status from the controlled final section.
    review_status = "PASS_WITH_WARNINGS"

    if "FINAL REVIEW STATUS:" in review_text:
        status_part = review_text.split(
            "FINAL REVIEW STATUS:",
            1,
        )[1].strip()

        first_line = status_part.splitlines()[0].strip()

        if first_line in {
            "PASS",
            "PASS_WITH_WARNINGS",
            "FAIL",
        }:
            review_status = first_line

    return {
        "review": {
            "analysis": review_text,
            "review_status": review_status,
        },
        "status": "REVIEW_COMPLETED",
    }