from langchain_huggingface import ChatHuggingFace

from multi_agent_ops.agents.research import create_llm
from multi_agent_ops.state import OpsState


def create_review_llm() -> ChatHuggingFace:
    """Create the Hugging Face model used by the review agent."""

    return create_llm()


llm = create_review_llm()


def review_agent(state: OpsState) -> dict:
    """Review research, data analysis, and business analysis."""

    problem = state["problem"]
    research_findings = state["research_findings"]
    data_findings = state["data_findings"]
    business_analysis = state["business_analysis"]

    prompt = f"""
You are a senior quality reviewer in a multi-agent
business operations system.

Your responsibility is to independently verify the work
produced by the research, data analysis, and business
analysis agents.

Business Problem:
{problem}

Research Findings:
{research_findings}

Data Findings:
{data_findings}

Business Analysis:
{business_analysis}


PERFORM THESE CHECKS:

1. FACTUAL CONSISTENCY

Check whether statements in the business analysis are
consistent with the research findings and available data.


2. NUMERICAL VALIDATION

Independently calculate important percentage changes
from the raw data.

For example:

Percentage change =
((new value - old value) / old value) * 100

Compare your calculations with every percentage reported
by the previous agents.

If a reported number is incorrect, explicitly identify:

- reported value
- correct value
- source values used for calculation


3. DATA-SUPPORTED CONCLUSIONS

Check whether the conclusions are actually supported
by the available data.

Do not treat correlation as proof of causation.


4. RESEARCH EVIDENCE

Check whether research findings actually support the
claims made in the business analysis.

Do not assume that a research article proves that the
same factor caused the problem in this specific business.


5. UNSUPPORTED CLAIMS

Identify statements that are hypotheses rather than
established facts.


6. MISSING INFORMATION

Identify information that would be required to establish
stronger causal conclusions.


7. OVERALL QUALITY

Determine whether the analysis is:

PASS
PASS_WITH_WARNINGS
FAIL


IMPORTANT RULES:

- Do not invent facts.
- Do not invent statistics.
- Do not introduce external evidence.
- Use only the information provided in the state.
- Independently verify numerical calculations.
- Be critical.
- Clearly distinguish evidence, inference, and hypothesis.
- If you find a numerical error, explicitly flag it.


Return the review using exactly these sections:

OVERALL ASSESSMENT:

STRENGTHS:
- ...

ISSUES IDENTIFIED:
- ...

UNSUPPORTED CLAIMS:
- ...

MISSING INFORMATION:
- ...

RECOMMENDED CORRECTIONS:
- ...

FINAL REVIEW STATUS:
PASS
or
PASS_WITH_WARNINGS
or
FAIL

Do not use JSON.
Do not use code blocks.
Do not add additional sections.
"""

    response = llm.invoke(prompt)

    return {
        "review": {
            "analysis": response.content,
        },
        "status": "REVIEW_COMPLETED",
    }