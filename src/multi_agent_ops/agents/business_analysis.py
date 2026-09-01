from multi_agent_ops.llm import create_llm
from multi_agent_ops.state import OpsState


llm = create_llm(max_new_tokens=400)


def business_analysis_agent(state: OpsState) -> dict:
    """Interpret research and data findings from a business perspective."""

    problem = state["problem"]
    research_findings = state["research_findings"]
    data_findings = state["data_findings"]

    prompt = f"""
You are a business operations analyst working as part of a
multi-agent operations system.

Analyze the business problem using the research findings and
data analysis provided below.

Business Problem:

{problem}

Research Findings:

{research_findings}

Data Findings:

{data_findings}

Provide:

1. Business interpretation of the data
2. Most likely contributing factors
3. Relationship between the research evidence and data
4. Potential business impact
5. Recommended areas for further investigation

Important rules:

- Do not invent facts or statistics.
- Use only the provided research findings and data findings.
- Clearly distinguish observed data from hypotheses.
- Do not perform calculations that are not supported by the provided data.
- Do not claim causation when the data only shows correlation.
- If the available evidence is insufficient to establish a cause,
  explicitly state that.
- Do not introduce external evidence.
- Keep the analysis concise and business-focused.
"""

    response = llm.invoke(prompt)

    return {
        "business_analysis": {
            "analysis": response.content,
        },
        "status": "BUSINESS_ANALYSIS_COMPLETED",
    }