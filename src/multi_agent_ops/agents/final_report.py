import os

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from multi_agent_ops.state import OpsState

load_dotenv()


def create_llm() -> ChatHuggingFace:
    """Create and configure the Hugging Face chat model."""

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError("HF_TOKEN is not configured.")

    endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        provider="featherless-ai",
        task="text-generation",
        max_new_tokens=500,
        temperature=0.1,
        huggingfacehub_api_token=hf_token,
    )

    return ChatHuggingFace(llm=endpoint)


llm = create_llm()


def final_report_agent(state: OpsState) -> dict:
    """Generate the final executive report from all agent findings."""

    problem = state["problem"]
    research_findings = state["research_findings"]
    data_findings = state["data_findings"]
    business_analysis = state["business_analysis"]
    review = state["review"]

    prompt = f"""
You are the final reporting analyst in a multi-agent business
operations system.

Create a concise, evidence-based executive report for the
following business problem.

Business Problem:
{problem}

Research Findings:
{research_findings}

Data Analysis:
{data_findings}

Business Analysis:
{business_analysis}

Review:
{review}

Structure the final report as:

1. Executive Summary
2. Problem
3. Key Findings
4. Data Evidence
5. Business Interpretation
6. Most Likely Root Causes
7. Business Impact
8. Recommended Actions
9. Further Investigation

Rules:

- Use only the information provided above.
- Do not invent facts, statistics, or sources.
- Clearly distinguish evidence from hypotheses.
- Preserve important numerical findings from the data analysis.
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