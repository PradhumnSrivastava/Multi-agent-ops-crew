import os

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from multi_agent_ops.llm import create_llm
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
    """Generate a review-aware executive report."""

    problem = state["problem"]
    research_findings = state["research_findings"]
    data_findings = state["data_findings"]
    business_analysis = state["business_analysis"]
    review = state["review"]

    prompt = f"""
You are the final reporting analyst in a multi-agent
business operations system.

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

The quality review is the final validation layer before reporting.

You must carefully apply the reviewer's findings.

If the reviewer identifies an unsupported claim:
- Do not present it as an established fact.

If the reviewer identifies a numerical inconsistency:
- Treat Data Findings as the authoritative numerical source.
- Verify the calculation before reporting it.

If the reviewer identifies an unsupported causal claim:
- Present it as a possible explanation, hypothesis, or association.
- Do not present correlation as causation.

If the reviewer identifies missing information:
- Include the relevant limitation or investigation requirement
  in the final report.

Do not hide important warnings identified by the reviewer.

Classify conclusions into three categories:

Evidence:
Directly supported by the provided data or research.

Inference:
A reasonable interpretation of the available evidence.

Hypothesis:
A possible explanation that requires further validation.

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
- Do not ignore review warnings.
- Keep the report concise and professional.
- Write for a business decision-maker.

Return only the final executive report.
"""

    response = llm.invoke(prompt)

    return {
        "final_report": {
            "report": response.content,
        },
        "status": "FINAL_REPORT_COMPLETED",
    }