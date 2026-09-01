import os

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from multi_agent_ops.state import OpsState


load_dotenv()


def create_llm() -> ChatHuggingFace:
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError("HF_TOKEN is not configured.")

    endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        provider="featherless-ai",
        task="text-generation",
        max_new_tokens=300,
        temperature=0.1,
        huggingfacehub_api_token=hf_token,
    )

    return ChatHuggingFace(llm=endpoint)


llm = create_llm()


def research_agent(state: OpsState) -> dict:
    problem = state["problem"]

    prompt = f"""
You are a research analyst.

Analyze the following business problem and provide a concise
research-oriented response.

Problem:
{problem}

Return:
1. Key areas that should be investigated
2. Relevant factors that may explain the problem
3. Important information that would be needed
4. Limitations of this initial analysis

Do not invent factual sources or statistics.
"""

    response = llm.invoke(prompt)

    return {
        "research_findings": {
            "analysis": response.content,
        },
        "status": "RESEARCH_COMPLETED",
    }