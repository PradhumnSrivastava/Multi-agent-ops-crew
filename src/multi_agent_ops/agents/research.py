import os

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from multi_agent_ops.state import OpsState, ResearchSource
from multi_agent_ops.tools.research_tools import web_search

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
        max_new_tokens=300,
        temperature=0.1,
        huggingfacehub_api_token=hf_token,
    )

    return ChatHuggingFace(llm=endpoint)


llm = create_llm()


def research_agent(state: OpsState) -> dict:
    """Research the business problem using web search and an LLM."""

    problem = state["problem"]

    search_results = web_search(
        problem,
        max_results=5,
    )

    sources = [
        ResearchSource(
            title=result.get("title", ""),
            url=result.get("url", ""),
            snippet=result.get("snippet", ""),
        )
        for result in search_results
    ]

    prompt = f"""
You are a research analyst working as part of a
multi-agent business operations system.

Analyze the following business problem using ONLY
the provided web search results.

Business Problem:
{problem}

Web Search Results:
{search_results}

Your task is to produce a concise research analysis.

Include:

1. Key Findings
2. Relevant Factors
3. Evidence from the Search Results
4. Information Still Requiring Investigation
5. Limitations

Important rules:

- Use only the provided search results as external evidence.
- Do not invent facts.
- Do not invent statistics.
- Do not invent sources.
- Do not claim that a hypothesis is a proven cause.
- Clearly distinguish evidence, inference, and hypothesis.
- If the search results do not provide enough evidence,
  explicitly say so.
- Keep the analysis concise and business-focused.
"""

    response = llm.invoke(prompt)

    return {
        "research_findings": {
            "analysis": response.content,
            "sources": [
                source.model_dump()
                for source in sources
            ],
        },
        "status": "RESEARCH_COMPLETED",
    }