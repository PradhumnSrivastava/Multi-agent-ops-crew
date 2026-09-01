from multi_agent_ops.llm import create_llm
from multi_agent_ops.state import OpsState, ResearchSource
from multi_agent_ops.tools.research_tools import web_search


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

Analyze the following business problem using the
provided web search results.

Business Problem:
{problem}

Web Search Results:
{search_results}

Your task is to produce a concise research analysis.

Include:

1. Key findings
2. Relevant factors that may explain the problem
3. Evidence from the provided search results
4. Information that still needs investigation
5. Limitations of this analysis

Important rules:

- Do not invent facts.
- Do not invent statistics.
- Do not invent sources.
- Only use the provided search results as external evidence.
- Clearly distinguish evidence from possible hypotheses.
- Do not treat a possible explanation as a proven cause.
- If the search results do not provide enough evidence,
  explicitly state that more investigation is required.

Return a professional research analysis suitable for
downstream business analysis.
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