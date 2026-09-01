from ddgs import DDGS


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web and return relevant results."""

    with DDGS() as ddgs:
        results = ddgs.text(
            query,
            max_results=max_results,
        )

    return [
        {
            "title": result.get("title"),
            "url": result.get("href"),
            "snippet": result.get("body"),
        }
        for result in results
    ]