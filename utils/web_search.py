import logging
from typing import Optional, Dict, List

from config import config

logger = logging.getLogger(__name__)


def tavily_search(query: str, max_results: int = 5) -> List[Dict]:
    if not config.TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY not configured")
        return []
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=config.TAVILY_API_KEY)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_answer=True,
        )
        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
            })
        return results
    except ImportError:
        logger.error("tavily not installed, run: pip install tavily")
        return []
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return []


def serpapi_search(query: str, max_results: int = 5) -> List[Dict]:
    if not config.SERPAPI_API_KEY:
        logger.warning("SERPAPI_API_KEY not configured")
        return []
    try:
        import requests
        params = {
            "api_key": config.SERPAPI_API_KEY,
            "q": query,
            "num": max_results,
            "engine": "google",
        }
        resp = requests.get("https://serpapi.com/search", params=params, timeout=15)
        data = resp.json()
        results = []
        for r in data.get("organic_results", [])[:max_results]:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "content": r.get("snippet", ""),
            })
        return results
    except Exception as e:
        logger.error(f"SerpAPI search failed: {e}")
        return []


def web_search(query: str, max_results: int = 5) -> List[Dict]:
    if config.TAVILY_API_KEY:
        return tavily_search(query, max_results)
    if config.SERPAPI_API_KEY:
        return serpapi_search(query, max_results)
    logger.warning("No web search API key configured (TAVILY_API_KEY or SERPAPI_API_KEY)")
    return []


def format_search_results(results: List[Dict], max_content_length: int = 500) -> str:
    if not results:
        return ""
    parts = []
    for i, r in enumerate(results[:5]):
        title = r.get("title", "")
        content = r.get("content", "")[:max_content_length]
        url = r.get("url", "")
        parts.append(f"[网络来源 {i+1}] {title}\n{content}\n来源: {url}")
    return "\n\n---\n\n".join(parts)
