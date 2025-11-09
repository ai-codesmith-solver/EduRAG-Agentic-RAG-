from langchain.tools import tool
from serpapi import GoogleSearch
from dotenv import load_dotenv
import os

load_dotenv()

@tool
def fetch_web_context(query: str, num_results: int = 3):
    """
    Fetch top Google search results using SerpAPI.

    Performs a real-time web search for the given query and returns a
    formatted text block containing titles, links, and snippets of the
    top results — useful for providing fresh context to EduRAG.

    Args:
        query (str): The topic or question to search.
        num_results (int): Number of results to fetch (default: 3).

    Returns:
        str: Combined search results with titles, URLs, and snippets.
    """

    params = {
        "engine": "google",
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "num": num_results
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    context = ""
    for i, res in enumerate(results.get("organic_results", []), 1):
        context += f"[Result {i}] {res.get('title')} - {res.get('link')}\n{res.get('snippet')}\n\n"
    return context