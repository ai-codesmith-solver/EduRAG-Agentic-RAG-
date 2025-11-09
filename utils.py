from langchain_docling import DoclingLoader
from langchain_community.document_loaders import TextLoader,UnstructuredMarkdownLoader,PyMuPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from serpapi import GoogleSearch
from dotenv import load_dotenv
import os


load_dotenv()


def get_llm():
    return ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API_KEY"))

def load_docs(path:str):
    if path.endswith('.txt'):
        loader=TextLoader(file_path=path,encoding='utf-8')
    elif path.endswith(".pdf"):
        loader = PyMuPDFLoader(file_path=path)
    elif path.endswith(".csv"):
        loader = DoclingLoader(file_path=path)
    elif path.endswith(".docx"):
        loader = DoclingLoader(file_path=path)
    elif path.startswith("http://") or path.startswith("https://"):
        loader = DoclingLoader(file_path=path)
    elif path.endswith(".md"):
        loader = UnstructuredMarkdownLoader(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    
    return loader.load()

def get_retrive(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


def fetch_pyq_context(query: str, num_results: int = 5):
    """
    Fetches real past-year exam questions (PYQs) related to the user's query
    using Google Search via SerpAPI.

    Args:
        query (str): The student's input question (e.g., "photosynthesis PYQs", "Newton's laws past year questions").
        num_results (int): Number of search results to retrieve.

    Returns:
        str: A formatted string containing summarized snippets of potential PYQ sources.
    """

    # 🔍 Enhanced query — focuses search on real PYQ sources
    search_query = (
        f"past year questions OR previous year papers OR PYQs "
        f"related to {query} site:byjus.com OR site:examrace.com "
        f"OR site:careers360.com OR site:toppr.com OR site:cbseacademic.nic.in"
    )

    params = {
        "engine": "google",
        "q": search_query,
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "num": num_results
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    if not results.get("organic_results"):
        return "No relevant past-year questions found for this topic."

    # 🧠 Build a contextual summary string
    context = f"🧾 Web Search Results for Past Year Questions on: {query}\n\n"
    for i, res in enumerate(results.get("organic_results", []), 1):
        title = res.get("title", "Untitled")
        link = res.get("link", "No link available")
        snippet = res.get("snippet", "")
        context += f"[Result {i}] {title}\n🔗 {link}\n📘 {snippet}\n\n"

    return context.strip()