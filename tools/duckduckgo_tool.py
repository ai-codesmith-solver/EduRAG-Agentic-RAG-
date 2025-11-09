from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import tool

@tool
def get_web_serach(topic:str)->str:
    """Searches the web for a topic and returns results as text."""

    search_web=DuckDuckGoSearchRun()
    result=search_web.invoke(topic)
    return result