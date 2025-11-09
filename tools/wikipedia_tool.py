from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool

@tool
def get_query_info(query:str)->str:
    """
    Fetch concise information from Wikipedia for a given student query.

    This function takes a query (usually a topic, concept, or question from a student),
    searches Wikipedia using the LangChain `WikipediaQueryRun` tool, and returns
    a summarized text result that can be used by EduRAG for explanation or reasoning.

    Parameters:
        query (str): The topic or question entered by the student.

    Returns:
        str: A concise, readable summary of the Wikipedia search result related to the query.
    """

    
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    result=wikipedia.run(query)
    return result