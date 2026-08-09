from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
#from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from typing import Any
import requests
import math
import os
from langchain_core.globals import set_verbose
from langgraph.checkpoint.memory import MemorySaver #Stores state inside the ram
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3 #already in python

from langgraph.types import interrupt, Command
#import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from dotenv import load_dotenv

#from langchain_core.tools import tool
#from langgraph.graph import StateGraph, START
from typing import Annotated, TypedDict
#from langgraph.graph.message import add_messages
#from langchain_core.messages import HumanMessage, BaseMessage
#from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
llm = ChatOpenAI(temperature = 0.7)
embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")



def ingestion_pipeline(file_path):
    loader = PyPDFLoader(file_path= file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    chunks = splitter.split_documents(docs)

    vector_store = Chroma.from_documents(
        documents= chunks,
        embedding= embeddings,
        persist_directory= "./rag_db",
        collection_metadata= {"hnsw:space": "cosine"}
        )

    # return vector_store

def retriever_pipeline():
    vector_store = Chroma(persist_directory= "./rag_db", embedding_function = embeddings)
    retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs = {"k": 4})
    return retriever


@tool
def rag_tool(query: str)-> str:
    """
    Retrieve relevant information from the PDF document.

    Use this tool when the user asks factual or conceptual questions
    that may be answered using the stored PDF documents.

    Args:
        query: The question or search query used to retrieve PDF content.
    """
    retriever = retriever_pipeline()
    documents = retriever.invoke(query)

    if not documents:
        return "No relevant information was found in the PDF."

    formatted_documents = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")

        formatted_documents.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(formatted_documents)


#Tools 
set_verbose(True)
search_tool = TavilySearch(
    max_results = 5,
    topic = "general",
    search_depth = "advanced"
)

@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """

    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=VS375O2SG6ABMZCY"
    r = requests.get(url)
    return r.json()

@tool
def purchase_stock(symbol: str, quantity: int)-> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    NOTE: This is a mock implementation:
    - No real brokerage API is called.
    - It simply returns a confirmation payload.
    """
    decision = interrupt(f"Approve this purchase of {quantity} stocks of {symbol}? yes/no")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    else:
        return {
                    "status": "cancelled",
                    "message": f"Purchase order was declined by human for {quantity} shares of {symbol}.",
                    "symbol": symbol,
                    "quantity": quantity,
                }


#Bind tools to LLM
tools = [search_tool, calculator, get_stock_price, rag_tool, purchase_stock]
llm_with_tools = llm.bind_tools(tools)

#state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot with access to several tools.\n\n"

            "Tool usage instructions:\n"
            "- Use `rag_tool` for questions about the uploaded PDF or document. "
            "Always retrieve relevant document content before answering PDF-related questions.\n"
            "- Use `search_tool` for current events, recent information, or information "
            "that requires an internet search.\n"
            "- Use `calculator` for mathematical calculations. Do not calculate complex "
            "expressions manually when the calculator is available.\n"
            "- Use `get_stock_price` when the user asks for the current price of a stock.\n" 

            "Answer general questions directly when no tool is required. "
            "Do not invent information from the uploaded document. "
            "If the user asks about a PDF but no document is available, ask them to upload a PDF. "
            "After receiving a tool result, provide a clear and helpful final answer."
        )
    )

    messages = [
        system_message,
        *state["messages"]
    ]
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}

tool_node = ToolNode(tools)  # Executes tool calls

conn = sqlite3.connect(database= "chatbot.db", check_same_thread=False) ##sqlite doesnt support multithreading by default but setting to False allows for multiple threads
checkpoint = SqliteSaver(conn)
# graph structure
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")

# If the LLM asked for a tool, go to ToolNode; else finish
graph.add_conditional_edges("chat_node", tools_condition)

graph.add_edge("tools", "chat_node")    
chatbot = graph.compile(checkpointer= checkpoint)

def get_all_threads():
    all_threads = set()
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config['configurable']['thread_id'])
    return list(all_threads)

