import os
import logging
from typing import Annotated, TypedDict
import httpx
import atexit
import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langchain_core.runnables import RunnableConfig

from .tools_voice import tools
from .memory import memory
from prompts.voice_agent_prompt import get_system_prompt

# -------------------------------------------------------------------
# SETUP
# -------------------------------------------------------------------

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_TOKENS = 1000
MAX_STEPS = 6
TIMEOUT = 30

# Validate env
if not os.getenv("LLM_API_KEY"):
    raise ValueError("LLM_API_KEY not set")

# LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Production - Petes Inn Resort"

# -------------------------------------------------------------------
# LLM
# -------------------------------------------------------------------

_http_client = httpx.AsyncClient(
    limits=httpx.Limits(
        max_connections=10,
        max_keepalive_connections=5,
        keepalive_expiry=30,
    ),
    timeout=httpx.Timeout(
        connect=5.0,
        read=TIMEOUT,
        write=10.0,
        pool=5.0,
    ),
)


llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4.1-mini",
    temperature=0.4,
    timeout=TIMEOUT,
    max_retries=2,
    max_tokens=500,
    http_async_client=_http_client,
)
llm_with_tools = llm.bind_tools(tools)

atexit.register(lambda: asyncio.get_event_loop().run_until_complete(_http_client.aclose()))

# -------------------------------------------------------------------
# STATE
# -------------------------------------------------------------------

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------------------------------------------------------
# MESSAGE FILTERING
# -------------------------------------------------------------------

def filter_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Keep only content, tool_calls, and tool results"""
    filtered = []
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            filtered.append(HumanMessage(content=msg.content))
        
        elif isinstance(msg, AIMessage):
            filtered.append(AIMessage(
                content=msg.content or "",
                tool_calls=getattr(msg, 'tool_calls', []),
            ))
        
        elif isinstance(msg, ToolMessage):
            filtered.append(ToolMessage(
                content=msg.content or "",
                tool_call_id=msg.tool_call_id,
                name=msg.name,
            ))
    
    return filtered

# -------------------------------------------------------------------
# LLM NODE
# -------------------------------------------------------------------

def llm_node(state: State, config: RunnableConfig) -> State:
    """Main LLM logic with token management"""
    
    try:
        sender_id = config["configurable"].get("thread_id", "unknown")
        step_count = config["configurable"].get("step_count", 0)
        
        # Prevent infinite loops
        if step_count >= MAX_STEPS:
            logger.warning(f"Max steps reached for {sender_id}")
            return {
                "messages": [AIMessage(
                    content="I'm having trouble completing this. Can you simplify your request?"
                )]
            }
        
        # Filter first, then trim
        clean = filter_messages(state["messages"])
        
        trimmed = trim_messages(
            clean,
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=MAX_TOKENS,
            start_on="human",
            end_on=("human", "tool"),
        )
        

        
        
        # Add system prompt
        system_prompt = get_system_prompt()
        conversation = [SystemMessage(content=system_prompt)] + trimmed
        
        # Log token usage
        tokens = count_tokens_approximately(conversation)
        logger.info(f"Tokens: {tokens} | Step: {step_count + 1} | User: {sender_id}")
        
        # Call LLM

        response = llm_with_tools.invoke(conversation)
        
        
        return {"messages": [response]}
    
    except TimeoutError:
        logger.error(f"Timeout for {sender_id}")
        return {"messages": [AIMessage(content="Request timed out. Try again?")]}
    
    except Exception as e:
        logger.error(f"Error: {type(e).__name__}: {e}")
        return {"messages": [AIMessage(content="An error occurred. Please try again.")]}

# -------------------------------------------------------------------
# GRAPH
# -------------------------------------------------------------------

builder = StateGraph(State)
builder.add_node("LLM", llm_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "LLM")
builder.add_conditional_edges("LLM", tools_condition)
builder.add_edge("tools", "LLM")

graph = builder.compile(checkpointer=memory)

logger.info("✅ Graph ready")
