from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from agent.state import AgentState
from agent.config import MODEL_NAME
from agent.prompts import SYSTEM_PROMPT
from mcp_server.tools import (
    read_file,
    list_files,
    search_files,
    delete_file,
    query_database,
    update_record,
)


@tool
def tool_read_file(path: str) -> str:
    """Read the contents of a file at the given path."""
    return read_file(path)


@tool
def tool_list_files(directory: str) -> str:
    """List all files under the given directory prefix."""
    return list_files(directory)


@tool
def tool_search_files(query: str) -> str:
    """Search file paths and contents for a query string (case-insensitive)."""
    return search_files(query)


@tool
def tool_delete_file(path: str) -> str:
    """Delete a file at the given path."""
    return delete_file(path)


@tool
def tool_query_database(sql_like: str) -> str:
    """Execute a simple SELECT query against the mock user database."""
    return query_database(sql_like)


@tool
def tool_update_record(record_id: int, field: str, value: str) -> str:
    """Update a field on a database record by id."""
    return update_record(record_id, field, value)


tools = [
    tool_read_file,
    tool_list_files,
    tool_search_files,
    tool_delete_file,
    tool_query_database,
    tool_update_record,
]

llm = ChatAnthropic(model=MODEL_NAME)
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: AgentState) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ToolNode runs tool calls as-is — no try/except, ToolError will crash the graph
tool_node = ToolNode(tools)

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

app = builder.compile()
