from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import tools_condition

from agent.state import AgentState
from agent.config import MODEL_NAME, MAX_RETRIES
from agent.prompts import SYSTEM_PROMPT, REFLECTION_PROMPT
from mcp_server.tools import (
    read_file,
    list_files,
    search_files,
    delete_file,
    query_database,
    update_record,
    ToolError,
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

tools_by_name = {
    "tool_read_file": read_file,
    "tool_list_files": list_files,
    "tool_search_files": search_files,
    "tool_delete_file": delete_file,
    "tool_query_database": query_database,
    "tool_update_record": update_record,
}

llm = ChatAnthropic(model=MODEL_NAME)
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: AgentState) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def execute_tools(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    name = tool_call["name"]
    args = tool_call["args"]
    call_id = tool_call["id"]
    trace = list(state.get("reasoning_trace", []))
    step = len(trace) + 1

    try:
        result = tools_by_name[name](**args)
        trace.append({"step": step, "node": "execute_tools", "decision": f"Tool {name} succeeded"})
        return {"messages": [ToolMessage(content=str(result), tool_call_id=call_id)], "retry_count": 0, "reasoning_trace": trace}
    except ToolError as e:
        retry_count = state.get("retry_count", 0) + 1
        if retry_count > MAX_RETRIES:
            trace.append({"step": step, "node": "execute_tools", "decision": f"Max retries reached for {name}"})
            content = f"Tool failed after retries: {e.message}. I'll explain what happened to the user."
        else:
            trace.append({"step": step, "node": "execute_tools", "decision": f"Tool {name} failed: {e.error_type} - will reflect"})
            content = REFLECTION_PROMPT.format(tool_name=name, tool_args=args, error_message=e.message)
        return {"messages": [ToolMessage(content=content, tool_call_id=call_id)], "retry_count": retry_count, "reasoning_trace": trace}


builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", execute_tools)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

app = builder.compile()
