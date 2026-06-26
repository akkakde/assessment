import pytest
from contextlib import contextmanager
from unittest.mock import patch
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.tool import ToolCall
from langgraph.checkpoint.memory import MemorySaver

from agent.config import USE_MOCK_LLM
from agent.graph import builder
from mcp_server.mock_data import MOCK_FILESYSTEM, MOCK_DATABASE


def _make_tool_call(name, args, call_id="call_1"):
    return AIMessage(
        content="",
        tool_calls=[ToolCall(name=name, args=args, id=call_id)],
    )


def _make_final_reply(content):
    return AIMessage(content=content)


@contextmanager
def maybe_mock_llm(responses):
    """Patch the LLM if USE_MOCK_LLM is True, otherwise run real."""
    if USE_MOCK_LLM:
        with patch("agent.graph.llm_with_tools") as mock_llm:
            mock_llm.invoke.side_effect = responses
            yield
    else:
        yield


# --- happy path: list files ---

def test_graph_list_files_happy_path():
    with maybe_mock_llm([
        _make_tool_call("tool_list_files", {"directory": "/data/reports/"}, "c1"),
        _make_final_reply("Found 2 files: q4_2025.csv and q3_2025.csv"),
    ]):
        from agent.graph import app
        result = app.invoke({"messages": [HumanMessage(content="List files in /data/reports/")]})

    assert result["messages"][-1].content != ""


# --- happy path: read file ---

def test_graph_read_file_happy_path():
    with maybe_mock_llm([
        _make_tool_call("tool_read_file", {"path": "/data/users.csv"}, "c2"),
        _make_final_reply("The file contains Alice and Bob."),
    ]):
        from agent.graph import app
        result = app.invoke({"messages": [HumanMessage(content="Read /data/users.csv")]})

    assert result["messages"][-1].content != ""


# --- happy path: query database ---

def test_graph_query_database_happy_path():
    with maybe_mock_llm([
        _make_tool_call("tool_query_database", {"sql_like": "SELECT * FROM users"}, "c3"),
        _make_final_reply("There are 3 users: Alice, Bob, Charlie."),
    ]):
        from agent.graph import app
        result = app.invoke({"messages": [HumanMessage(content="Show all users")]})

    assert result["messages"][-1].content != ""


# --- happy path: search files ---

def test_graph_search_files_happy_path():
    with maybe_mock_llm([
        _make_tool_call("tool_search_files", {"query": "Widget"}, "c4"),
        _make_final_reply("Found files containing Widget."),
    ]):
        from agent.graph import app
        result = app.invoke({"messages": [HumanMessage(content="Search for Widget")]})

    assert result["messages"][-1].content != ""


# --- update_record is now destructive: graph pauses for HITL approval ---

def test_graph_update_record_requires_approval():
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "gr-upd"}}
    with patch("agent.graph.llm_with_tools") as mock_llm:
        mock_llm.invoke.return_value = _make_tool_call("tool_update_record", {"record_id": 2, "field": "status", "value": "active"}, "c5")
        graph.invoke({"messages": [HumanMessage(content="Set Bob's status to active")]}, config)

    graph_state = graph.get_state(config)
    assert graph_state.next
    assert "update_record" in graph_state.tasks[0].interrupts[0].value["action"]


# --- ToolError on safe tool is handled via reflection (no crash) ---

def test_graph_crashes_on_tool_error():
    """read_file with missing path is caught by execute_tools and reflected — no crash."""
    with maybe_mock_llm([
        _make_tool_call("tool_read_file", {"path": "/nonexistent/file.txt"}, "c6"),
        _make_final_reply("The file was not found."),
    ]):
        from agent.graph import app
        result = app.invoke({"messages": [HumanMessage(content="Read /nonexistent/file.txt")]})

    assert result["messages"][-1].content != ""


# --- delete_file is destructive: graph pauses for HITL approval ---

def test_graph_delete_requires_approval():
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "gr-del"}}
    with patch("agent.graph.llm_with_tools") as mock_llm:
        mock_llm.invoke.return_value = _make_tool_call("tool_delete_file", {"path": "/data/ghost.csv"}, "c7")
        graph.invoke({"messages": [HumanMessage(content="Delete /data/ghost.csv")]}, config)

    graph_state = graph.get_state(config)
    assert graph_state.next
    assert "delete_file" in graph_state.tasks[0].interrupts[0].value["action"]
