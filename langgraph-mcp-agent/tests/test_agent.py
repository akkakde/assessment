from unittest.mock import patch
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.tool import ToolCall
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent.graph import execute_tools, builder
from agent.config import MAX_RETRIES


def _state(tool_name, args, call_id="c1", retry_count=0, trace=None):
    """Build a minimal AgentState with a single pending tool call."""
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[ToolCall(name=tool_name, args=args, id=call_id)],
            )
        ],
        "retry_count": retry_count,
        "reasoning_trace": trace or [],
    }


# --- happy path ---

def test_happy_path():
    state = _state("tool_list_files", {"directory": "/data/reports/"})
    result = execute_tools(state)

    content = result["messages"][0].content
    assert "q4_2025" in content
    assert "q3_2025" in content
    assert result["retry_count"] == 0
    assert any("succeeded" in e["decision"] for e in result["reasoning_trace"])


# --- tool failure recovery ---

def test_tool_failure_recovery():
    state = _state("tool_read_file", {"path": "/data/nonexistent.csv"})
    result = execute_tools(state)

    content = result["messages"][0].content
    assert "failed" in content.lower() or "nonexistent" in content.lower() or "wrong" in content.lower()
    assert result["retry_count"] == 1
    assert any("failed" in e["decision"] for e in result["reasoning_trace"])


# --- max retries graceful degradation ---

def test_max_retries_graceful():
    state = _state(
        "tool_read_file",
        {"path": "/impossible/path/file.xyz"},
        retry_count=MAX_RETRIES,
    )
    result = execute_tools(state)

    content = result["messages"][0].content
    assert "failed after retries" in content.lower()
    assert "explain" in content.lower()
    assert any("Max retries" in e["decision"] for e in result["reasoning_trace"])


# --- HITL: destructive action is blocked until approval ---

def test_hitl_blocks_destructive():
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t1"}}
    ai_msg = AIMessage(content="", tool_calls=[ToolCall(name="tool_delete_file", args={"path": "/data/users.csv"}, id="c1")])

    with patch("agent.graph.llm_with_tools") as mock_llm:
        mock_llm.invoke.return_value = ai_msg
        graph.invoke({"messages": [HumanMessage(content="Delete /data/users.csv")]}, config)

    graph_state = graph.get_state(config)
    assert graph_state.next
    interrupt_val = graph_state.tasks[0].interrupts[0].value
    assert "delete_file" in interrupt_val["action"]
    assert "Approve?" in interrupt_val["message"]


# --- HITL: safe actions complete without interruption ---

def test_hitl_allows_safe_actions():
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t2"}}

    with patch("agent.graph.llm_with_tools") as mock_llm:
        mock_llm.invoke.side_effect = [
            AIMessage(content="", tool_calls=[ToolCall(name="tool_read_file", args={"path": "/data/users.csv"}, id="c1")]),
            AIMessage(content="Here is the file content."),
        ]
        result = graph.invoke({"messages": [HumanMessage(content="Read /data/users.csv")]}, config)

    assert not graph.get_state(config).next
    assert not any("approve" in e["node"] for e in result["reasoning_trace"])
