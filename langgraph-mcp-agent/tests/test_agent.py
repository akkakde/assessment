from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.tool import ToolCall

from agent.graph import execute_tools
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
    # agent should not crash and message should contain reflection context
    assert "failed" in content.lower() or "nonexistent" in content.lower() or "wrong" in content.lower()
    assert result["retry_count"] == 1
    assert any("failed" in e["decision"] for e in result["reasoning_trace"])


# --- max retries graceful degradation ---

def test_max_retries_graceful():
    # simulate retry_count already at MAX_RETRIES so next failure hits the limit
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
