import json
import argparse
from unittest.mock import patch
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import ToolCall

from agent.config import USE_MOCK_LLM
from agent.graph import app


def _build_mock_llm(user_message: str):
    """Return a callable that the mock LLM will use.

    First call always issues a tool call. Second call sees the ToolMessage
    result (including any error/reflection) and returns a final reply.
    """
    msg = user_message.lower()

    # Decide which tool call to make based on the user message
    if "list" in msg:
        first_call = AIMessage(content="", tool_calls=[ToolCall(name="tool_list_files", args={"directory": "/data/"}, id="m1")])
    elif "nonexistent" in msg or "impossible" in msg:
        # Extract path from message or use a clearly missing one
        first_call = AIMessage(content="", tool_calls=[ToolCall(name="tool_read_file", args={"path": "/data/nonexistent.txt"}, id="m2")])
    elif "read" in msg:
        first_call = AIMessage(content="", tool_calls=[ToolCall(name="tool_read_file", args={"path": "/data/users.csv"}, id="m3")])
    elif "active" in msg or "database" in msg:
        first_call = AIMessage(content="", tool_calls=[ToolCall(name="tool_query_database", args={"sql_like": "SELECT * FROM users WHERE status='active'"}, id="m4")])
    elif "search" in msg:
        first_call = AIMessage(content="", tool_calls=[ToolCall(name="tool_search_files", args={"query": msg.split()[-1]}, id="m5")])
    else:
        # No tool needed — reply directly
        def direct_reply(messages):
            return AIMessage(content="I can help with file and database operations. Try asking me to list files or query the database.")
        return direct_reply

    invoke_count = [0]

    def mock_invoke(messages):
        invoke_count[0] += 1
        if invoke_count[0] == 1:
            return first_call
        # Second call: read the ToolMessage content and compose a final reply
        tool_result = next((m.content for m in reversed(messages) if isinstance(m, ToolMessage)), "")
        if "failed" in tool_result.lower() or "wrong" in tool_result.lower() or "Analyze" in tool_result:
            return AIMessage(content=f"I tried to access the file but it doesn't exist. Here's what the tool reported:\n\n{tool_result}\n\nYou may want to use 'list files' or 'search files' to find what you're looking for.")
        return AIMessage(content=f"Here is the result:\n\n{tool_result}")

    return mock_invoke


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Path to input JSON file")
    parser.add_argument("message", nargs="?", default="List all files in /data/")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            input_data = json.load(f)
        user_message = input_data["messages"][0]["content"]
    else:
        user_message = args.message
        input_data = {"messages": [{"role": "user", "content": user_message}]}

    if USE_MOCK_LLM:
        with patch("agent.graph.llm_with_tools") as mock_llm:
            mock_llm.invoke.side_effect = _build_mock_llm(user_message)
            result = app.invoke(input_data)
    else:
        result = app.invoke(input_data)

    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
