import json
import argparse
from unittest.mock import patch
from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolCall

from agent.config import USE_MOCK_LLM
from agent.graph import app


def _mock_llm(user_message: str):
    """Return a scripted tool call + final reply based on the user message."""
    msg = user_message.lower()

    if "list" in msg:
        return [
            AIMessage(content="", tool_calls=[ToolCall(name="tool_list_files", args={"directory": "/data/"}, id="m1")]),
            AIMessage(content="I found the following files in /data/:\n- /data/reports/q4_2025.csv\n- /data/reports/q3_2025.csv\n- /data/users.csv\n- /data/config.yaml"),
        ]
    if "read" in msg or "users.csv" in msg:
        return [
            AIMessage(content="", tool_calls=[ToolCall(name="tool_read_file", args={"path": "/data/users.csv"}, id="m2")]),
            AIMessage(content="The file /data/users.csv contains 2 users: Alice (admin) and Bob (viewer)."),
        ]
    if "active" in msg or "database" in msg:
        return [
            AIMessage(content="", tool_calls=[ToolCall(name="tool_query_database", args={"sql_like": "SELECT * FROM users WHERE status='active'"}, id="m3")]),
            AIMessage(content="Active users: Alice (admin) and Charlie (editor)."),
        ]
    if "nonexistent" in msg:
        return [
            AIMessage(content="", tool_calls=[ToolCall(name="tool_read_file", args={"path": "/nonexistent/file.txt"}, id="m4")]),
        ]
    return [AIMessage(content="I can help with file and database operations. Try asking me to list files or query the database.")]


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
            mock_llm.invoke.side_effect = _mock_llm(user_message)
            result = app.invoke(input_data)
    else:
        result = app.invoke(input_data)

    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
