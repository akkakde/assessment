import json
import argparse
from unittest.mock import patch
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent.config import USE_MOCK_LLM
from agent.graph import builder


def _build_mock_llm(user_message: str):
    """Return a callable that the mock LLM will use.

    First call always issues a tool call. Second call sees the ToolMessage
    result (including any error/reflection) and returns a final reply.
    """
    msg = user_message.lower()

    if "old_backup" in msg:
        calls = [
            AIMessage(content="I'll delete /data/old_backup.csv first. This is a destructive action — requesting your approval.", tool_calls=[ToolCall(name="tool_delete_file", args={"path": "/data/old_backup.csv"}, id="m0a")]),
            AIMessage(content="The file doesn't exist. Let me search to confirm.", tool_calls=[ToolCall(name="tool_search_files", args={"query": "old_backup"}, id="m0b")]),
            AIMessage(content="Confirmed — no old_backup file found. Now listing reports.", tool_calls=[ToolCall(name="tool_list_files", args={"directory": "/data/reports/"}, id="m0c")]),
            AIMessage(content="Here's what I found:\n\n1. The file /data/old_backup.csv doesn't exist — it may have been previously deleted.\n2. The latest report in /data/reports/ is q4_2025.csv.\n\nWould you like me to read the Q4 2025 report?"),
        ]
        idx = [0]
        def multi_step(messages):
            r = calls[idx[0]]
            idx[0] = min(idx[0] + 1, len(calls) - 1)
            return r
        return multi_step
    elif "list" in msg:
        first_call = AIMessage(content="", tool_calls=[ToolCall(name="tool_list_files", args={"directory": "/data/"}, id="m1")])
    elif "delete" in msg:
        first_call = AIMessage(content="", tool_calls=[ToolCall(name="tool_delete_file", args={"path": "/data/users.csv"}, id="m6")])
    elif "update" in msg:
        first_call = AIMessage(content="", tool_calls=[ToolCall(name="tool_update_record", args={"record_id": 1, "field": "status", "value": "inactive"}, id="m7")])
    elif "nonexistent" in msg or "impossible" in msg:
        first_call = AIMessage(content="", tool_calls=[ToolCall(name="tool_read_file", args={"path": "/data/nonexistent.txt"}, id="m2")])
    elif "read" in msg:
        first_call = AIMessage(content="", tool_calls=[ToolCall(name="tool_read_file", args={"path": "/data/users.csv"}, id="m3")])
    elif "active" in msg or "database" in msg:
        first_call = AIMessage(content="", tool_calls=[ToolCall(name="tool_query_database", args={"sql_like": "SELECT * FROM users WHERE status='active'"}, id="m4")])
    elif "search" in msg:
        first_call = AIMessage(content="", tool_calls=[ToolCall(name="tool_search_files", args={"query": msg.split()[-1]}, id="m5")])
    else:
        def direct_reply(messages):
            return AIMessage(content="I can help with file and database operations. Try asking me to list files or query the database.")
        return direct_reply

    invoke_count = [0]

    def mock_invoke(messages):
        invoke_count[0] += 1
        if invoke_count[0] == 1:
            return first_call
        tool_result = next((m.content for m in reversed(messages) if isinstance(m, ToolMessage)), "")
        if "denied" in tool_result.lower():
            return AIMessage(content="Understood. The action was cancelled as requested.")
        if "failed" in tool_result.lower() or "Analyze" in tool_result:
            return AIMessage(content=f"I tried to access the file but it doesn't exist. Here's what the tool reported:\n\n{tool_result}\n\nYou may want to use 'list files' or 'search files' to find what you're looking for.")
        return AIMessage(content=f"Here is the result:\n\n{tool_result}")

    return mock_invoke


def _serialize_messages(messages):
    out = []
    for m in messages:
        role = getattr(m, "type", "unknown")
        entry = {"role": role, "content": m.content}
        if getattr(m, "tool_calls", None):
            entry["tool_calls"] = [{"name": tc["name"], "args": tc["args"]} for tc in m.tool_calls]
        out.append(entry)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Path to input JSON file")
    parser.add_argument("--save", action="store_true", help="Save output to examples/output.json")
    parser.add_argument("message", nargs="?", default="List all files in /data/")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            input_data = json.load(f)
        user_message = input_data["messages"][0]["content"]
    else:
        user_message = args.message
        input_data = {"messages": [{"role": "user", "content": user_message}]}

    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "cli-1"}}
    mock_callable = _build_mock_llm(user_message) if USE_MOCK_LLM else None

    def invoke(payload):
        if USE_MOCK_LLM:
            with patch("agent.graph.llm_with_tools") as mock_llm:
                mock_llm.invoke.side_effect = mock_callable
                return graph.invoke(payload, config)
        return graph.invoke(payload, config)

    result = invoke(input_data)

    while graph.get_state(config).next:
        interrupt_val = graph.get_state(config).tasks[0].interrupts[0].value
        print(f"\n{interrupt_val['message']}")
        user_input = input("Your decision: ").strip()
        result = invoke(Command(resume=user_input))

    print(result["messages"][-1].content)

    if args.save:
        output = {
            "messages": _serialize_messages(result["messages"]),
            "reasoning_trace": result.get("reasoning_trace", []),
        }
        with open("examples/output.json", "w") as f:
            json.dump(output, f, indent=2)
        print("\nOutput saved to examples/output.json")


if __name__ == "__main__":
    main()
