# LangGraph MCP Agent

A LangGraph agent that interacts with a mock file system and database through
MCP-style tool functions, with intelligent error recovery and human-in-the-loop
safety gates.

## What this solves

- **Tool errors** (FileNotFound, QueryError, etc.) are handled with a reflection
  loop — the agent reasons about the failure and tries an alternative approach
- **Destructive actions** (delete_file, update_record, and others) require human
  approval before execution via LangGraph's `interrupt()` mechanism
- **All decisions** are logged in a `reasoning_trace` for explainability

## Project structure

```
langgraph-mcp-agent/
├── agent/
│   ├── config.py       # MODEL_NAME, MAX_RETRIES, DESTRUCTIVE_ACTIONS, USE_MOCK_LLM
│   ├── graph.py        # LangGraph graph: agent → approve/tools → agent
│   ├── prompts.py      # SYSTEM_PROMPT, REFLECTION_PROMPT
│   └── state.py        # AgentState: messages, retry_count, reasoning_trace
├── mcp_server/
│   ├── mock_data.py    # MOCK_FILESYSTEM, MOCK_DATABASE
│   └── tools.py        # read_file, list_files, search_files, delete_file, query_database, update_record
├── tests/
│   ├── test_agent.py   # Node-level tests: happy path, failure recovery, HITL
│   ├── test_graph.py   # Graph-level tests: full flow, approval gate
│   └── test_tools.py   # Unit tests for all MCP tool functions
├── examples/
│   ├── input.json      # Example multi-step input (HITL + error recovery)
│   └── output.json     # Expected output with full message history and trace
├── test_report/        # Test result snapshots per phase
├── .env.example        # Environment variable template
├── main.py             # CLI entry point
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env to set your API key and USE_MOCK_LLM flag
```

## Configuration

All config lives in `.env`:

| Variable | Default | Description |
|---|---|---|
| `USE_MOCK_LLM` | `true` | Use scripted mock LLM (no API key needed) |
| `MODEL_NAME` | `claude-haiku-4-5-20251001` | Anthropic model to use |
| `ANTHROPIC_API_KEY` | — | Required when `USE_MOCK_LLM=false` |

## Running

```bash
# Default message
python main.py

# Custom message
python main.py "List all files in /data/reports/"

# From input file
python main.py --file examples/input.json

# From input file and save output
python main.py --file examples/input.json --save
```

## How the graph works

```
START → agent → route_tool_call
                    ├── safe action   → execute_tools → agent → ...
                    ├── destructive   → approve (interrupt) → execute_tools → agent → ...
                    └── no tool call  → END
```

**Error recovery**: when `execute_tools` catches a `ToolError`, it injects a
`REFLECTION_PROMPT` into the message history so the LLM can reason about what
went wrong and try a different approach. After `MAX_RETRIES` failures, the agent
gives up gracefully and explains the situation to the user.

**HITL approval**: `interrupt()` pauses the graph and surfaces the pending action
to the user. Responding `yes` resumes execution; anything else cancels the action
and routes back to the agent.

## Running tests

```bash
# All tests
pytest tests/ -v

# With real LLM (set USE_MOCK_LLM=false and ANTHROPIC_API_KEY in .env first)
pytest tests/ -v
```

## Example scenario

`examples/input.json` tests all three capabilities in one run:

```json
{
  "messages": [{
    "role": "user",
    "content": "Delete the file at /data/old_backup.csv, then find the latest report in /data/reports/"
  }]
}
```

1. Agent calls `delete_file` → **HITL gate fires**, asks for approval
2. User approves → file doesn't exist → **error recovery**, agent reflects
3. Agent searches for the file → confirms it's missing
4. Agent lists `/data/reports/` → returns `q4_2025.csv` as the latest report

See `examples/output.json` for the full expected message history and reasoning trace.
