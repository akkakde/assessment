import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.environ.get("MODEL_NAME", "claude-haiku-4-5-20251001")
MAX_RETRIES = 2
DESTRUCTIVE_ACTIONS = {
    "tool_delete_file",
    "tool_update_record",
    "tool_delete_record",
    "tool_drop_table",
    "tool_truncate_table",
    "tool_write_file",
    "tool_overwrite_file",
    "tool_execute_sql",
}
USE_MOCK_LLM = os.environ.get("USE_MOCK_LLM", "true").lower() == "true"
