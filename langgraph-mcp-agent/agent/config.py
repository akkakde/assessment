import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.environ.get("MODEL_NAME", "claude-haiku-4-5-20251001")
MAX_RETRIES = 2
DESTRUCTIVE_ACTIONS = {"delete_file", "update_record"}
USE_MOCK_LLM = os.environ.get("USE_MOCK_LLM", "true").lower() == "true"
