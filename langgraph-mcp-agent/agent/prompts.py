SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a file system and database.\n"
    "You can read, search, list, and delete files, and query or update database records.\n"
    "Always explain what you're doing before taking action.\n"
    "If a user asks for something, use the appropriate tool."
)

REFLECTION_PROMPT = (
    "The tool '{tool_name}' was called with arguments {tool_args} but failed with error: {error_message}\n"
    "\n"
    "Analyze what went wrong and decide your next action:\n"
    "- If the file path was wrong, try searching for the file instead\n"
    "- If a query failed, try a simpler query format\n"
    "- If the resource doesn't exist, inform the user\n"
    "- Do NOT retry the exact same call with the same arguments\n"
    "\n"
    "Available tools: read_file, list_files, search_files, delete_file, query_database, update_record\n"
    "\n"
    "What should you try next?"
)
