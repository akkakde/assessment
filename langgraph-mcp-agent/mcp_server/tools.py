from mcp_server.mock_data import MOCK_FILESYSTEM, MOCK_DATABASE


class ToolError(Exception):
    def __init__(self, error_type: str, message: str):
        self.error_type = error_type
        self.message = message
        super().__init__(message)


def read_file(path: str) -> str:
    """Return the contents of a file at the given path.
    Raises ToolError if the file does not exist."""
    if path not in MOCK_FILESYSTEM:
        raise ToolError("FileNotFound", f"No file at {path}")
    return MOCK_FILESYSTEM[path]


def list_files(directory: str) -> str:
    """List all files under the given directory prefix.
    Returns a newline-separated list of matching paths."""
    matches = [p for p in MOCK_FILESYSTEM if p.startswith(directory)]
    if not matches:
        return f"No files found in {directory}"
    return "\n".join(matches)


def search_files(query: str) -> str:
    """Search file paths and contents for a query string (case-insensitive).
    Returns a newline-separated list of matching paths."""
    q = query.lower()
    matches = [
        path for path, content in MOCK_FILESYSTEM.items()
        if q in path.lower() or q in content.lower()
    ]
    if not matches:
        return f"No files matching '{query}'"
    return "\n".join(matches)


def delete_file(path: str) -> str:
    """Delete a file at the given path from the filesystem.
    Raises ToolError if the file does not exist."""
    if path not in MOCK_FILESYSTEM:
        raise ToolError("FileNotFound", f"No file at {path}")
    del MOCK_FILESYSTEM[path]
    return f"Deleted {path}"


def query_database(sql_like: str) -> str:
    """Execute a simple SELECT query against the mock user database.
    Supports basic WHERE field='value' filtering."""
    sql = sql_like.strip()
    if not sql.upper().startswith("SELECT"):
        raise ToolError("QueryError", "Invalid query format")

    rows = list(MOCK_DATABASE)

    if "WHERE" in sql.upper():
        where_idx = sql.upper().index("WHERE")
        condition = sql[where_idx + 5:].strip()
        if "=" not in condition:
            raise ToolError("QueryError", "Invalid query format")
        parts = condition.split("=", 1)
        field = parts[0].strip()
        value = parts[1].strip().strip("'\"")
        rows = [r for r in rows if str(r.get(field, "")) == value]

    if not rows:
        return "No records found"

    header = " | ".join(rows[0].keys())
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(" | ".join(str(v) for v in row.values()))
    return "\n".join(lines)


def update_record(record_id: int, field: str, value: str) -> str:
    """Update a field on a database record by id.
    Raises ToolError if the record or field does not exist."""
    for record in MOCK_DATABASE:
        if record["id"] == record_id:
            if field not in record:
                raise ToolError("InvalidField", f"Field '{field}' not in record")
            record[field] = value
            return f"Updated record {record_id}: {field} = {value}"
    raise ToolError("RecordNotFound", f"No record with id {record_id}")
