import pytest
from mcp_server.mock_data import MOCK_FILESYSTEM, MOCK_DATABASE
from mcp_server.tools import (
    ToolError,
    read_file,
    list_files,
    search_files,
    delete_file,
    query_database,
    update_record,
)


def setup_function():
    MOCK_FILESYSTEM.setdefault("/data/reports/q4_2025.csv", "id,product,revenue\n1,Widget A,50000\n2,Widget B,62000\n3,Widget C,31000")
    MOCK_FILESYSTEM.setdefault("/data/reports/q3_2025.csv", "id,product,revenue\n1,Widget A,45000\n2,Widget B,58000")
    MOCK_FILESYSTEM.setdefault("/data/users.csv", "id,name,email,role\n1,Alice,alice@acme.com,admin\n2,Bob,bob@acme.com,viewer")
    MOCK_FILESYSTEM.setdefault("/data/config.yaml", "app:\n  name: AcmeApp\n  version: 2.1.0\n  debug: false")

    for record in MOCK_DATABASE:
        if record["id"] == 2:
            record["status"] = "inactive"
            record["role"] = "viewer"


def test_read_file_valid():
    content = read_file("/data/users.csv")
    assert "Alice" in content
    assert "admin" in content


def test_read_file_invalid_raises():
    with pytest.raises(ToolError) as exc_info:
        read_file("/nonexistent/file.txt")
    assert exc_info.value.error_type == "FileNotFound"


def test_list_files_returns_matching():
    result = list_files("/data/reports/")
    assert "/data/reports/q4_2025.csv" in result
    assert "/data/reports/q3_2025.csv" in result


def test_list_files_nonexistent_directory():
    result = list_files("/nonexistent/dir/")
    assert "No files found" in result


def test_search_files_by_content():
    result = search_files("Widget")
    assert "/data/reports/q4_2025.csv" in result


def test_search_files_by_name():
    result = search_files("users")
    assert "/data/users.csv" in result


def test_delete_file_removes_it():
    MOCK_FILESYSTEM["/data/temp.txt"] = "temporary"
    result = delete_file("/data/temp.txt")
    assert "Deleted" in result
    with pytest.raises(ToolError) as exc_info:
        delete_file("/data/temp.txt")
    assert exc_info.value.error_type == "FileNotFound"


def test_query_database_all_users():
    result = query_database("SELECT * FROM users")
    assert "Alice" in result
    assert "Bob" in result
    assert "Charlie" in result


def test_query_database_where_filter():
    result = query_database("SELECT * FROM users WHERE status='active'")
    assert "Alice" in result
    assert "Bob" not in result


def test_update_record_changes_value():
    result = update_record(2, "status", "active")
    assert "Updated record 2" in result
    active_result = query_database("SELECT * FROM users WHERE status='active'")
    assert "Bob" in active_result


def test_update_record_bad_id_raises():
    with pytest.raises(ToolError) as exc_info:
        update_record(999, "status", "active")
    assert exc_info.value.error_type == "RecordNotFound"
