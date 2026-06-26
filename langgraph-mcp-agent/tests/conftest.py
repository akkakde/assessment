import copy
import pytest
from mcp_server import mock_data

_orig_filesystem = copy.deepcopy(mock_data.MOCK_FILESYSTEM)
_orig_database = copy.deepcopy(mock_data.MOCK_DATABASE)


@pytest.fixture(autouse=True)
def reset_mock_data():
    """Restore shared mock globals before each test to prevent cross-test pollution."""
    mock_data.MOCK_FILESYSTEM.clear()
    mock_data.MOCK_FILESYSTEM.update(copy.deepcopy(_orig_filesystem))
    mock_data.MOCK_DATABASE[:] = copy.deepcopy(_orig_database)
