"""Shared fixtures for IDA Chat plugin tests.

Opens a test binary with IDA Pro for integration testing.
"""

import shutil
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
TEST_BINARY = TESTS_DIR / "data" / "Practical Malware Analysis Lab 01-01.exe_"


@pytest.fixture(scope="session")
def test_binary(tmp_path_factory) -> Path:
    """Copy test binary to temp dir to avoid IDB conflicts."""
    work = tmp_path_factory.mktemp("ida_chat_tests")
    dest = work / TEST_BINARY.name
    shutil.copy(TEST_BINARY, dest)
    return dest


@pytest.fixture(scope="session")
def db(test_binary):
    """Open the test binary with IDA Pro and yield the Database."""
    from ida_domain import Database
    from ida_domain.database import IdaCommandOptions

    options = IdaCommandOptions(auto_analysis=True, new_database=False)
    with Database.open(str(test_binary), options, save_on_close=False) as database:
        yield database


@pytest.fixture(scope="session")
def sandbox(db):
    """Create an IdaSandbox for the test database."""
    from ida_codemode_sandbox import IdaSandbox
    return IdaSandbox(db)
