"""Shared fixtures for ida-chat-plugin tests.

Provides:
- test_binary: path to the shared PMA Lab01-01 test binary
- db: session-scoped IDA database (real IDA Pro analysis)
- sandbox: IdaSandbox backed by the real DB
"""

import shutil
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_DIR = TESTS_DIR.parent
TEST_BINARY = (
    REPO_DIR / "deps" / "idawilli" / "tests" / "data"
    / "Practical Malware Analysis Lab 01-01.exe_"
)


@pytest.fixture(scope="session")
def test_binary(tmp_path_factory) -> Path:
    """Copy the test binary to a temp dir to avoid IDB conflicts."""
    work = tmp_path_factory.mktemp("ida_chat")
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
    """Create an IdaSandbox backed by the real database."""
    from ida_codemode_sandbox import IdaSandbox
    return IdaSandbox(db)
