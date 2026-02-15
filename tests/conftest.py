"""Shared fixtures for ida-chat-plugin tests.

Opens the test binary with real IDA Pro analysis.
"""

import shutil
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
# Re-use the test binary from the codemode sandbox package
SANDBOX_TEST_DATA = TESTS_DIR.parent / "deps" / "idawilli" / "ida-codemode" / "packages" / "ida-codemode-sandbox" / "tests" / "data"
TEST_BINARY = SANDBOX_TEST_DATA / "Practical Malware Analysis Lab 01-01.exe_"


@pytest.fixture(scope="session")
def test_binary(tmp_path_factory) -> Path:
    """Copy the shared test binary to a temp dir to avoid IDB conflicts."""
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
