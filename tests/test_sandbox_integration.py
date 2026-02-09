"""Tests for codemode sandbox integration with the chat plugin.

Verifies that IdaSandbox can replace raw exec() as the script executor
in the chat core.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestSandboxAsExecutor:
    """Test IdaSandbox.execute() as a drop-in for the chat core's script_executor."""

    def test_execute_returns_string(self, sandbox):
        """execute() returns a string, matching the (str) -> str interface."""
        result = sandbox.execute('print("hello")')
        assert isinstance(result, str)
        assert "hello" in result

    def test_execute_with_ida_functions(self, sandbox):
        """execute() has access to all 28 IDA analysis functions."""
        result = sandbox.execute('info = get_binary_info()\nprint(info["architecture"])')
        assert "metapc" in result

    def test_execute_enumerate_functions(self, sandbox):
        """execute() can enumerate functions from the database."""
        result = sandbox.execute(
            'fns = enumerate_functions()\n'
            'print(f"Found {len(fns)} functions")\n'
            'for f in fns[:3]:\n'
            '    print(f"  {f[\'name\']} at {hex(f[\'address\'])}")'
        )
        assert "Found" in result
        assert "functions" in result

    def test_execute_error_handling(self, sandbox):
        """execute() returns error message instead of raising."""
        result = sandbox.execute("1 / 0")
        assert "Script error" in result
        assert "runtime" in result

    def test_execute_syntax_error(self, sandbox):
        """execute() handles syntax errors gracefully."""
        result = sandbox.execute("def")
        assert "Script error" in result
        assert "syntax" in result

    def test_execute_no_file_access(self, sandbox):
        """Sandbox blocks file operations (open is not a sandbox builtin)."""
        result = sandbox.execute('open("/etc/passwd")')
        assert "Script error" in result

    def test_execute_no_eval(self, sandbox):
        """Sandbox blocks eval() (not a sandbox builtin)."""
        result = sandbox.execute('eval("1+1")')
        assert "Script error" in result


class TestSandboxSystemPrompt:
    """Test the system prompt and API reference generation."""

    def test_system_prompt_available(self):
        from ida_codemode_sandbox import IdaSandbox
        prompt = IdaSandbox.system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100
        assert "enumerate_functions" in prompt

    def test_api_reference_available(self):
        from ida_codemode_sandbox import IdaSandbox
        ref = IdaSandbox.api_reference()
        assert isinstance(ref, str)
        assert "get_binary_info" in ref
        assert "disassemble_function" in ref

    def test_api_reference_covers_all_functions(self):
        from ida_codemode_sandbox import IdaSandbox
        ref = IdaSandbox.api_reference()
        expected_functions = [
            "get_binary_info", "enumerate_functions", "get_function_by_name",
            "disassemble_function", "decompile_function", "get_function_signature",
            "get_callers", "get_callees", "get_basic_blocks",
            "get_xrefs_to", "get_xrefs_from",
            "enumerate_strings", "get_string_at",
            "enumerate_segments",
            "enumerate_names", "get_name_at", "demangle_name",
            "enumerate_imports", "enumerate_entries",
            "read_bytes", "find_bytes", "get_disassembly_at", "get_instruction_at",
            "is_code_at", "is_data_at", "is_valid_address",
            "get_comment_at", "random_int",
        ]
        for fn in expected_functions:
            assert fn in ref, f"Missing function in API reference: {fn}"


class TestSandboxAnalysis:
    """End-to-end analysis through the sandbox."""

    def test_comprehensive_analysis(self, sandbox):
        """Run a multi-step analysis script through the sandbox."""
        code = """\
info = get_binary_info()
print("Binary: " + info["module"])
print("Arch: " + info["architecture"])
print("Bitness: " + str(info["bitness"]))

functions = enumerate_functions()
print("Functions: " + str(len(functions)))

strings = enumerate_strings()
print("Strings: " + str(len(strings)))

segments = enumerate_segments()
print("Segments: " + str(len(segments)))

imports = enumerate_imports()
print("Imports: " + str(len(imports)))

# Analyze first function
if len(functions) > 0:
    f = functions[0]
    print("First function: " + f["name"])
    disasm = disassemble_function(f["address"])
    print("  Instructions: " + str(len(disasm)))
    blocks = get_basic_blocks(f["address"])
    print("  Basic blocks: " + str(len(blocks)))
"""
        result = sandbox.execute(code)
        assert "Binary:" in result
        assert "Arch: metapc" in result
        assert "Functions:" in result
        assert "Strings:" in result
        assert "Segments:" in result
        assert "Imports:" in result
        assert "First function:" in result
