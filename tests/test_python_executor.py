from service.mcp_server.tools.python_executor import execute_python_code


def test_python_executor_eval_expression():
    result = execute_python_code("1 + 2")
    assert result["success"] is True
    assert result["output"] == "3"

def test_python_executor_print_statement():
    result = execute_python_code("print('hello')")
    assert result["success"] is True
    assert result["output"] == "hello"

def test_python_executor_blocks_unsafe_builtin():
    result = execute_python_code("open('README.md').read()")
    assert result["success"] is False
