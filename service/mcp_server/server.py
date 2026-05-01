from mcp.server.fastmcp import FastMCP
from service.mcp_server.tools.python_executor import execute_python_code


mcp = FastMCP("MiniGPT Debug Copilot Tools")

@mcp.tool()
def python_executor(code: str) -> str:
    return execute_python_code(code)

if __name__ == "__main__":
    mcp.run()
