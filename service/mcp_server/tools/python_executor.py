import contextlib
import io
import time
from dataclasses import dataclass


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: str | None
    error: str | None
    latency_ms: int

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "latency_ms": self.latency_ms
        }
    
def execute_python_code(code: str) -> dict:
    start = time.perf_counter()

    stdout = io.StringIO()

    try:
        local_vars = {}
        SAFE_BUILTINS = {
            "print": print,
            "len": len,
            "range": range,
            "sum": sum,
            "min": min,
            "max": max,
        }
        # capture anything printed
        with contextlib.redirect_stdout(stdout):
            try:
                result = eval(code, {"__builtins__": SAFE_BUILTINS}, local_vars)
                output = stdout.getvalue()
                if result is not None:
                    output += str(result)
            except SyntaxError:
                exec(code, {"__builtins__": {}}, local_vars)
                output = stdout.getvalue()
        latency_ms = int((time.perf_counter() - start) * 1000)
        
        return ToolResult(
            tool_name="python_executor",
            success=True,
            output=output.strip(),
            error=None,
            latency_ms=latency_ms
        ).to_dict()
    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)

        return ToolResult(
            tool_name="python_executor",
            success=False,
            output=None,
            error=str(e),
            latency_ms=latency_ms
        ).to_dict()
