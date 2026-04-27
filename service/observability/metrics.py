import time
from dataclasses import dataclass, field


@dataclass
class InferernceMetric:
    prompt_length: str
    output_length: str
    latency_ms: int
    success: bool
    timestamp: float = field(default_factory=time.time)

class MetricsStore:
    def __init__(self) -> None:
        self.requests: list[InferernceMetric] = []

    def record_request(
        self,
        prompt_length: str,
        output_length: str,
        latency_ms: int,
        success: bool
    ) -> None:
        self.requests.append(
            InferernceMetric(
                prompt_length=prompt_length,
                output_length=output_length,
                latency_ms=latency_ms,
                success=success
            )
        )
    
    def _percentile(self, values, pct) -> float:
        if not values:
            return None
        values = sorted(values)
        idx = int((pct / 100) * (len(values) - 1))
        return round(values[idx], 2)

    def snapshot(self) -> dict:
        total_requests = len(self.requests)
        success_requests = sum(1 for req in self.requests if req.success)
        error_requests = total_requests - success_requests

        latencies = [req.latency_ms for req in self.requests]
        avg_latency_ms = round(sum(latencies) / len(latencies), 2) if latencies else None
        p95_latency_ms = self._percentile(latencies, 95)

        avg_prompt_length = round(sum(req.prompt_length for req in self.requests) / total_requests, 2) if total_requests else 0.0
        avg_output_length = round(sum(req.output_length for req in self.requests) / total_requests, 2) if total_requests else 0.0

        success_rate = (success_requests / total_requests) if total_requests else 0.0
        error_rate = (error_requests / total_requests) if total_requests else 0.0   

        return {
            "requests": {
                "total": total_requests,
                "success_requests": success_requests,
                "error_requests": error_requests
            },
            "latency_ms": {
                "avg": avg_latency_ms,
                "p95": p95_latency_ms
            },
            "avg_prompt_length": avg_prompt_length,
            "avg_output_length": avg_output_length,
            "rates": {
                "success_rate": success_rate,
                "error_rate": error_rate
            }
        }
