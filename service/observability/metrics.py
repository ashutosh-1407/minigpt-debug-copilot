class MetricsStore:
    def __init__(self) -> None:
        self.total_requests = 0

    def record_request(self) -> None:
        self.total_requests += 1
    
    def snapshot(self) -> dict:
        return {
            "total_requests": self.total_requests
        }
