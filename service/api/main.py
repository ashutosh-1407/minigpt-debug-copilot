from fastapi import FastAPI
from service.schemas.generate import GenerateRequest, GenerateResponse


app = FastAPI(title="MiniGPT Debug Copilot")

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    # Placeholder implementation for Day 1
    return GenerateResponse(
        answer="Model inference not implemented yet.",
        latency_ms=0,
        model_version="not-loaded"
    )
