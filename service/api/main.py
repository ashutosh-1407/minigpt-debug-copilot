import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from service.schemas.generate import GenerateRequest, GenerateResponse
from service.inference.model_loader import LoadedMiniGPT
from service.inference.generator import MiniGPTGenerator


@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded_model = LoadedMiniGPT()
    app.state.generator = MiniGPTGenerator(loaded_model)
    yield

app = FastAPI(
    title="MiniGPT Debug Copilot",
    lifespan=lifespan
)

@app.get("/")
def homepage() -> str:
    return "Welcome to MiniGPT transformer based debug copilot"

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    start = time.perf_counter()
    generator: MiniGPTGenerator = app.state.generator
    answer = generator.generate(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens
    )
    latency_ms = int((time.perf_counter() - start) * 1000)
    return GenerateResponse(
        answer=answer,
        latency_ms=latency_ms,
        model_version=generator.loaded_model.model_version,
        tokenize_type=generator.loaded_model.tokenizer_type
    )
