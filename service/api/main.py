import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from service.schemas.generate import GenerateRequest, GenerateResponse
from service.inference.model_loader import LoadedMiniGPT
from service.inference.generator import MiniGPTGenerator
from service.observability.metrics import MetricsStore
from service.observability.logger import generate_request_id, log_event


@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded_model = LoadedMiniGPT()
    app.state.generator = MiniGPTGenerator(loaded_model)
    app.state.metrics = MetricsStore()
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
    request_id = generate_request_id()
    start = time.perf_counter()

    generator: MiniGPTGenerator = app.state.generator
    metrics: MetricsStore = app.state.metrics

    log_event(
        event_name="generation_started",
        request_id=request_id,
        prompt_length=len(request.prompt),
        max_new_tokens=request.max_new_tokens,
        model_version=generator.loaded_model.model_version
    )
    
    try:
        answer = generator.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        metrics.record_request(
            prompt_length=len(request.prompt),
            output_length=len(answer),
            latency_ms=latency_ms,
            success=True
        )

        log_event(
            event_name="generation_completed",
            request_id=request_id,
            output_length=len(answer),
            latency_ms=latency_ms,
            success=True
        )

        return GenerateResponse(
            answer=answer,
            latency_ms=latency_ms,
            model_version=generator.loaded_model.model_version,
            tokenize_type=generator.loaded_model.tokenizer_type
        )
    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        
        metrics.record_request(
            prompt_length=len(request.prompt),
            output_length=0,
            latency_ms=latency_ms,
            success=False
        )
        
        log_event(
            event_name="generation_failed",
            request_id=request_id,
            error=str(e),
            latency_ms=latency_ms,
            success=False
        )

        raise

@app.get("/metrics")
def metrics() -> dict:
    metrics_store: MetricsStore = app.state.metrics
    return metrics_store.snapshot()
