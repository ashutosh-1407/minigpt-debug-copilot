# MiniGPT Debug Copilot

A domain-specialized debugging assistant built on top of Andrej Karpathy's mini-GPT, extended into a production-style system with training, inference, routing, and tool-augmented reasoning.

---

## Project Goal

Build a compact transformer-based debugging copilot capable of assisting with:

- Python errors  
- FastAPI issues  
- Docker problems  
- SQL debugging  
- log interpretation  
- backend troubleshooting  

---

## Why this project

This project demonstrates end-to-end ownership of an LLM system, including:

- transformer-level model understanding  
- dataset design and training pipeline  
- inference API and serving layer  
- prompt routing and orchestration  
- tool-augmented generation  
- evaluation and observability  

---

## Features

- Attention-based transformer adapted from a minimal GPT implementation  
- Debugging-domain dataset and preprocessing pipeline  
- BPE tokenization with checkpointed fine-tuning  
- FastAPI-based inference service (`/generate`, `/batch_generate`)  
- Prompt routing for debugging categories  
- Tool-augmented generation using a Python executor  
- Structured logging and `/metrics` endpoint  
- Tool-aware evaluation harness (tool usage + answer correctness)

---

## Architecture

User Request
→ FastAPI API Layer
→ Prompt Router
→ Tool Router
→ (Optional) Python Executor
→ MiniGPT Generator
→ Response
→ Logging + Metrics

---

## Project Structure

minigpt-debug-copilot/
├── config/                    # Global configuration (paths, settings)
│   ├── paths.py
│   └── settings.py
│
├── data/
│   ├── raw/                   # Raw dataset (input JSONL)
│   └── processed/             # Train/val splits + conversation format
│
├── docs/                      # Design docs and walkthroughs
│   ├── baseline_run.md
│   ├── dataset_design.md
│   ├── implementation_plan.md
│   ├── model_walkthrough.md
│   └── project_scope.md
│
├── evaluation/
│   ├── datasets/              # Eval prompts with expected tool behavior
│   ├── reports/               # Generated evaluation reports
│   └── run_generation_eval.py # Tool-aware evaluation script
│
├── model/
│   ├── base/minigpt/          # Core transformer model + trainer
│   ├── training/              # Training configs and data loaders
│   ├── tokenizer/             # Tokenizer utilities (BPE/char)
│   └── checkpoints/           # Saved model checkpoints
│
├── scripts/
│   └── preprocess_dataset.py  # Dataset preprocessing script
│
├── service/
│   ├── api/                   # FastAPI app entrypoint
│   ├── inference/             # Model loading + generation logic
│   ├── orchestration/         # Prompt + tool routing logic
│   ├── observability/         # Metrics + structured logging
│   ├── schemas/               # Request/response models
│   └── mcp_server/            # MCP tool server + tools
│
├── tests/                     # Unit tests (tools, health, etc.)
│
├── requirements.txt
└── README.md

config → model → service → evaluation
           ↑         
scripts → data

---

## How to Run

```bash
pip install -r requirements.txt
uvicorn service.api.main:app --reload
```

## Sample Requests

### Generate

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Why is my Docker container exiting immediately?","max_new_tokens":80}'
```

### Tool-augmented (Python execution)

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What does this return: len([1,2,3])?"}'
```

### Batch Generate

```bash
curl -X POST http://127.0.0.1:8000/batch_generate \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"prompt": "Why is Docker exiting?", "max_new_tokens": 80},
      {"prompt": "What does this return: len([1,2,3])?", "max_new_tokens": 80}
    ]
  }'
```

### Metrics

```bash
curl http://127.0.0.1:8000/metrics
```

## Evaluation

The system includes a tool-aware evaluation pipeline that measures:

- tool selection accuracy
- direct answer correctness
- generation quality for explanation prompts

```bash
python3 -m evaluation.run_generation_eval
```

## Deployment Note

Render deployment was evaluated but deferred due to PyTorch runtime memory constraints under a 500 MB limit.

The system is designed for local or higher-memory deployments, with future options including:

- Hugging Face Spaces
- Modal / RunPod
- AWS ECS / EC2
- Separate model worker architecture

## Future Improvements

- Expand dataset from 100 → 1,000+ examples
- Add save-best-checkpoint by validation loss
- Improve tokenizer/model quality experiments
- Add sandboxed Python execution
- Add more MCP tools
- Add streaming responses
- Add retrieval augmentation (RAG)
- Add model optimization / quantization
