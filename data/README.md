# MiniGPT Debug Dataset v1

Starter dataset for the Engineering Debugging Copilot project.

## Files
- `train.jsonl` — 90 examples
- `val.jsonl` — 10 examples
- `debug_examples_v1.jsonl` — 100 total examples
- `train_conversations.txt` — conversation-formatted training text
- `val_conversations.txt` — conversation-formatted validation text

## Format
Each JSONL line is:
{"prompt": "...", "response": "..."}

Conversation format is:
User: <prompt>
Assistant: <response>

## Coverage
This dataset mixes:
- raw error snippets
- natural debugging questions

Main categories:
- Python/runtime errors
- FastAPI and API failures
- Docker and deployment issues
- SQL/database issues
- pandas/data issues
- pytest/testing problems
- async/networking issues
- ML/tensor shape issues

## Note
This is a synthetic starter dataset built from realistic engineering debugging patterns, including several issues aligned with your recent LLM/RAG project work.
