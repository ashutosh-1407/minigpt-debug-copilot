# MiniGPT Debug Copilot

A domain-specialized debugging assistant built on top of Andrej Karpathy's mini-GPT foundation, extended into a production-style platform with:

- domain fine-tuning
- FastAPI inference APIs
- MCP-based tool integration
- evaluation pipeline
- structured observability

## Project Goal

Build a compact transformer-based debugging copilot that can assist with:
- Python errors
- FastAPI issues
- Docker problems
- SQL debugging
- log interpretation
- backend troubleshooting

## Why this project

This project is meant to understand and own project pieces across:
- model understanding
- training pipeline design
- inference serving
- tool orchestration
- evaluation
- production-style observability

## MVP Scope

- Karpathy mini-GPT base adapted locally
- debugging-oriented dataset
- fine-tuned checkpoint
- `/generate` API
- request logging and metrics
- evaluation harness
- at least one MCP-integrated tool

## Future Improvements

- BPE tokenizer comparison
- streaming responses
- batch inference
- caching
- multiple MCP tools
- retrieval integration
- quantization / optimization

## Initial Domain

Engineering Debugging Copilot

## Project Structure

(To be documented as implementation progresses)

## Architecture Preview

User Request
→ FastAPI API Layer
→ Prompt Router
→ MiniGPT Inference Engine
→ Optional MCP Tool Execution
→ Response Generation
→ Metrics + Logging + Evaluation
