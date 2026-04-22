mkdir -p data/raw
mkdir -p data/processed
mkdir -p docs
mkdir -p evaluation/datasets
mkdir -p evaluation/reports
mkdir -p model/base
mkdir -p model/checkpoints
mkdir -p model/tokenizer
mkdir -p model/training
mkdir -p scripts
mkdir -p service/api
mkdir -p service/inference
mkdir -p service/mcp_server/tools
mkdir -p service/observability
mkdir -p service/orchestration
mkdir -p service/schemas
mkdir -p tests
touch README.md
touch .gitignore
touch requirements.txt

touch service/__init__.py
touch service/api/__init__.py
touch service/inference/__init__.py
touch service/observability/__init__.py
touch service/orchestration/__init__.py
touch service/schemas/__init__.py