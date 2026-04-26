from pathlib import Path


PROJECT_PATH = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_PATH / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_PATH / "model"
CHECKPOINTS_DIR = MODEL_DIR / "checkpoints"

EVALUATION_DIR = PROJECT_PATH / "evaluation"
EVAL_PROMPTS_DIR = EVALUATION_DIR / "datasets"
REPORTS_DIR = EVALUATION_DIR / "reports"

SERVICE_DIR = PROJECT_PATH / "service"
DOCS_DIR = PROJECT_PATH / "docs"
