from dataclasses import dataclass
from config.paths import CHECKPOINTS_DIR


@dataclass(frozen=True)
class AppSettings:
    model_version: str = "debug-copilot-v1"
    checkpoint_path: str = str(CHECKPOINTS_DIR / "debug-copilot-v1.pt")
    tokenizer_type: str = "bpe"
    encoding_name: str = "gpt2"
    default_max_new_tokens: int = 120
