from dataclasses import dataclass
from model.utils import get_device


@dataclass
class TrainingConfig:
    train_data_path: str = "data/processed/train_conversations.txt"
    val_data_path: str = "data/processed/val_conversations.txt"
    checkpoint_dir: str = "model/checkpoints"

    batch_size: int = 16
    block_size: int = 128
    max_iters: int = 1200
    eval_interval: int = 100
    eval_iters: int = 50
    learning_rate: float = 3e-4

    n_embed: int = 96
    n_head: int = 4
    n_layer: int = 3
    dropout: float = 0.3

    device: str = get_device()
    model_version: str = "debug-copilot-v1"
