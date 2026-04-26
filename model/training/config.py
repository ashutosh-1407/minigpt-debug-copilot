from dataclasses import dataclass, field
from model.utils import get_device
from config.paths import PROCESSED_DATA_DIR, CHECKPOINTS_DIR
from config.settings import AppSettings


@dataclass
class TrainingConfig:
    settings: AppSettings = field(default=AppSettings)

    train_data_path: str = str(PROCESSED_DATA_DIR / "train_conversations.txt")
    val_data_path: str = str(PROCESSED_DATA_DIR / "val_conversations.txt")
    checkpoint_dir: str = str(CHECKPOINTS_DIR)

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
    
    @property
    def model_version(self):
        return self.settings.model_version
    
    @property
    def tokenizer_type(self):
        return self.settings.tokenizer_type
    
    @property
    def encoding_name(self):
        return self.settings.encoding_name
