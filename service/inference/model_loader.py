from pathlib import Path
import torch
import tiktoken
from model.base.minigpt.model import MiniGPTLanguageModel
from model.utils import get_device
from config.settings import AppSettings


class LoadedMiniGPT:
    def __init__(self, settings: AppSettings = AppSettings):
        self.settings = settings
        device = get_device()
        checkpoint_path = Path(settings.checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        config = checkpoint["config"]
        self.tokenizer_type = checkpoint["tokenizer_type"]
        encoding_name = checkpoint["encoding_name"]
        self.model_version = config["model_version"]

        if self.tokenizer_type != "bpe":
            raise ValueError("This inference loader currently expects a BPE checkpoint.")

        self.encoder = tiktoken.get_encoding(encoding_name)

        self.model = MiniGPTLanguageModel(
            block_size=config["block_size"],
            vocab_size=self.encoder.n_vocab,
            n_embed=config["n_embed"],
            n_head=config["n_head"],
            n_layer=config["n_layer"],
            dropout=config["dropout"],
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

    def encode(self, text: str) -> list[int]:
        return self.encoder.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.encoder.decode(tokens)
