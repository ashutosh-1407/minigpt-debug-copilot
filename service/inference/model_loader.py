from pathlib import Path
import torch
import tiktoken
from model.base.minigpt.model import MiniGPTLanguageModel
from model.utils import get_device
from model.tokenizer.char_tokenizer import CharTokenizer
from config.settings import AppSettings


class LoadedMiniGPT:
    def __init__(self, settings: AppSettings | None = None):
        self.settings = settings or AppSettings()
        device = get_device()
        checkpoint_path = Path(self.settings.checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        config = checkpoint["config"]
        self.tokenizer_type = checkpoint["tokenizer_type"]
        self.encoding_name = checkpoint["encoding_name"]
        self.model_version = config["settings"].model_version

        if self.tokenizer_type == "bpe":
            self.tokenizer = tiktoken.get_encoding(self.encoding_name)
            vocab_size = self.tokenizer.n_vocab
        elif self.tokenizer_type == "char":
            self.tokenizer = CharTokenizer(
                text=None,
                stoi=checkpoint["stoi"],
                itos=checkpoint["itos"]
            )
            vocab_size = self.tokenizer.vocab_size
        else:
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")
        
        self.model = MiniGPTLanguageModel(
            block_size=config["block_size"],
            vocab_size=vocab_size,
            n_embed=config["n_embed"],
            n_head=config["n_head"],
            n_layer=config["n_layer"],
            dropout=config["dropout"],
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)
