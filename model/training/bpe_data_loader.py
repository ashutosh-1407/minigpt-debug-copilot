from pathlib import Path
import torch
import tiktoken


class BPEDatasetProcessor:
    def __init__(self, train_data_path: str, val_data_path: str, block_size: int, batch_size: int, device: str = "cpu", encoding_name: str = "gpt2"):
        train_text = Path(train_data_path).read_text(encoding="utf-8")
        val_text = Path(val_data_path).read_text(encoding="utf-8")

        self.encoder = tiktoken.get_encoding(encoding_name)

        self.vocab_size = self.encoder.n_vocab

        self.train_data = torch.tensor(self.encode(train_text), dtype=torch.long)
        self.val_data = torch.tensor(self.encode(val_text), dtype=torch.long)

        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

    def encode(self, text: str) -> list[int]:
        return self.encoder.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.encoder.decode(tokens)
    
    def get_batch(self, split: str):
        data = self.train_data if split == "train" else self.val_data

        if len(data) <= self.block_size:
            raise ValueError(
                f"{split} data is too small for block_size={self.block_size}. "
                f"Data length={len(data)}"
            )
        
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])
        return x.to(self.device), y.to(self.device)
