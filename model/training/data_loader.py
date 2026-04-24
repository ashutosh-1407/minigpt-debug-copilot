from pathlib import Path
import torch


class CharDatasetProcessor:
    def __init__(self, train_data_path: str, val_data_path: str, block_size: int, batch_size: int, device: str = "cpu"):
        train_text = Path(train_data_path).read_text(encoding="utf-8")
        val_text = Path(val_data_path).read_text(encoding="utf-8")

        full_text = train_text + val_text
        chars = sorted(list(set(full_text)))
        self.vocab_size = len(chars)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        self.train_data = torch.tensor(self.encode(train_text), dtype=torch.long)
        self.val_data = torch.tensor(self.encode(val_text), dtype=torch.long)

        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, tokens: list[int]):
        return "".join([self.itos[i] for i in tokens])
    
    def get_batch(self, split: str):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i: i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1: i + self.block_size + 1] for i in ix])
        return x.to(self.device), y.to(self.device)
