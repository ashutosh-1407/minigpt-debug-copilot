from pathlib import Path
import torch
from model.tokenizer.char_tokenizer import CharTokenizer
from model.tokenizer.bpe_tokenizer import BPETokenizer


class TrainingDataLoader:
    def __init__(self, train_data_path: str, val_data_path: str, block_size: int, batch_size: int, device: str = "cpu", tokenizer_type: str = "bpe", encoding_name: str = "gpt2"):
        train_text = Path(train_data_path).read_text(encoding="utf-8")
        val_text = Path(val_data_path).read_text(encoding="utf-8")
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        if tokenizer_type == "bpe":
            self.tokenizer = BPETokenizer(encoding_name)
        elif tokenizer_type == "char":
            self.tokenizer = CharTokenizer(train_text + val_text)
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
        
        self.vocab_size = self.tokenizer.vocab_size

        self.train_data = torch.tensor(self.encode(train_text), dtype=torch.long)
        self.val_data = torch.tensor(self.encode(val_text), dtype=torch.long)   

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)
    
    def get_batch(self, split: str):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i: i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1: i + self.block_size + 1] for i in ix])
        return x.to(self.device), y.to(self.device)
