import torch


class CharDatasetProcessor:
    def __init__(self, file_path, block_size, batch_size, device):
        self.file_path = file_path
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        self.text = self._load_text()
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
    
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        self.data = torch.tensor(self.encode(self.text), dtype=torch.long)

        n = int(0.9 * len(self.data))
        self.train_data = self.data[: n]
        self.val_data = self.data[n: ]

    def _load_text(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, values):
        return "".join([self.itos[i] for i in values])
    
    def get_batch(self, split):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i: i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1: i + self.block_size + 1] for i in ix])
        return x.to(self.device), y.to(self.device)
