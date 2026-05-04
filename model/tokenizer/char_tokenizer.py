class CharTokenizer:
    def __init__(self, text: str | None = None, stoi: dict[str, int] | None = None, itos: dict[int, str] | None = None):
        if stoi is not None and itos is not None:
            self.stoi = stoi
            self.itos = {int(k): v for k, v in itos.items()}
            self.vocab_size = len(self.stoi)
            return
        if text is None:
            raise ValueError(f"Either text or stoi/itos mappings must be provided")
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]
    
    def decode(self, tokens: list[int]) -> str:
        return "".join([self.itos[token] for token in tokens])
