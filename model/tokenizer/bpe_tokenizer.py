import tiktoken


class BPETokenizer:
    def __init__(self, encoding_name: str = "gpt2"):
        self.encoder = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.encoder.n_vocab

    def encode(self, text: str) -> list[int]:
        return self.encoder.encode(text)
    
    def decode(self, tokens: list[int]) -> str:
        return self.encoder.decode(tokens)
