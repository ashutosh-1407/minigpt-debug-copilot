from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


class LoadedPretrainedModel:
    def __init__(self, checkpoint_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        self.model_version = self.model.config.model_type
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.tokenizer_type = type(self.tokenizer).__name__
        self.model.eval()
