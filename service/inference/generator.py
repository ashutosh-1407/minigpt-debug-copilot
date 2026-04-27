import torch
from service.inference.model_loader import LoadedMiniGPT
from model.utils import get_device


class MiniGPTGenerator:
    def __init__(self, loaded_model: LoadedMiniGPT):
        self.device = get_device()
        self.loaded_model = loaded_model

    def build_prompt(self, user_prompt: str) -> str:
        if user_prompt.strip().startswith("User:"):
            return user_prompt
        return f"User: {user_prompt.strip()}\n Assistant:"
    
    def generate(self, prompt: str, max_new_tokens=None) -> str:
        if max_new_tokens is None:
            max_new_tokens = self.loaded_model.settings.default_max_new_tokens

        formatted_prompt = self.build_prompt(prompt)

        context = torch.tensor(
            [self.loaded_model.encode(formatted_prompt)],
            dtype=torch.long,
            device=self.device
        )

        with torch.no_grad():
            generated = self.loaded_model.model.generate(
                context,
                max_new_tokens=max_new_tokens,
            )[0].tolist()
        
        decoded = self.loaded_model.decode(generated)

        if "Assistant:" in decoded:
            return decoded.split("Assistant:", 1)[1].strip()

        return decoded.strip()
