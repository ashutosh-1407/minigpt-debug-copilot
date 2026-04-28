import torch
from service.inference.model_loader import LoadedMiniGPT
from service.orchestration.prompt_templates import build_prompt
from service.orchestration.router import classify_prompt
from model.utils import get_device


class MiniGPTGenerator:
    def __init__(self, loaded_model: LoadedMiniGPT):
        self.device = get_device()
        self.loaded_model = loaded_model

    def generate(self, prompt: str, max_new_tokens=None) -> str:
        if max_new_tokens is None:
            max_new_tokens = self.loaded_model.settings.default_max_new_tokens

        route = classify_prompt(prompt)
        formatted_prompt = build_prompt(
            prompt=prompt, 
            route=route
        )

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
            answer =  decoded.split("Assistant:", 1)[1].strip()
        else:
            answer = decoded.strip()

        return answer, route
