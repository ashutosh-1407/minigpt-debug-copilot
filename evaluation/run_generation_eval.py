import json
from datetime import datetime
from pathlib import Path
import torch
from config.paths import EVAL_PROMPTS_DIR, REPORTS_DIR
from model.utils import get_device
from service.inference.model_loader import LoadedMiniGPT


EVAL_PROMPTS_PATH = EVAL_PROMPTS_DIR / "debug_eval_prompts.jsonl"

def load_eval_prompts(path: Path) -> list[dict]:
    prompts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    return prompts

# def decode(tokens: list[int], itos: dict[int, str]) -> str:
#     return "".join(itos[int(i)] for i in tokens)

# def encode(text: str, stoi: dict[str, int]) -> list[int]:
#     unknown_chars = set(text) - set(stoi.keys())
#     if unknown_chars:
#         raise ValueError(f"Prompt contains unknown chars: {unknown_chars}")
#     return [stoi[ch] for ch in text]

def main() -> None:
    device = get_device()
    loaded_model = LoadedMiniGPT()

    eval_prompts = load_eval_prompts(EVAL_PROMPTS_PATH)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / f"generation_eval_{timestamp}.md"

    with output_path.open("w", encoding="utf-8") as f:
        f.write("# BPE Eval Report\n\n")
        CHECKPOINT_PATH = loaded_model.settings.checkpoint_path
        checkpoint_relative = Path(CHECKPOINT_PATH).relative_to(Path(CHECKPOINT_PATH).parent.parent)
        f.write(f"Checkpoint: `{checkpoint_relative}`\n\n")
        f.write(f"Device: `{device}`\n\n")
        f.write(f"Tokenizer: `{loaded_model.settings.encoding_name}`\n\n")

        for item in eval_prompts:
            prompt_id = item["id"]
            prompt = item["prompt"]

            context = torch.tensor(
                [loaded_model.encode(prompt)],
                dtype=torch.long,
                device=device
            )

            with torch.no_grad():
                generated = loaded_model.model.generate(
                    idx=context, 
                    max_new_tokens=180
                )[0].tolist()
            
            text = loaded_model.decode(generated)

            f.write(f"## {prompt_id}\n\n")
            f.write("### Prompt\n\n")
            f.write("```text\n")
            f.write(prompt)
            f.write("\n```\n\n")
            f.write("### Output\n\n")
            f.write("```text\n")
            f.write(text)
            f.write("\n```\n\n")

    print(f"Wrote eval report: {output_path}")

if __name__ == "__main__":
    main()
