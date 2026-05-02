import json
from datetime import datetime
from pathlib import Path
import torch
from config.paths import EVAL_PROMPTS_DIR, REPORTS_DIR
from model.utils import get_device
from service.inference.model_loader import LoadedMiniGPT
from service.inference.generator import MiniGPTGenerator


EVAL_PROMPTS_PATH = EVAL_PROMPTS_DIR / "debug_eval_prompts.jsonl"

def load_eval_prompts(path: Path) -> list[dict]:
    prompts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            cleaned_line = line
            if cleaned_line:
                prompts.append(json.loads(cleaned_line))
    return prompts

# def decode(tokens: list[int], itos: dict[int, str]) -> str:
#     return "".join(itos[int(i)] for i in tokens)

# def encode(text: str, stoi: dict[str, int]) -> list[int]:
#     unknown_chars = set(text) - set(stoi.keys())
#     if unknown_chars:
#         raise ValueError(f"Prompt contains unknown chars: {unknown_chars}")
#     return [stoi[ch] for ch in text]

def main() -> None:
    tool_total = 0
    tool_correct = 0
    direct_total = 0
    direct_correct = 0

    device = get_device()
    loaded_model = LoadedMiniGPT()
    generator = MiniGPTGenerator(loaded_model)

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

            with torch.no_grad():
                answer, _, tool_used, tool_result = generator.generate(
                    prompt=prompt, 
                    max_new_tokens=180
                )
            
            expected_tool = item.get("expected_tool")
            if expected_tool is not None:
                tool_total += 1
                if tool_used == expected_tool:
                    tool_correct += 1
            
            if "direct_answer" in item:
                direct_total += 1
                if item["direct_answer"] in answer:
                    direct_correct += 1

            f.write(f"## {prompt_id}\n\n")
            f.write("### Prompt\n\n")
            f.write("```text\n")
            f.write(prompt)
            f.write("\n```\n\n")
            f.write("### Output\n\n")
            f.write("```text\n")
            f.write(answer)
            f.write("\n```\n\n")
            f.write("\n# Summary\n\n")
            if tool_total > 0:
                f.write(f"Tool usage accuracy: {tool_correct} / {tool_total}\n\n")
            if direct_total > 0:
                f.write(f"Direct answer accuracy: {direct_correct} / {direct_total}\n\n")

    print(f"Wrote eval report: {output_path}")

if __name__ == "__main__":
    main()
