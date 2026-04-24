import json
import random
from pathlib import Path


INPUT_PATH = Path("data/processed/debug_examples_v1.jsonl")
TRAIN_JSONL_PATH = Path("data/processed/train.jsonl")
VAL_JSONL_PATH = Path("data/processed/val.jsonl")
TRAIN_TXT_PATH = Path("data/processed/train_conversations.txt")
VAL_TXT_PATH = Path("data/processed/val_conversations.txt")

VAL_RATIO = 0.1
SEED = 42


def load_examples(path: Path) -> list[dict]:
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            if "prompt" not in record or "response" not in record:
                raise ValueError(f"Missing prompt/response on line {line_number}")

            prompt = record["prompt"].strip()
            response = record["response"].strip()

            if not prompt or not response:
                raise ValueError(f"Empty prompt/response on line {line_number}")

            examples.append({
                "prompt": prompt,
                "response": response,
            })

    return examples


def to_conversation_text(example: dict) -> str:
    return f"User: {example['prompt']}\nAssistant: {example['response']}\n\n"


def write_jsonl(path: Path, examples: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def write_conversations(path: Path, examples: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(to_conversation_text(example))


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_PATH}")

    examples = load_examples(INPUT_PATH)

    if len(examples) < 10:
        raise ValueError("Need at least 10 examples before splitting train/val.")

    random.seed(SEED)
    random.shuffle(examples)

    val_size = max(1, int(len(examples) * VAL_RATIO))
    val_examples = examples[:val_size]
    train_examples = examples[val_size:]

    TRAIN_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(TRAIN_JSONL_PATH, train_examples)
    write_jsonl(VAL_JSONL_PATH, val_examples)
    write_conversations(TRAIN_TXT_PATH, train_examples)
    write_conversations(VAL_TXT_PATH, val_examples)

    print(f"Total examples: {len(examples)}")
    print(f"Train examples: {len(train_examples)}")
    print(f"Val examples: {len(val_examples)}")
    print(f"Wrote: {TRAIN_JSONL_PATH}")
    print(f"Wrote: {VAL_JSONL_PATH}")
    print(f"Wrote: {TRAIN_TXT_PATH}")
    print(f"Wrote: {VAL_TXT_PATH}")


if __name__ == "__main__":
    main()
