from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable
from config.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR


DEBUB_QA_PATH = RAW_DATA_DIR / "debug_qa.jsonl"
ERRORS_PATH = RAW_DATA_DIR / "errors.txt"

TRAIN_PATH = PROCESSED_DATA_DIR / "train_conversations.txt"
VAL_PATH = PROCESSED_DATA_DIR / "val_conversations.txt"

VAL_RATIO = 0.1
SEED = 42

def format_conversations(user_text: str, assistant_text: str) -> str:
    return f"User: {user_text.strip()}\nAssistant: {assistant_text.strip()}\n<END>\n"

def load_debug_qa(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    
    examples: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_number}: {line}") from e
            question = row.get("question")
            answer = row.get("answer")
            if not question or not answer:
                raise ValueError(f"Line {line_number} must contain question and answer")
            examples.append(format_conversations(
                user_text=question,
                assistant_text=answer
            ))
    return examples

def parse_error_block(block: str) -> tuple[str, str]:
    marker = "Expected diagnosis:"
    if marker not in block:
        raise ValueError(f"Error block mising '{marker}':\n{block}")

    error_text, diagnosis = block.split(marker, maxsplit=1)
    user_text = (
        "Debug this error:\n"
        + error_text.strip()
    )
    assistant_text = diagnosis.strip()
    return user_text, assistant_text

def load_errors(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    
    examples: list[str] = []
    raw_text = path.read_text(encoding="utf-8")
    blocks = [block.strip() for block in raw_text.split("---") if block.strip()]
    for block in blocks:
        user_text, assistant_text = parse_error_block(block)
        examples.append(format_conversations(
            user_text=user_text,
            assistant_text=assistant_text
        ))
    return examples

def train_val_split(examples: list[str]) -> tuple[list[str], list[str]]:
    if not examples:
        raise ValueError("No examples found.")
    
    shuffled = examples[:]
    random.Random(SEED).shuffle(shuffled)
    val_size = max(1, int(len(examples) * VAL_RATIO))
    val_examples = shuffled[: val_size]
    train_examples = shuffled[val_size: ]
    return train_examples, val_examples

def write_examples(path: Path, examples: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(example)
            f.write("\n")

def main() -> None:
    qa_examples = load_debug_qa(DEBUB_QA_PATH)
    error_examples = load_errors(ERRORS_PATH)

    all_examples = qa_examples + error_examples

    train_examples, val_examples = train_val_split(all_examples)

    write_examples(TRAIN_PATH, train_examples)
    write_examples(VAL_PATH, val_examples)

    print(f"Loaded QA examples: {len(qa_examples)}")
    print(f"Loaded error examples: {len(error_examples)}")
    print(f"Total examples: {len(all_examples)}")
    print(f"Wrote train examples: {len(train_examples)} -> {TRAIN_PATH}")
    print(f"Wrote val examples: {len(val_examples)} -> {VAL_PATH}")

if __name__ == "__main__":
    main()
