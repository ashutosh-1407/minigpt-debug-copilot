import json
from datasets import Dataset
from transformers import AutoTokenizer


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

def prepare_dataset(debug_qa_file_path: str, errors_file_path: str, tokenizer, max_length: int = 256):
    texts = []
    with open(debug_qa_file_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts.append(f"User: {row['question']}\nAssistant: {row['answer']}\n<END>")

    with open(errors_file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
        blocks = [block.strip() for block in raw_text.split("---") if block.strip()]
        for block in blocks:
            user_text, assistant_text = parse_error_block(block)
            texts.append(f"User: {user_text}\nAssistant: {assistant_text}\n<END>")

    raw_dataset = Dataset.from_dict({"text": texts}).train_test_split(test_size=0.1)

    def tokenize_func(batch):
        return tokenizer(
            batch["text"], 
            truncation=True, 
            max_length=max_length, 
            padding="max_length"
        )

    tokenized_dataset = raw_dataset.map(tokenize_func, batched=True, remove_columns=["text"])
    return tokenized_dataset
