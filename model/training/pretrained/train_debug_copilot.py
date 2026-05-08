from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from model.training.pretrained.data_loader import prepare_dataset


def get_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def train(model, tokenizer, dataset: Dataset, output_path: str) -> None:
    args = TrainingArguments(
        output_dir=output_path,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=50,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        save_total_limit=2,
        report_to="none" # Prevents syncing to external logs unless configured
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    trainer.train()
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

def run_training() -> None:
    # --- Configuration ---
    CFG = {
        "model_name": "distilgpt2",
        "debug_qa_data_path": "data/raw/debug_qa.jsonl",
        "errors_data_path": "data/raw/errors.txt",
        "output_dir": "model/checkpoints/gpt2-debug-final"
    }

    # --- Execution ---
    model, tokenizer = get_model_and_tokenizer(CFG["model_name"])
    tokenized_data = prepare_dataset(CFG["debug_qa_data_path"], CFG["errors_data_path"], tokenizer)
    train(model, tokenizer, tokenized_data, CFG["output_dir"])

if __name__ == "__main__":
    run_training()