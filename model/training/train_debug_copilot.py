import os
from pathlib import Path
import torch
from model.training.config import TrainingConfig
from model.training.bpe_data_loader import BPEDatasetProcessor
from model.base.minigpt.model import MiniGPTLanguageModel
from model.base.minigpt.trainer import Trainer


def main() -> None:
    config = TrainingConfig()
    device = config.device

    data_processor = BPEDatasetProcessor(
        train_data_path=config.train_data_path,
        val_data_path=config.val_data_path,
        block_size=config.block_size,
        batch_size=config.batch_size,
        device=device
    )

    print(f"Device: {device}")
    print(f"Vocab size: {data_processor.vocab_size}")
    print(f"Train chars: {len(data_processor.train_data)}")
    print(f"Val chars: {len(data_processor.val_data)}")

    model = MiniGPTLanguageModel(
        block_size=config.block_size,
        vocab_size=data_processor.vocab_size,
        n_embed=config.n_embed,
        n_head=config.n_head,
        n_layer=config.n_layer,
        dropout=config.dropout
    )
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    trainer = Trainer(
        model=model,
        data_processor=data_processor,
        optimizer=optimizer,
        eval_iters=config.eval_iters
    )

    trainer.train(
        max_iters=config.max_iters,
        eval_interval=config.eval_interval
    )

    # save the checkpoint
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(config.checkpoint_dir) / f"{config.model_version}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            # "stoi": data_processor.stoi,
            # "itos": data_processor.itos,
            "config": config.__dict__,
            "tokenizer_type": config.tokenizer_type,
            "encoding_name": config.encoding_name,
        },
        checkpoint_path
    )
    print(f"Saved checkpoint: {checkpoint_path}")

    print("\n\nGenerating text")
    # context = torch.zeros((1, 1), dtype=torch.long, device=device)
    # generated = model.generate(context, max_new_tokens=200)[0].tolist()
    # print(data_processor.decode(generated))

    prompt = "User: Why is my Docker container exiting immediately?\nAssistant:"

    context = torch.tensor(
        [data_processor.encode(prompt)],
        dtype=torch.long,
        device=device,
    )

    generated = model.generate(context, max_new_tokens=200)[0].tolist()
    print(data_processor.decode(generated))

if __name__ == "__main__":
    main()
