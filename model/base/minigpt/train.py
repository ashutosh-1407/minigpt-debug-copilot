import torch
from model.base.minigpt.config import (
    BATCH_SIZE,
    BLOCK_SIZE,
    MAX_ITERS,
    EVAL_INTERVAL,
    LEARNING_RATE,
    EVAL_ITERS,
    N_EMBED,
    N_HEAD,
    N_LAYER,
    DROPOUT,
    DEVICE,
    SEED,
)
from model.base.minigpt.data_loader import CharDatasetProcessor
from model.base.minigpt.model import MiniGPTLanguageModel
from model.base.minigpt.trainer import Trainer


def main():
    print(f"Device available: {DEVICE}")
    torch.manual_seed(SEED)

    data_processor = CharDatasetProcessor(
        file_path="model/base/minigpt/input.txt",
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )

    model = MiniGPTLanguageModel(
        block_size=BLOCK_SIZE,
        vocab_size=data_processor.vocab_size,
        n_embed=N_EMBED,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=DROPOUT
    )
    model.to(DEVICE)

    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    trainer = Trainer(
        model=model,
        data_processor=data_processor,
        optimizer=optimizer,
        eval_iters=EVAL_ITERS
    )

    trainer.train(
        max_iters=MAX_ITERS, 
        eval_interval=EVAL_INTERVAL
    )

    print("\n\nGenerating text")
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    generated = model.generate(context, max_new_tokens=500)[0].tolist()
    print(data_processor.decode(generated))

if __name__ == "__main__":
    main()
