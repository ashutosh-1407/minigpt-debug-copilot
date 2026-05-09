import argparse
from model.training.base.train_debug_copilot import run_training as base_run_training
from model.training.pretrained.train_debug_copilot import run_training as pretrained_run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model",
        choices=["base", "pretrained"],
        required=True
    )

    parser.add_argument(
        "--tokenizer",
        choices=["char", "bpe"],
        default="bpe"
    )

    args = parser.parse_args()

    if args.model == "base":
        base_run_training(args.tokenizer)
    elif args.model == "pretrained":
        pretrained_run_training()

if __name__ == "__main__":
    main()
