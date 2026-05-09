import argparse
from service.inference.base_generator import generate_base
from service.inference.pretrained_generator import generate_pretrained


def main():
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

    parser.add_argument(
        "--prompt", 
        required=True
    )

    args = parser.parse_args()

    if args.model == "base":
        answer = generate_base(
            prompt=args.prompt,
            tokenizer_type=args.tokenizer,
        )
    elif args.model == "pretrained":
        answer = generate_pretrained(prompt=args.prompt)

    print(answer)

if __name__ == "__main__":
    main()
