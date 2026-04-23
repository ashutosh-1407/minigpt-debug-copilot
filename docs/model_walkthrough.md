# Karpathy Mini-GPT Walkthrough

## Objective
Understand the code deeply enough to modify, fine-tune, instrument, and serve it.

## Questions to Answer
- What is the input format? - (B, T)
- Where are token embeddings defined? - in model.py -> self.token_embedding_table
- Where are positional embeddings defined? - in model.py -> position_embedding_table
- Where is self-attention implemented? - in model.py -> Head, MultiHeadAttention Block
- Where is loss computed? - in model.py -> F.cross_entropy(logits, targets) (logits are reshaped before loss)
- Where does generation happen? - in model.py -> generate()
- What can be reused as-is? -> in model.py -> embeddings, attention blocks, feedforward layers, lm_head
- What needs adaptation for domain fine-tuning? - see below.
- Where can observability hooks be added? - see below.

## Training Flow

Raw text
→ tokenization
→ input batch creation
→ embedding lookup
→ positional embedding addition
→ transformer blocks
→ logits projection
→ loss computation
→ optimizer step

Input shape:
(B, T)

B = batch size
T = sequence length

Embedding output:
(B, T, C)

C = embedding dimension

Token embedding converts token ids into dense vectors.

Adds token position awareness.

Attention computes token-to-token dependency weights.

lm_head projects hidden states into vocabulary logits.

Loss is computed only when targets are provided.
Inference skips loss path.

Generation repeatedly:
- crops context
- predicts next token
- samples next token
- appends token

What needs adaptation for domain fine-tuning?
training data, training schedule, context length, tokenizer, checkpoint loading, evaluation prompts
- training corpus
- batching pipeline
- training schedule
- checkpoint strategy
- evaluation prompts
- optional tokenizer improvements

Where can observability hooks be added?
- around generate() for inference latency
- around forward() during training loss logging
- around checkpoint save/load

## Extension Points for Project

1. Replace raw text input with debugging dataset
2. Save checkpoints into model/checkpoints
3. Expose generate() through FastAPI
4. Add request latency timing
5. Add tool decision layer before generation
