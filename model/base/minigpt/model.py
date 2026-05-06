import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """ one head of self attention """

    def __init__(self, n_embed, head_size, block_size, dropout):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x):
        _, T, _ = x.shape
        q = self.query(x) # (B, T, C)
        k = self.key(x) # (B, T, C)

        # compute attention scores or affinities
        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5 # (B, T, T) @ (B, T, C) --> (B, T, C)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # perform weighed aggregation of values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) --> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple head of attention running in parallel """

    def __init__(self, n_embed, n_head, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, head_size, block_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by non-linearity """

    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embed, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_embed, n_head, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class MiniGPTLanguageModel(nn.Module):

    def __init__(self, block_size, vocab_size, n_embed, n_head, n_layer, dropout):
        super().__init__()
        self.block_size = block_size

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        token_embed = self.token_embedding_table(idx) # (B, T, C)
        position_embed = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, C)
        x = token_embed + position_embed # (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, end_token_id=None):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to last block_size tokens
            idx_cond = idx[:, -self.block_size: ]
            # get the predictions
            logits, loss = self(idx_cond)
            
            # 1. Focus on the last predicted token and apply "Temperature"
            # We only care about the next word (last position in the sequence).
            # Dividing by 0.3 (low temperature) makes the model more confident/predictable.
            logits = logits[:, -1, :] / 0.3  # Shape: [Batch, Vocab_Size]

            # 2. Apply Repetition Penalty
            # Look at every token already in the current sequence (idx[0]).
            # Divide their scores by 1.2 to make them less likely to be picked again.
            # We iterate through the batch (b) to handle each sequence's history independently.
            for b in range(logits.size(0)):
                # idx[b] gives the history for the b-th item in the batch
                for token in set(idx[b].tolist()):
                    if logits[b, token] > 0:
                        logits[b, token] /= 1.2
                    else:
                        logits[b, token] *= 1.2  # Makes negative scores even "worse"

            # 3. Top-K Filtering (K=5)
            # Find the scores of the top 5 most likely next tokens.
            v, _ = torch.topk(logits, min(5, logits.size(-1)))

            # 4. Zero-out low probability options
            # If a token's score is lower than the 5th best score, set it to -Infinity.
            # This ensures that after Softmax, only the Top 5 tokens have a chance to be chosen.
            logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            # append sampled index to the running sequence
            if end_token_id is not None and idx_next.item() == end_token_id:
                break
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
