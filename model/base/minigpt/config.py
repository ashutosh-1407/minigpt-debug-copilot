import torch


# hyperparameters
BATCH_SIZE = 64
BLOCK_SIZE = 256
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
EVAL_ITERS = 200

N_EMBED = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 1337
