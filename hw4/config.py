import torch

RANDOM_SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'

# Part 1 hyperparameters
BATCH_SIZE_1 = 128
LEARNING_RATE_1 = 1.5

# Part 2 hyperparameters
BATCH_SIZE_2 = 64
LEARNING_RATE_2 = 0.8
