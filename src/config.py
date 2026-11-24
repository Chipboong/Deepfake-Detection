
import torch

# Hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
BATCH_SIZE = 32
NUM_WORKERS = 4
IMAGE_SIZE = 380

# Training
EPOCHS_FROZEN = 2
EPOCHS_FINETUNE = 8
TOTAL_EPOCHS = EPOCHS_FROZEN + EPOCHS_FINETUNE

# Learning rates
HEAD_LR = 3e-4
BACKBONE_LR = 1e-5

# Optimization
WEIGHT_DECAY = 1e-4

# Checkpoint
CHECKPOINT_PATH = "checkpoints/efficientnet_b7_best.pth"

# Mixed precision
USE_AMP = DEVICE.type == "cuda"
