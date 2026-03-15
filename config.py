"""
Configuration file for RNN chaos analysis experiment
"""
# Training parameters
BATCH_SIZE = 128
MAX_EPOCHS = 2000
SEQUENCE_LENGTH = 500
EMBEDDING_DIM = 32
MAX_VOCAB_SIZE = 4000
NUM_CLASSES = 1

EMBEDDING_FIX = True
FC_FIX = False
PRETRAINED_CHECKPOINT = 'checkpoints/pretrained1.pt'

# RNN specific parameters
HIDDEN_SIZE = 60
LEARNING_RATE = 0.0003

# Data split
TRAIN_RATIO = 0.7
TEST_RATIO = 0.3

# Asymptotic analysis parameters
ZERO_INPUT_TIMESTEPS = 2000
NUM_TEST_SAMPLES = 500
MACHINE_PRECISION_THRESHOLD = -15

# FTLE (Benettin) parameters
# - eps: initial perturbation magnitude (should be small, but not underflow)
# - window_length: renormalize every N zero-drive steps
# - burn_in: skip the first N zero-drive steps when accumulating the exponent
FTLE_EPS = 1e-6
FTLE_WINDOW_LENGTH = 10
FTLE_BURN_IN = 200

# Device and reproducibility
RANDOM_SEED = 666
DEVICE = 'cuda'

# File paths
DATA_PATH = './data/'
RESULTS_PATH = './results/'
CHECKPOINT_PATH = './checkpoints/'

# Visualization parameters
FIGURE_DPI = 300
