"""
Configuration file for RNN chaos analysis experiment
"""
# Training parameters
BATCH_SIZE = 128
MAX_EPOCHS = 1200
SEQUENCE_LENGTH = 500
EMBEDDING_DIM = 64
MAX_VOCAB_SIZE = 4000
NUM_CLASSES = 1

EMBEDDING_FIX = False
FC_FIX = False
PRETRAINED_CHECKPOINT = 'checkpoints/pretrained_gru.pt'

# RNN specific parameters
HIDDEN_SIZE = 128
LEARNING_RATE = 3e-4
WARMUP_EPOCHS = 100
WARMUP_INIT_FACTOR = 0.01

# RNN cell type: 'lstm', 'gru', or 'rnn'
RNN_CELL_TYPE = 'rnn'

# Data split
TRAIN_RATIO = 0.7
TEST_RATIO = 0.3

# Asymptotic analysis parameters
ZERO_INPUT_TIMESTEPS = 2000
NUM_TEST_SAMPLES = 500
MACHINE_PRECISION_THRESHOLD = -15

# FTLE (Benettin) parameters
FTLE_EPS = [1e-7, 4e-7, 7e-7, 1e-6, 4e-6, 7e-6, 1e-5, 4e-5, 7e-5, 1e-4]
FTLE_WINDOW_LENGTHS = [10]
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
