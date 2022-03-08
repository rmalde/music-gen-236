# DATA
SAMPLE_RATE = 16000
SLICE_LEN = 65536

RAW_DATA_DIR = "raw_data/"

PREPROCESSED_DATA_DIR = "preprocessed_data/"

# MODELS
NOISE_DIM = 100

# TRAINING
BATCH_SIZE = 16
EPOCHS = 5000
EPOCHS_PER_SAVE = 1000

LR_G = 1e-5
LR_D = 3e-5 # you can use with discriminator having a larger learning rate than generator instead of using n_critic updates ttur https://arxiv.org/abs/1706.08500
BETA1 = 0.5
BETA2 = 0.9

PENALTY_COEFF = 10

MODEL_OUTPUT_DIR = "outputs/"



