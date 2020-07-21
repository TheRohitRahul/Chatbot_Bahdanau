import torch
import os


DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
FORCE_CPU = False

CHARS_TO_FILTER = ".,-?'!~@#$%^&*+-|\\\"<>:()"
TERMS_TO_REPLACE = ["<u>","</u>", "<i>", "</i>", "<b>", "</b>"]
DATASET_PATH = ''
TEACHER_FORCING_RATIO = 50

NUM_EPOCHS = 20000
LEARNING_RATE = 1e-3
BATCH_SIZE = 96

MAX_WORD_LENTH = 15
MAX_NUM_WORDS = 15
EMBEDDING_DIM = 300
NUM_LAYERS_ENCODER = 2
HIDDEN_DIM = 256

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
PRINT_EVERY = 500

TRAIN_RESUME = ""
MODEL_NAME = "fixed_batch_length"
MODEL_SAVE_FOLDER = './models'
DICT_PICKLE_NAME = "dict_pickle"
MODEL_PATH = os.path.join(MODEL_SAVE_FOLDER, MODEL_NAME)

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

if FORCE_CPU:
    DEVICE = 'cpu'
print("working on {}".format(DEVICE))
