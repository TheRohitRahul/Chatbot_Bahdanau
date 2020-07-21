import torch.nn as nn
import torch
import numpy as np
import os
import pickle
from tqdm import tqdm

from batchloader import BatchLoader
from parse_dataset import data_parse_main

from main_model import EncoderDecoder
from config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, PAD_TOKEN, DEVICE, PRINT_EVERY,MODEL_PATH, TRAIN_RESUME

from utils import chatbot_output_convert, test_input_convert

def test(model_path = TRAIN_RESUME):
    
    word_to_num, num_to_word = [], []
    if model_path != "":
        print("Loading dictionary")
        with open(os.path.join(MODEL_PATH, "dict_pickle"), "rb") as f:
            dict_pickle = pickle.load(f)
            word_to_num, num_to_word = dict_pickle["word_to_num"], dict_pickle["num_to_word"]
    else:
        print("No Model specified exiting")
        exit()
        
    vocab_size = len(word_to_num)

    chatbot_model = EncoderDecoder(vocab_size).to(DEVICE)
    if model_path != "":
        print("loading model from path {}".format(model_path))
        chatbot_model.load_state_dict(torch.load(model_path))
    else:
        print("No model specified exiting")
    
    should_exit = False
    while(not should_exit):
        
        chatbot_model.eval()
        
        input_tensor, lengths = test_input_convert(word_to_num)
        input_tensor = input_tensor.to(DEVICE)         
        
        batch_num = input_tensor.size(0)
        time_steps = input_tensor.size(1)

        preds = None
        with torch.no_grad():
            preds = chatbot_model(input_tensor, lengths, target=None, is_train=False)
        
        all_outputs = chatbot_output_convert(num_to_word, preds)
        print (all_outputs[0])

if __name__ == "__main__":

    test()
