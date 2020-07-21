import torch.nn as nn
import torch
import numpy as np
import os
import pickle
from tqdm import tqdm
from torchtext.data.metrics import bleu_score

from batchloader import BatchLoader
from parse_dataset import data_parse_main

from main_model import EncoderDecoder
from config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, PAD_TOKEN, DEVICE, PRINT_EVERY,MODEL_PATH, TRAIN_RESUME, MAX_NUM_WORDS, EOS_TOKEN, SOS_TOKEN

def encode(word_to_num):
    
    strings_to_validate = ["hello", "how are you", "who are you",
     "hi", "what do you want", "aur bata", "where is mama","what is your name",
     "suggest me a horror movie", "do you believe in nihilism", "goodbye", "what do you like to do"]
    
    lengths = torch.LongTensor(len(strings_to_validate))
    
    for i in range(len(strings_to_validate)):
        all_words = strings_to_validate[i].split(" ")
        lengths[i] = (len(all_words))

    max_length = MAX_NUM_WORDS
    all_input_strings = torch.LongTensor(len(strings_to_validate), max_length).fill_(0)
    
    for i in range(len(strings_to_validate)):
        all_words = strings_to_validate[i].split(" ")
        
        assert len(all_words) < max_length, "input string too long"

        for j in range(len(all_words)):
            all_input_strings[i,j] = word_to_num[all_words[j]]
        
    return all_input_strings, strings_to_validate , lengths 
    
def decode(num_to_word, output_tensor, do_argmax=True):
    all_outputs = []

    for i in range(output_tensor.size(0)):

        single_output = []
        for j in range(output_tensor.size(1)):
            element = None
            if do_argmax:
                element = torch.argmax(output_tensor[i][j]).item()
            else:
                element = output_tensor[i][j]
            element = int(element)
            if element == EOS_TOKEN or element == PAD_TOKEN or element == SOS_TOKEN:
                break

            single_output.append(num_to_word[element])
        single_output = " ".join(single_output)
        all_outputs.append(single_output)

    return all_outputs

def calculate_bleu_score(ground_truth, predictions):
    for i in range(len(ground_truth)):
        ground_truth[i] = [ground_truth[i].split(" ")]
    
    for i in range(len(predictions)):
        predictions[i] = predictions[i].split(' ')

    return bleu_score(predictions, ground_truth)

def print_all(input_strings, output_strings):
    print("-"*50)
    for i in range(len(input_strings)):
         
        print("YOU : {}".format(input_strings[i]))
        print("BOT : {}".format(output_strings[i]))
    
    print("-"*50)

def train(model_path = TRAIN_RESUME):
    
    word_to_num, num_to_word, paired_data = [], [], []
    if model_path != "":
        print("Loading dictionary")
        word_to_num, num_to_word, paired_data = data_parse_main(no_save = True)
        with open(os.path.join(MODEL_PATH, "dict_pickle"), "rb") as f:
            dict_pickle = pickle.load(f)
            word_to_num, num_to_word = dict_pickle["word_to_num"], dict_pickle["num_to_word"]
    else:
        print("creating dictionary from scratch")
        word_to_num, num_to_word, paired_data = data_parse_main(no_save = False)
        
    vocab_size = len(word_to_num)

    train_loader = BatchLoader(word_to_num, num_to_word, paired_data)
    train_examples = len(train_loader)

    chatbot_model = EncoderDecoder(vocab_size).to(DEVICE)
    if model_path != "":
        print("loading model from path {}".format(model_path))
        chatbot_model.load_state_dict(torch.load(model_path))
    else:
        print("Creating model from scratch")
    critirion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    optimizer = torch.optim.AdamW(chatbot_model.parameters(), lr=LEARNING_RATE)

    best_loss = np.inf
    best_bleu = -np.inf
    for epoch in range(NUM_EPOCHS):
        
        epoch_bleu = []
        epoch_loss = []

        chatbot_model.train()
        
        for i in tqdm(range(train_examples)):
            optimizer.zero_grad()
           
            input_sentences, input_lengths, output_sentences, output_lengths = train_loader.get_batch()
            
            input_sentences = input_sentences.to(DEVICE)
            input_lengths = input_lengths.to(DEVICE)
            output_sentences = output_sentences.to(DEVICE)
            output_lengths = output_lengths.to(DEVICE)

            batch_num = input_sentences.size(0)
            time_steps = input_sentences.size(1)

            preds = chatbot_model(input_sentences, input_lengths, target=output_sentences, is_train=True)
            

            ground_truth_sent = decode(num_to_word, output_sentences[:,1:], do_argmax=False)
            predictions = decode(num_to_word, preds)
            bleu_sc = calculate_bleu_score(ground_truth_sent, predictions)

            output_sentences = output_sentences[:,1:].contiguous().view(-1)
            loss = critirion(preds.view(-1, preds.shape[-1]), output_sentences)
            epoch_loss.append(loss.item())
            epoch_bleu.append(bleu_sc)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(chatbot_model.parameters(), 50)
            optimizer.step()

        chatbot_model.eval()
        input_tensor, strings_to_validate, lengths = encode(word_to_num)
        
        input_tensor = input_tensor.to(DEVICE)
        lengths = lengths.to(DEVICE)
        
        output_tensor = None
        with torch.no_grad():
            output_tensor = chatbot_model(input_tensor, lengths, target=None, is_train=False)
        all_outputs = decode(num_to_word, output_tensor)
        print_all(strings_to_validate, all_outputs)
        chatbot_model.train()

        total_epoch_loss = np.mean(epoch_loss)
        total_epoch_bleu = np.mean(epoch_bleu)
        print("Bleu Score is : {}\nLoss is : {}".format(total_epoch_bleu, total_epoch_loss))
        
        
        if total_epoch_loss < best_loss:
            best_loss = total_epoch_loss

        if total_epoch_bleu > best_bleu:
            best_bleu = total_epoch_bleu
            print("bleu score increased saving model")
            model_path = os.path.join(MODEL_PATH, "epoch_{}_loss_{:.2f}_bleu_score_{:.2f}.pt".format(epoch, total_epoch_loss, total_epoch_bleu))
            torch.save(chatbot_model.state_dict(), model_path)
        
        else:
            print("bleu score not increased so not saving model")
            print("\nBleu Score for the epoch is {} and the best bleu score is {}".format(total_epoch_bleu, best_bleu))
        
        

if __name__ == "__main__":
    train()
