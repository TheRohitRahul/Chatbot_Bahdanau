import os
import torch
import numpy as np

from config import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN

def get_tsv_as_paired_data(tsv_path):
    paired_data = []
    with open(tsv_path) as f:
        all_lines = f.read().split("\n")
        
    paired_data = [ element.split("\t") for element in all_lines]
    return paired_data


def test_input_convert(word_to_num):
    input_sentence = input("Enter : ")
    input_sentence = input_sentence.split(" ")
    
    input_num = []
    for an_element in input_sentence:
        if an_element not in word_to_num:
            print("{} not present in vocabulary exiting ....")
            exit()
        
        elif an_element.lower() == "exit":
            print("exiting code")
            exit()
        
        else:
            input_num.append(word_to_num[an_element])
    
    length = torch.LongTensor([len(input_num)])
    return torch.LongTensor([input_num]), length
    


def chatbot_output_convert(num_to_word, output_tensor):
    all_outputs = []

    for i in range(output_tensor.size(0)):

        single_output = []
        for j in range(output_tensor.size(1)):
            element = torch.argmax(output_tensor[i][j]).item()
            
            element = int(element)
            if element == EOS_TOKEN or element == PAD_TOKEN or element == SOS_TOKEN:
                break

            single_output.append(num_to_word[element])
        single_output = " ".join(single_output)
        all_outputs.append(single_output)

    return all_outputs