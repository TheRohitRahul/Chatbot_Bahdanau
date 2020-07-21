import os
import math
import torch
from random import randint
from config import PAD_TOKEN, BATCH_SIZE, SOS_TOKEN, EOS_TOKEN, MAX_WORD_LENTH

class BatchLoader(object):
    def __init__(self, word_to_num, num_to_word, paired_data, batch_size=BATCH_SIZE):
        
        self.word_to_num = word_to_num 
        self.num_to_word = num_to_word
        self.batch_size = batch_size

        self.paired_data = sorted(paired_data, reverse = True, key = self.sort_on_label_length)
        self.example_len = len(self.paired_data)
        self.len = math.ceil(self.example_len/self.batch_size)

        

    def get_batch(self):
        
        start_point = randint(0, self.len - 1)*self.batch_size
        end_point = min(self.example_len - 1 , start_point + self.batch_size)
        
        input_sentences, input_lengths, output_sentences, output_lengths = self.get_data_tensor(start_point, end_point)
        # print("input_sentence  : ", self.paired_data[start_point])
        # print("output_sentence : ", self.paired_data[end_point])
        # print("input tensor    : ", input_sentences[0])
        # print("Input length    : ", input_lengths)
        # print("output tensor   : ", output_sentences[0])
        # print("ouput length    : ", output_lengths)
        # exit()

        
        return input_sentences, input_lengths, output_sentences, output_lengths

    def __len__(self):
        return self.len*2

    def get_data_tensor(self, start_point, end_point):
        input_lengths = []
        output_lengths = []
        
        example_batch = self.paired_data[start_point : end_point]

        input_sentences = []
        output_sentences = []

        for an_example in example_batch:
            inp_array = [SOS_TOKEN]
            rep_array = [SOS_TOKEN]

            inp = an_example[0]
            reply = an_example[1]
            inp_words = inp.split(" ")
            rep_words = reply.split(" ")

            # Processing input sentence
            for inp_wrd in inp_words:
                inp_array.append(self.word_to_num[inp_wrd])

            # Processing Reply
            for rep_wrd in rep_words:
                rep_array.append(self.word_to_num[rep_wrd])

            inp_array.append(EOS_TOKEN)
            input_lengths.append(len(inp_array))
            input_sentences.append(inp_array)
            
            rep_array.append(EOS_TOKEN)
            #for eos
            output_lengths.append(len(rep_array))
            output_sentences.append(rep_array)
            
        # Now that the examples are processed we would need to pad them with PAD_TOKEN

        inp_max = max(input_lengths)

        out_max = MAX_WORD_LENTH + 1#max(output_lengths)

        input_sentences = self.pad_to_max(input_sentences, input_lengths, inp_max)
        output_sentences = self.pad_to_max(output_sentences, output_lengths, out_max)
        

        input_lengths = torch.LongTensor(input_lengths)
        output_lengths = torch.LongTensor(output_lengths)
        input_sentences = torch.LongTensor(input_sentences)
        output_sentences = torch.LongTensor(output_sentences)

        return input_sentences, input_lengths, output_sentences, output_lengths


    def pad_to_max(self, array, array_len, array_max):
        for i in range(len(array)):
            element_len = array_len[i]
            diff = array_max - element_len
            for _ in range(diff):
                array[i].append(PAD_TOKEN)

        return array

    def sort_on_label_length(self, element):
        return len(element[1].split(" "))