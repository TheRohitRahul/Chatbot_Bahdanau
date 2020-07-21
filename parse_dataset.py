import os 
import pickle
import pdb

from config import DATASET_PATH, CHARS_TO_FILTER,\
     MODEL_PATH, EOS_TOKEN, SOS_TOKEN, PAD_TOKEN, DICT_PICKLE_NAME, TERMS_TO_REPLACE, MAX_NUM_WORDS
from utils import get_tsv_as_paired_data

def make_word_list(paired_data):
    word_list = []
    for [statement, reply] in paired_data:
        word_list.extend(statement.split(" "))
        word_list.extend(reply.split(" "))

    word_list = set(word_list)
    
    return word_list

def create_and_save_dict(word_list):

    word_to_num = { word : num + 3 for num, word in enumerate(word_list)}
    word_to_num["<PAD>"] = PAD_TOKEN 
    word_to_num["<SOS>"] = SOS_TOKEN
    word_to_num["<EOS>"] = EOS_TOKEN

    num_to_word = {value : key for key, value in word_to_num.items()}

    dict_pickle_path = os.path.join(MODEL_PATH, DICT_PICKLE_NAME)
    with open(dict_pickle_path, "wb") as f:
        pickle.dump({"word_to_num" : word_to_num, "num_to_word" : num_to_word}, f)

    # print("word to num")
    # print(word_to_num)
    # print("num to word")
    # print(num_to_word)

    return word_to_num, num_to_word

def filter_chars_from_paired_data(paired_data):
    print("Total Number of sentences before processing are : {}".format(len(paired_data)))
    num_errors = 0
    sentence_too_long = 0

    to_remove = []
    for i in range(len(paired_data)):
        # if "nihilism" in paired_data[i][0] or "nihilism" in paired_data[i][1]:
        #     pdb.set_trace() 
        if len(paired_data[i]) != 2:
            num_errors += 1
            to_remove.append(i) 
            continue

        if len(paired_data[i][0].split(" ")) > MAX_NUM_WORDS - 1 or len(paired_data[i][1].split(" ")) > MAX_NUM_WORDS - 1:
            sentence_too_long += 1
            to_remove.append(i)
            continue


        statement, reply = paired_data[i]
        for a_char in CHARS_TO_FILTER:
            statement = statement.replace(a_char, "")
            reply = reply.replace(a_char, "")
        
        statement = statement.split()
        if "" in statement:
            statement.remove("")
        statement = " ".join(statement).lower()
        
        reply = reply.split()
        if "" in reply:
            reply.remove("")
        reply = " ".join(reply).lower()

        paired_data[i] = [statement, reply]

    for i in range(len(to_remove)- 1, -1, -1):
        del paired_data[to_remove[i]]
    

    print("Found {} errors in the dataset and removed the examples\nFound {} sentences that were too long to keep".format(num_errors, sentence_too_long))
    print("Total Number of sentences after processing are : {}".format(len(paired_data)))
    return paired_data

def remove_terms(paired_data):
    for i in range(len(paired_data)):
        if len(paired_data[i]) != 2:
            continue

        statment, reply = paired_data[i]
        for a_term in TERMS_TO_REPLACE:
            statment = statment.replace(a_term, "")
            reply = reply.replace(a_term, "")
        paired_data[i] = [statment, reply]

    return paired_data

def data_parse_main(no_save = False, file_path = DATASET_PATH):
    print("Reading dataset from {}".format(file_path))
    
    paired_data = get_tsv_as_paired_data(file_path)
    paired_data = remove_terms(paired_data)
    paired_data = filter_chars_from_paired_data(paired_data)
    
    word_list = make_word_list(paired_data)
    
    word_to_num, num_to_word = None, None
    if no_save:
        print("not creating dict")
    else:
        print("Creating dict and saving")
        word_to_num, num_to_word = create_and_save_dict(word_list)
    
    print("Total words in vocab : {}".format(len(word_list)))

    return word_to_num, num_to_word, paired_data

if __name__ == "__main__":
    data_parse_main()