import torch
import torch.nn as nn

from encoder import Encoder
from decoder import AttentionDecoder

from config import MAX_NUM_WORDS, EMBEDDING_DIM

class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size):
        super(EncoderDecoder, self).__init__()
        
        self.embedding_layer = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.encoder = Encoder(vocab_size, self.embedding_layer)
        self.decoder = AttentionDecoder(vocab_size, self.embedding_layer)

    def forward(self, inp, lengths, target= None, is_train = True):
        
        max_length = MAX_NUM_WORDS
        
        encoder_hidden, enc_hn, enc_cn = self.encoder(inp, lengths)
        preds = self.decoder(target, encoder_hidden, enc_hn, enc_cn, max_length = max_length, is_train=is_train)

        return preds

if __name__ == "__main__":
    vocab_size = 9
    random_vec = [[1,2,3,4,0,0,0,0,0], [2,3,4,1,7,7,8,2,4], [1,2,3,4,6,5,4,0,0]]
    lengths = torch.LongTensor([4,9, 7])
    inp_vec = torch.LongTensor(random_vec)
    print(inp_vec.size())
    enc = EncoderDecoder(vocab_size)
    out = enc(inp_vec, lengths, is_train=False)
    print(out.size())
    print(out)
