import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS_ENCODER


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_layer):
        super(Encoder, self).__init__()
        self.embedding_layer = embedding_layer
        self.layer_norm = nn.LayerNorm(EMBEDDING_DIM)
        self.rnn = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, num_layers=NUM_LAYERS_ENCODER, 
                    dropout=0.2, bidirectional=True, batch_first=True)

        self.linear = nn.Linear(HIDDEN_DIM*2, HIDDEN_DIM)


    def forward(self, inp, lengths):
        embedded = self.embedding_layer(inp)
        # print(embedded.size())
        embed_normed = self.layer_norm(embedded)
        # print(embed_normed.size())
        embed_packed = pack_padded_sequence(embed_normed, lengths, batch_first=True, enforce_sorted=False)
        # print(embed_packed)
        rnn_out, (hn, cn) = self.rnn(embed_packed)
        unpacked, _ = pad_packed_sequence(rnn_out, batch_first=True)
        # print(unpacked.size())
        # print(unpacked)
        out = self.linear(unpacked)

        return out, hn, cn

if __name__ == "__main__":
    vocab_size = 9
    random_vec = [[1,2,3,4,0,0,0,0,0], [2,3,4,1,7,7,8,2,4], [1,2,3,4,6,5,4,0,0]]
    lengths = torch.LongTensor([4,9, 7])
    inp_vec = torch.LongTensor(random_vec)
    print(inp_vec.size())
    enc = Encoder(vocab_size)
    out = enc(inp_vec, lengths)
    print(out.size())

