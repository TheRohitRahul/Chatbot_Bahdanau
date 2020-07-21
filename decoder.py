import torch.nn as nn
import torch
import torch.nn.functional as F
from random import randint

from config import HIDDEN_DIM, EMBEDDING_DIM, MAX_NUM_WORDS ,DEVICE, NUM_LAYERS_ENCODER, TEACHER_FORCING_RATIO,SOS_TOKEN

class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_layer):
        super(AttentionDecoder, self).__init__()

        self.project_hn = nn.Linear(NUM_LAYERS_ENCODER*2, 1)
        self.project_cn = nn.Linear(NUM_LAYERS_ENCODER*2, 1)

        self.vocab_size = vocab_size
        self.embedding_layer = embedding_layer
        self.layer_norm_enc = nn.LayerNorm(HIDDEN_DIM)
        self.layer_norm_embed = nn.LayerNorm(EMBEDDING_DIM)
        self.attention_cell = AttentionCell()
        self.use_teacher_forcing = False
        self.output_layer = nn.Linear(HIDDEN_DIM, vocab_size)

    def forward(self, targets, encoder_hidden, enc_hn, enc_cn, max_length = MAX_NUM_WORDS, is_train = True):
        
        batch_size = encoder_hidden.size(0)
        decoder_hidden = self.init_hidden(enc_hn, enc_cn)#(
            #torch.FloatTensor(batch_size, HIDDEN_DIM).fill_(0).to(DEVICE),
            #torch.FloatTensor(batch_size, HIDDEN_DIM).fill_(0).to(DEVICE)
        #)

        
        num_steps = max_length
        encoder_hidden = self.layer_norm_enc(encoder_hidden)
        
        preds = torch.zeros(batch_size, max_length, self.vocab_size).to(DEVICE)
        output_hidden = torch.zeros(batch_size, max_length, HIDDEN_DIM).to(DEVICE)

        if randint(0,100) < TEACHER_FORCING_RATIO:
            self.use_teacher_forcing = True
        else:
            self.use_teacher_forcing = False

        if is_train:
            if self.use_teacher_forcing:
                for i in range(num_steps):
                    
                    word_in = targets[:, i]
                    embedded_word = self.embedding_layer(word_in)
                    embedded_word = self.layer_norm_embed(embedded_word)
                    decoder_hidden, alphas = self.attention_cell(decoder_hidden, encoder_hidden, embedded_word)
                    output_hidden[:, i, :] = decoder_hidden[0]
                preds = self.output_layer(output_hidden)
            
            else:
                word_in = torch.LongTensor(batch_size).fill_(SOS_TOKEN).to(DEVICE) 

                for i in range(num_steps):
                    embedded_word = self.embedding_layer(word_in)
                    embedded_word = self.layer_norm_embed(embedded_word)
                    decoder_hidden, alphas = self.attention_cell(decoder_hidden, encoder_hidden, embedded_word)
                    output = self.output_layer(decoder_hidden[0])
                    preds[:, i, :] = output
                    word_in = output.max(1)[1]

        else:
            word_in = torch.LongTensor(batch_size).fill_(SOS_TOKEN).to(DEVICE) 

            for i in range(num_steps):
                embedded_word = self.embedding_layer(word_in)
                embedded_word = self.layer_norm_embed(embedded_word)
                decoder_hidden, alphas = self.attention_cell(decoder_hidden, encoder_hidden, embedded_word)
                output = self.output_layer(decoder_hidden[0])
                preds[:, i, :] = output
                word_in = output.max(1)[1]
               

        return preds

    def init_hidden(self, encoder_hn, encoder_cn):
        encoder_hn = encoder_hn.permute(1,2,0)
        encoder_cn = encoder_cn.permute(1,2,0)
        
        decoder_cn = self.project_cn(encoder_cn).squeeze(2)
        decoder_hn = self.project_hn(encoder_hn).squeeze(2)      

        return (decoder_hn, decoder_cn)

class AttentionCell(nn.Module):
    def __init__(self):
        super(AttentionCell, self).__init__()

        self.project_query = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.project_key = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)

        self.compute_energy = nn.Linear(HIDDEN_DIM, 1, bias=False)

        self.rnn = nn.LSTMCell(HIDDEN_DIM + EMBEDDING_DIM, HIDDEN_DIM)

    def forward(self, prev_hidden, encoder_hidden, embedding_input):

        projected_encoder_hidden = self.project_key(encoder_hidden)
        query = self.project_query(prev_hidden[0]).unsqueeze(1)

        energy_feed = torch.tanh(query + projected_encoder_hidden)

        energies = self.compute_energy(energy_feed)

        alphas = F.softmax(energies, dim=1)

        context = torch.bmm(alphas.permute(0,2,1), encoder_hidden).squeeze(1)

        
        concat_context = torch.cat([context, embedding_input], dim=1)
        decoder_out = self.rnn(concat_context, prev_hidden)

        return decoder_out, alphas        