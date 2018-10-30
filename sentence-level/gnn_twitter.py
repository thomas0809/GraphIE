import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import string
import os
import json
import time
from ModelUtils import RNN, CRF
from attention import MultiHeadAttention
from gnn import *


def masked_mean(x, mask):
    # x: (batch, num_sent, -1)
    # mask: (batch, num_sent)
    # return (batch, -1)
    mask = mask.float()
    length = mask.sum(dim=1)
    y = torch.bmm(mask.unsqueeze(1), x).squeeze(1)
    length = length.unsqueeze(1).expand_as(y)
    length = length + length.eq(0).float()
    return y / length

def masked_max(x, mask):
    # x: (batch, num_sent, -1)
    # mask: (batch, num_sent)
    # return (batch, -1)
    mask = mask.unsqueeze(2).expand_as(x).float()
    y = x * mask
    return torch.max(y, dim=1)[0]



class GNN_Twitter(nn.Module):

    def __init__(self, word_vocab_size, char_vocab_size, d_output, args, pretrained_emb=None):
        super(GNN_Twitter, self).__init__()
        self.model = args.model

        self.char_cnn = Character_CNN(char_vocab_size, args.d_char_embed, args.n_filter, args.filter_sizes)
        self.word_emb = nn.Embedding(word_vocab_size, args.d_word_embed)
        if pretrained_emb is not None:
            self.word_emb.weight.data.copy_(pretrained_emb)
            self.word_emb.weight.requires_grad = False
        
        self.d_graph = args.d_graph
        self.gnn_layer = nn.ModuleList()
        d_in = args.n_filter * len(args.filter_sizes) + args.d_word_embed
        
        self.encoder_lstm = Word_LSTM(d_in, args.d_graph[0])

        if self.model in ['lstm-lstm', 'lstm-gcn-lstm', 'lstm-gat-lstm']:
            self.decoder_lstm = Word_LSTM(args.d_graph[0], args.d_graph[-1])
        else:
            self.decoder_lstm = None
        
            # self.final_layer = nn.Linear(args.d_graph[0]+args.d_graph[-1],args.d_graph[-1])
        # elif self.final == 'attn':
        #     self.final_layer = MultiHeadAttention(n_head=4, d_input=args.d_graph[0]+args.d_graph[-1], 
        #                             d_model=args.d_graph[-1], d_input_v=args.d_graph[0]+args.d_graph[-1])
        
        d_in = args.d_graph[0]
        for d_out in args.d_graph:
            if 'gcn' in self.model:
                self.gnn_layer.append(GNN_Layer(d_in, d_out, globalnode=False))
            if 'gat' in self.model:
                self.gnn_layer.append(GNN_Att_Layer(n_head=4, d_input=d_in, d_model=d_out, globalnode=False))
            d_in = d_out
        
        self.entity_classification = args.entity_classification
        
        if 'concat' not in self.model:
            self.out_linear1 = nn.Linear(d_in, d_in)
        else:
            self.out_linear1 = nn.Linear(d_in*2, d_in)
        self.out_linear2 = nn.Linear(d_in, d_output)
        
        self.drop = nn.Dropout(p=args.dropout)
        if self.entity_classification:
            self.log_softmax = nn.LogSoftmax(dim=1)
        else:
            self.log_softmax = nn.LogSoftmax(dim=2)

        self.crf = args.crf
        if self.crf:
            self.crf_layer = CRF(d_output)

        self.clear_time()

    def clear_time(self):
        self.cnn_time, self.lstm_time, self.gcn_time = 0, 0, 0
        self.encoder_lstm.clear_time()
        if self.decoder_lstm:
            self.decoder_lstm.clear_time()

    def print_time(self):
        print('cnn %f  lstm %f  gcn %f'%(self.cnn_time, self.lstm_time, self.gcn_time))
        # print([layer.lstm.lstm_time for layer in self.lstm_layer])


    def forward(self, data_char, data_word, length, mask, adj, entity_mask=None):

        # here, batch_size = num_user, docu_len = num_sent
        batch_size, docu_len, sent_len, word_len = data_char.size()
        
        # word representation
        start = time.time()
        char_emb = self.char_cnn(data_char.long().view(-1, word_len), mask.view(-1))
        self.cnn_time += time.time()-start

        char_emb = char_emb.view(batch_size*docu_len, sent_len, -1)
        word_emb = self.word_emb(data_word.view(batch_size*docu_len, -1))
        word_emb = word_emb.view(batch_size * docu_len, sent_len, -1)
        h_word = torch.cat((char_emb, word_emb), dim=2)

        length = length.data.view(batch_size * docu_len)
        mask = mask.view(batch_size * docu_len, sent_len)
        sent_mask = Variable(length.ne(0).view(batch_size, docu_len))

        # encoder
        start = time.time()
        h_word = self.drop(h_word)
        h_word, h_sent = self.encoder_lstm(h_word, length, mask, None)
        self.lstm_time += time.time()-start
        
        # graph neural network
        start = time.time() 
        h_gcn = None
        if len(self.gnn_layer) > 0:
            h_sent = h_sent.view(batch_size, docu_len, -1)
            h_gcn = masked_mean(h_sent, sent_mask).unsqueeze(0)
            adj = adj.unsqueeze(0)
            for i in range(len(self.gnn_layer)):
                h_gcn = self.gnn_layer[i](h_gcn, adj)
            h_gcn = h_gcn.view(batch_size, -1).unsqueeze(1).expand(-1, docu_len, -1)
            h_gcn = h_gcn.contiguous().view(batch_size * docu_len, -1)
        self.gcn_time += time.time()-start

        # decoder
        start = time.time()
        if self.decoder_lstm is not None:
            h_word, h_sent = self.decoder_lstm(h_word, length, mask, h_gcn)
        if self.model in ['lstm-gcn-concat', 'lstm-gat-concat']:
            h_gcn = h_gcn.unsqueeze(1).expand(-1, sent_len, -1)
            h_word = torch.cat([h_word, h_gcn], dim=2)
        self.lstm_time += time.time()-start

        h_word = h_word.view(batch_size, docu_len, sent_len, -1)
        h = h_word[0]

        if self.entity_classification:
        # ## h: num_sent, sent_len, d_input
        # ## entity_mask: num_sent, sent_len
            num_sent, sent_len = entity_mask.size()
            h = h[:num_sent, :sent_len]
            h = masked_mean(h, entity_mask)

        # if self.output_type == 'entity':
        #     ## h: num_sent, d_input
        #     ## entity_sent_mask: num_entity, num_sent
        #     num_entity, num_sent = entity_sent_mask.size()
        #     h = h.unsqueeze(0).expand(num_entity, -1, -1)
        #     h = masked_mean(h, entity_sent_mask)

        h = self.drop(h)
        h = self.out_linear1(h)
        h = F.relu(h)
        h = self.out_linear2(h)

        # output = self.log_softmax(h)

        return h




