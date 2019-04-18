__author__ = 'jindi'

import torch
import torch.nn as nn
from .._functions import LockedDropout, WeightDrop
from ..utils import *

class WeightDropLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, 
                dropout=(0., 0., 0., 0.), bidirectional=True, variational=True):
        super(WeightDropLSTM, self).__init__()
        # Modified LockedDropout that support batch first arrangement
        self.batch_first = batch_first
        self.lockdrop = LockedDropout(batch_first=batch_first)
        self.idrop, self.hdrop, self.wdrop, self.odrop = dropout # input, hidden, weight, output
        self.num_layers = num_layers
        self.rnns = [
            nn.LSTM(input_size if l == 0 else hidden_size,
                   hidden_size // 2, num_layers=1, batch_first=batch_first, 
                   bidirectional=bidirectional, dropout=0)
            for l in range(num_layers)
        ]
        if self.wdrop:
            self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=self.wdrop, variational=variational)
                         for rnn in self.rnns]
        self.rnns = nn.ModuleList(self.rnns)
        
    def forward(self, input, length=None, mask=None):
        raw_output = self.lockdrop(input, self.idrop)
        new_hidden, new_cell_state = [], []
        raw_outputs, outputs = [], []
        for l, rnn in enumerate(self.rnns):
            if length is not None:
                raw_output, _, rev_order, mask = prepare_rnn_seq(raw_output, length, hx=None, 
                                                                    masks=mask, batch_first=self.batch_first)
            raw_output, (new_h, new_c) = rnn(raw_output)
            if length is not None:
                raw_output, _ = recover_rnn_seq(raw_output, rev_order, hx=None, batch_first=self.batch_first)
            raw_outputs.append(raw_output)
            if l != self.num_layers - 1:
                raw_output = self.lockdrop(raw_output, self.hdrop)
                outputs.append(raw_output)         
            new_hidden.append(new_h)
            new_cell_state.append(new_c)
        hidden = torch.cat(new_hidden, 1)
        cell_state = torch.cat(new_cell_state, 1)
        final_output = self.lockdrop(raw_output, self.odrop)
        outputs.append(final_output)

        if length is not None:
            return final_output, (mask, hidden, cell_state, raw_outputs, outputs)
        else:
            return final_output, (hidden, cell_state, raw_outputs, outputs)
