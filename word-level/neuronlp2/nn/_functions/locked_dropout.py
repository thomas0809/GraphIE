__author__ = 'jindi'

import torch
import torch.nn as nn
from torch.autograd import Variable

class LockedDropout(nn.Module):
    def __init__(self, batch_first=True):
        super(LockedDropout, self).__init__()
        self.batch_first = batch_first

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        if self.batch_first:
        	m = x.data.new(x.size(0), 1,  x.size(2)).bernoulli_(1 - dropout)
        else:
        	m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x