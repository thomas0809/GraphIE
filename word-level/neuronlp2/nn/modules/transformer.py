import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from .._functions import GELU

from efficiency.log import show_var

import pdb

PAD_ID_WORD = 1
max_doc_n_words = 1534 + 2


class AttEncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, n_head, d_graph, d_inner_hid, d_k, d_v, p_gcn):
        super(AttEncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_graph, d_k, d_v, dropout=p_gcn)
        self.pos_ffn = PositionwiseFeedForward(d_graph, d_inner_hid, dropout=p_gcn)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


''' Define the sublayers in encoder/decoder layer '''


def get_attn_padding_mask(seq_q, seq_k):
    ''' Indicate the padding-related part to mask '''
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.eq(PAD_ID_WORD).unsqueeze(1)  # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k)  # bxsqxsk
    return pad_attn_mask


def get_attn_adj_mask(adjs):
    adjs_mask = adjs.ne(0)  # batch*n_node*n_node
    # torch.set_printoptions(precision=None, threshold=float('inf'))
    # pdb.set_trace()

    n_neig = adjs_mask.sum(dim=2)
    adjs_mask[:, :, 0] += n_neig.eq(0)  # this is for making PAD not all zeros

    return adjs_mask.eq(0)


def adj_normalization(adj):
    '''
    symmetric normalization of adjacency matrix, i.e.
    :param adj:
    :return: D^{-0.5} \dot (A+I) \dot D^{-0.5}

    '''
    rowsum = torch.clamp(adj.sum(-1), min=1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)

    diag = torch.zeros_like(adj)
    diag.as_strided(d_inv_sqrt.size(), [diag.stride(0), diag.size(2) + 1]).copy_(d_inv_sqrt)
    normed = torch.bmm(torch.bmm(diag, adj), diag)
    return normed


class PositionEncoder(nn.Module):

    def __init__(self, d_graph, mode="lookup"):
        super(PositionEncoder, self).__init__()

        self.mode = mode
        max_n_node = max_doc_n_words
        d_pos = 1
        if self.mode == "lookup":
            self.position_enc = nn.Embedding(max_n_node, d_graph, padding_idx=0)
            self.position_enc.weight.data = self._position_encoding_init(max_n_node, d_graph)
        elif self.mode == "linear":
            self.position_enc = nn.Linear(d_pos, d_graph)

    def _position_encoding_init(self, n_position, d_pos_vec):
        ''' Init the sinusoid position encoding table '''

        # keep dim 0 for padding token position encoding zero vector
        position_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
            if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        return torch.from_numpy(position_enc).type(torch.FloatTensor)

    def forward(self, pos, h_gcn):

        # x: (N, input_dim)
        if self.mode == "lookup":
            pos_enc = self.position_enc(pos)
            pos_enc = pos_enc.squeeze(2)
        elif self.mode == "linear":
            pos_enc = F.tanh(self.position_enc(pos))
        elif self.mode == "none":
            pos_enc = torch.zeros_like(h_gcn)

        return pos_enc


class Linear(nn.Module):
    ''' Simple Linear layer with xavier init '''

    def __init__(self, d_in, d_out, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class Bottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.view(size[0], size[1], -1)


class BottleLinear(Bottle, Linear):
    ''' Perform the reshape routine before and after a linear projection '''
    pass


class BottleSoftmax(Bottle, nn.Softmax):
    ''' Perform the reshape routine before and after a softmax operation'''
    pass


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1,
                 use_residual=True):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization(d_model)
        self.proj = BottleLinear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual

        init.xavier_normal_(self.w_qs)
        init.xavier_normal_(self.w_ks)
        init.xavier_normal_(self.w_vs)

    def forward(self, q, k, v, attn_mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        residual = q if self.use_residual else 0

        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_value = v.size()

        # get v_s
        v_s = v.repeat(n_head, 1, 1)  # (n_head*mb_size) x len_v x d_value
        if self.use_residual:
            v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_value)  # n_head x (mb_size*len_v) x d_model
            v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)  # (n_head*mb_size) x len_v x d_v

        # treat as a (n_head) size batch
        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_q) x d_model
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_k) x d_model

        # treat the result as a (n_head * mb_size) size batch
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)  # (n_head*mb_size) x len_q x d_k
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)  # (n_head*mb_size) x len_k x d_k

        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        attn_mask = attn_mask.repeat(n_head, 1, 1) if attn_mask is not None else attn_mask
        outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=attn_mask)

        if self.use_residual:
            # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
            outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1)
            # project back to residual size
            outputs = self.proj(outputs)
        else:
            outputs = outputs.mean(0, True)
            attns = attns.mean(0, True)

        outputs = self.dropout(outputs)

        if self.use_residual:
            outputs = self.layer_norm(outputs + residual)

        return outputs, attns


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1)  # position-wise
        self.layer_norm = LayerNormalization(d_hid)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.elu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return self.layer_norm(output + residual)


class LayerNormalization(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = BottleSoftmax(dim=-1)

    def forward(self, q, k, v, attn_mask=None, show_net=False):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        # cos sim needs normalization

        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                'Attention mask shape {} mismatch ' \
                'with Attention logit tensor shape ' \
                '{}.'.format(attn_mask.size(), attn.size())

            attn.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn)  # attn: [32, 27, 27]
        if attn_mask is not None:
            attn.data.masked_fill_(attn_mask, 0)  # convert NaN to 0

        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class WeightedScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_model, attn_dropout=0.1, ff_layers=1, ff_drop_p=0.2,
                 comb_mode=2):
        super(WeightedScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = BottleSoftmax(dim=-1)

        self.ff_layers = ff_layers
        self.linear1 = nn.Linear(d_model, d_model)
        self.ff_dropout = nn.Dropout(ff_drop_p)
        if ff_layers == 2:
            self.linear2 = nn.Linear(d_model, d_model)
            self.elu = GELU()

        if comb_mode == 1:
            self.comb_att_n_init_adj = add_n_norm
        elif comb_mode == 2:
            self.comb_att_n_init_adj = learn_n_norm
        elif comb_mode == 0:
            self.comb_att_n_init_adj = use_init_adj
        self.comb_mode = comb_mode

    def _fc(self, lin, q, k, use_elu=True):

        # assume q == k

        q_out = lin(q)
        q_out = self.ff_dropout(q_out)

        if use_elu:
            q_out = self.elu(q_out)

        return q_out, q_out

    def forward(self, q, k, v, attn_mask=None, show_net=False):
        assert len(q.size()) == 3

        if self.ff_layers == 1:
            q_out, k_out = self._fc(self.linear1, q, k, use_elu=False)
            if show_net:
                show_var(["self.linear1"])

        elif self.ff_layers == 2:
            q_out, k_out = self._fc(self.linear1, q, k)
            q_out, k_out = self._fc(self.linear2, q_out, k_out, use_elu=False)
            if show_net:
                show_var(["self.linear1", "self.linear2"])


        if show_net:
            print("bmm --> self.dropout")
            show_var(["self.comb_att_n_init_adj"])

        attn = torch.bmm(q_out, k_out.transpose(1, 2))  # / self.temper

        # cos sim needs normalization

        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                'Attention mask shape {} mismatch ' \
                'with Attention logit tensor shape ' \
                '{}.'.format(attn_mask.size(), attn.size())

            attn.masked_fill_(attn_mask, -float('inf'))

        if attn_mask is not None:
            attn.data.masked_fill_(attn_mask, 0)  # convert NaN to 0

        attn = self.dropout(attn)
        import pdb; pdb.set_trace()
        output = self.comb_att_n_init_adj(attn, v)

        return output, attn


def add_n_norm(attn, v):
    output = attn + v
    output = adj_normalization(output)
    return output


def learn_n_norm(attn, v):
    output = adj_normalization(attn)
    return output


def use_init_adj(attn, v):
    return v


class ConcatProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_model, attn_dropout=0.1, ff_drop_p=0.2, use_elu=False):
        super(ConcatProductAttention, self).__init__()
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = BottleSoftmax(dim=-1)

        self.linear1 = nn.Linear(d_model, 1)
        self.linear2 = nn.Linear(d_model, 1)
        self.ff_dropout = nn.Dropout(ff_drop_p)

        if use_elu:
            self.elu = GELU()

    def _fc(self, lin, q, use_elu=False):

        q_out = lin(q)
        q_out = self.ff_dropout(q_out)
        if use_elu:
            q_out = self.elu(q_out)

        return q_out

    def forward(self, q, k, v, attn_mask=None, show_net=False):

        batch, sent_len, dim = q.size()

        q_out = self._fc(self.linear1, q)
        k_out = self._fc(self.linear2, k)

        k_out = k_out.permute(0, 2, 1)

        q_out = q_out.expand(batch, sent_len, sent_len)
        k_out = k_out.expand(batch, sent_len, sent_len)

        attn = q_out + k_out

        # cos sim needs normalization

        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                'Attention mask shape {} mismatch ' \
                'with Attention logit tensor shape ' \
                '{}.'.format(attn_mask.size(), attn.size())

            attn.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn)  # attn: [32, 27, 27]
        if attn_mask is not None:
            attn.data.masked_fill_(attn_mask, 0)  # convert NaN to 0

        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn
