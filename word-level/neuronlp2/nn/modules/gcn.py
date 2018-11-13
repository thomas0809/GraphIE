import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import AttEncoderLayer, get_attn_padding_mask, get_attn_adj_mask, \
    LayerNormalization, PositionEncoder, MultiHeadAttention, ScaledDotProductAttention, \
    WeightedScaledDotProductAttention, ConcatProductAttention
from .._functions import GELU

from efficiency.log import show_var


class GCN(nn.Module):
    def __init__(self, gcn_model, n_gcn_layer, d_graph, d_in, p_gcn,
                 n_head=4, d_inner_hid=256, d_k=32, d_v=32,
                 position_enc_mode="lookup", globalnode=False,
                 adj_attn_type=''):
        super(GCN, self).__init__()

        self.gcn_model = gcn_model
        n_graph = 3 if gcn_model == 'gnn' else 1
        self.adj_attn_type = adj_attn_type

        self.to_graph = nn.Linear(d_in, d_graph)
        self.dropout_gcn_in = nn.Dropout(p_gcn[0])

        if adj_attn_type == 'cossim':
            self.adj_attn = ScaledDotProductAttention(d_graph)
        elif adj_attn_type == 'multihead':
            self.adj_attn = MultiHeadAttention(n_head, d_graph, d_k, d_v, dropout=0.1,
                                               use_residual=False)
        elif adj_attn_type == 'flex_cossim':
            self.adj_attn = WeightedScaledDotProductAttention(d_graph)
        elif adj_attn_type == 'flex_cossim2':
            self.adj_attn = WeightedScaledDotProductAttention(d_graph, ff_layers=2)
        elif adj_attn_type == 'concat':
            self.adj_attn = ConcatProductAttention(d_graph)

        self.position_enc = PositionEncoder(d_graph, mode=position_enc_mode)

        self.gcn_layers = nn.ModuleList([
            GNN_Layer(d_graph, d_graph, p_gcn[1], n_graph=n_graph)
            if gcn_model in ['gnn', 'gnn1']

            else GNN_Att_Layer(n_head=n_head, d_input=d_graph, d_model=d_graph,
                               globalnode=globalnode)  # GNN_Pos_Att_Layer(d_graph, d_graph)
            if gcn_model == 'gnnattn'

            else AttEncoderLayer(n_head, d_graph, d_inner_hid, d_k, d_v, p_gcn[1])
            if 'transformer' in gcn_model

            else None

            for _ in range(n_gcn_layer)

        ])

    def forward(self, h_gcn, adjs, doc_word_mask, return_edge=False, show_net=False):

        gcn_model = self.gcn_model

        posi = [sent.nonzero().view(-1) for sent in doc_word_mask]
        posi = torch.stack(posi) + 1

        batch_size, n_node = h_gcn.size()[:2]

        slf_attn_pad_mask = get_attn_padding_mask(doc_word_mask, doc_word_mask)

        if len(self.gcn_layers) > 0:
            if show_net:
                print("<")
                print("[Net_gcn] gcn prep:")
                show_var(["self.to_graph", "F.elu", "self.dropout_gcn_in", "self.position_enc"])

            h_gcn = self.dropout_gcn_in(F.elu(self.to_graph(h_gcn)))

            # h_gcn = torch.cat((init_enc, pos_enc), dim=2)
            h_gcn = h_gcn + self.position_enc(posi, h_gcn)


            if self.adj_attn_type:
                if show_net:
                    show_var(["self.adj_attn_type"])
                old_adjs = adjs.clone()
                coref_adj = old_adjs[:, 0]
                new_adj, attns = self.adj_attn(h_gcn, h_gcn, coref_adj, show_net=show_net)
                adjs[:, 0] = new_adj

        if return_edge:
            edge_weights = []
        adjs = adjs if gcn_model in ['gnn', 'gnn1'] else adjs.squeeze(1)

        slf_attn_adj_mask = get_attn_adj_mask(adjs)

        opts = (adjs, doc_word_mask) if gcn_model in ['gnn', 'gnn1'] \
            else (adjs,) if gcn_model == 'gnnattn' \
            else (slf_attn_pad_mask,) if gcn_model == 'transformer' \
            else (slf_attn_adj_mask,) if gcn_model == 'transformer_graph' \
            else None

        for layer_i, layer in enumerate(self.gcn_layers):
            if show_net:
                print("gcn [{m}] layer {i}: {l}".format(m=gcn_model, i=layer_i, l=layer))
                print(">gcn")

            h_gcn, edge_weig = layer(h_gcn, *opts)
            if return_edge:
                edge_weights += [edge_weig]

        h_gcn = h_gcn.view(batch_size, n_node, -1)

        if return_edge:
            edge_weights = torch.stack(edge_weights)

        if return_edge:
            return h_gcn, torch.stack(edge_weights)
        else:
            return h_gcn,


class GNN_Layer(nn.Module):

    def __init__(self, d_input, d_model, p_gcn, n_graph=1):
        super(GNN_Layer, self).__init__()
        self.linear_gcn = nn.Linear(d_input * n_graph, d_model)
        self.n_graph = n_graph

        self.elu = GELU()
        self.norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(p=p_gcn)

    def forward(self, x, adjs, mask=None):
        # x: (batch, N, d_input)
        # adjs: (batch, n_graph, N, N)
        assert len(adjs.size()) == 4
        batch, n_node, _ = x.size()
        assert adjs.size(1) == self.n_graph

        h = x.clone()
        x = x.unsqueeze(1).expand(-1, self.n_graph, -1, -1)
        h_gcn = torch.matmul(adjs, x).transpose(1, 2).contiguous().view(batch, n_node,
                                                                        -1)  # (batch, N, n_graph * d_input)

        # self.linear_gcn.weight.transpose(1,0).data.size():[384, 128]

        h_gcn = self.linear_gcn(h_gcn)
        d = adjs.sum(dim=3).sum(dim=1).unsqueeze(2)
        d = d + d.eq(0).float()
        h = h + h_gcn / d  # (batch, N, d_model)

        h = self.elu(h)
        h = self.norm(h)
        h = self.dropout(h)
        return h, None
        # _h1 = self.sparse_mm(a1, x)
        # _h2 = self.sparse_mm(a2, x)
