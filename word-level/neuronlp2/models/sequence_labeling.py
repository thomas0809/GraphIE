__author__ = 'jindi'

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..nn import ChainCRF, GCN, WeightDropLSTM
from ..nn import utils, embedded_dropout
from ..io.Constants import PAD_ID_WORD
# from allennlp.modules.elmo import Elmo, batch_to_ids
from efficiency.log import show_var

options_file = '../../data/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json'
weight_file = '../../data/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'


class DocModel(nn.Module):

    def __init__(self):
        super(DocModel, self).__init__()

    def _doc2sent(self, doc_word, doc_char, doc_target=None, show_net=False):
        if show_net:
            print("[Net] _doc2sent")

        batch_size, n_sent, sent_len, word_len = doc_char.size()
        doc_mask = doc_word.ne(PAD_ID_WORD)
        doc_length = doc_mask.sum(-1)
        doc_n_sent = (doc_length > 0).sum(-1)  # (batch_size,)

        input_word = doc_word.view(batch_size * n_sent, sent_len)
        input_char = doc_char.view(batch_size * n_sent, sent_len, word_len)
        target = doc_target.view(
            batch_size * n_sent, sent_len) if doc_target is not None else None
        mask = doc_mask.view(batch_size * n_sent, sent_len)
        length = doc_length.view(batch_size * n_sent)

        input_word = input_word[length > 0]
        input_char = input_char[length > 0]
        target = target[length > 0] if doc_target is not None else None
        mask = mask[length > 0]
        length = length[length > 0]

        return input_word, input_char, target, mask.float(), length, doc_n_sent

    def _sent2word(self, sent_flat, mask_sent_flat, doc_n_sent, show_net=False):
        '''

        :param sent_flat:
        :param mask_sent_flat:
        :param doc_n_sent:
        :return:
        This has been debugged by calling _sent2word and _word2sent consecutively, nothing changed
        '''
        if show_net:
            print("[Net] _sent2word")

        # sent_flat is [n_sent, sent_len, dim]
        # mask_sent_flat is [n_sent, sent_len]
        batch_size = len(doc_n_sent)

        # [batch,] for doc start idx
        doc_sent_st = [doc_n_sent[:i].sum() for i in range(batch_size)]
        # word_enc for packed_doc [batch, n_word, dim]
        words_in_doc = [sent_flat[st:st + lens][mask_sent_flat[st:st + lens] != 0]
                        for st, lens in zip(doc_sent_st, doc_n_sent)]

        # words in doc [batch, n_word, dim]
        # mask of words_in_doc [batch, n_word]
        pad_w_in_doc, pad_mask_w_in_doc = utils.list2padseq(
            words_in_doc, doc_n_sent)

        return pad_w_in_doc, pad_mask_w_in_doc

    def _word2sent(self, word_in_doc, mask_w_in_doc, sent_len, mask_sent_flat, show_net=False):
        if show_net:
            print("[Net] _word2sent")

        batch_size, n_words, h_dim = word_in_doc.size()

        words_flat = word_in_doc.view(batch_size * n_words, h_dim)
        mask_flat = mask_w_in_doc.view(batch_size * n_words)
        words_flat = words_flat[mask_flat != 0]
        assert len(words_flat) == sent_len.sum()

        n_sent = len(sent_len)
        idx = [sent_len[:i].sum() for i in range(n_sent)]
        # flattened sents [n_sent, sent_len, dim]
        sents = [words_flat[st:end] for st, end in zip(idx, idx[1:] + [None])]

        # flattened sents [n_sents, sent_len, h_dim]
        # flattened sents mask [n_sents, sent_len]
        sents, _ = utils.list2padseq(sents, sent_len, padding_value=0)

        return sents

    def forward(self, doc_word, doc_char, doc_target=None):
        return self._doc2sent(doc_word, doc_char, doc_target)


class BiRecurrentConv(DocModel):

    def __init__(self, word_dim, num_words, char_dim, num_chars, char_hidden_size, kernel_size, rnn_mode, encoder_mode,
                 hidden_size, num_layers, num_labels,
                 char_method='cnn', tag_space=0, embedd_word=None, embedd_char=None, use_elmo=False, p_em_vec=0.0,
                 p_em=0.33, p_in=0.33, p_tag=0.5, p_rnn=(0.5, 0.5, 0.5), initializer=None):
        super(BiRecurrentConv, self).__init__()

        self.word_embedd = nn.Embedding(num_words, word_dim, _weight=embedd_word)
        self.char_embedd = nn.Embedding(num_chars, char_dim, _weight=embedd_char)

        # ELMO module
        if use_elmo:
            self.elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0.5)
        else:
            self.elmo = None

        # embedding vector dropout
        self.p_em_vec = p_em_vec

        # dropout word
        self.dropout_em = nn.Dropout2d(p=p_em)
        # standard dropout
        self.dropout_rnn_in = nn.Dropout(p=p_rnn[0])
        self.dropout_rnn_out = nn.Dropout(p=p_rnn[2])

        if rnn_mode == 'RNN':
            self.RNN = nn.RNN
        elif rnn_mode == 'LSTM':
            self.RNN = nn.LSTM
        elif rnn_mode == 'GRU':
            self.RNN = nn.GRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)

        self.char_method = char_method
        self.encoder_mode = encoder_mode

        if char_method == 'cnn':
            self.char_conv1d = nn.Conv1d(char_dim, char_hidden_size, kernel_size, padding=kernel_size // 2)
        else:
            self.char_rnn = self.RNN(char_dim, char_hidden_size // 2, num_layers=num_layers,
                                     batch_first=True, bidirectional=True, dropout=p_rnn[1])

        if self.elmo:
            input_hidden_size = word_dim + char_hidden_size + 1024
        else:
            input_hidden_size = word_dim + char_hidden_size

        if encoder_mode == 'cnn':
            self.sent_conv1d_layer1 = nn.Conv1d(input_hidden_size, hidden_size, kernel_size, padding=kernel_size // 2)
            self.sent_conv1d_layer2 = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2)
        else:
            self.sent_rnn = self.RNN(input_hidden_size, hidden_size // 2, num_layers=num_layers,
                                     batch_first=True, bidirectional=True, dropout=p_rnn[1])

        # if encoder_mode == 'cnn':
        #     out_dim = hidden_size + word_dim + char_hidden_size
        # else:
        # out_dim = hidden_size
        out_dim = hidden_size
        self.tag_space = tag_space

        if tag_space:
            self.dropout_tag = nn.Dropout(p_tag)
            self.lstm_to_tag_space = nn.Linear(out_dim, tag_space)
            out_dim = tag_space
        self.dense_softmax = nn.Linear(out_dim, num_labels)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.nll_loss = nn.NLLLoss(size_average=False, reduce=False)

        self.initializer = initializer
        self.reset_parameters()

    def reset_parameters(self):
        if self.initializer is None:
            return

        for name, parameter in self.named_parameters():
            if name.find('embedd') == -1:
                if parameter.dim() == 1:
                    nn.init.constant_(parameter, 0.)
                else:
                    self.initializer(parameter)

    def _get_word_enc(self, input_word_orig, input_word, input_char, mask=None, length=None, show_net=False):
        # hack length from mask
        # we do not hack mask from length for special reasons.
        # Thus, always provide mask if it is necessary.
        if length is None and mask is not None:
            length = mask.sum(dim=1).long()

        if self.p_em_vec:
            word = embedded_dropout(self.word_embedd, input_word, dropout=self.p_em_vec if self.training else 0)
            char = embedded_dropout(self.char_embedd, input_char, dropout=self.p_em_vec if self.training else 0)
        else:
            # [batch, length, word_dim]
            word = self.word_embedd(input_word)
            # [batch, length, char_length, char_dim]
            char = self.char_embedd(input_char)

        char_size = char.size()

        if self.char_method == 'cnn':
            # first transform to [batch *length, char_length, char_dim]
            # then transpose to [batch * length, char_dim, char_length]
            char = char.view(char_size[0] * char_size[1], char_size[2], char_size[3]).transpose(1, 2)
            # put into cnn [batch*length, char_filters, char_length]
            # then put into maxpooling [batch * length, char_filters]
            char, _ = self.char_conv1d(char).max(dim=2)
            # reshape to [batch, length, char_filters]
            char = torch.tanh(char).view(char_size[0], char_size[1], -1)
        else:
            # first transform to [batch *length, char_length, char_dim]
            char = char.view(char_size[0] * char_size[1], char_size[2], char_size[3])
            # put into rnn module and get the last hidden state
            _, (char, _) = self.char_rnn(char)
            # reshape to [batch, length, char_hidden_size]
            char = char.view(char_size[0], char_size[1], -1)

        # apply dropout word on input
        word = self.dropout_em(word)
        char = self.dropout_em(char)

        # concatenate word and char [batch, length, word_dim+char_filter]
        # choose whether to concatenate the ELMO embeddings
        if self.elmo:
            elmo_embeddings = self.elmo(batch_to_ids(input_word_orig))
            # TODO: the coefficient for elmo needs to be tuned
            input = torch.cat([word, char, 0.1 * elmo_embeddings], dim=2)
        else:
            input = torch.cat([word, char], dim=2)

        if show_net:
            print("[Net] _get_word_enc: torch.cat([word {}, char {}]".format(word.shape[-1], char.shape[-1]))
            show_var(["self.dropout_em"])
        return input, length

    def _get_rnn_enc(self, input, length, mask, hx, show_net=False):
        if show_net:
            print('[Net] _get_rnn_enc')
            show_var(["self.dropout_rnn_in"])

        # apply dropout rnn input
        input = self.dropout_rnn_in(input)

        # use lstm or cnn to encode the sentence at token level
        if self.encoder_mode == 'lstm':
            # prepare packed_sequence
            if length is not None:
                seq_input, hx, rev_order, mask = utils.prepare_rnn_seq(input, length, hx=hx, masks=mask,
                                                                       batch_first=True)
                seq_output, hn = self.sent_rnn(seq_input, hx=hx)
                output, hn = utils.recover_rnn_seq(seq_output, rev_order, hx=hn, batch_first=True)
                if show_net:
                    print("utils.prepare_rnn_seq()")
                    show_var(["self.sent_rnn"])
            else:
                output, hn = self.sent_rnn(input, hx=hx)
                if show_net:
                    show_var(["self.sent_rnn"])
        else:
            _, _, _, mask = utils.prepare_rnn_seq(input, length, hx=hx, masks=mask, batch_first=True)
            if length is not None:
                max_len = length.max()
                input = input[:, :max_len, :]

            # first transpose to [batch, hidden_size, length]
            input = input.transpose(1, 2)
            # then send into the first cnn layer
            output = torch.relu(self.sent_conv1d_layer1(input))
            # then second cnn layer
            output = torch.relu(self.sent_conv1d_layer2(output))
            # transpose to [batch, length, hidden_size]
            output = output.transpose(1, 2)

            # output = torch.cat([input.transpose(1, 2), output], dim=2)

            hn = None

        # apply dropout for the output of rnn
        output = self.dropout_rnn_out(output)
        if show_net:
            show_var(["self.dropout_rnn_out"])

        return output, hn

    def _get_rnn_output(self, input_word_orig, input_word, input_char,
                        mask=None, length=None, hx=None, show_net=False):

        input, length = self._get_word_enc(
            input_word_orig, input_word, input_char, mask=mask, length=length, show_net=show_net)

        output, hn = self._get_rnn_enc(input, length, mask, hx, show_net=show_net)

        if self.tag_space:
            # [batch, length, tag_space]
            output = self.dropout_tag(F.elu(self.lstm_to_tag_space(output)))
            if show_net:
                print("[Net] to_tag")
                show_var(["self.lstm_to_tag_space"])
                show_var(["F.elu"])
                show_var(["self.dropout_tag"])

        return output, hn, mask, length

    def forward(self, input_word, input_char, mask=None, length=None, hx=None):
        if len(input_word.size()) == 3:
            # input_word is the packed sents [n_sent, sent_len]
            input_word, input_char, _, mask, length, _ = self._doc2sent(
                input_word, input_char)
        # output from rnn [batch, length, tag_space]
        output, _, mask, length = self._get_rnn_output(input_word_orig, input_word, input_char, mask=mask,
                                                       length=length, hx=hx)
        return output, mask, length

    def loss(self, input_word_orig, input_word, input_char, target, mask=None, length=None, hx=None, leading_symbolic=0,
             show_net=False):
        # [batch, length, tag_space]
        output, mask, length = self.forward(input_word_orig, input_word, input_char, mask=mask, length=length, hx=hx)
        # [batch, length, num_labels]
        output = self.dense_softmax(output)
        # preds = [batch, length]
        _, preds = torch.max(output[:, :, leading_symbolic:], dim=2)
        preds += leading_symbolic

        output_size = output.size()
        # [batch * length, num_labels]
        output_size = (output_size[0] * output_size[1], output_size[2])
        output = output.view(output_size)

        if length is not None and target.size(1) != mask.size(1):
            max_len = length.max()
            target = target[:, :max_len].contiguous()

        if mask is not None:
            return (self.nll_loss(self.logsoftmax(output), target.view(-1)) * mask.contiguous().view(
                -1)).sum() / mask.sum(), \
                   (torch.eq(preds, target).type_as(mask) * mask).sum(), preds
        else:
            num = output_size[0] * output_size[1]
            return self.nll_loss(self.logsoftmax(output), target.view(-1)).sum() / num, \
                   (torch.eq(preds, target).type_as(output)).sum(), preds


class BiRecurrentConvCRF(BiRecurrentConv):
    def __init__(self, word_dim, num_words, char_dim, num_chars, char_hidden_size, kernel_size, rnn_mode, encoder_mode,
                 hidden_size, num_layers, num_labels,
                 char_method='cnn', tag_space=0, embedd_word=None, embedd_char=None, use_elmo=False, p_em_vec=0.0,
                 p_em=0.33, p_in=0.33, p_tag=0.5, p_rnn=(0.5, 0.5, 0.5), bigram=False, initializer=None):
        super(BiRecurrentConvCRF, self).__init__(word_dim, num_words, char_dim, num_chars, char_hidden_size,
                                                 kernel_size, rnn_mode, encoder_mode, hidden_size, num_layers,
                                                 num_labels, char_method=char_method, tag_space=tag_space,
                                                 embedd_word=embedd_word, embedd_char=embedd_char,
                                                 use_elmo=use_elmo, p_em=p_em, p_in=p_in, p_tag=p_tag, p_rnn=p_rnn,
                                                 initializer=initializer)

        out_dim = tag_space if tag_space else hidden_size * 2
        self.crf = ChainCRF(out_dim, num_labels, bigram=bigram)
        self.dense_softmax = None
        self.logsoftmax = None
        self.nll_loss = None
        self.char_method = char_method

    def forward(self, input_word_orig, input_word, input_char, _, target, mask=None, length=None, hx=None,
                leading_symbolic=0):
        return self.loss(input_word_orig, input_word, input_char, _, target, mask=mask, length=length, hx=hx,
                         leading_symbolic=leading_symbolic)

    def loss(self, input_word_orig, input_word, input_char, _, target, mask=None,
             length=None, hx=None, leading_symbolic=0, show_net=False):

        if len(input_word.size()) == 3:
            # input_word is the packed sents [n_sent, sent_len]
            input_word, input_char, target, mask, length, doc_n_sent = self._doc2sent(
                input_word, input_char, target, show_net=show_net)

        # output from rnn [batch, length, tag_space]
        output, _, mask, length = self._get_rnn_output(input_word_orig, input_word, input_char, mask=mask,
                                                       length=length, hx=hx, show_net=show_net)

        if length is not None:
            max_len = length.max()
            target = target[:, :max_len]

        # [batch, length, num_label,  num_label]
        return self.crf.loss(output, target, mask=mask).mean()

    def decode(self, input_word_orig, input_word, input_char, _, target=None, mask=None, length=None, hx=None,
               leading_symbolic=0):
        if len(input_word.size()) == 3:
            # input_word is the packed sents [n_sent, sent_len]
            input_word, input_char, target, sent_mask, length, doc_n_sent = self._doc2sent(
                input_word, input_char, target)
        # output from rnn [batch, length, tag_space]
        output, _, mask, length = self._get_rnn_output(input_word_orig, input_word, input_char, mask=mask,
                                                       length=length, hx=hx)

        if target is None:
            return self.crf.decode(output, mask=mask, leading_symbolic=leading_symbolic), None

        if length is not None:
            max_len = length.max()
            target = target[:, :max_len]

        preds = self.crf.decode(output, mask=mask, leading_symbolic=leading_symbolic)
        if mask is None:
            return preds, torch.eq(preds, target).float().sum()
        else:
            return preds, (torch.eq(preds, target).float() * mask).sum()


class BiRecurrentConvGraphCRF(BiRecurrentConvCRF):

    def __init__(self, word_dim, num_words, char_dim, num_chars, char_hidden_size, kernel_size, rnn_mode, encoder_mode,
                 hidden_size, num_layers, num_labels,
                 gcn_model, n_head, d_graph, d_inner_hid, d_k, d_v, p_gcn, n_gcn_layer,
                 d_out, post_lstm=1, mask_singles=False, position_enc_mode="lookup", adj_attn='',
                 adj_loss_lambda=0,
                 char_method='cnn', tag_space=0, embedd_word=None, embedd_char=None, use_elmo=False, p_em_vec=0.0,
                 p_em=0.33, p_in=0.33, p_tag=0.5, p_rnn=(0.5, 0.5, 0.5), p_rnn2=(0.5, 0.5, 0.5), bigram=False,
                 initializer=None):

        super(BiRecurrentConvGraphCRF, self).__init__(word_dim, num_words, char_dim, num_chars, char_hidden_size,
                                                      kernel_size, rnn_mode, encoder_mode, hidden_size, num_layers,
                                                      num_labels, char_method=char_method,
                                                      tag_space=tag_space, embedd_word=embedd_word,
                                                      embedd_char=embedd_char, use_elmo=use_elmo, p_em_vec=p_em_vec,
                                                      p_em=p_em,
                                                      p_in=p_in, p_tag=p_tag, p_rnn=p_rnn, bigram=bigram,
                                                      initializer=initializer)
        self.post_lstm = post_lstm
        self.mask_singles = mask_singles

        self.gcn = GCN(gcn_model, n_gcn_layer, d_graph, hidden_size, p_gcn,
                       n_head=n_head, d_inner_hid=d_inner_hid, d_k=d_k, d_v=d_v,
                       position_enc_mode=position_enc_mode, globalnode=False, adj_attn_type=adj_attn)

        d_rnn2_in = d_graph if n_gcn_layer > 0 else hidden_size
        if post_lstm:
            self.dropout_rnn2_in = nn.Dropout(p_rnn2[0])
            self.rnn2 = self.RNN(d_rnn2_in, d_out // 2, num_layers=num_layers,
                                 batch_first=True, bidirectional=True, dropout=p_rnn2[1])
            self.dropout_rnn2_out = nn.Dropout(p_rnn2[2])

        if tag_space:
            if not post_lstm:
                d_out = d_rnn2_in
            self.dropout_tag = nn.Dropout(p_tag)
            self.to_tag_space = nn.Linear(d_out, tag_space)

    def _get_rnn_enc2(self, encoding, length, mask, hx, show_net=False):
        if show_net:
            print("<")
            print("[Net] _get_rnn_enc2")
            show_var(["self.dropout_rnn2_in"])
        # prepare packed_sequence
        encoding = self.dropout_rnn2_in(encoding)

        if length is not None:
            seq_input, hx, rev_order, mask = utils.prepare_rnn_seq(
                encoding, length, hx=hx, masks=mask, batch_first=True)
            seq_output, hn = self.rnn2(seq_input, hx=hx)
            output, hn = utils.recover_rnn_seq(
                seq_output, rev_order, hx=hn, batch_first=True)
            if show_net:
                print("utils.prepare_rnn_seq()")
                show_var(["self.rnn2"])
        else:
            # output from rnn_out [batch, length, hidden_size]
            output, hn = self.rnn2(encoding, hx=hx)
            if show_net:
                show_var(["self.rnn2"])

        output = self.dropout_rnn2_out(output)
        if show_net:
            show_var(["self.dropout_rnn2_out"])

        return output, hn

    def forward(self, input_word_orig, input_word, input_char, adjs, target, mask=None, length=None, hx=None,
                leading_symbolic=0, return_edge=False):
        assert len(input_word.size()) == 3, "the input is not document level"
        # input_word is the packed sents [n_sent, sent_len]

        return input_word[0]

        return self.loss(input_word_orig, input_word, input_char, adjs, target, mask=mask, length=length, hx=hx,
                         leading_symbolic=leading_symbolic, return_edge=return_edge)

    def _get_gcn_output(self, input_word_orig, input_word, input_char, adjs, target=None, mask=None, length=None,
                        hx=None, leading_symbolic=0, return_edge=False, show_net=False, graph_types=['coref']):
        if "wonderful" in graph_types:
            gold_adj = adjs[:, -1, :].clone()
            gnn_adjs = adjs[:, :-1, :]

        mask_singles = self.mask_singles

        assert len(input_word.size()) == 3, "the input is not document level"
        # input_word is the packed sents [n_sent, sent_len]
        input_word, input_char, target, sent_mask, length, doc_n_sent = self._doc2sent(
            input_word, input_char, target, show_net=show_net)

        # input: [n_sent, sent_len, enc_dim]
        input, length = self._get_word_enc(
            input_word_orig, input_word, input_char, mask=sent_mask, length=length, show_net=show_net)

        # output from rnn [n_sent, sent_len, enc_dim]
        sent_output, hn = self._get_rnn_enc(input, length, sent_mask, hx, show_net=show_net)

        # flatten sents to words [batch, n_word, dim]
        # mask for packed_doc [batch, n_word]
        output, doc_word_mask = self._sent2word(sent_output, sent_mask, doc_n_sent, show_net=show_net)

        # enc for non-repetitive words

        if mask_singles:
            if show_net:
                print("[Net] Block singles from here.")

            coref_ix = 0
            # single is 1, repetitive word is 0
            single_mask = gnn_adjs[:, coref_ix].sum(-1, keepdim=True).eq(0).float()
            sent_single_mask = self._word2sent(single_mask, doc_word_mask, length, sent_mask, show_net=show_net)
            singles = sent_output * sent_single_mask.expand_as(sent_output)
            if self.tag_space:
                # [batch, length, tag_space]
                singles = self.dropout_tag(
                    F.elu(self.lstm_to_tag_space(singles)))
                if show_net:
                    print("singles -> self.lstm_to_tag_space")
                singles = singles * sent_single_mask.expand_as(singles)

            # [batch, n_word, d_graph]
            output = output * (1 - single_mask).expand_as(output)

        # go thru gcn [batch, n_word, d_graph]
        h_gcn, *_ = self.gcn(output, gnn_adjs,
                             doc_word_mask, return_edge=return_edge, show_net=show_net)

        output = self._word2sent(h_gcn, doc_word_mask, length, sent_mask, show_net=show_net)

        if self.post_lstm:
            # output from rnn [n_sent, sent_len, enc_dim]
            output, hn = self._get_rnn_enc2(output, length, sent_mask, hx, show_net=show_net)

        # output from rnn_out [batch, length, tag_space]
        output = self.dropout_tag(F.elu(self.to_tag_space(output)))
        if show_net:
            print("<")
            print("[Net] to_tag")
            show_var(["self.to_tag_space"])
            show_var(["F.elu"])
            show_var(["self.dropout_tag"])
            print(">")

        if mask_singles:
            output = output * (1 - sent_single_mask).expand_as(output)
            output = output + singles  # repetive word enc + single word enc
            if show_net:
                print("[Net] output + singles")

        if length is not None:
            max_len = length.max()
            target = target[:, :max_len]

        adj_loss = self._adj_loss(gnn_adjs[:, 0, :], gold_adj) if "wonderful" in graph_types else 0
        return output, target, sent_mask, length, adj_loss

    def _adj_loss(self, coref_adj, gold_adj):
        '''
        This is the same as an average of element_wise cross_entropy
        The only constraint is (coref_adj.shape == gold_adj.shape)
        :param coref_adj: a matrix of 0~1 values
        :param gold_adj: a matrix, (gold_adj.sum(-1) == 1).all() == True
        :return: loss
        '''
        loss_fn = nn.BCELoss()
        softmax = nn.Softmax()

        assert (coref_adj.shape == gold_adj.shape)
        # assert len(coref_adj.shape) == 2
        # assert (coref_adj.sum(-1) == 1).all()
        # assert (gold_adj.sum(-1) == 1).all()

        coref_adj_for_comp = torch.clamp(coref_adj, 0., 1.)

        return loss_fn(coref_adj_for_comp, gold_adj)

    def loss(self, input_word_orig, input_word, input_char, adjs, target, graph_types=['coref'],
             lambda1=1., lambda2=0.,
             mask=None, length=None, hx=None, leading_symbolic=0, return_edge=False, show_net=False):
        output, target, sent_mask, _, adj_loss = self._get_gcn_output(input_word_orig, input_word, input_char, adjs,
                                                                      target,
                                                                      mask=mask, length=length, hx=hx,
                                                                      leading_symbolic=leading_symbolic,
                                                                      return_edge=return_edge,
                                                                      show_net=show_net, graph_types=graph_types)

        # [batch, length, num_label,  num_label]
        # import pdb;
        # pdb.set_trace()
        ner_loss = self.crf.loss(output, target, mask=sent_mask).mean()
        total_loss = lambda1 * ner_loss + lambda2 * adj_loss
        return total_loss, (ner_loss, adj_loss)

    def decode(self, input_word_orig, input_word, input_char, adjs, target=None, mask=None, length=None, hx=None,
               leading_symbolic=0, graph_types=['coref']):
        # output from rnn [batch, length, tag_space]

        output, target, sent_mask, length, _ = self._get_gcn_output(input_word_orig, input_word, input_char, adjs,
                                                                    target,
                                                                    mask=mask, length=length, hx=hx,
                                                                    leading_symbolic=leading_symbolic,
                                                                    graph_types=graph_types)

        if target is None:
            return self.crf.decode(output, mask=sent_mask, leading_symbolic=leading_symbolic), None

        preds = self.crf.decode(output, mask=sent_mask,
                                leading_symbolic=leading_symbolic)
        if mask is None:
            return preds, torch.eq(preds, target).float().sum()
        else:
            return preds, (torch.eq(preds, target).float() * sent_mask).sum()
