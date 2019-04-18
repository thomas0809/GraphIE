__author__ = 'max'

import os
import json
import torch
import random
import numpy as np
from pathlib import Path

from copy import deepcopy
from scipy.sparse import coo_matrix

from .instance import DependencyInstance, NERInstance, GraphInstance
from .instance import Sentence

from .make_graph import read_graph, normalize_adj
from . import utils
from .utils import MAX_CHAR_LENGTH

from .Constants import PAD_WORD, PAD_ID_CHAR, PAD_ID_WORD, PAD_ID_POSI, PAD_ID_NER


class CoNLL03Reader(object):
    def __init__(self, file_path, word_alphabet, char_alphabet, ner_alphabet):
        self.__source_file = open(file_path, 'r')
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__ner_alphabet = ner_alphabet

    def close(self):
        self.__source_file.close()

    def getNext(self, normalize_digits=True):
        line = self.__source_file.readline()
        # skip multiple blank lines.
        while len(line) > 0 and len(line.strip()) == 0:
            line = self.__source_file.readline()
        if len(line) == 0:
            return None

        lines = []
        while len(line.strip()) > 0:
            line = line.strip()

            lines.append(line.split(' '))
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        words = []
        word_ids = []
        char_seqs = []
        char_id_seqs = []
        ner_tags = []
        ner_ids = []

        for tokens in lines:
            chars = []
            char_ids = []
            for char in tokens[1]:
                chars.append(char)
                char_ids.append(self.__char_alphabet.get_index(char))
            if len(chars) > utils.MAX_CHAR_LENGTH:
                chars = chars[:utils.MAX_CHAR_LENGTH]
                char_ids = char_ids[:utils.MAX_CHAR_LENGTH]
            char_seqs.append(chars)
            char_id_seqs.append(char_ids)

            word = utils.DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
            ner = tokens[-1]

            words.append(word)  # log the original word
            word_ids.append(self.__word_alphabet.get_index(word))

            ner_tags.append(ner)
            ner_ids.append(self.__ner_alphabet.get_index(ner))

        return NERInstance(Sentence(words, word_ids, char_seqs, char_id_seqs),
                           ner_tags, ner_ids)


class CoNLL03DocReader(object):

    def __init__(self, file_path, word_alphabet, char_alphabet,
                 ner_alphabet,
                 coref_edge_filt='', coref_edge_type='all',
                 coref_word_edge='all', coref_sent_edge='', coref_sent_thres=0.2,
                 save_path='', cheat_densify=False):

        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__ner_alphabet = ner_alphabet

        self.data_i = 0
        self.save_path = save_path
        self.data = self.get_data(file_path, coref_edge_filt, coref_edge_type,
                                  coref_word_edge, coref_sent_edge, coref_sent_thres, save_path,
                                  cheat_densify=cheat_densify)

    def get_data(self, file_path, coref_edge_filt, coref_edge_type,
                 coref_word_edge, coref_sent_edge, coref_sent_thres,
                 save_path, cheat_densify=False):
        file = save_path + '_graph.json'
        data = []
        if Path(file).is_file():
            print('[Info] Loading graph:', file)
            with open(file, 'r') as f:
                data = [json.loads(line.strip())
                        for line in f]

        else:
            data = read_graph(file_path, self.__ner_alphabet,
                              coref_edge_filt, coref_edge_type,
                              coref_word_edge, coref_sent_edge, coref_sent_thres,
                              save_path, terms=[], keep_word_ixs='non_eos', cheat_densify=cheat_densify)
        # data[0] is meta
        # data[1:] is all documents
        return data

    def _preprocess_adjs(self, adjs, model, graph_types, edges):
        if True:
            for i in range(len(adjs)):
                np.fill_diagonal(adjs[i], 0)

        # pdb.set_trace()

        if model in ['gnn1', 'transformer_graph', 'gnnattn']:
            adjs = adjs.sum(axis=0, keepdims=True)
            adjs[0] = normalize_adj(adjs[0]).toarray()
        elif edges["coref"]:
            adjs[graph_types.index("coref")] = normalize_adj(
                adjs[graph_types.index("coref")]).toarray()

        if "wonderful" in edges:
            adjs[graph_types.index("wonderful")] = normalize_adj(
                adjs[graph_types.index("wonderful")]).toarray()

        if model == 'gnn_coref':
            adjs = adjs[graph_types.index(
                "coref"): graph_types.index("coref") + 1]

    def getNext(self, unk_replace=0., normalize_digits=True, model="transformer_graph"):

        if unk_replace:
            raise NotImplementedError("unk_replace is to be implemented")

        self.data_i += 1
        if self.data_i >= len(self.data):
            return None

        graph_types = self.data[0]["graph_types"]
        raw_n_graph = len(graph_types)

        line = deepcopy(self.data[self.data_i])

        words = line["word"]
        ner_tags = line["tag"]
        feat_tags = line["feat"]
        edges = line["edge"]

        # (1) ner, feat

        feat_ids = [[self.__ner_alphabet.get_index(
            feat)] for feat in feat_tags]
        feat_ids = np.array(feat_ids, dtype=int)

        n_sent = len(words)
        sent_len = max(len(sent) for sent in words)

        # (2) word, posi
        word_ids = np.zeros((n_sent, sent_len), dtype=int) + PAD_ID_WORD
        ner_ids = np.zeros((n_sent, sent_len), dtype=int) + PAD_ID_NER
        posi = np.zeros((n_sent, sent_len, 2), dtype=int) + PAD_ID_POSI

        for sent_i in range(n_sent):
            for word_i, w in enumerate(words[sent_i]):
                w = utils.DIGIT_RE.sub("0", w) if normalize_digits else w
                word_ids[sent_i][word_i] = self.__word_alphabet.get_index(w)

                ner = ner_tags[sent_i][word_i]
                ner_ids[sent_i][word_i] = self.__ner_alphabet.get_index(ner)

                posi[sent_i][word_i] = np.array(
                    [sent_i + 1, word_i + 1])  # so that PAD is meaningful

        words_flat = [w for sent in words for w in sent]
        word_len = min(max([len(w) for w in words_flat]), MAX_CHAR_LENGTH)
        doc_n_words = len(words_flat)

        n_node = doc_n_words

        # (3) char
        words_cutoff = [[w[:word_len] for w in sent] for sent in words]
        chars = np.zeros((n_sent, sent_len, word_len), dtype=int) + PAD_ID_CHAR
        for sent_i in range(n_sent):
            for word_i in range(len(words_cutoff[sent_i])):
                for char_i, c in enumerate(words_cutoff[sent_i][word_i]):
                    chars[sent_i][word_i][
                        char_i] = self.__char_alphabet.get_index(c)
        # (4) adj
        adjs = np.zeros((raw_n_graph, n_node, n_node))

        for graph_i, graph_type in enumerate(graph_types):
            e = edges[graph_type]
            if e:
                row, col = zip(*e)
                adjs[graph_i] = coo_matrix(
                    ([1] * len(row), (row, col)), shape=(n_node, n_node)).toarray()

        # preprocess adjs
        self._preprocess_adjs(adjs, model, graph_types, edges)

        # (5) words
        words_padded = np.vstack([
            np.pad(sent,
                   pad_width=[(0, sent_len - len(sent))],
                   mode='constant', constant_values=PAD_WORD)
            for sent in words])

        return GraphInstance(word_ids, chars, feat_ids,
                             posi, adjs, ner_ids,
                             n_sent, sent_len, word_len, doc_n_words, n_node,
                             words_padded)


class DataLoader():

    def __init__(self, dataset, meta_info, batch_size=1, shuffle=False, sort=True, total_batch=0,
                 device=torch.device('cpu')):
        self.meta_info = meta_info
        self.dataset = sorted(
            dataset, key=lambda x: x.length()) if sort else dataset
        self.data_len = len(dataset)
        self.batch_size = batch_size

        batch_heads = list(range(0, len(dataset), batch_size))
        self.batch_itervals = [(i, i + batch_size) for i in batch_heads]

        if isinstance(total_batch, str):
            assert total_batch.endswith("x"), "it should be '<num_epoch>x' "
            total_epoch = int(total_batch[:-1])
            total_batch = len(self.batch_itervals) * total_epoch
        self.total_batch = max(total_batch, len(self.batch_itervals))
        self.iter_cnt = 0

        self.shuffle = shuffle
        self.device = device

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.total_batch

    def shuffle_dataset(self):
        random.shuffle(self.batch_itervals)

    def sample(self, batch_i=None):
        start_idx, end_idx = random.choice(self.batch_itervals) if batch_i is None \
            else self.batch_itervals[batch_i]
        batch = self.dataset[start_idx: end_idx]
        return self.pad_batch(batch)

    def next(self):

        if self.iter_cnt < self.total_batch:
            which_interval = self.iter_cnt % len(self.batch_itervals)
            if which_interval == 0 and self.shuffle:
                self.shuffle_dataset()
            start_idx, end_idx = self.batch_itervals[which_interval]
            self.iter_cnt += 1
            self.which_interval = which_interval

            batch = self.dataset[start_idx: end_idx]
            return self.pad_batch(batch)

        else:
            self.iter_cnt = 0
            if self.shuffle:
                self.shuffle_dataset()
            raise StopIteration()

    def pad_batch(self, batch):
        device = self.device

        max_n_sent = max([x.n_sent for x in batch])
        max_sent_len = max([x.sent_len for x in batch])
        max_word_len = max([x.word_len for x in batch])
        max_doc_n_words = max([x.doc_n_words for x in batch])
        max_n_node = max([x.n_node for x in batch])

        chars = torch.stack([
            torch.LongTensor(np.pad(x.chars,
                                    pad_width=[(0, max_n_sent - x.n_sent), (0, max_sent_len - x.sent_len),
                                               (0, max_word_len - x.word_len)],
                                    mode='constant', constant_values=PAD_ID_CHAR))
            for x in batch])
        word_ids = torch.stack([
            torch.LongTensor(np.pad(x.word_ids,
                                    pad_width=[
                                        (0, max_n_sent - x.n_sent), (0, max_sent_len - x.sent_len)],
                                    mode='constant', constant_values=PAD_ID_WORD))
            for x in batch])
        # pdb.set_trace()

        posi = torch.stack([
            torch.LongTensor(np.pad(x.posi,
                                    pad_width=[
                                        (0, max_n_sent - x.n_sent), (0, max_sent_len - x.sent_len), (0, 0)],
                                    mode='constant', constant_values=PAD_ID_POSI))
            for x in batch])

        ner_ids = torch.stack([
            torch.from_numpy(np.pad(x.ner_ids,
                                    pad_width=[
                                        (0, max_n_sent - x.n_sent), (0, max_sent_len - x.sent_len)],
                                    mode='constant', constant_values=PAD_ID_NER))
            for x in batch])
        feat_ids = torch.stack([
            torch.LongTensor(np.pad(x.feat_ids,
                                    pad_width=[
                                        (0, max_doc_n_words - x.doc_n_words), (0, 0)],
                                    mode='constant', constant_values=PAD_ID_NER))
            for x in batch])
        adjs = torch.stack([
            torch.FloatTensor(np.pad(x.adjs,
                                     pad_width=[
                                         (0, 0), (0, max_n_node - x.n_node), (0, max_n_node - x.n_node)],
                                     mode='constant', constant_values=0))
            for x in batch])

        words_en = np.vstack([
            np.pad(x.words_en,
                   pad_width=[
                       (0, max_n_sent - x.n_sent), (0, max_sent_len - x.sent_len)],
                   mode='constant', constant_values=PAD_WORD)
            for x in batch])

        # with torch.no_grad():
        return {"chars": chars.to(device),
                    "word_ids": word_ids.to(device),
                    "posi": posi.to(device),
                    "ner_ids": ner_ids.to(device),
                    "feat_ids": feat_ids.to(device),
                    "adjs": adjs.to(device),
                    "words_en": words_en}
