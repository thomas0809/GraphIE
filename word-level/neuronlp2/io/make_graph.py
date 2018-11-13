__author__ = 'max'

import argparse
import torch
import json
import os
import sys
import pdb

import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix

from collections import OrderedDict, Counter
from itertools import permutations, combinations
from nltk.corpus import stopwords

from efficiency.log import fwrite, show_var

from .Constants import EOS_WORD, DATA_PAD_WORD, PAD_NER


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = coo_matrix(adj)
    rowsum = np.maximum(np.array(adj.sum(1)), 1)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def _listify_eos(para, eos_ixs=None):
    if not eos_ixs:
        eos_ixs = [i + 1 for i, x in enumerate(para) if x == EOS_WORD]
    return [para[st:end] for st, end in zip([0] + eos_ixs, eos_ixs)], eos_ixs


def _read_words_from_file(file, keep_case=True, cheat_densify=False):
    ''' Convert file into word seq lists and vocab '''

    word_insts = []
    tag_insts = []
    feat_insts = []

    with open(file) as f:
        field = f.readline().strip()
        assert field in ["\t".join(["char_i", "word_i", "doc_i", "d_w_i", "word", "tag", "dic_tag"]), "\t".join(
            ["char_i", "word_i", "doc_i", "d_w_i", "d_sent_i", "s_word_i", "word", "tag", "dic_tag"])]
        field = field.split('\t')
        content = f.readlines()

    doc_i_fld = field.index("doc_i")
    d_w_i_fld = field.index("d_w_i")
    src_fld = field.index("word")
    tgt_fld = field.index("tag")
    dic_fld = field.index("dic_tag")

    for line_i, line in enumerate(content):
        if not line:
            print("[Info] Line{} of {} is empty.".format(line_i, len(content)))
            continue
        toks = line.strip().split('\t')
        doc_i, d_w_i, word, tag, dic_tag = (
            toks[i] for i in [doc_i_fld, d_w_i_fld, src_fld, tgt_fld, dic_fld])

        feat = PAD_NER if dic_tag == DATA_PAD_WORD else dic_tag

        if not keep_case:
            word = word.lower()
        doc_i = int(doc_i)
        d_w_i = int(d_w_i)
        try:
            word_insts[doc_i] += [word]
            tag_insts[doc_i] += [tag]
            feat_insts[doc_i] += [feat]
        except IndexError:
            word_insts += [[word]]
            tag_insts += [[tag]]
            feat_insts += [[feat]]

        assert word_insts[doc_i][d_w_i] == word

    n_bad_sents = 0
    for doc_i in range(len(word_insts)):
        words = word_insts[doc_i]
        tags = tag_insts[doc_i]
        feats = feat_insts[doc_i]

        word_insts[doc_i], eos_ixs = _listify_eos(words)
        tag_insts[doc_i], _ = _listify_eos(tags, eos_ixs)
        feat_insts[doc_i], _ = _listify_eos(feats, eos_ixs)

        sent_tags = [set(sent) for sent in tag_insts[doc_i]]
        bad_sent_i = [sent_i for sent_i, s_t in enumerate(
            sent_tags) if (s_t == set('O')) and cheat_densify]
        n_bad_sents += len(bad_sent_i)

        word_insts[doc_i] = [sent for i, sent in enumerate(
            word_insts[doc_i]) if i not in bad_sent_i]
        tag_insts[doc_i] = [sent for i, sent in enumerate(
            tag_insts[doc_i]) if i not in bad_sent_i]
        feat_insts[doc_i] = [sent for i, sent in enumerate(
            feat_insts[doc_i]) if i not in bad_sent_i]

    print('[Info] Get {} instances from {}'.format(len(word_insts), file))
    print('[Info] Deleted {} useless instances'.format(n_bad_sents))

    return word_insts, tag_insts, feat_insts


def read_graph(file, ner_alphabet,
               coref_edge_filt, coref_edge_type,
               c_word_edge, c_sent_edge, c_sent_thres,
               save_path, terms=[], keep_word_ixs='non_eos', keep_case=True, cheat_densify=False):
    word_insts, tag_insts, feat_insts = _read_words_from_file(
        file, keep_case=keep_case, cheat_densify=cheat_densify)

    all_data = []

    doc_n_words = []
    doc_n_sents = []
    sent_len = set()
    word_len = set()

    coref_mats = ''
    coref_val = []
    coref_dens = []

    for doc_i, (para, tag, feat) in enumerate(zip(word_insts, tag_insts, feat_insts)):

        # get meta data
        para_flat = [word for sent in para for word in sent]
        tag_flat = [t for ta in tag for t in ta]
        feat_flat = [f for fea in feat for f in fea]
        doc_n_words += [len(tag_flat)]
        doc_n_sents += [len(para)]
        sent_len |= set(len(sent) for sent in para)
        word_len |= set(len(word) for word in para_flat)

        keep_word_ixs = \
            [i for i in range(len(para_flat))
             if tag_flat[i] not in ["O"]] \
                if coref_edge_filt == 'ib_tgt' \
                else [i for i in range(len(para_flat))
                      if feat_flat[i] not in ["O"]] \
                if coref_edge_filt == 'ib_feat' \
                else [i for i in range(len(para_flat))
                      if para_flat[i] in terms] \
                if coref_edge_filt == 'term' \
                else [i for i in range(len(para_flat))
                      if para_flat[i] != EOS_WORD]

        # get content data
        doc = {}
        doc["word"] = para
        doc["tag"] = tag
        doc["feat"] = feat_flat
        if c_sent_edge != '':
            _, eos_ixs = _listify_eos(para_flat)

            doc["edge"], coref_mat = _find_sent_edges(
                para, keep_word_ixs, eos_ixs, c_sent_edge, c_sent_thres, keep_stopwords_coref=False)
            coref_mats += np.array_str(coref_mat) + '\n'
            coref_val += coref_mat[np.nonzero(coref_mat)].tolist()
            coref_dens += [len(doc['edge']['coref']) /
                           doc_n_sents / doc_n_sents]
            doc["coref_groups"] = []
        else:
            doc["edge"], doc["coref_groups"] = _find_edges(para_flat, tag_flat, set(
                keep_word_ixs), coref_edge_type, c_word_edge, keep_stopwords_coref=False,
                                                           coref_edge_filt=coref_edge_filt)
            coref_dens.append(
                (len(doc["edge"]["coref"]) - len(para_flat)) / (len(para_flat) * (len(para_flat) - 1)))
        doc["id"] = doc_i

        all_data.append(doc)

    if False:  # c_sent_edge:
        fwrite(coref_mats, save_path + '_coref_sent.matrix')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        bins = list(range(22))
        plt.hist(coref_val, bins=bins, density=True)
        plt.savefig(save_path + '_coref_sent_hist.png')

    meta = {"max_doc_n_words": max(doc_n_words),
            "max_doc_n_sents": max(doc_n_sents),
            "max_word_len": max(word_len),
            "max_sent_len": max(sent_len),
            "graph_types": list(sorted(doc["edge"].keys()))
            }
    all_data = [meta] + all_data
    all_data_text = [json.dumps(i) + '\n' for i in all_data]
    fwrite(''.join(all_data_text), save_path + '_graph.json')

    print("[Info] Average density of coref: %.2f" %
          (sum(coref_dens) / len(coref_dens)))
    return all_data


def _find_sent_edges(para, keep_word_ixs, eos_ixs, c_sent_edge, c_sent_thres, keep_stopwords_coref=False):
    stop_words = set(stopwords.words('english'))

    sent_range = [(st + 1, end + 1)
                  for st, end in zip([-1] + eos_ixs, eos_ixs)]
    keep_word_ixs_in_sents = [
        [ix - st for ix in keep_word_ixs if (ix >= st) and (ix < end)] for st, end in sent_range]

    sents = [[word for word_i, word in enumerate(sent)
              if (word_i in keep_word_ixs_in_sents[sent_i]) and ((word not in stop_words) or keep_stopwords_coref)]
             for sent_i, sent in enumerate(para)]
    sents = [Counter(sent) for sent in sents]
    n_sents = len(para)
    assert n_sents == len(sents)

    coref_mat = np.zeros((n_sents, n_sents))

    if c_sent_edge == 'all':
        return {"coref": [[i, j] for i in range(n_sents) for j in range(n_sents)]}, np.ones((n_sents, n_sents))

    assert c_sent_edge == "nonzero"
    thres = c_sent_thres
    thres = int(n_sents * (n_sents - 1) / 2 * thres)
    thres += 1

    for i, j in combinations(list(range(n_sents)), 2):
        coref_mat[i, j] = _numDups(sents[i], sents[j])
    pdb.set_trace()

    def argsort_mat(arr):
        sorted_ixs = np.dstack(np.unravel_index(
            np.argsort(-arr.ravel()), arr.shape))[0].tolist()
        sorted_ixs = set(tuple(pair) for pair in sorted_ixs)
        nonzero_ixs = np.transpose(np.nonzero(arr)).tolist()
        nonzero_ixs = set(tuple(pair) for pair in nonzero_ixs)
        return list(sorted_ixs & nonzero_ixs)

    # coref_mat = np.maximum(coref_mat, coref_mat.T)
    coref_pairs = argsort_mat(coref_mat)[: thres]
    coref_pairs += [(j, i) for i, j in coref_pairs]
    coref_pairs += [(i, i) for i in range(n_sents)]

    return {"coref": coref_pairs}, coref_mat


def _numDups(a, b):
    if len(a) > len(b):
        return sum(min(a[ak], av) for ak, av in b.items())

    return sum(min(b[ak], av) for ak, av in a.items())


def _find_gold_edges(paragraph, tags, keep_word_ixs, coref_edge_type, c_word_edge):
    lower_para = [i.lower() for i in paragraph]
    word_tag = [i for i in tags if i != "O"]
    targets = OrderedDict.fromkeys(word_tag).keys()
    coref_groups = [[i for i, word in enumerate(lower_para)
                     if (tags[i] == target)]
                    for target in targets]

    # amend coref_groups
    coref_groups = [group for group in coref_groups if len(group) > 1]

    coref_groups_ib = [group for group in coref_groups if (
            set(group) & keep_word_ixs)]
    # import pdb
    # pdb.set_trace()
    coref_groups = coref_groups_ib

    if coref_edge_type == "all":
        coref_pairs = [pair for group in coref_groups for pair in list(
            permutations(group, 2))]
    elif coref_edge_type == "star":
        coref_pairs = [(group[0], group[ix])
                       for group in coref_groups for ix in range(1, len(group))]
    elif coref_edge_type == "":
        coref_pairs = []
    if c_word_edge != "all":
        insent_coref, intersent_coref = _inner_coref(paragraph, coref_pairs)
        coref_pairs = intersent_coref if c_word_edge == "inter" else insent_coref

    coref_pairs += [(i, i) for i in range(len(paragraph))]
    return coref_pairs


def _find_edges(paragraph, tags, keep_word_ixs, coref_edge_type, c_word_edge,
                keep_stopwords_coref=False, coref_edge_filt=''):
    '''
    Turn a paragraph into a dict of "forward", "backward", "coref"
        "forward" is a list of tuples, where each tuple is an ordered pair
    '''

    # Get coref edges
    stop_words = set(stopwords.words('english'))
    lower_para = [i.lower() for i in paragraph]

    if coref_edge_filt in ['ib_feat', 'term', '']:
        targets = OrderedDict.fromkeys(lower_para).keys()
        coref_groups = [[i for i, word in enumerate(lower_para)
                         if (word == target) and ((word not in stop_words) or keep_stopwords_coref)]
                        for target in targets]
    elif coref_edge_filt == 'ib_tgt_tag':
        word_tag = [i for i in tags if i != "O"]
        targets = OrderedDict.fromkeys(word_tag).keys()
        coref_groups = [[i for i, word in enumerate(lower_para)
                         if (tags[i] == target)]
                        for target in targets]
    elif coref_edge_filt == 'ib_tgt_tag_o':
        word_tag = [i for i in tags]
        targets = OrderedDict.fromkeys(word_tag).keys()
        coref_groups = [[i for i, word in enumerate(lower_para)
                         if (tags[i] == target)]
                        for target in targets]

    else:
        word_tag = [i for i in zip(lower_para, tags) if i[1] != "O"]
        targets = OrderedDict.fromkeys(word_tag).keys()
        coref_groups = [[i for i, word in enumerate(lower_para)
                         if ((word, tags[i]) == target)]
                        for target in targets]

    coref_groups = [group for group in coref_groups if len(group) > 1]

    coref_groups_ib = [group for group in coref_groups if (
            set(group) & keep_word_ixs)]
    # import pdb
    # pdb.set_trace()
    coref_groups = coref_groups_ib

    if coref_edge_type == "all":
        coref_pairs = [pair for group in coref_groups for pair in list(
            permutations(group, 2))]
    elif coref_edge_type == "star":
        coref_pairs = [(group[0], group[ix])
                       for group in coref_groups for ix in range(1, len(group))]
    elif coref_edge_type == "":
        coref_pairs = []
    if c_word_edge != "all":
        insent_coref, intersent_coref = _inner_coref(paragraph, coref_pairs)
        coref_pairs = intersent_coref if c_word_edge == "inter" else insent_coref

    coref_pairs += [(i, i) for i in range(len(paragraph))]

    # Get forward, backward edges
    ixs = list(range(len(paragraph)))
    forw = list(zip(ixs, ixs[1:]))
    backw = list(zip(ixs[1:], ixs))
    gold_pairs = _find_gold_edges(paragraph, tags, keep_word_ixs, coref_edge_type, c_word_edge)

    return {"forward": forw, "rear": backw, "coref": coref_pairs, "wonderful": gold_pairs}, \
           coref_groups


def _inner_coref(paragraph, coref_pairs):
    _, eos_ixs = _listify_eos(paragraph)
    sents_st_end = [list(range(st + 1, end + 1))
                    for st, end in zip([-1] + eos_ixs, eos_ixs)]
    insent_pairs = [list(permutations(sent, 2)) for sent in sents_st_end]
    insent_pairs = set(pair for sent in insent_pairs for pair in sent)
    all_pairs = set(coref_pairs)

    insent_coref = all_pairs & insent_pairs
    intersent_coref = all_pairs - insent_pairs
    insent_coref = sorted(list(insent_coref))
    intersent_coref = sorted(list(intersent_coref))
    return insent_coref, intersent_coref
