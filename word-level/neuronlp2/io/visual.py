# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np

from efficiency.log import fwrite
import os
import pdb
import random
import json
import torch
from .Constants import PAD_ID_WORD


def plot_att(adjs, word_txt, record, image_name, epoch=0, select=None):
    ori_adjs = adjs.clone()
    att_arr = ori_adjs[0, 0]
    if select is None:
        select = att_arr.nonzero()[:, 0].cpu().numpy()
        select = np.sort(np.unique(select))

    try:
        import matplotlib
    except ModuleNotFoundError:
        return

    record.plot_img(epoch, None, caption=' '.join(word_txt[:30]), att_arr=att_arr, image_name=image_name,
                    xticks=word_txt, yticks=word_txt, part=select)
    return select


def plot_att_change(batch_doc, network, record, save_img_path, uid='temp',
                    epoch=0, device=torch.device('cpu'), word_alphabet=None, show_net=False, graph_types=['coref']):
    char, word, posi, labels, feats, adjs = [batch_doc[i].to(device) for i in
                                             ["chars", "word_ids", "posi", "ner_ids", "feat_ids", "adjs"]]
    word_txt = []
    if word_alphabet:
        doc = word[0][word[0] != PAD_ID_WORD]
        word_txt = [word_alphabet.get_instance(w) for w in doc]

    adjs_cp = adjs.clone()

    # save adj to file
    print_thres = adjs.size(-1) * adjs.size(-2) + 1000
    torch.set_printoptions(threshold=print_thres)

    # check adj_old, adj_new
    # select = plot_att(adjs_cp, word_txt, record, epoch=epoch)

    network.loss(None, word, char, adjs_cp, labels, show_net=show_net, graph_types=graph_types)
    # plot_att(adjs_cp, word_txt, record, epoch=epoch, select=select)
