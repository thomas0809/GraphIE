__author__ = 'jindi'

import os.path
import random
import torch
import numpy as np
from .reader import CoNLL03DocReader, DataLoader
from .alphabet import Alphabet
from .logger import get_logger
from . import utils
from .Constants import PAD_WORD, PAD_POS, PAD_CHUNK, PAD_NER, PAD_CHAR, _START_VOCAB, UNK_ID, PAD_ID_WORD, PAD_ID_CHAR, PAD_ID_TAG, NUM_SYMBOLIC_TAGS

# Special vocabulary symbols - we always put them at the start.

_buckets = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 140]


def create_alphabets(alphabet_directory, train_path, data_paths=None, max_vocabulary_size=50000, embedd_dict=None,
                     min_occurence=1, normalize_digits=True):

    def expand_vocab():
        vocab_set = set(vocab_list)

        for data_path in data_paths:
            # logger.info("Processing data: %s" % data_path)
            with open(data_path, 'r') as file:
                line = file.readline()
                fields = line.strip().split('\t')
                field_w = fields.index('word')
                field_t = fields.index('tag')
                for line in file:

                    line = line.strip()
                    if len(line) == 0:
                        continue

                    tokens = line.split('\t')
                    word = tokens[field_w]
                    for char in word:
                        char_alphabet.add(char)

                    word = utils.DIGIT_RE.sub("0", word) if normalize_digits else word
                    ner = tokens[field_t]

                    ner_alphabet.add(ner)

                    if word not in vocab_set and (word in embedd_dict or word.lower() in embedd_dict):
                        vocab_set.add(word)
                        vocab_list.append(word)

    logger = get_logger("Create Alphabets")
    word_alphabet = Alphabet('word', use_default_value=True, singleton=True)
    char_alphabet = Alphabet('character', use_default_value=True)
    ner_alphabet = Alphabet('ner')

    if not os.path.isdir(alphabet_directory):
        logger.info("Creating Alphabets: %s" % alphabet_directory)

        char_alphabet.add(PAD_CHAR)
        ner_alphabet.add(PAD_NER)

        vocab2count = dict()
        with open(train_path, 'r') as file:
            line = file.readline()
            fields = line.strip().split('\t')
            field_w = fields.index('word')
            field_t = fields.index('tag')
            for line in file:

                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split('\t')
                word = tokens[field_w]
                for char in word:
                    char_alphabet.add(char)

                word = utils.DIGIT_RE.sub("0", word) if normalize_digits else word
                ner = tokens[field_t]

                ner_alphabet.add(ner)

                if word in vocab2count:
                    vocab2count[word] += 1
                else:
                    vocab2count[word] = 1
        # collect singletons
        singletons = set([word for word, count in vocab2count.items() if count <= min_occurence])

        # if a singleton is in pretrained embedding dict, set the count to min_occur + c
        if embedd_dict is not None:
            for word in vocab2count.keys():
                if word in embedd_dict or word.lower() in embedd_dict:
                    vocab2count[word] += min_occurence

        vocab_list = _START_VOCAB + sorted(vocab2count, key=vocab2count.get, reverse=True)
        logger.info("Total Vocabulary Size: %d" % len(vocab_list))
        logger.info("Total Singleton Size:  %d" % len(singletons))
        vocab_list = [word for word in vocab_list if word in _START_VOCAB or vocab2count[word] > min_occurence]
        logger.info("Total Vocabulary Size (w.o rare words): %d" % len(vocab_list))

        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]

        if data_paths is not None and embedd_dict is not None:
            expand_vocab()

        for word in vocab_list:
            word_alphabet.add(word)
            if word in singletons:
                word_alphabet.add_singleton(word_alphabet.get_index(word))

        word_alphabet.save(alphabet_directory)
        char_alphabet.save(alphabet_directory)
        ner_alphabet.save(alphabet_directory)
    else:
        print("[Info] Loading existing alphabet at:", alphabet_directory)
        word_alphabet.load(alphabet_directory)
        char_alphabet.load(alphabet_directory)
        ner_alphabet.load(alphabet_directory)

    word_alphabet.close()
    char_alphabet.close()
    ner_alphabet.close()
    logger.info("Word Alphabet Size (Singleton): %d (%d)" % (word_alphabet.size(), word_alphabet.singleton_size()))
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("NER Alphabet Size: %d" % ner_alphabet.size())
    return word_alphabet, char_alphabet, ner_alphabet

def read_data(source_path, word_alphabet, char_alphabet, ner_alphabet, graph_model,
              batch_size, shuffle=False, ori_order=False, total_batch=0, unk_replace=0.,
              normalize_digits=True, device=torch.device('cpu'), save_path='',
              coref_edge_filt='', coref_edge_type='all',
              coref_word_edge='all', coref_sent_edge='', coref_sent_thres=0.2,
              cheat_densify=False):

    data = []

    print('Reading data from %s' % source_path)

    reader = CoNLL03DocReader(source_path, word_alphabet, char_alphabet, ner_alphabet,
                              coref_edge_filt=coref_edge_filt, coref_edge_type=coref_edge_type,
                              coref_word_edge=coref_word_edge, coref_sent_edge=coref_sent_edge, coref_sent_thres=coref_sent_thres,
                              save_path=save_path, cheat_densify=cheat_densify)
    inst = reader.getNext(unk_replace, normalize_digits, graph_model)
    while inst is not None:

        if reader.data_i % 10000 == 0:
            print("reading data: %d" % reader.data_i)

        data += [inst]

        inst = reader.getNext(unk_replace, normalize_digits, graph_model)

    print("Total number of data: %d" % reader.data_i)

    dataloader = DataLoader(data, reader.data[0], batch_size=batch_size, shuffle=shuffle, sort=not ori_order, total_batch=total_batch, device=device)
    return dataloader
