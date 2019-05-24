from __future__ import print_function, division
import re
import sys
from collections import Counter, OrderedDict
import argparse

from efficiency.log import *

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
O_TAG = "O"
BOD_WORD = '-DOCSTART-'


def add_eos(content):
    full_txt = []
    full_tag = []
    doc_txt = []
    doc_tag = []
    for line_ix, line in enumerate(content):
        if line.startswith(BOD_WORD):
            doc_txt = []
            doc_tag = []
        if line:
            word, _, _, tag = line.split()
            doc_txt += [word]
            doc_tag += [tag]
        elif content[line_ix - 1]:
            doc_txt += [EOS_WORD]
            doc_tag += [O_TAG]

        end_of_doc = False
        if line_ix + 1 == len(content):
            end_of_doc = True
        elif content[line_ix + 1].startswith(BOD_WORD):
            end_of_doc = True

        if end_of_doc:
            full_txt += [doc_txt]
            full_tag += [doc_tag]

    return full_txt, full_tag


def reformat(full_txt, full_tag, mer_f):
    all_words = [word for doc in full_txt for word in doc]
    all_ibos = [word for doc in full_tag for word in doc]
    text = ' '.join(all_words) + ' '

    assert len(all_words) == len(all_ibos)
    assert len(text) == sum(len(w) + 1 for w in all_words), "the length difference: " + show_var(
        ["sum(len(w) + 1 for w in all_words) ", "len(text)"])

    c_cnt = 0
    c_w_word_ibo_dict = OrderedDict()
    w_cnt = 0
    for abst_i, abst in enumerate(full_txt):
        abst = ' '.join(abst)
        if not abst:
            print('[Info] Line {} is empty.'.format(abst_i))
            continue
        d_w_i = 0
        eos_ixs = [m.start() + len(EOS_WORD) for m in re.finditer(re.escape(EOS_WORD), abst)]

        sents = [abst[st:end] for st, end in zip([0] + eos_ixs, eos_ixs)]
        sents = [s.strip() for s in sents]
        for d_s_i, sent in enumerate(sents):
            for s_w_i, word in enumerate(sent.split()):
                assert word == all_words[w_cnt]
                assert text[c_cnt: c_cnt + len(word)] == word
                c_w_word_ibo_dict[c_cnt] = (w_cnt, abst_i, d_w_i, d_s_i, s_w_i, word, all_ibos[w_cnt])
                c_cnt += len(word) + 1
                w_cnt += 1
                d_w_i += 1
        assert d_w_i == len(abst.split())

    assert c_cnt == len(text)
    assert w_cnt == len(all_words)

    assert len(all_words) == len(all_ibos)
    assert len(text) == sum(len(w) + 1 for w in all_words), "the length difference: " + show_var(
        ["sum(len(w) + 1 for w in all_words) ", "len(text)"])

    c_w_word_ibo_dict_text = [
        "\t".join([str(c_ix), str(w_ix), str(d_i), str(d_w_i), str(d_s_i), str(s_w_i), word, tag, '<blank>']) + '\n' for
        c_ix, (w_ix, d_i, d_w_i, d_s_i, s_w_i, word, tag) in c_w_word_ibo_dict.items()]
    c_w_word_ibo_dict_text = "\t".join(
        ["char_i", "word_i", "doc_i", "d_w_i", "d_sent_i", "s_word_i", "word", "tag", "dic_tag"]) + '\n' + ''.join(
        c_w_word_ibo_dict_text)
    fwrite(c_w_word_ibo_dict_text, mer_f)


def read_file(file):
    with open(file) as f:
        content = [line.strip() for line in f]
    if not content[0].startswith(BOD_WORD):
        print('[Error] Your source file does not have -DOCSTART- symbol (to signal the start of a document).')
        sys.exit(1)
    cleaned = []
    for line_ix, line in enumerate(content):
        if (not line) and (not content[line_ix - 1]):
            continue
        cleaned += [line]
    return cleaned


def main():
    parser = argparse.ArgumentParser(
        description='NER Data Preprocessor')
    parser.add_argument('--data_folder', type=str, default='data/dset/03co/')
    parser.add_argument('--train_file_name', type=str, default='train.txt')
    parser.add_argument('--valid_file_name', type=str, default='valid.txt')
    parser.add_argument('--test_file_name', type=str, default='test.txt')
    args = parser.parse_args()

    folder = args.data_folder
    file_lookup = {
        'raw_train': args.train_file_name,
        'raw_valid': args.valid_file_name,
        'raw_test': args.test_file_name,
        'reformatted_train': 'train.c_w_d_dw_ds_sw_word_ibo_dic',
        'reformatted_valid': 'valid.c_w_d_dw_ds_sw_word_ibo_dic',
        'reformatted_test': 'test.c_w_d_dw_ds_sw_word_ibo_dic',
    }

    for typ in ['train', 'valid', 'test']:
        file_raw = folder + file_lookup['raw_' + typ]
        file_out = folder + file_lookup['reformatted_' + typ]

        content = read_file(file_raw)
        full_txt, full_tag = add_eos(content)
        reformat(full_txt, full_tag, file_out)


if __name__ == "__main__":
    main()

'''
# input file
-DOCSTART- -X- O O

CRICKET NNP I-NP O
- : O O
LEICESTERSHIRE NNP I-NP I-ORG
TAKE NNP I-NP O
OVER IN I-PP O
AT NNP I-NP O
TOP NNP I-NP O
AFTER NNP I-NP O
INNINGS NNP I-NP O
VICTORY NN I-NP O
. . O O

LONDON NNP I-NP I-LOC
1996-08-30 CD I-NP O
'''
