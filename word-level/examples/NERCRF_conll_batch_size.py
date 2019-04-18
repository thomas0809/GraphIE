from __future__ import print_function

__author__ = 'max'
"""
Implementation of Bi-directional LSTM-CNNs-CRF model for NER.
"""

import sys
import os

sys.path.append(".")
sys.path.append("..")
homedir = os.path.expanduser('~')

import time
import argparse
import uuid
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from neuronlp2.io import get_logger, conll03_data, CoNLL03Writer
from neuronlp2.models import BiRecurrentConvCRF, BiWeightDropRecurrentConvCRF
from neuronlp2 import utils

all_batch_sizes = [21, 9, 10, 5, 15, 6, 16, 16, 4, 29, 8, 16, 9, 10, 7, 22, 10, 10, 19, 8, 8, 7, 21, 6, 11, 7, 14, 7, 18, 26, 10, 12, 13, 14, 14, 11, 25, 25, 9, 4, 5, 32, 5, 97, 11, 14, 13, 29, 79, 11, 57, 21, 44, 7, 9, 6, 12, 8, 27, 53, 11, 8, 8, 14, 5, 19, 13, 15, 8, 15, 9, 25, 30, 17, 6, 13, 12, 7, 8, 19, 10, 8, 6, 6, 6, 6, 13, 7, 8, 8, 9, 14, 11, 9, 19, 14, 12, 17, 29, 14, 11, 29, 8, 8, 24, 12, 11, 12, 20, 20, 12, 9, 10, 7, 40, 27, 11, 13, 9, 11, 13, 10, 7, 14, 40, 26, 8, 18, 11, 8, 18, 16, 7, 6, 32, 6, 36, 6, 9, 28, 22, 9, 12, 10, 60, 18, 40, 8, 6, 9, 5, 10, 6, 5, 7, 16, 9, 6, 26, 25, 198, 11, 14, 11, 6, 15, 12, 25, 15, 6, 11, 6, 7, 10, 31, 9, 13, 25, 7, 11, 6, 5, 10, 13, 6, 13, 5, 10, 9, 6, 6, 20, 10, 12, 15, 8, 12, 14, 20, 10, 9, 8, 23, 11, 14, 10, 10, 11, 33, 10, 25, 15, 12, 15, 16, 8, 8, 8, 15, 8, 7, 8, 17, 12, 10, 4, 9, 9, 9, 12, 12, 29, 15, 6, 11, 10, 18, 12, 11, 7, 14, 29, 10, 13, 14, 7, 51, 21, 17, 22, 37, 31, 12, 61, 7, 13, 8, 13, 43, 13, 6, 20, 60, 22, 6, 9, 12, 8, 36, 26, 16, 29, 27, 34, 21, 5, 17, 11, 12, 6, 16, 6, 11, 19, 12, 11, 10, 10, 6, 10, 11, 14, 13, 8, 18, 28, 8, 6, 11, 12, 12, 28, 11, 7, 5, 8, 10, 17, 19, 39, 12, 9, 14, 33, 33, 37, 65, 29, 40, 14, 65, 15, 20, 10, 5, 7, 9, 56, 33, 7, 18, 5, 11, 5, 8, 11, 9, 9, 10, 15, 34, 34, 39, 39, 37, 17, 22, 13, 60, 22, 19, 21, 13, 8, 14, 10, 28, 9, 26, 24, 26, 26, 9, 25, 12, 34, 38, 35, 7, 10, 25, 29, 38, 16, 31, 27, 7, 10, 12, 8, 6, 11, 6, 9, 7, 21, 10, 23, 5, 11, 36, 20, 7, 7, 10, 22, 6, 37, 36, 27, 8, 12, 16, 10, 4, 8, 8, 28, 21, 25, 48, 5, 41, 5, 20, 35, 10, 20, 17, 30, 19, 31, 35, 31, 37, 14, 20, 7, 46, 5, 19, 20, 13, 21, 38, 28, 65, 26, 55, 12, 14, 8, 18, 5, 42, 4, 7, 7, 16, 7, 30, 24, 11, 17, 12, 14, 10, 14, 14, 12, 7, 10, 6, 9, 7, 10, 8, 11, 10, 8, 7, 8, 22, 13, 6, 13, 11, 11, 46, 10, 10, 8, 12, 34, 36, 10, 6, 17, 15, 15, 10, 23, 31, 4, 12, 18, 19, 9, 13, 10, 6, 7, 15, 11, 10, 6, 9, 8, 21, 16, 6, 34, 10, 38, 8, 10, 9, 10, 11, 11, 14, 7, 81, 39, 63, 9, 12, 21, 19, 91, 60, 12, 20, 16, 38, 6, 12, 6, 20, 29, 8, 6, 18, 14, 9, 25, 22, 33, 9, 13, 8, 16, 5, 28, 15, 30, 13, 17, 21, 13, 13, 11, 17, 5, 7, 9, 8, 6, 16, 12, 10, 18, 7, 16, 7, 8, 7, 10, 12, 15, 8, 31, 30, 18, 7, 9, 7, 16, 11, 12, 7, 9, 14, 11, 20, 27, 29, 11, 16, 8, 16, 44, 14, 9, 15, 16, 8, 16, 12, 10, 15, 5, 8, 9, 15, 7, 23, 31, 9, 9, 9, 14, 13, 8, 8, 10, 15, 16, 14, 11, 9, 19, 21, 5, 10, 11, 61, 28, 6, 28, 25, 53, 23, 5, 35, 10, 10, 8, 12, 9, 6, 24, 11, 6, 12, 5, 10, 22, 6, 13, 6, 29, 45, 60, 21, 24, 7, 11, 11, 8, 6, 6, 19, 10, 19, 22, 11, 22, 36, 30, 33, 26, 16, 18, 11, 5, 36, 7, 9, 16, 8, 17, 21, 12, 13, 10, 9, 9, 37, 14, 7, 10, 7, 9, 4, 14, 14, 10, 5, 6, 11, 13, 11, 17, 10, 18, 7, 8, 8, 26, 9, 17, 14, 11, 12, 8, 9, 20, 7, 11, 9, 12, 15, 39, 29, 12, 23, 10, 12, 25, 6, 13, 6, 19, 20, 14, 10, 24, 8, 9, 10, 38, 6, 33, 8, 8, 8, 5, 11, 17, 10, 5, 9, 27, 6, 7, 12, 7, 17, 14, 13, 8, 20, 13, 7, 13, 13, 7, 18, 9, 10, 7, 14, 11, 15, 15, 8, 10, 7, 11, 12, 10, 18, 13, 9, 12, 20, 14, 20, 7, 26, 26, 5, 30, 13, 39, 27, 7, 13, 7, 6, 9, 8, 12, 19, 23, 49, 56, 20, 21, 27, 7, 7, 36, 17, 6, 8, 6, 9, 5, 6, 8, 13, 14, 15, 6, 6, 9, 15, 12, 5, 20, 14, 14, 9, 13, 6, 6, 12, 11, 5, 12, 15, 7, 9, 11, 12, 10, 7, 32, 13, 15, 25, 5, 13, 13, 25, 13, 10, 11, 4, 6, 14, 11, 7, 28, 9, 6, 8, 12, 7, 13, 9, 21, 11, 30, 12, 15, 6, 8, 9, 26, 7, 7, 6, 7, 14, 8, 22, 8, 6, 8, 33, 5, 19, 15, 17, 7, 10, 6, 9, 12, 10, 11, 7, 10, 8, 8, 19, 8, 26, 9, 11, 8, 17, 8, 10, 10, 11, 14, 13, 12, 47, 18, 8, 13, 4, 26, 6, 7, 16, 24, 14, 15, 24, 9]
confined_batch_sizes = [size for size in all_batch_sizes if size >= 15 and size <=35]

# evaluate the NER score using official scorer from CONLL-2003 competition
def evaluate(output_file, score_file, evaluate_raw_format=False, o_tag='O'):
    if evaluate_raw_format:
        os.system("examples/eval/conll03eval.v2 -r -o %s < %s > %s" % (o_tag, output_file, score_file))
    else:
        os.system("examples/eval/conll03eval.v2 -o %s < %s > %s" % (o_tag, output_file, score_file))
    with open(score_file, 'r') as fin:
        fin.readline()
        line = fin.readline()
        fields = line.split(";")
        acc = float(fields[0].split(":")[1].strip()[:-1])
        precision = float(fields[1].split(":")[1].strip()[:-1])
        recall = float(fields[2].split(":")[1].strip()[:-1])
        f1 = float(fields[3].split(":")[1].strip())
    return acc, precision, recall, f1


def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate""" 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # Arguments parser
    parser = argparse.ArgumentParser(description='Tuning with DNN Model for NER')
    # Model Hyperparameters
    parser.add_argument('--mode', choices=['RNN', 'LSTM', 'GRU'], help='architecture of rnn', default='LSTM')
    parser.add_argument('--encoder_mode', choices=['cnn', 'lstm'], help='Encoder type for sentence encoding', default='lstm')
    parser.add_argument('--char_method', choices=['cnn', 'lstm'], help='Method to create character-level embeddings', required=True)
    parser.add_argument('--hidden_size', type=int, default=128, help='Number of hidden units in RNN for sentence level')
    parser.add_argument('--char_hidden_size', type=int, default=30, help='Output character-level embeddings size')
    parser.add_argument('--char_dim', type=int, default=30, help='Dimension of Character embeddings')
    parser.add_argument('--tag_space', type=int, default=0, help='Dimension of tag space')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers of RNN')
    parser.add_argument('--dropout', choices=['std', 'weight_drop'], help='Dropout method', default='weight_drop')
    parser.add_argument('--p_em', type=float, default=0.33, help='dropout rate for input embeddings')
    parser.add_argument('--p_in', type=float, default=0.33, help='dropout rate for input of RNN model')
    parser.add_argument('--p_rnn', nargs=2, type=float, required=True, help='dropout rate for RNN')
    parser.add_argument('--p_out', type=float, default=0.33, help='dropout rate for output layer')
    parser.add_argument('--bigram', action='store_true', help='bi-gram parameter for CRF')

    # Data loading and storing params
    parser.add_argument('--embedding_dict', help='path for embedding dict')
    parser.add_argument('--dataset_name', type=str, default='alexa', help='Which dataset to use')
    parser.add_argument('--train', type=str, required=True, help='Path of train set')  
    parser.add_argument('--dev', type=str, required=True, help='Path of dev set')  
    parser.add_argument('--test', type=str, required=True, help='Path of test set')  
    parser.add_argument('--results_folder', type=str, default='results', help='The folder to store results')
    parser.add_argument('--tmp_folder', type=str, default='tmp', help='The folder to store tmp files')
    parser.add_argument('--alphabets_folder', type=str, default='data/alphabets', help='The folder to store alphabets files')
    parser.add_argument('--result_file_name', type=str, default='hyperparameters_tuning', help='File name to store some results')
    parser.add_argument('--result_file_path', type=str, default='results/hyperparameters_tuning', help='File name to store some results')

    # Training parameters
    parser.add_argument('--cuda', action='store_true', help='whether using GPU')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of sentences in each batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Base learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.95, help='Decay rate of learning rate')
    parser.add_argument('--schedule', type=int, default=3, help='schedule for learning rate decay')
    parser.add_argument('--gamma', type=float, default=0.0, help='weight for l2 regularization')
    parser.add_argument('--max_norm', type=float, default=1., help='Max norm for gradients')
    parser.add_argument('--gpu_id', type=int, nargs='+', required=True, help='which gpu to use for training')

    # Misc
    parser.add_argument('--embedding', choices=['glove', 'senna', 'alexa'], help='Embedding for words', required=True)
    parser.add_argument('--restore', action='store_true', help='whether restore from stored parameters')
    parser.add_argument('--save_checkpoint', type=str, default='', help='the path to save the model')
    parser.add_argument('--o_tag', type=str, default='O', help='The default tag for outside tag')
    parser.add_argument('--unk_replace', type=float, default=0., help='The rate to replace a singleton word with UNK')
    parser.add_argument('--evaluate_raw_format', action='store_true', help='The tagging format for evaluation')
    parser.add_argument('--batch_sizes_confine', action='store_true', help='whether confine the batch sizes choices')

    args = parser.parse_args()

    logger = get_logger("NERCRF")

    # rename the parameters
    mode = args.mode
    encoder_mode = args.encoder_mode
    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    char_hidden_size = args.char_hidden_size
    char_method = args.char_method
    learning_rate = args.learning_rate
    momentum = 0.9
    decay_rate = args.decay_rate
    gamma = args.gamma
    max_norm = args.max_norm
    schedule = args.schedule
    dropout = args.dropout
    p_em = args.p_em
    p_rnn = tuple(args.p_rnn)
    p_in = args.p_in
    p_out = args.p_out
    unk_replace = args.unk_replace
    bigram = args.bigram
    embedding = args.embedding
    embedding_path = args.embedding_dict
    dataset_name = args.dataset_name
    result_file_name = args.result_file_name
    evaluate_raw_format = args.evaluate_raw_format
    o_tag = args.o_tag
    restore = args.restore
    save_checkpoint = args.save_checkpoint
    gpu_id = args.gpu_id
    results_folder = args.results_folder
    tmp_folder = args.tmp_folder
    alphabets_folder = args.alphabets_folder
    use_elmo = False
    p_em_vec = 0.
    result_file_path = args.result_file_path
    batch_sizes_confine = args.batch_sizes_confine

    if batch_sizes_confine:
        batch_sizes = confined_batch_sizes
    else:
        batch_sizes = all_batch_sizes
    print('Batch sizes used: ', ','.join(map(str, batch_sizes)))

    score_file = "%s/score_gpu_%s" % (tmp_folder, '-'.join(map(str, gpu_id)))

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)  
    if not os.path.exists(alphabets_folder):
        os.makedirs(alphabets_folder)

    embedd_dict, embedd_dim = utils.load_embedding_dict(embedding, embedding_path)

    logger.info("Creating Alphabets")
    word_alphabet, char_alphabet, ner_alphabet = conll03_data.create_alphabets("{}/{}/".format(alphabets_folder, dataset_name), train_path, data_paths=[dev_path, test_path],
                                                                 embedd_dict=embedd_dict, max_vocabulary_size=50000)

    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("NER Alphabet Size: %d" % ner_alphabet.size())

    logger.info("Reading Data")
    device = torch.device('cuda') if args.cuda else torch.device('cpu')
    print(device)

    data_train = conll03_data.read_data_to_tensor(train_path, word_alphabet, char_alphabet, ner_alphabet, device=device)
    num_data = sum(data_train[1])
    num_labels = ner_alphabet.size()

    data_dev = conll03_data.read_data_to_tensor(dev_path, word_alphabet, char_alphabet, ner_alphabet, device=device)
    data_test = conll03_data.read_data_to_tensor(test_path, word_alphabet, char_alphabet, ner_alphabet, device=device)

    writer = CoNLL03Writer(word_alphabet, char_alphabet, ner_alphabet)

    def construct_word_embedding_table():
        scale = np.sqrt(3.0 / embedd_dim)
        table = np.empty([word_alphabet.size(), embedd_dim], dtype=np.float32)
        table[conll03_data.UNK_ID, :] = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
        oov = 0
        for word, index in word_alphabet.items():
            if word in embedd_dict:
                embedding = embedd_dict[word]
            elif word.lower() in embedd_dict:
                embedding = embedd_dict[word.lower()]
            else:
                embedding = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        print('oov: %d' % oov)
        return torch.from_numpy(table)

    word_table = construct_word_embedding_table()
    logger.info("constructing network...")

    char_dim = args.char_dim
    window = 3
    num_layers = args.num_layers
    tag_space = args.tag_space
    initializer = nn.init.xavier_uniform_
    if args.dropout == 'std':
        network = BiRecurrentConvCRF(embedd_dim, word_alphabet.size(), char_dim, char_alphabet.size(), char_hidden_size, window, mode, encoder_mode, hidden_size, num_layers, num_labels,
                                     tag_space=tag_space, embedd_word=word_table, use_elmo=use_elmo, p_em_vec=p_em_vec, p_em=p_em, p_in=p_in, p_out=p_out, p_rnn=p_rnn, bigram=bigram, initializer=initializer)
    elif args.dropout == 'var':
        network = BiVarRecurrentConvCRF(embedd_dim, word_alphabet.size(), char_dim, char_alphabet.size(), char_hidden_size, window, mode, encoder_mode, hidden_size, num_layers, num_labels,
                                        tag_space=tag_space, embedd_word=word_table, use_elmo=use_elmo, p_em_vec=p_em_vec, p_em=p_em, p_in=p_in, p_out=p_out, p_rnn=p_rnn, bigram=bigram, initializer=initializer)
    else:
        network = BiWeightDropRecurrentConvCRF(embedd_dim, word_alphabet.size(), char_dim, char_alphabet.size(), char_hidden_size, window, mode, encoder_mode, hidden_size, num_layers, num_labels,
                                        tag_space=tag_space, embedd_word=word_table, p_em=p_em, p_in=p_in, p_out=p_out, p_rnn=p_rnn, bigram=bigram, initializer=initializer)

    network = network.to(device)

    lr = learning_rate
    optim = SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)
    # optim = Adam(network.parameters(), lr=lr, weight_decay=gamma, amsgrad=True)
    nn.utils.clip_grad_norm_(network.parameters(), max_norm)
    logger.info("Network: %s, encoder_mode=%s, num_layer=%d, hidden=%d, char_hidden_size=%d, char_method=%s, tag_space=%d, crf=%s" % \
        (mode, encoder_mode, num_layers, hidden_size, char_hidden_size, char_method, tag_space, 'bigram' if bigram else 'unigram'))
    logger.info("training: l2: %f, (#training data: %d, batch: %d, unk replace: %.2f)" % (gamma, num_data, batch_size, unk_replace))
    logger.info("dropout(in, out, rnn): (%.2f, %.2f, %s)" % (p_in, p_out, p_rnn))

    num_batches = num_data / batch_size + 1
    dev_f1 = 0.0
    dev_acc = 0.0
    dev_precision = 0.0
    dev_recall = 0.0
    test_f1 = 0.0
    test_acc = 0.0
    test_precision = 0.0
    test_recall = 0.0
    best_epoch = 0
    best_test_f1 = 0.0
    best_test_acc = 0.0
    best_test_precision = 0.0
    best_test_recall = 0.0
    best_test_epoch = 0.0
    for epoch in range(1, num_epochs + 1):
        print('Epoch %d (%s(%s), learning rate=%.4f, decay rate=%.4f (schedule=%d)): ' % (epoch, mode, args.dropout, lr, decay_rate, schedule))

        train_err = 0.
        train_total = 0.

        start_time = time.time()
        num_back = 0
        network.train()
        for batch in range(1, num_batches + 1):
            batch_size = random.sample(batch_sizes, 1)[0]
            _, word, char, labels, masks, lengths = conll03_data.get_batch_tensor(data_train, batch_size, unk_replace=unk_replace)

            optim.zero_grad()
            loss = network.loss(_, word, char, labels, mask=masks)
            loss.backward()
            optim.step()

            with torch.no_grad():
                num_inst = word.size(0)
                train_err += loss * num_inst
                train_total += num_inst

            time_ave = (time.time() - start_time) / batch
            time_left = (num_batches - batch) * time_ave

            # update log
            if batch % 20 == 0:
                sys.stdout.write("\b" * num_back)
                sys.stdout.write(" " * num_back)
                sys.stdout.write("\b" * num_back)
                log_info = 'train: %d/%d loss: %.4f, time left (estimated): %.2fs' % (batch, num_batches, train_err / train_total, time_left)
                sys.stdout.write(log_info)
                sys.stdout.flush()
                num_back = len(log_info)

        sys.stdout.write("\b" * num_back)
        sys.stdout.write(" " * num_back)
        sys.stdout.write("\b" * num_back)
        print('train: %d loss: %.4f, time: %.2fs' % (num_batches, train_err / train_total, time.time() - start_time))

        # evaluate performance on dev data
        with torch.no_grad():
            network.eval()
            tmp_filename = '%s/gpu_%s_dev' % (tmp_folder, '-'.join(map(str, gpu_id)))
            writer.start(tmp_filename)

            for batch in conll03_data.iterate_batch_tensor(data_dev, batch_size):
                _, word, char, labels, masks, lengths = batch
                preds, _ = network.decode(_, word, char, target=labels, mask=masks, leading_symbolic=conll03_data.NUM_SYMBOLIC_TAGS)
                writer.write(word.cpu().numpy(), preds.cpu().numpy(), labels.cpu().numpy(), lengths.cpu().numpy())
            writer.close()
            acc, precision, recall, f1 = evaluate(tmp_filename, score_file, evaluate_raw_format, o_tag)
            print('dev acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%' % (acc, precision, recall, f1))

            if dev_f1 < f1:
                dev_f1 = f1
                dev_acc = acc
                dev_precision = precision
                dev_recall = recall
                best_epoch = epoch

                # evaluate on test data when better performance detected
                tmp_filename = '%s/gpu_%s_test' % (tmp_folder, '-'.join(map(str, gpu_id)))
                writer.start(tmp_filename)

                for batch in conll03_data.iterate_batch_tensor(data_test, batch_size):
                    _, word, char, labels, masks, lengths = batch
                    preds, _ = network.decode(_, word, char, target=labels, mask=masks, leading_symbolic=conll03_data.NUM_SYMBOLIC_TAGS)
                    writer.write(word.cpu().numpy(), preds.cpu().numpy(), labels.cpu().numpy(), lengths.cpu().numpy())
                writer.close()
                test_acc, test_precision, test_recall, test_f1 = evaluate(tmp_filename, score_file, evaluate_raw_format, o_tag)
                if best_test_f1 < test_f1:
                    best_test_acc, best_test_precision, best_test_recall, best_test_f1 = test_acc, test_precision, test_recall, test_f1
                    best_test_epoch = epoch

            print("best dev  acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d)" % (dev_acc, dev_precision, dev_recall, dev_f1, best_epoch))
            print("best test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d)" % (test_acc, test_precision, test_recall, test_f1, best_epoch))
            print("overall best test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d)" % (best_test_acc, best_test_precision, best_test_recall, best_test_f1, best_test_epoch))

        if epoch % schedule == 0:
            lr = learning_rate / (1.0 + epoch * decay_rate)
            optim = SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)    

    with open(result_file_path, 'a') as ofile:
        ofile.write("best dev  acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d)\n" % (dev_acc, dev_precision, dev_recall, dev_f1, best_epoch))
        ofile.write("best test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d)\n" % (test_acc, test_precision, test_recall, test_f1, best_epoch))
        ofile.write("overall best test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d)\n\n" % (best_test_acc, best_test_precision, best_test_recall, best_test_f1, best_test_epoch))
    print('Training finished!')


if __name__ == '__main__':
    main()
