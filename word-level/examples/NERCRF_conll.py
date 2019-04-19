from __future__ import print_function, division

__author__ = 'zhijing'
"""
Implementation of Bi-directional LSTM-CNNs-CRF model for NER.
"""

import sys
import os

sys.path.append(".")
sys.path.append("..")
homedir = os.path.expanduser('~')

import json
import time
import argparse
import random

from decimal import Decimal

import numpy as np
import torch
import torch.nn as nn
from neuronlp2.nn import Optimizer
from neuronlp2.io import get_logger, LogInfo, conll03_data, CoNLL03Writer, \
    TensorboardLossRecord, LossRecorder, plot_att_change
from neuronlp2.models import BiRecurrentConvCRF, BiRecurrentConvGraphCRF
from neuronlp2 import utils

from efficiency.log import show_time, show_var, fwrite


# evaluate the NER score using official scorer from CONLL-2003 competition
def evaluate(output_file, score_file, evaluate_raw_format=False, o_tag='O'):
    if evaluate_raw_format:
        os.system("./examples/eval/conll03eval.v2 -r -o %s < %s > %s" %
                  (o_tag, output_file, score_file))
    else:
        os.system("./examples/eval/conll03eval.v2 -o %s < %s > %s" %
                  (o_tag, output_file, score_file))
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
    parser.add_argument('--encoder_mode', choices=['cnn', 'lstm'], help='Encoder type for sentence encoding',
                        default='lstm')
    parser.add_argument('--char_method', choices=['cnn', 'lstm'], help='Method to create character-level embeddings',
                        required=True)
    parser.add_argument('--hidden_size', type=int, default=128, help='Number of hidden units in RNN for sentence level')
    parser.add_argument('--char_hidden_size', type=int, default=30, help='Output character-level embeddings size')
    parser.add_argument('--char_dim', type=int, default=30, help='Dimension of Character embeddings')
    parser.add_argument('--tag_space', type=int, default=0, help='Dimension of tag space')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers of RNN')
    parser.add_argument('--dropout', choices=['std', 'weight_drop', 'gcn'], help='Dropout method',
                        default='weight_drop')
    parser.add_argument('--p_em', type=float, default=0.33, help='dropout rate for input embeddings')
    parser.add_argument('--p_in', type=float, default=0.33, help='dropout rate for input of RNN model')
    parser.add_argument('--p_rnn', nargs=3, type=float, required=True, help='dropout rate for RNN')
    parser.add_argument('--p_tag', type=float, default=0.33, help='dropout rate for output layer')
    parser.add_argument('--bigram', action='store_true', help='bi-gram parameter for CRF')

    parser.add_argument('--adj_attn', choices=['cossim', 'flex_cossim', 'flex_cossim2', 'concat', '', 'multihead'],
                        default='')

    # Data loading and storing params
    parser.add_argument('--embedding_dict', help='path for embedding dict')
    parser.add_argument('--dataset_name', type=str, default='alexa', help='Which dataset to use')
    parser.add_argument('--train', type=str, required=True, help='Path of train set')
    parser.add_argument('--dev', type=str, required=True, help='Path of dev set')
    parser.add_argument('--test', type=str, required=True, help='Path of test set')
    parser.add_argument('--results_folder', type=str, default='results', help='The folder to store results')
    parser.add_argument('--alphabets_folder', type=str, default='data/alphabets',
                        help='The folder to store alphabets files')

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

    parser.add_argument('--learning_rate_gcn', type=float, default=5e-4, help='Base learning rate')
    parser.add_argument('--gcn_warmup', type=int, default=200, help='Base learning rate')
    parser.add_argument('--pretrain_lstm', type=float, default=10, help='Base learning rate')

    parser.add_argument('--adj_loss_lambda', type=float, default=0.)
    parser.add_argument('--lambda1', type=float, default=1.)
    parser.add_argument('--lambda2', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=None)

    # Misc
    parser.add_argument('--embedding', choices=['glove', 'senna', 'alexa'], help='Embedding for words', required=True)
    parser.add_argument('--restore', action='store_true', help='whether restore from stored parameters')
    parser.add_argument('--save_checkpoint', type=str, default='', help='the path to save the model')
    parser.add_argument('--o_tag', type=str, default='O', help='The default tag for outside tag')
    parser.add_argument('--unk_replace', type=float, default=0., help='The rate to replace a singleton word with UNK')
    parser.add_argument('--evaluate_raw_format', action='store_true', help='The tagging format for evaluation')

    parser.add_argument('--show_network', action='store_true', help='whether to display the network structure')
    parser.add_argument('--smooth', action='store_true', help='whether to skip all pdb break points')

    parser.add_argument('--uid', type=str, default='temp')
    parser.add_argument('--misc', type=str, default='')

    args = parser.parse_args()
    show_var(['args'])

    uid = args.uid
    results_folder = args.results_folder
    dataset_name = args.dataset_name
    use_tensorboard = True

    save_dset_dir = '{}../dset/{}/graph'.format(results_folder, dataset_name[:4])
    result_file_path = '{}/{dataset}_{uid}_result'.format(results_folder, dataset=dataset_name[:4], uid=uid)

    save_loss_path = '{}/{dataset}_{uid}_loss'.format(results_folder, dataset=dataset_name[:4], uid=uid)
    save_lr_path = '{}/{dataset}_{uid}_lr'.format(results_folder, dataset=dataset_name[:4], uid='temp')
    save_tb_path = '{}/tensorboard/'.format(results_folder)

    logger = get_logger("NERCRF")
    loss_recorder = LossRecorder(uid=uid)
    record = TensorboardLossRecord(use_tensorboard, save_tb_path, uid=uid)

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
    p_tag = args.p_tag
    unk_replace = args.unk_replace
    bigram = args.bigram
    embedding = args.embedding
    embedding_path = args.embedding_dict
    evaluate_raw_format = args.evaluate_raw_format
    o_tag = args.o_tag
    restore = args.restore
    save_checkpoint = args.save_checkpoint
    gpu_id = args.gpu_id
    alphabets_folder = args.alphabets_folder
    use_elmo = False
    p_em_vec = 0.

    learning_rate_gcn = args.learning_rate_gcn
    gcn_warmup = args.gcn_warmup
    pretrain_lstm = args.pretrain_lstm

    adj_loss_lambda = args.adj_loss_lambda
    lambda1 = args.lambda1
    lambda2 = args.lambda2

    if args.smooth:
        import pdb
        pdb.set_trace = lambda: None

    graph_model = 'gnn'
    coref_edge_filt = ''

    cheat_densify = False
    train_order = False
    misc = "{}".format(str(args.misc))

    score_file = "{}/{dataset}_{uid}_score".format(results_folder, dataset=dataset_name[:4], uid=uid)

    for folder in [results_folder, alphabets_folder, save_dset_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    def set_seed(seed):
        if not seed:
            seed = int(show_time())
        print("[Info] seed set to: {}".format(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    set_seed(args.seed)

    embedd_dict, embedd_dim = utils.load_embedding_dict(embedding, embedding_path)

    logger.info("Creating Alphabets")
    word_alphabet, char_alphabet, ner_alphabet = conll03_data.create_alphabets(
        "{}/{}/".format(alphabets_folder, dataset_name), train_path, data_paths=[dev_path, test_path],
        embedd_dict=embedd_dict, max_vocabulary_size=50000)

    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("NER Alphabet Size: %d" % ner_alphabet.size())

    logger.info("Reading Data")
    device = torch.device('cuda') if args.cuda else torch.device('cpu')
    print(device)

    data_train = conll03_data.read_data(train_path, word_alphabet, char_alphabet,
                                        ner_alphabet,
                                        graph_model, batch_size, ori_order=train_order,
                                        total_batch="{}x".format(num_epochs + 1),
                                        unk_replace=unk_replace, device=device,
                                        save_path=save_dset_dir + '/train', coref_edge_filt=coref_edge_filt,
                                        cheat_densify=cheat_densify)
    # , shuffle=True,
    num_data = data_train.data_len
    num_labels = ner_alphabet.size()
    graph_types = data_train.meta_info['graph_types']

    data_dev = conll03_data.read_data(dev_path, word_alphabet, char_alphabet,
                                      ner_alphabet,
                                      graph_model, batch_size, ori_order=True, unk_replace=unk_replace, device=device,
                                      save_path=save_dset_dir + '/dev',
                                      coref_edge_filt=coref_edge_filt, cheat_densify=cheat_densify)

    data_test = conll03_data.read_data(test_path, word_alphabet, char_alphabet,
                                       ner_alphabet,
                                       graph_model, batch_size, ori_order=True, unk_replace=unk_replace, device=device,
                                       save_path=save_dset_dir + '/test',
                                       coref_edge_filt=coref_edge_filt, cheat_densify=cheat_densify)

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

    p_gcn = [0.5, 0.5]

    d_graph = 256
    d_out = 256
    d_inner_hid = 128
    d_k = 32
    d_v = 32
    n_head = 4
    n_gcn_layer = 1

    p_rnn2 = [0.0, 0.5, 0.5]

    adj_attn = args.adj_attn
    mask_singles = True
    post_lstm = 1
    position_enc_mode = 'none'

    adj_memory = False

    if dropout == 'gcn':
        network = BiRecurrentConvGraphCRF(embedd_dim, word_alphabet.size(), char_dim, char_alphabet.size(),
                                          char_hidden_size, window, mode, encoder_mode, hidden_size, num_layers,
                                          num_labels,
                                          graph_model, n_head, d_graph, d_inner_hid, d_k, d_v, p_gcn, n_gcn_layer,
                                          d_out, post_lstm=post_lstm, mask_singles=mask_singles,
                                          position_enc_mode=position_enc_mode, adj_attn=adj_attn,
                                          adj_loss_lambda=adj_loss_lambda,
                                          tag_space=tag_space, embedd_word=word_table,
                                          use_elmo=use_elmo, p_em_vec=p_em_vec, p_em=p_em, p_in=p_in, p_tag=p_tag,
                                          p_rnn=p_rnn, p_rnn2=p_rnn2,
                                          bigram=bigram, initializer=initializer)

    elif dropout == 'std':
        network = BiRecurrentConvCRF(embedd_dim, word_alphabet.size(), char_dim, char_alphabet.size(), char_hidden_size,
                                     window, mode, encoder_mode, hidden_size, num_layers, num_labels,
                                     tag_space=tag_space, embedd_word=word_table, use_elmo=use_elmo, p_em_vec=p_em_vec,
                                     p_em=p_em, p_in=p_in, p_tag=p_tag, p_rnn=p_rnn, bigram=bigram,
                                     initializer=initializer)
    elif dropout == 'var':
        network = BiVarRecurrentConvCRF(embedd_dim, word_alphabet.size(), char_dim, char_alphabet.size(),
                                        char_hidden_size, window, mode, encoder_mode, hidden_size, num_layers,
                                        num_labels,
                                        tag_space=tag_space, embedd_word=word_table, use_elmo=use_elmo,
                                        p_em_vec=p_em_vec, p_em=p_em, p_in=p_in, p_out=p_out, p_rnn=p_rnn,
                                        bigram=bigram, initializer=initializer)
    else:
        network = BiWeightDropRecurrentConvCRF(embedd_dim, word_alphabet.size(), char_dim, char_alphabet.size(),
                                               char_hidden_size, window, mode, encoder_mode, hidden_size, num_layers,
                                               num_labels,
                                               tag_space=tag_space, embedd_word=word_table, p_em=p_em, p_in=p_in,
                                               p_out=p_out, p_rnn=p_rnn, bigram=bigram, initializer=initializer)

    # whether restore from trained model
    if restore:
        network.load_state_dict(torch.load(save_checkpoint + '_best.pth'))  # load trained model

    logger.info("cuda()ing network...")

    network = network.to(device)
    if True:
        if dataset_name == '03conll' and data_dev.data_len > 26:
            sample = data_dev.pad_batch(data_dev.dataset[25:26])
        else:
            sample = data_dev.pad_batch(data_dev.dataset[:1])
        plot_att_change(sample, network, record, save_tb_path + 'att/', uid='temp', epoch=0, device=device,
                        word_alphabet=word_alphabet, show_net=args.show_network,
                        graph_types=data_train.meta_info['graph_types'])
        # import pdb; pdb.set_trace()

    logger.info("finished cuda()ing network...")

    lr = learning_rate
    lr_gcn = learning_rate_gcn
    optim = Optimizer('sgd', 'adam', network, dropout, lr=learning_rate,
                      lr_gcn=learning_rate_gcn,
                      wd=0., wd_gcn=0., momentum=momentum, lr_decay=decay_rate, schedule=schedule,
                      gcn_warmup=gcn_warmup,
                      pretrain_lstm=pretrain_lstm)
    nn.utils.clip_grad_norm_(network.parameters(), max_norm)
    logger.info(
        "Network: %s, encoder_mode=%s, num_layer=%d, hidden=%d, char_hidden_size=%d, char_method=%s, tag_space=%d, crf=%s" % \
        (mode, encoder_mode, num_layers, hidden_size, char_hidden_size, char_method, tag_space,
         'bigram' if bigram else 'unigram'))
    logger.info("training: l2: %f, (#training data: %d, batch: %d, unk replace: %.2f)" % (
        gamma, num_data, batch_size, unk_replace))
    logger.info("dropout(in, out, rnn): (%.2f, %.2f, %s)" % (p_in, p_tag, p_rnn))

    num_batches = num_data // batch_size + 1
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

    loss_recorder.start(save_loss_path, mode='w', misc=misc)
    fwrite('', save_lr_path)
    fwrite(json.dumps(vars(args)) + '\n', result_file_path)

    for epoch in range(1, num_epochs + 1):
        show_var(['misc'])

        lr_state = 'Epoch %d (uid=%s, lr=%.2E, lr_gcn=%.2E, decay rate=%.4f): ' % (
            epoch, uid, Decimal(optim.curr_lr), Decimal(optim.curr_lr_gcn), decay_rate)
        print(lr_state)
        fwrite(lr_state[:-2] + '\n', save_lr_path, mode='a')

        train_err = 0.
        train_err2 = 0.
        train_total = 0.

        start_time = time.time()
        num_back = 0
        network.train()
        for batch_i in range(1, num_batches + 1):

            batch_doc = data_train.next()
            char, word, posi, labels, feats, adjs, words_en = [batch_doc[i] for i in [
                "chars", "word_ids", "posi", "ner_ids", "feat_ids", "adjs", "words_en"]]

            sent_word, sent_char, sent_labels, sent_mask, sent_length, _ = network._doc2sent(
                word, char, labels)

            optim.zero_grad()

            adjs_into_model = adjs if adj_memory else adjs.clone()

            loss, (ner_loss, adj_loss) = network.loss(None, word, char, adjs_into_model, labels,
                                                      graph_types=graph_types, lambda1=lambda1, lambda2=lambda2)

            # loss = network.loss(_, sent_word, sent_char, sent_labels, mask=sent_mask)
            loss.backward()
            optim.step()

            with torch.no_grad():
                num_inst = sent_mask.size(0)
                train_err += ner_loss * num_inst
                train_err2 += adj_loss * num_inst
                train_total += num_inst

            time_ave = (time.time() - start_time) / batch_i
            time_left = (num_batches - batch_i) * time_ave

            # update log
            if batch_i % 20 == 0:
                sys.stdout.write("\b" * num_back)
                sys.stdout.write(" " * num_back)
                sys.stdout.write("\b" * num_back)
                log_info = 'train: %d/%d loss1: %.4f, loss2: %.4f, time left (estimated): %.2fs' % (
                    batch_i, num_batches, train_err / train_total, train_err2 / train_total, time_left)
                sys.stdout.write(log_info)
                sys.stdout.flush()
                num_back = len(log_info)

            optim.update(epoch, batch_i, num_batches, network)

        sys.stdout.write("\b" * num_back)
        sys.stdout.write(" " * num_back)
        sys.stdout.write("\b" * num_back)
        print('train: %d loss: %.4f, loss2: %.4f, time: %.2fs' % (
            num_batches, train_err / train_total, train_err2 / train_total, time.time() - start_time))

        # evaluate performance on dev data
        with torch.no_grad():
            network.eval()
            tmp_filename = "{}/{dataset}_{uid}_output_dev".format(results_folder, dataset=dataset_name[:4], uid=uid)

            writer.start(tmp_filename)

            for batch in data_dev:
                char, word, posi, labels, feats, adjs, words_en = [batch[i] for i in [
                    "chars", "word_ids", "posi", "ner_ids", "feat_ids", "adjs", "words_en"]]
                sent_word, sent_char, sent_labels, sent_mask, sent_length, _ = network._doc2sent(
                    word, char, labels)

                preds, _ = network.decode(
                    None, word, char, adjs.clone(), target=labels, leading_symbolic=conll03_data.NUM_SYMBOLIC_TAGS,
                    graph_types=graph_types)
                # preds, _ = network.decode(_, sent_word, sent_char, target=sent_labels, mask=sent_mask,
                #                           leading_symbolic=conll03_data.NUM_SYMBOLIC_TAGS)
                writer.write(sent_word.cpu().numpy(), preds.cpu().numpy(), sent_labels.cpu().numpy(),
                             sent_length.cpu().numpy())
            writer.close()
            acc, precision, recall, f1 = evaluate(tmp_filename, score_file, evaluate_raw_format, o_tag)
            print('dev acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%' % (acc, precision, recall, f1))

            # plot loss and attention
            record.plot_loss(epoch, train_err / train_total, f1)

            plot_att_change(sample, network, record, save_tb_path + 'att/', uid="{}_{:03d}".format(uid, epoch),
                            epoch=epoch, device=device,
                            word_alphabet=word_alphabet, show_net=False, graph_types=graph_types)

            if dev_f1 < f1:
                dev_f1 = f1
                dev_acc = acc
                dev_precision = precision
                dev_recall = recall
                best_epoch = epoch

                # evaluate on test data when better performance detected
                tmp_filename = "{}/{dataset}_{uid}_output_test".format(results_folder, dataset=dataset_name[:4],
                                                                       uid=uid)
                writer.start(tmp_filename)

                for batch in data_test:
                    char, word, posi, labels, feats, adjs, words_en = [batch[i] for i in [
                        "chars", "word_ids", "posi", "ner_ids", "feat_ids", "adjs", "words_en"]]
                    sent_word, sent_char, sent_labels, sent_mask, sent_length, _ = network._doc2sent(
                        word, char, labels)

                    preds, _ = network.decode(
                        None, word, char, adjs.clone(), target=labels, leading_symbolic=conll03_data.NUM_SYMBOLIC_TAGS,
                        graph_types=graph_types)
                    # preds, _ = network.decode(_, sent_word, sent_char, target=sent_labels, mask=sent_mask,
                    #                           leading_symbolic=conll03_data.NUM_SYMBOLIC_TAGS)

                    writer.write(sent_word.cpu().numpy(), preds.cpu().numpy(), sent_labels.cpu().numpy(),
                                 sent_length.cpu().numpy())
                writer.close()
                test_acc, test_precision, test_recall, test_f1 = evaluate(tmp_filename, score_file, evaluate_raw_format,
                                                                          o_tag)
                if best_test_f1 < test_f1:
                    best_test_acc, best_test_precision, best_test_recall, best_test_f1 = test_acc, test_precision, test_recall, test_f1
                    best_test_epoch = epoch

                # save the model parameters
                if save_checkpoint:
                    torch.save(network.state_dict(), save_checkpoint + '_best.pth')

            print("best dev  acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d)" % (
                dev_acc, dev_precision, dev_recall, dev_f1, best_epoch))
            print("best test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d)" % (
                test_acc, test_precision, test_recall, test_f1, best_epoch))
            print("overall best test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d)" % (
                best_test_acc, best_test_precision, best_test_recall, best_test_f1, best_test_epoch))

        # optim.update(epoch, 1, num_batches, network)
        loss_recorder.write(epoch, train_err / train_total, train_err2 / train_total,
                            Decimal(optim.curr_lr), Decimal(optim.curr_lr_gcn), f1, best_test_f1, test_f1)
    with open(result_file_path, 'a') as ofile:
        ofile.write("best dev  acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d)\n" % (
            dev_acc, dev_precision, dev_recall, dev_f1, best_epoch))
        ofile.write("best test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d)\n" % (
            test_acc, test_precision, test_recall, test_f1, best_epoch))
        ofile.write("overall best test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d)\n\n" % (
            best_test_acc, best_test_precision, best_test_recall, best_test_f1, best_test_epoch))

    record.close()

    print('Training finished!')


if __name__ == '__main__':
    main()
