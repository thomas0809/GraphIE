import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from data import read_data, DataLoader
from gnn import GNN
import numpy as np
import random
import json
import time
import math
import sys
import os
from tqdm import tqdm
from constants import *

import argparse

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 20)')
parser.add_argument('--batch', type=int, default=1, help='mini-batch size')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--wd', type=float, default=0., help='weight decay (default: 0)')
parser.add_argument('--cuda', action='store_true', default=True, help='enable CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--patience', type=int, default=5, help='number of times to observe worsening validation set error before giving up')
parser.add_argument('--d_embed', type=int, default=64, help='character embedding dimension')
parser.add_argument('--filter_sizes', type=str, default='2,3,4', help='character cnn filter size')
parser.add_argument('--n_filter', type=int, default=64, help='character cnn filter number')
parser.add_argument('--d_pos_embed', type=int, default=32, help='additional feature')
parser.add_argument('--d_graph', type=str, default='128', help='lstm and graph layer dimension')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout probability')
parser.add_argument('--model', type=str, default='lstm-gcn-lstm', choices=['lstm', 'lstm-lstm', 'lstm-gcn-lstm', 'lstm-rgcn-lstm', 'lstm-gat-lstm'], help='model')
parser.add_argument('--crf', action='store_true', default=False, help='final crf')
parser.add_argument('--save_path', type=str, default='models/output', help='output name')
parser.add_argument('--data_path', type=str, default='./', help='data path')
parser.add_argument('--case', type=str, default='final_train_case.txt,final_valid_case.txt,final_test_case.txt', help='case list file')
parser.add_argument('--globalnode', action='store_true', default=False, help='add global node')

parser.add_argument('--test', action='store_true', default=False, help='test mode')
# parser.add_argument('--testmodel', type=str, default='', help='test model')
# parser.add_argument('--testbase', type=str, default='./', help='test base path')
# parser.add_argument('--testcase', type=str, default='tmp_test.txt')

args = parser.parse_args()

args.filter_sizes = [int(x) for x in args.filter_sizes.split(',')]
args.d_graph = [int(x) for x in args.d_graph.split(',')]
# if args.testmodel == '':
#     args.testmodel = args.output+'.model'
# if args.model == 'lstm+g':
#     args.final = 'linear'

label2id = tag2id
d_output = len(label2id)

print(args)
result_obj = {}

def acc_to_str(acc):
    s = ['%s:%.3f'%(label, acc[label]) for label in acc]
    return ' '.join(s)

def to_var(tensors, cuda=True):
    if cuda:
        return [Variable(t, requires_grad=False).cuda() for t in tensors]
    else:
        return [Variable(t, requires_grad=False) for t in tensors]


def evaluate(model, dataloader, output=False, args=args):
    # (hit, pred_cnt, gold_cnt)
    count = {label:[0, 0, 0] for label in label2id if label != 'O'}
    n_correct = 0
    n_total = 0
    current_case_id = None
    output_file = None
    model.eval()
    eval_log = open(args.save_path+'_eval.log','w')
    for tensors, batch in tqdm(dataloader, file=eval_log, mininterval=30): # tqdm(dataloader, leave=False):
        data, data_word, pos, length, mask, label, adjs = to_var(tensors, cuda=args.cuda)
        batch_size, docu_len, sent_len, word_len = data.size()

        logit = model(data, data_word, pos, length, mask, adjs).view(-1, d_output)

        if args.crf:
            logit = logit.view(batch_size*docu_len, sent_len, -1)
            mask = mask.view(batch_size*docu_len, -1)
            _, pred = model.crf_layer.viterbi_decode(logit, mask)
            pred = pred.data
        else:
            pred = logit.max(dim=1)[1]

        if output:
            prob = F.softmax(logit.view(batch_size, docu_len, sent_len, -1), dim=3).data
            for i, data in enumerate(batch):
                if data.case_id != current_case_id:
                    current_case_id = data.case_id
                    if model.crf:
                        filename = 'prob_%s_crf.json' % model.model
                    else:
                        filename = 'prob_%s.json' % model.model
                    output_file = open(data.path+filename, 'w')
                obj = {'id':{'case':data.case_id,'doc':data.doc_id,'page':data.page_id}, 'prob':[]}
                for j in range(data.num_sent):
                    obj['prob'].append(prob[i,j,:len(data.sents[j])].tolist())
                output_file.write(json.dumps(obj) + '\n')

        mask = mask.data
        label = label.data
        pred = pred.contiguous().view(-1).masked_select(mask.view(-1))
        gold = label.contiguous().view(-1).masked_select(mask.view(-1))

        for label, idx in label2id.items():
            if label == 'O':
                continue
            pred_cnt = pred.eq(idx).sum()
            gold_cnt = gold.eq(idx).sum()
            hit = pred.eq(idx).mul(gold.eq(idx)).sum()
            count[label][0] += int(hit)
            count[label][1] += int(pred_cnt)
            count[label][2] += int(gold_cnt)

        n_total += length.data.sum()
        n_correct += pred.eq(gold).sum()

    eval_log.close()
    os.remove(args.save_path+'_eval.log')

    prec, recall, f1 = {}, {}, {}
    for label in count:
        prec[label] = float(count[label][0]) / max(count[label][1],1)
        recall[label] = float(count[label][0]) / max(count[label][2],1)
        if prec[label] * recall[label] == 0:
            f1[label] = 0
        else:
            f1[label] = 2*prec[label]*recall[label]/(prec[label]+recall[label])

    return float(n_correct)/float(n_total), prec, recall, f1


def train(dataset):

    print('random seed:', args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False

    cross_res = {label:[] for label in label2id if label != 'O'}

    for cross_valid in range(1):

        # print('cross_valid', cross_valid)

        model = GNN(word_vocab_size=WORD_VOCAB_SIZE, char_vocab_size=CHAR_VOCAB_SIZE, d_output=d_output, args=args)
        model.cuda()
        # print vocab_size

        # print('split dataset')
        # dataset.split_train_valid_test_bycase([0.5, 0.1, 0.4], 5, cross_valid)
        print('train:', len(dataset.train), 'valid:', len(dataset.valid), 'test:', len(dataset.test))
        sys.stdout.flush()
        
        train_dataloader = DataLoader(dataset.train, batch_size=args.batch, shuffle=True)
        valid_dataloader = DataLoader(dataset.valid, batch_size=args.batch)
        test_dataloader = DataLoader(dataset.test,  batch_size=args.batch)

        weight = torch.zeros(len(label2id))
        for label, idx in label2id.items():
            weight[idx] = 1 if label == 'O' else 2
        loss_function = nn.CrossEntropyLoss(weight.cuda(), reduce=False)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

        best_acc = -1
        wait = 0
        batch_cnt = 0

        for epoch in range(args.epochs):
            total_loss = 0
            pending_loss = None
            model.train()
            # random.shuffle(dataset.train)
            load_time, forward_time, backward_time = 0, 0, 0
            model.clear_time()
            
            train_log = open(args.save_path+'_train.log','w')
            for tensors, batch in tqdm(train_dataloader, file=train_log, mininterval=60):
                # print(batch[0].case_id, batch[0].doc_id, batch[0].page_id)
                start = time.time()
                data, data_word, pos, length, mask, label, adjs = to_var(tensors, cuda=args.cuda)
                batch_size, docu_len, sent_len, word_len = data.size()
                load_time += (time.time()-start)

                start = time.time()
                logit = model(data, data_word, pos, length, mask, adjs)
                forward_time += (time.time()-start)
                
                start = time.time()
                if args.crf:
                    logit = logit.view(batch_size*docu_len, sent_len, -1)
                    mask = mask.view(batch_size*docu_len, -1)
                    length = length.view(batch_size*docu_len)
                    label = label.view(batch_size*docu_len, -1)
                    loss = -model.crf_layer.loglikelihood(logit, mask, length, label)
                    loss = torch.masked_select(loss, torch.gt(length, 0)).mean() 
                else:
                    loss = loss_function(logit.view(-1, d_output), label.view(-1))
                    loss = torch.masked_select(loss, mask.view(-1)).mean()
                total_loss += loss.data.sum()
                # print(total_loss, batch[0].case_id, batch[0].doc_id, batch[0].page_id)
                if math.isnan(total_loss):
                    print('Loss is NaN!')
                    exit()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                backward_time += (time.time()-start)

                batch_cnt += 1
                if batch_cnt % 20000 != 0:
                    continue
                # print('load %f   forward %f   backward %f'%(load_time, forward_time, backward_time))
                # model.print_time()
                valid_acc, valid_prec, valid_recall, valid_f1 = evaluate(model, valid_dataloader, args=args)
                
                print('Epoch %d:  Train Loss: %.3f  Valid Acc: %.5f' % (epoch, total_loss, valid_acc))
                # print(acc_to_str(valid_f1))
                # scheduler.step()

                acc = np.mean(list(valid_f1.values())) # valid_acc
                print(acc)
                if acc >= best_acc:
                    obj = {'args':args, 'model':model.state_dict()}
                    torch.save(obj, args.save_path+'.model')
                    result_obj['valid_prec'] = np.mean(list(valid_prec.values()))
                    result_obj['valid_recall'] = np.mean(list(valid_recall.values()))
                    result_obj['valid_f1'] = np.mean(list(valid_f1.values()))
                wait = 0 if acc > best_acc else wait+1
                best_acc = max(acc, best_acc)

                model.train()
                sys.stdout.flush()
                if wait >= args.patience:
                    break
            
            train_log.close()
            os.remove(args.save_path+'_train.log')

            if wait >= args.patience:
                break

        obj = torch.load(args.save_path+'.model')
        model.load_state_dict(obj['model'])

        test(test_dataloader, model)

    # print("Cross Validation Result:")
    # for label in cross_res:
    #     cross_res[label] = np.mean(cross_res[label])
    # print(acc_to_str(cross_res))
    return cross_res


def test(test_dataloader, model=None):

    if model is None:
        obj = torch.load(args.save_path+'.model', map_location=lambda storage, loc:storage)
        train_args = obj['args']
        model = GNN(word_vocab_size=WORD_VOCAB_SIZE, char_vocab_size=CHAR_VOCAB_SIZE, d_output=d_output, args=train_args)
        model.load_state_dict(obj['model'])
        model.cuda()
        print('Model loaded.')

    test_acc, test_prec, test_recall, test_f1 = evaluate(model, test_dataloader, output=True, args=args)
    print('######## prec   : ', acc_to_str(test_prec))
    print('######## recall : ', acc_to_str(test_recall))
    print('######## f1     : ', acc_to_str(test_f1))
    prec, recall, f1 = np.mean(list(test_prec.values())), np.mean(list(test_recall.values())), np.mean(list(test_f1.values()))
    print(prec, recall, f1)
    result_obj['test_prec'] = prec
    result_obj['test_recall'] = recall
    result_obj['test_f1'] = f1
    result_obj['test_info'] = '\n'.join([acc_to_str(test_prec), acc_to_str(test_recall), acc_to_str(test_f1)])
    # result_obj['tmp_test_f1'] = mean_test_f1

if __name__ == '__main__':

    dataset = read_data(args.data_path, args.case, args.model)
    print('Data loaded.')
    sys.stdout.flush()
    
    if not args.test:
        train(dataset)
        # dataset = read_data(args.testbase, args.testcase, args.model)
        # test(dataset)
        for attr, value in sorted(args.__dict__.items()):
            result_obj[attr] = value
        json.dump(result_obj, open(args.save_path+'_results.json','w'))
    else:
        test_dataloader = DataLoader(dataset.data, batch_size=args.batch)
        test(test_dataloader)

# result = {}
# cross_res = train(args.seed)
# for label in cross_res:
#     if label not in result:
#         result[label] = []
#     result[label].append(cross_res[label])

# print("Final Result")
# for label in result:
#     print(label, np.mean(result[label]), np.std(result[label]))





