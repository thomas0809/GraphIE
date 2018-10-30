import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from data_twitter import *
from gnn_twitter import GNN_Twitter
import numpy as np
import random
import json
import time
import math
import sys
import os
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 20)')
parser.add_argument('--batch', type=int, default=1, help='mini-batch size')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--wd', type=float, default=0., help='weight decay (default: 0)')
parser.add_argument('--cuda', action='store_true', default=True, help='enable CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--patience', type=int, default=5, help='number of times to observe worsening validation set error before giving up')
parser.add_argument('--d_char_embed', type=int, default=64, help='character embedding dimension')
parser.add_argument('--d_word_embed', type=int, default=50, help='character embedding dimension')
parser.add_argument('--filter_sizes', type=str, default='2,3,4', help='character cnn filter size')
parser.add_argument('--n_filter', type=int, default=64, help='character cnn filter number')
parser.add_argument('--d_pos_embed', type=int, default=32, help='additional feature')
parser.add_argument('--d_graph', type=str, default='128', help='lstm and graph layer dimension')
parser.add_argument('--weight_balance', type=float, default=1., help='weight balance')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout probability')
parser.add_argument('--model', type=str, default='lstm-gcn-lstm', choices=['lstm', 'lstm-lstm', 'lstm-gcn-lstm', 'lstm-gat-lstm'], help='model')
parser.add_argument('--crf', action='store_true', default=False, help='final crf')
parser.add_argument('--save_path', type=str, default='models/output', help='output name')
parser.add_argument('--data_path', type=str, default='twitter_data/tmp/', help='data path')
parser.add_argument('--task', type=str, default='education', choices=['education','job'])
parser.add_argument('--entity_classification', action='store_true', default=False, help='entity classification')
parser.add_argument('--test', action='store_true', default=False, help='test mode')

args = parser.parse_args()

args.filter_sizes = [int(x) for x in args.filter_sizes.split(',')]
args.d_graph = [int(x) for x in args.d_graph.split(',')]

d_output = D_TAG

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


def run_model(model, ego, loss_function, predict=False, args=args):
    # start = time.time()
    data_char, data_word, mask, length, adj, label = to_var(
        [ego.data_char, ego.data_word, ego.mask, ego.length, ego.adj, ego.label], cuda=args.cuda)
    if args.entity_classification == True:
        [label, entity_mask] = to_var([ego.entity_label, ego.entity_mask], cuda=args.cuda)
    else:
        entity_mask = None
    # load_time += (time.time()-start)

    # start = time.time()
    logit = model(data_char, data_word, length, mask, adj, entity_mask)
    # forward_time += (time.time()-start)

    if args.crf:
        loss = -model.crf_layer.loglikelihood(logit, mask[0], length[0], label)
        loss = torch.masked_select(loss, torch.gt(length[0], 0)).mean()
    else:
        loss = loss_function(logit.view(-1, d_output), label.view(-1))
        if args.entity_classification:
            loss = loss.mean()
        else:
            loss = torch.masked_select(loss, mask[0].view(-1)).mean()

    if predict:
        if args.crf:
            _, pred = model.crf_layer.viterbi_decode(logit, mask[0])
            pred = pred.data
        elif not args.entity_classification:
            pred = logit.max(dim=2)[1].data
        else:
            logit = logit[:ego.c_user.num_sent]
            pred = logit.max(dim=1)[1].data
    else:
        pred = None

    return logit, loss, pred


def evaluate(model, dataset, output=False, args=args):
    # (hit, pred_cnt, gold_cnt)
    count = {TAG_B: [0, 0, 0], TAG_I: [0, 0, 0]}
    entity_count = [0, 0, 0]
    n_correct = 0
    n_total = 0
    
    if output:
        pred_file = open(args.save_path+'_pred.json', 'w')

    weight = torch.FloatTensor([1] + [args.weight_balance] * (d_output - 1))
    loss_function = nn.CrossEntropyLoss(weight.cuda(), reduction='none')
    
    model.eval()
    total_loss = 0
    eval_log = open(args.save_path+'_eval.log', 'w')
    for ego in tqdm(dataset, file=eval_log,  mininterval=10): 
        
        logit, loss, pred = run_model(model, ego, loss_function, predict=True, args=args)
        total_loss += loss.data.sum()

        if not args.entity_classification:
            c_user = ego.c_user
            obj = {'id': c_user.id, 'results':{}}
            res = obj['results']
            entity_count[2] += c_user.num_pos_entity_match
            for i in range(c_user.num_sent):
                m = min(len(c_user.text[i]), len(pred[i]))
                l = 0
                while l < m:
                    if pred[i,l] in [TAG_B, TAG_I]:
                        r = l
                        while r+1<m and pred[i,r+1] == TAG_I:
                            r += 1
                        entity = ' '.join(c_user.text[i][l:r+1])
                        l = r
                        if entity not in res:
                            res[entity] = 0
                        res[entity] += 1
                        entity_count[1] += 1
                        if entity in c_user.pos_entity:
                            entity_count[0] += 1
                    l += 1
        
        if output:
            pred_file.write(json.dumps(obj)+'\n')

        label = ego.label
        pred = pred.contiguous().view(-1).cpu()
        gold = label.contiguous().view(-1)

        if not args.entity_classification:
            mask = ego.mask[0]
            pred = pred.masked_select(mask.view(-1))
            gold = gold.masked_select(mask.view(-1))

        for tag in [TAG_B, TAG_I]:
            pred_cnt = int(pred.eq(tag).sum())
            gold_cnt = int(gold.eq(tag).sum())
            hit = int(pred.eq(tag).mul(gold.eq(tag)).sum())
            count[tag][0] += hit
            count[tag][1] += pred_cnt
            count[tag][2] += gold_cnt

        n_total += pred.size()[0]
        n_correct += pred.eq(gold).sum()

    eval_log.close()
    os.remove(args.save_path+'_eval.log')

    prec_list = []
    recall_list = []
    f1_list = []
    
    for tag in [TAG_B, TAG_I]:
        prec = float(count[tag][0]) / max(count[tag][1],1)
        recall = float(count[tag][0]) / max(count[tag][2],1)
        f1 = 2*prec*recall / max(prec+recall,1e-9)
        prec_list.append(prec)
        recall_list.append(recall)
        f1_list.append(f1)
    
    prec = np.mean(prec_list)
    recall = np.mean(recall_list)
    f1 = np.mean(f1_list)

    e_prec = float(entity_count[0]) / max(entity_count[1],1)
    e_recall = float(entity_count[0]) / max(entity_count[2],1)
    e_f1 = 2*e_prec*e_recall / max(e_prec+e_recall,1e-9)

    return total_loss, float(n_correct)/n_total, prec, recall, f1, e_prec, e_recall, e_f1


def train(dataset):


    CROSS_VALID = 5
    cross_valid_res = np.zeros((CROSS_VALID, 6))

    for cross_valid in range(CROSS_VALID):


        print('\n========================================================================\n')
        print('cross_valid', cross_valid)

        print('split dataset')

        # dataset.split_train_valid_test([0.6, 0.2, 0.2], 5, cross_valid)
        dataset.split_dataset(args.data_path+args.task+'_split.json', cross_valid)
        print('train:', len(dataset.train), 'valid:', len(dataset.valid), 'test:', len(dataset.test))
        
        model = GNN_Twitter(word_vocab_size=WORD_VOCAB_SIZE, char_vocab_size=CHAR_VOCAB_SIZE, d_output=d_output, args=args,
                            pretrained_emb=glove_emb)
        model.cuda()
        
        weight = torch.FloatTensor([1] + [args.weight_balance] * (d_output - 1))
        loss_function = nn.CrossEntropyLoss(weight.cuda(), reduction='none')
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

        best_acc = -1
        total_loss = 0
        wait = 0
        iters = 0
        model.train()

        for epoch in range(args.epochs):
            
            train_log = open(args.save_path+'_train.log','w')
            for ego in tqdm(dataset.train, file=train_log,  mininterval=10):
                iters += 1
                logit, loss, pred = run_model(model, ego, loss_function, args=args)
                total_loss += loss.data.sum()
                
                if math.isnan(total_loss):
                    print('Loss is NaN!')
                    exit()

                # start = time.time()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # backward_time += (time.time()-start)
                
                if iters % 5000 != 0:
                    continue
                # print('load %f   forward %f   backward %f'%(load_time, forward_time, backward_time))
                # model.print_time()
                train_acc = 0
                valid_loss, valid_acc, valid_prec, valid_recall, valid_f1, e_prec, e_recall, e_f1 = evaluate(model, dataset.valid, args=args)
                print('Epoch %d:  Train Loss: %.3f  Valid Loss: %.3f  Valid: %.3f' \
                    % (epoch, total_loss, valid_loss, valid_acc))
                # print(acc_to_str(train_f1))
                print('Valid (tag)    - prec:%.3f recall:%.3f f1:%.3f'%(valid_prec, valid_recall, valid_f1))
                print('Valid (entity) - prec:%.3f recall:%.3f f1:%.3f'%(e_prec, e_recall, e_f1))
                # print(acc_to_str(test_f1))
                # scheduler.step()

                acc = e_f1
                print(acc)
                sys.stdout.flush()
                if acc >= best_acc:
                    obj = {'args':args, 'model':model.state_dict()}
                    torch.save(obj, args.save_path+'.model')
                    result_obj['valid_prec'] = e_prec
                    result_obj['valid_recall'] = e_recall
                    result_obj['valid_f1'] = e_f1
                wait = 0 if acc > best_acc else wait+1
                best_acc = max(acc, best_acc)
                if wait >= args.patience:
                    break

                model.train()

            train_log.close()
            os.remove(args.save_path+'_train.log')

            if wait >= args.patience:
                break
            
        obj = torch.load(args.save_path+'.model')
        model.load_state_dict(obj['model'])

        # test_dataloader = DataLoader(dataset.test,  batch_size=args.batch)
        test_prec, test_recall, test_f1, e_prec, e_recall, e_f1 = test(dataset.test, model)
        cross_valid_res[cross_valid] = [test_prec, test_recall, test_f1, e_prec, e_recall, e_f1]

    print(cross_valid_res)
    print('cross validation results')
    mean_res = cross_valid_res.mean(0)
    print(mean_res)

    result_obj['test_prec']     = mean_res[0]
    result_obj['test_recall']   = mean_res[1]
    result_obj['test_f1']       = mean_res[2]
    result_obj['entity_prec']   = mean_res[3]
    result_obj['entity_recall'] = mean_res[4]
    result_obj['entity_f1']     = mean_res[5]

    return 


def test(dataset, model=None):

    if model is None:
        obj = torch.load(args.save_path+'.model', map_location=lambda storage, loc:storage)
        train_args = obj['args']
        model = GNN_Twitter(word_vocab_size=WORD_VOCAB_SIZE, char_vocab_size=CHAR_VOCAB_SIZE, d_output=d_output, args=train_args)
        model.load_state_dict(obj['model'])
        model.cuda()
        print('Model loaded.')

    test_loss, test_acc, test_prec, test_recall, test_f1, e_prec, e_recall, e_f1 = evaluate(model, dataset, output=True, args=args)
    prec, recall, f1 = test_prec, test_recall, test_f1
    print('tagging', prec, recall, f1)
    print('entity', e_prec, e_recall, e_f1)
    result_obj['test_prec'] = prec
    result_obj['test_recall'] = recall
    result_obj['test_f1'] = f1
    result_obj['entity_prec'] = e_prec
    result_obj['entity_recall'] = e_recall
    result_obj['entity_f1'] = e_f1
    result_obj['test_info'] = ''

    return test_prec, test_recall, test_f1, e_prec, e_recall, e_f1



if __name__ == '__main__':

    print('random seed:', args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    graph = ('gcn' in args.model) or ('gat' in args.model)
    dataset = read_twitter_data(args.data_path+args.task+'_user.json', args.data_path+args.task+'_graph.json', args.data_path+args.task+'_vocab.json', graph)
    print('Data loaded.')

    if not args.test:
        train(dataset)
        for attr, value in sorted(args.__dict__.items()):
            result_obj[attr] = value
        json.dump(result_obj, open(args.save_path+'_results.json','w'))




