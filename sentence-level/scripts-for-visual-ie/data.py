import string
import os
import json
import torch
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp
import random
from tqdm import tqdm
from multiprocessing import Pool
from constants import *
import sys


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.maximum(np.array(adj.sum(1)), 1)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


class DataPoint():

    def __init__(self, obj, model):

        self.case_id = obj['id']['case']
        self.doc_id  = obj['id']['doc']
        self.page_id = obj['id']['page']

        # input_word = []
        G = {'v':obj['edge']['v'], 'h':obj['edge']['h']}
        N = len(obj['sent'])
        for i in range(N):
            if len(obj['sent'][i]) == 0:
                obj['sent'][i].append(' ')
                obj['tag'][i].append(0)
        sents = obj['sent']
        words = [w for sent in sents for w in sent]
        positions = obj['pos']
        tags  = obj['tag']

        self.max_word_len = max([len(w) for w in words])
        self.max_sent_len = max([len(s) for s in sents][:MAX_SENT_LEN])
        self.num_sent = len(sents)

        self.valid = (len(sents)>0 and len(words)>0)
        self.max_word_len = min(self.max_word_len, MAX_WORD_LEN)
        self.max_sent_len = min(self.max_sent_len, MAX_SENT_LEN)
        self.num_sent = min(self.num_sent, MAX_DOCU_LEN)

        self.num_word = len(words)
        self.num_feat = len(positions[0])
        self.sents = sents
        # self.words = words

        data = np.zeros((self.num_sent, self.max_sent_len, self.max_word_len), dtype=int)
        data_word = np.zeros((self.num_sent, self.max_sent_len), dtype=int)
        label = np.zeros((self.num_sent, self.max_sent_len), dtype=int)
        pos = np.array(positions)[:self.num_sent]
        # length = np.array([len(sent) for sent in sents])
        mask = np.zeros((self.num_sent, self.max_sent_len), dtype=int)
        
        # word_id = {}
        for i in range(self.num_sent):
            for j in range(len(sents[i])):
                if j >= self.max_sent_len:
                    break
                data_word[i,j] = get_word_id(sents[i][j])
                for k, c in enumerate(sents[i][j]):
                    if k >= self.max_word_len:
                        break
                    data[i,j,k] = get_char_id(c)
                if tags[i][j] in ignore_attrs_id:
                    tags[i][j] = 0
                label[i,j] = tags[i][j]
                mask[i,j] = 1
                # word_id[sents[i][j]] = i * self.max_sent_len + j
        #self.input = torch.nn.utils.rnn.pack_padded_sequence(Variable(data).cuda(), length, batch_first=True)
        self.data = data
        self.data_word = data_word
        self.pos = pos
        self.label = label
        self.length = mask.sum(axis=1)
        self.mask = mask

        # wordid2id = {order[i]:i for i in range(n)}
        # self.words = [words[i] for i in order]

        n = self.num_sent

        G['v'] = [(u,v) for u,v in G['v'] if u<n and v < n]
        G['h'] = [(u,v) for u,v in G['h'] if u<n and v < n]

        neighbor = [[-1, -1, -1, -1] for i in range(n)]   # up down left right

        for u,v in G['v']:
            ux1, uy1, ux2, uy2 = positions[u]
            vx1, vy1, vx2, vy2 = positions[v]
            if uy1 < vy1:
                neighbor[u][1] = v
                neighbor[v][0] = u
            else:
                neighbor[u][0] = v
                neighbor[v][1] = u

        for u,v in G['h']:
            ux1, uy1, ux2, uy2 = positions[u]
            vx1, vy1, vx2, vy2 = positions[v]
            if ux1 < vx1:
                neighbor[u][3] = v
                neighbor[v][2] = u
            else:
                neighbor[u][3] = v
                neighbor[v][2] = u

        neighbor_mask = np.zeros((4, n, n))
        for i in range(n):
            for j in range(4):
                if neighbor[i][j] != -1:
                    neighbor_mask[j,i,neighbor[i][j]] = 1

        if model == 'lstm-gcn-lstm':
            neighbor_mask = neighbor_mask.sum(axis=0, keepdims=True)

        self.adjs = neighbor_mask

        # if len(G['v']) == 0:
        #     adj_v = np.zeros((n, n))
        # else:
        #     row = [u for u,v in G['v'] if u<n and v<n]
        #     col = [v for u,v in G['v'] if u<n and v<n]
        #     data = [1. for u,v in G['v'] if u<n and v<n]
        #     mat = coo_matrix((data, (row, col)), shape=(n, n))
        #     adj_v = (mat + mat.transpose())

        # if len(G['h']) == 0:
        #     adj_h = np.zeros((n, n))
        # else:
        #     row = [u for u,v in G['h'] if u<n and v<n]
        #     col = [v for u,v in G['h'] if u<n and v<n]
        #     data = [1. for u,v in G['h'] if u<n and v<n]
        #     mat = coo_matrix((data, (row, col)), shape=(n, n))
        #     adj_h = (mat + mat.transpose())

        # self.adj = normalize_adj(adj_h + adj_v).toarray()
        # self.adj_h = normalize_adj(adj_h).toarray()
        # self.adj_v = normalize_adj(adj_v).toarray()

        self.path = None



def read_graph(tp):
    path, model = tp
    res = []
    if model in ['lstm', 'lstm-lstm']:
        filename = 'textline_bio.json'
    else:
        filename = 'graph_bio.json'
    f = open(path + filename)
    for line in f:
        obj = json.loads(line)
        if len(obj['sent']) == 0 or int(obj['id']['page']) > 5:
            continue
        try:
            x = DataPoint(obj, model)
        except:
            print('LOAD ERROR:', path)
            continue
        x.path = path
        if x.valid:
            res.append(x)
    return res

# def read_textline(path):
#     res = []
#     f = open(path + 'textline.json')
#     for line in f:
#         obj = json.loads(line)
#         if len(obj['sent']) == 0 or int(obj['id']['page']) > 5:
#             continue
#         try:
#             x = DataPoint(obj)
#         except:
#             print('LOAD ERROR:', path)
#             continue
#         x.path = path
#         if x.valid:
#             res.append(x)
#     return res


class DataSet():

    def __init__(self, base, case_list, model):

        self.data = []
        self.char_vocab_size = CHAR_VOCAB_SIZE
        self.case2path = {}
        
        train_filename, valid_filename, test_filename = case_list.split(',')
        train_lines = open(base+train_filename).readlines()
        train_path_list = [base+path.strip() for path in train_lines]
        valid_lines = open(base+valid_filename).readlines()
        valid_path_list = [base+path.strip() for path in valid_lines]
        test_lines = open(base+test_filename).readlines()
        test_path_list  = [base+path.strip() for path in test_lines]

        def load_path_list(path_list):
            pool = Pool(processes=50)
            # lines = open(base+case_list).readlines()
            # path_list = [base+path.strip() for path in lines]
            path_list = [(path, model) for path in path_list]

            result = []

            n = len(path_list)
            k = 20
            chunk = n // k

            for i in range(k):
                # print('load %d of %d' % (i+1, k))
                # sys.stdout.flush()
                if i != k-1:
                    subpath_list = path_list[i*chunk:(i+1)*chunk]
                else:
                    subpath_list = path_list[i*chunk:]
                result += pool.map(read_graph, subpath_list)
                # break

            pool.close()

            return result

        self.train = [x  for lst in load_path_list(train_path_list) for x in lst]
        self.valid = [x  for lst in load_path_list(valid_path_list) for x in lst]
        self.test  = [x  for lst in load_path_list(test_path_list)  for x in lst]

        self.data = self.train + self.valid + self.test

        for x in self.data:
            self.case2path[x.case_id] = x.path
        
        # self.train = self.data
        # self.valid = []
        # self.test = []

        self.order = None


    def split_train_valid_test(self, ratio, split, offset):
        n = len(self.data)
        if self.order == None:
            self.order = list(range(n))
            random.shuffle(self.order)
            # print(self.order)
            # p = 9369319
            # self.order = [i*p%n for i in range(n)]
            # print(self.order)

        order = self.order[int(n*offset/split):n] + self.order[:int(n*offset/split)]
        train_size = int(n*ratio[0])
        valid_size = int(n*ratio[1])
        self.train = [self.data[i] for i in order[:train_size]]
        self.valid = [self.data[i] for i in order[train_size:train_size+valid_size]]
        self.test = [self.data[i] for i in order[train_size+valid_size:]]

    def split_train_valid_test_bycase(self, ratio, split, offset):
        case_id_list = list(set([x.case_id for x in self.data]))
        random.shuffle(case_id_list)
        n = len(case_id_list)
        train_size = int(n*ratio[0])
        valid_size = int(n*ratio[1])
        train_set = set(case_id_list[:train_size])
        valid_set = set(case_id_list[train_size:train_size+valid_size])
        test_set = set(case_id_list[train_size+valid_size:])
        self.train = [x for x in self.data if x.case_id in train_set]
        self.valid = [x for x in self.data if x.case_id in valid_set]
        self.test = [x for x in self.data if x.case_id in test_set]
        print('train case', len(train_set), 'valid case', len(valid_set), 'test case', len(test_set))
        def output_case(f, case_id):
            for cid in case_id:
                f.write(self.case2path[cid] + '\n')
        output_case(open('train_case.txt', 'w'), train_set)
        output_case(open('valid_case.txt', 'w'), valid_set)
        output_case(open('test_case.txt', 'w'), test_set)
        




class DataLoader():

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iter = len(dataset) // batch_size + (len(dataset) % batch_size > 0)
        self.iter_cnt = 0
        self.shuffle = shuffle

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.num_iter

    def shuffle_dataset(self):
        random.shuffle(self.dataset)

    def next(self):
        if self.iter_cnt < self.num_iter:
            start_idx = self.iter_cnt * self.batch_size
            end_idx = min(len(self.dataset), start_idx + self.batch_size)
            self.iter_cnt += 1
            batch = self.dataset[start_idx:end_idx]
            max_docu_len = max([x.num_sent for x in batch])
            max_sent_len = max([x.max_sent_len for x in batch])
            max_word_len = max([x.max_word_len for x in batch])
            data = torch.stack([
                torch.ByteTensor(np.pad(x.data, 
                    pad_width=[(0,max_docu_len-x.num_sent),(0,max_sent_len-x.max_sent_len),(0,max_word_len-x.max_word_len)], 
                    mode='constant', constant_values=0))
                for x in batch])
            data_word = torch.stack([
                torch.LongTensor(np.pad(x.data_word, 
                    pad_width=[(0,max_docu_len-x.num_sent),(0,max_sent_len-x.max_sent_len)], mode='constant', constant_values=0))
                for x in batch])
            pos = torch.stack([
                torch.FloatTensor(np.pad(x.pos,
                    pad_width=[(0,max_docu_len-x.num_sent),(0,0)], mode='constant', constant_values=0))
                for x in batch])
            length = torch.stack([
                torch.from_numpy(np.pad(x.length,
                    pad_width=[(0,max_docu_len-x.num_sent)], mode='constant', constant_values=0))
                for x in batch])
            mask = torch.stack([
                torch.ByteTensor(np.pad(x.mask,
                    pad_width=[(0,max_docu_len-x.num_sent),(0,max_sent_len-x.max_sent_len)],
                    mode='constant', constant_values=0))
                for x in batch])
            label = torch.stack([
                torch.from_numpy(np.pad(x.label,
                    pad_width=[(0,max_docu_len-x.num_sent),(0,max_sent_len-x.max_sent_len)],
                    mode='constant', constant_values=0))
                for x in batch])
            adjs = torch.stack([
                torch.FloatTensor(np.pad(x.adjs,
                    pad_width=[(0,0),(0,max_docu_len-x.num_sent),(0,max_docu_len-x.num_sent)],
                    mode='constant', constant_values=0))
                for x in batch])
            # neighbor_mask = torch.stack([
            #     torch.ByteTensor(np.pad(x.neighbor_mask,
            #         pad_width=[(0,0),(0,max_docu_len-x.num_sent),(0,max_docu_len-x.num_sent)],
            #         mode='constant', constant_values=0))
            #     for x in batch])
            # adj_h = torch.stack([
            #     torch.FloatTensor(np.pad(x.adj_h,
            #         pad_width=[(0,max_docu_len-x.num_sent),(0,max_docu_len-x.num_sent)],
            #         mode='constant', constant_values=0))
            #     for x in batch])
            # adj_v = torch.stack([
            #     torch.FloatTensor(np.pad(x.adj_v,
            #         pad_width=[(0,max_docu_len-x.num_sent),(0,max_docu_len-x.num_sent)],
            #         mode='constant', constant_values=0))
            #     for x in batch])
            return [data, data_word, pos, length, mask, label, adjs], batch
        else:
            self.iter_cnt = 0
            if self.shuffle:
                self.shuffle_dataset()
            raise StopIteration()



def read_data(base, case_list='selected_case.txt', model='lstm-gcn-lstm'):
    return DataSet(base, case_list, model)







def read_page(tp):
    path, lineid, model = tp
    f = open(path)
    line = f.readlines()[lineid]
    obj = json.loads(line)
    if len(obj['sent']) == 0 or int(obj['id']['page']) > 5:
        return None
    try:
        x = DataPoint(obj, model)
        if x.valid:
            return x
    except:
        pass
    return None


class ExpDataSet():

    def __init__(self, template_lists, n_page, model):

        self.data = []

        filename = 'graph.json' if model != 'lstm' else 'textline.json'

        pool = Pool(processes=16)

        for template, _list in template_lists.items():
            case_cnt = {}
            for i in range(0, len(_list), n_page*2):
                sublist = _list[i:i+n_page*2]
                for j in range(len(sublist)):
                    sublist[j][0] += filename
                res = pool.map(read_page, sublist)
                for j in range(len(res)):
                    if res[j] is not None:
                        case_id = res[j].case_id
                        if case_id in case_cnt or len(case_cnt) < n_page:
                            case_cnt[case_id] = True
                            self.data.append(res[j])
                if len(case_cnt) >= n_page:
                    break

        self.train = self.data
        self.valid = []
        self.test = []

    def split_train_valid(self, ratio):
        n_train = int(ratio[0]*len(self.data))
        random.shuffle(self.data)
        self.train = self.data[:n_train]
        self.valid = self.data[n_train:]



def read_exp_data(base, n_template, n_page, case_list='full_selected_case.txt', model='gnn'):
    # {'FollowUp': 152211, 'LightBlue': 19329, 'TableForm': 3878, 'WRB_Form': 132, 'BayerAG_Table': 195219, 'Blue': 176605, 'CSACDO_Form': 26681, 'Oracle': 318200, 'Mail': 605916}
    train_template = ['FollowUp', 'BayerAG_Table', 'LightBlue', 'TableForm'][:n_template]
    test_template = ['Blue', 'Oracle']

    lines = open(base+case_list).readlines()
    path_list = [base+path.strip() for path in lines]
    case2path = {path.split('/')[-2]:path for path in path_list}

    template2list = {}

    f = open('extracted_files/case_page_type.txt')
    for line in f:
        a = line.strip().split('\t')
        if a[-1] not in template2list:
            template2list[a[-1]] = []
        case = a[0].split('/')[0]
        if case not in case2path:
            continue
        template2list[a[-1]].append([case2path[case], int(a[1]), model])

    random.seed(1)
    # print('load test data')
    # sys.stdout.flush()
    # test_dataset = ExpDataSet([template2list[t] for t in test_template], 1000, model)

    # for x in test_dataset.data[:10]:
    #     print(x.case_id)
    # print()

    print('load train data')
    sys.stdout.flush()
    train_dataset = ExpDataSet({t:template2list[t] for t in train_template}, n_page, model)

    return train_dataset






