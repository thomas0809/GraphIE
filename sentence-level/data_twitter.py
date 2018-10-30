import string
import os
import json
import torch
import torchtext
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp
import random
from tqdm import tqdm
from multiprocessing import Pool


vocab = ' ' + string.ascii_letters + string.digits + string.punctuation

character2id = {c:idx for idx, c in enumerate(vocab)}
UNK_CHAR = len(character2id)
CHAR_VOCAB_SIZE = len(vocab) + 1

def get_char_id(c):
    return character2id[c] if c in character2id else UNK_CHAR

WORD_VOCAB_SIZE = 10000

PAD_WORD = 0
UNK_WORD = 1
word2id = {'PAD': PAD_WORD, 'UNK': UNK_WORD}

TAG_O = 0
TAG_B = 1
TAG_I = 2

D_TAG = 3

D_WORD_EMB = 50
glove_emb = torch.zeros((WORD_VOCAB_SIZE, D_WORD_EMB))
GloVe_vocab = torchtext.vocab.GloVe(name='twitter.27B', dim=D_WORD_EMB)


def load_word_vocab(vocab_file):
    sorted_word = json.load(open(vocab_file))
    words = [t[0] for t in sorted_word[:WORD_VOCAB_SIZE]]
    for w in words:
        word2id[w] = len(word2id)
        if len(word2id) == WORD_VOCAB_SIZE:
            break
    cnt = 0
    for w,_id in word2id.items():
        if w not in GloVe_vocab.stoi:
            cnt += 1
        glove_emb[_id] = GloVe_vocab[w]
    print("%d words don't have pretrained embedding" % cnt)


def get_word_id(w):
    w = w.lower()
    return word2id[w] if w in word2id else UNK_WORD


MAX_WORD_LEN = 16
MAX_SENT_LEN = 50
MAX_NUM_USER = 32

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.maximum(np.array(adj.sum(1)), 1)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


class User():

    def __init__(self, obj):

        self.id = obj['id']
        self.name = obj['name']
        self.full_name = obj['full_name']
        self.gold_ans = obj['label']
        self.pos_entity = obj['pos_entity']

        self.text = []
        self.match = []

        self.n_pos_tweet = len(obj['pos_tweet'])
        self.n_neg_tweet = len(obj['neg_tweet'])

        for tweet in obj['pos_tweet'] + obj['neg_tweet']:
            self.text.append(tweet['sent'])
            self.match.append(tweet['match'])

        self.num_pos_entity_match = sum([len(tweet['match']) for tweet in obj['pos_tweet']])
            
        self.max_sent_len = max([len(s) for s in self.text])
        self.max_word_len = max([len(w) for s in self.text for w in s])
        self.num_sent = len(self.text)

        self.max_word_len = min(self.max_word_len, MAX_WORD_LEN)
        self.max_sent_len = min(self.max_sent_len, MAX_SENT_LEN)

        data_char = np.zeros((self.num_sent, self.max_sent_len, self.max_word_len), dtype=int)
        data_word = np.zeros((self.num_sent, self.max_sent_len), dtype=int)
        label = np.zeros((self.num_sent, self.max_sent_len), dtype=int)
        mask = np.zeros((self.num_sent, self.max_sent_len), dtype=int)

        entity_mask = np.zeros((self.num_sent, self.max_sent_len), dtype=int)
        entity_label = np.array([1]*self.n_pos_tweet + [0]*self.n_neg_tweet, dtype=int)

        for i in range(self.num_sent):
            for j in range(len(self.text[i])):
                if j >= self.max_sent_len:
                    break
                mask[i,j] = 1
                data_word[i,j] = get_word_id(self.text[i][j])
                for k, c in enumerate(self.text[i][j]):
                    if k >= self.max_word_len:
                        break
                    data_char[i,j,k] = get_char_id(c)
            for l, r in self.match[i]:
                entity_mask[i, l:r+1] = 1
                if entity_label[i] == 1:
                    label[i, l:r+1] = TAG_I
                    label[i, l] = TAG_B

        self.data_char = data_char
        self.data_word = data_word
        self.label = label
        self.mask = mask
        self.length = mask.sum(axis=1)

        self.entity_mask = entity_mask
        self.entity_label = entity_label


class EgoNetwork():

    def __init__(self, users, edges):

        if len(users) > MAX_NUM_USER:
            users = users[:MAX_NUM_USER]

        self.max_num_sent = max([x.num_sent for x in users])
        self.max_sent_len = max([x.max_sent_len for x in users])
        self.max_word_len = max([x.max_word_len for x in users])

        max_num_sent, max_sent_len, max_word_len = self.max_num_sent, self.max_sent_len, self.max_word_len

        self.users = users
        self.c_user = users[0]

        self.data_char = torch.stack([
                torch.ByteTensor(np.pad(x.data_char, 
                    pad_width=[(0,max_num_sent-x.num_sent),(0,max_sent_len-x.max_sent_len),(0,max_word_len-x.max_word_len)], 
                    mode='constant', constant_values=0))
                for x in users])
        self.data_word = torch.stack([
                torch.LongTensor(np.pad(x.data_word, 
                    pad_width=[(0,max_num_sent-x.num_sent),(0,max_sent_len-x.max_sent_len)], mode='constant', constant_values=0))
                for x in users])
        self.mask = torch.stack([
                torch.ByteTensor(np.pad(x.mask,
                    pad_width=[(0,max_num_sent-x.num_sent),(0,max_sent_len-x.max_sent_len)],
                    mode='constant', constant_values=0))
                for x in users])
        self.length = torch.stack([
                torch.from_numpy(np.pad(x.length,
                    pad_width=[(0,max_num_sent-x.num_sent)], mode='constant', constant_values=0))
                for x in users])

        x = users[0]
        self.label = torch.from_numpy(np.pad(x.label, 
                    pad_width=[(0,max_num_sent-x.num_sent),(0,max_sent_len-x.max_sent_len)], 
                    mode='constant', constant_values=0))

        self.entity_mask = torch.ByteTensor(x.entity_mask)
        self.entity_label = torch.from_numpy(x.entity_label)

        n = len(users)
        if len(edges) == 0:
            self.adj = np.zeros((n, n))
        else:
            edges = [(u,v) for u,v in edges if u < n and v < n]
            row = [u for u,v in edges]
            col = [v for u,v in edges]
            data = [1. for u,v in edges]
            mat = coo_matrix((data, (row, col)), shape=(n, n))
            self.adj = mat.toarray()
            # self.adj = normalize_adj(mat).toarray()

        self.adj = torch.FloatTensor(self.adj)
        




def read_user(line):
    obj = json.loads(line)
    return User(obj)

class DataSet():

    def __init__(self, user_file, graph_file, graph):

        self.char_vocab_size = CHAR_VOCAB_SIZE
        
        pool = Pool(processes=16)
        lines = open(user_file).readlines()#[:10]
        self.users = pool.map(read_user, lines)

        uid2id = {u.id:idx for idx, u in enumerate(self.users)}

        self.data = []

        lines = open(graph_file).readlines()#[:10]
        for line in lines:
            obj = json.loads(line)
            if graph:
                users = [self.users[uid2id[uid]] for uid in obj['node']]
                edges = obj['edge']
            else:
                users = [self.users[uid2id[obj['center']]]]
                edges = []
            self.data.append(EgoNetwork(users, edges))

        # self.data= self.data[:50]

        self.train = self.data
        self.valid = []
        self.test = []

        self.order = None


    def split_train_valid_test(self, ratio, split, offset):
        n = len(self.data)
        if self.order == None:
            self.order = list(range(n))
            random.shuffle(self.order)

        # f = open('split.json', 'w')
        # split_list = []
        # for offset in range(split):
        order = self.order[int(n*offset/split):n] + self.order[:int(n*offset/split)]
        train_size = int(n*ratio[0])
        valid_size = int(n*ratio[1])
        self.train = [self.data[i] for i in order[:train_size]]
        self.valid = [self.data[i] for i in order[train_size:train_size+valid_size]]
        self.test = [self.data[i] for i in order[train_size+valid_size:]]
        obj = {}
        obj['train'] = [ego.c_user.id for ego in self.train]
        obj['valid'] = [ego.c_user.id for ego in self.valid]
        obj['test'] = [ego.c_user.id for ego in self.test]
            # split_list.append(obj)
        # json.dump(split_list, f)


    def split_dataset(self, filename, _id):
        f = open(filename)
        split_list = json.load(f)
        f.close()
        obj = split_list[_id]
        uid2ego = {ego.c_user.id: ego for ego in self.data}
        self.train = [uid2ego[uid] for uid in obj['train']]
        self.valid = [uid2ego[uid] for uid in obj['valid']]
        self.test = [uid2ego[uid] for uid in obj['test']]
        # train_set = set(obj['train'])
        # valid_set = set(obj['valid'])
        # test_set = set(obj['test'])
        # self.train = [ego for ego in self.data if ego.c_user.id in train_set]
        # self.valid = [ego for ego in self.data if ego.c_user.id in valid_set]
        # self.test  = [ego for ego in self.data if ego.c_user.id in test_set]

        



def read_twitter_data(user_file, graph_file, vocab_file, graph):
    load_word_vocab(vocab_file)
    return DataSet(user_file, graph_file, graph)






















'''

class User():

    def __init__(self, obj):

        self.id = obj['id']
        self.name = obj['name']
        self.full_name = obj['full_name']
        self.gold_ans = obj['label']

        self.text = []
        self.tag = []
        self.entity = []
        self.match = []
        for entity in obj['entity']:
            tag = obj['entity'][entity]['tag']
            l = len(self.text)
            for tweet in obj['entity'][entity]['tweet']:
                self.text.append(tweet['sent'])
                self.tag.append(tag)
                self.match.append(tweet['match'])
            r = len(self.text) - 1
            self.entity.append((entity, tag, (l, r)))

        self.max_sent_len = max([len(s) for s in self.text])
        self.max_word_len = max([len(w) for s in self.text for w in s])
        self.num_sent = len(self.text)
        self.num_entity = len(self.entity)

        self.max_word_len = min(self.max_word_len, MAX_WORD_LEN)
        self.max_sent_len = min(self.max_sent_len, MAX_SENT_LEN)

        data_char = np.zeros((self.num_sent, self.max_sent_len, self.max_word_len), dtype=int)
        data_word = np.zeros((self.num_sent, self.max_sent_len), dtype=int)
        label = np.zeros(self.num_sent, dtype=int)
        mask = np.zeros((self.num_sent, self.max_sent_len), dtype=int)
        # in each sentence, which words are the entity words
        entity_mask = np.zeros((self.num_sent, self.max_sent_len), dtype=int)
        # which sentences are for each entity
        entity_sent_mask = np.zeros((self.num_entity, self.num_sent), dtype=int)
        entity_label = np.zeros(self.num_entity, dtype=int)

        for i in range(self.num_sent):
            label[i] = self.tag[i]
            for j in range(len(self.text[i])):
                if j >= self.max_sent_len:
                    break
                mask[i,j] = 1
                data_word[i,j] = get_word_id(self.text[i][j])
                for k, c in enumerate(self.text[i][j]):
                    if k >= self.max_word_len:
                        break
                    data_char[i,j,k] = get_char_id(c)
            l, r = self.match[i][0]
            entity_mask[i, l:r+1] = 1 
        
        for i, (entity, tag, (l, r)) in enumerate(self.entity):
            entity_sent_mask[i, l:r+1] = 1
            entity_label[i] = tag

        self.data_char = data_char
        self.data_word = data_word
        self.label = label
        self.mask = mask
        self.length = mask.sum(axis=1)
        self.entity_mask = entity_mask
        self.entity_sent_mask = entity_sent_mask
        self.entity_label = entity_label


class EgoNetwork():

    def __init__(self, users, edges):

        self.max_num_sent = max([x.num_sent for x in users])
        self.max_sent_len = max([x.max_sent_len for x in users])
        self.max_word_len = max([x.max_word_len for x in users])

        max_num_sent, max_sent_len, max_word_len = self.max_num_sent, self.max_sent_len, self.max_word_len

        self.users = users
        self.c_user = users[0]

        self.data_char = torch.stack([
                torch.ByteTensor(np.pad(x.data_char, 
                    pad_width=[(0,max_num_sent-x.num_sent),(0,max_sent_len-x.max_sent_len),(0,max_word_len-x.max_word_len)], 
                    mode='constant', constant_values=0))
                for x in users])
        self.data_word = torch.stack([
                torch.LongTensor(np.pad(x.data_word, 
                    pad_width=[(0,max_num_sent-x.num_sent),(0,max_sent_len-x.max_sent_len)], mode='constant', constant_values=0))
                for x in users])
        self.mask = torch.stack([
                torch.ByteTensor(np.pad(x.mask,
                    pad_width=[(0,max_num_sent-x.num_sent),(0,max_sent_len-x.max_sent_len)],
                    mode='constant', constant_values=0))
                for x in users])
        self.length = torch.stack([
                torch.from_numpy(np.pad(x.length,
                    pad_width=[(0,max_num_sent-x.num_sent)], mode='constant', constant_values=0))
                for x in users])

        self.label = torch.from_numpy(users[0].label)
        self.entity_mask = torch.ByteTensor(users[0].entity_mask)
        self.entity_sent_mask = torch.ByteTensor(users[0].entity_sent_mask)
        self.entity_label = torch.from_numpy(users[0].entity_label)

        n = len(users)
        if len(edges) == 0:
            self.adj = np.zeros((n, n))
        else:
            row = [u for u,v in edges]
            col = [v for u,v in edges]
            data = [1. for u,v in edges]
            mat = coo_matrix((data, (row, col)), shape=(n, n))
            self.adj = normalize_adj(mat).toarray()

        self.adj = torch.FloatTensor(self.adj)

'''
