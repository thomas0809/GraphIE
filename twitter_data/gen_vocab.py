import sys
import argparse
import json
import random
import operator

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='education')
args = parser.parse_args()

vocab = {}

for line in open(args.task+'_user.json'):
    obj = json.loads(line)
    for s in obj['pos_tweet'] + obj['neg_tweet']:
        for w in s['sent']:
        	w = w.lower()
        	if w not in vocab:
        		vocab[w] = 0
        	vocab[w] += 1

sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)
json.dump(sorted_vocab, open('%s_vocab.json'%args.task, 'w'))
