import json
import operator
from tqdm import tqdm

vocab = {}

case = open('full_selected_case.txt')

for line in tqdm(case):
    path = line.strip()
    f = open(path + 'graph.json')
    for line in f:
        obj = json.loads(line)
        for sent in obj['sent']:
            for w in sent:
                w = w.lower()
                if w not in vocab:
                    vocab[w] = 0
                vocab[w] += 1

sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)
json.dump(sorted_vocab, open('visual_vocab.json','w'))

print(len(sorted_vocab))
output = open('vocab.txt', 'w')
for t in sorted_vocab:
    output.write(t[0] + '\n')
