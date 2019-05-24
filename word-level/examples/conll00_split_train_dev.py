import os
import sys
import random

infile_path = sys.argv[1]
dev_size = int(sys.argv[2])

infile_dir = '/'.join(infile_path.split('/')[:-1])
outfile_train = os.path.join(infile_dir, 'train.txt')
outfile_dev = os.path.join(infile_dir, 'dev.txt')

all_sentences = []
with open(infile_path, 'r') as ifile:
	 sentence = ''
	 line_idx = 1
	 for line in ifile:
		if line.strip():
			sentence = sentence + '{} '.format(line_idx) + line
			line_idx += 1
		else:
			all_sentences.append(sentence)
			sentence = ''
			line_idx = 1

print('Total train data: {}'.format(len(all_sentences)))

random.seed(2018)
dev = random.sample(all_sentences, dev_size)
dev_set = set(dev)
train = [sentence for sentence in all_sentences if sentence not in dev_set]

with open(outfile_train, 'w') as ofile_train:
	for sentence in train:
		ofile_train.write(sentence+'\n')

with open(outfile_dev, 'w') as ofile_dev:
	for sentence in dev:
		ofile_dev.write(sentence+'\n')