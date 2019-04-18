import os
import sys
import random

infile_train = sys.argv[1]
infile_dev = sys.argv[2]

outfile_train = infile_train + '.resized'
outfile_dev = infile_dev + '.resized'

all_sentences = []
with open(infile_train, 'r') as ifile:
	 sentence = ''
	 for line in ifile:
		if line.strip():
			sentence += line
		else:
			all_sentences.append(sentence)
			sentence = ''

with open(infile_dev, 'r') as ifile:
	 sentence = ''
	 for line in ifile:
		if line.strip():
			sentence += line
		else:
			all_sentences.append(sentence)
			sentence = ''

print('Total train data: {}'.format(len(all_sentences)))

dev_boundary = 55508
train = all_sentences[:dev_boundary]
dev = all_sentences[dev_boundary:]

with open(outfile_train, 'w') as ofile_train:
	for sentence in train:
		ofile_train.write(sentence+'\n')

with open(outfile_dev, 'w') as ofile_dev:
	for sentence in dev:
		ofile_dev.write(sentence+'\n')