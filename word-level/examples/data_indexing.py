import sys
import os

infile_path = sys.argv[1]

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

with open(infile_path, 'w') as ofile:
	for sentence in all_sentences:
		ofile.write(sentence+'\n')