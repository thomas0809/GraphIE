import os

with open('data/conll2003/eng.train.bio.conll', 'r') as ifile, \
	 open('data/conll2003/eng.train.bio.conll.simple', 'w') as ofile:
	 for line in ifile:
	 	if line.strip():
	 		line = line.strip().split()
	 		ofile.write(' '.join(line[0:2]+line[-1:])+'\n')
	 	else:
	 		ofile.write(line)

with open('data/conll2003/eng.dev.bio.conll', 'r') as ifile, \
	 open('data/conll2003/eng.dev.bio.conll.simple', 'w') as ofile:
	 for line in ifile:
	 	if line.strip():
	 		line = line.strip().split()
	 		ofile.write(' '.join(line[0:2]+line[-1:])+'\n')
	 	else:
	 		ofile.write(line)

with open('data/conll2003/eng.test.bio.conll', 'r') as ifile, \
	 open('data/conll2003/eng.test.bio.conll.simple', 'w') as ofile:
	 for line in ifile:
	 	if line.strip():
	 		line = line.strip().split()
	 		ofile.write(' '.join(line[0:2]+line[-1:])+'\n')
	 	else:
	 		ofile.write(line)
