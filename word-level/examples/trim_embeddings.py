import os
import sys
homedir = os.path.expanduser('~')

domain_names = ['music', 'global']
file_paths = [os.path.join(homedir, 'data/alexa', domain + '.' + set_name) for domain in domain_names
              for set_name in ['train', 'dev', 'test']]

vocabs = set()
for file_path in file_paths:
    with open(file_path, 'r') as ifile:
        for line in ifile:
            if line.strip():
                vocabs.add(line.strip().split()[1].lower())

embedding_file = sys.argv[1]
embedding_file_out = sys.argv[2]

with open(embedding_file, 'r') as ifile, open(embedding_file_out, 'w') as ofile:
    for line in ifile:
        if line.strip():
            if line.strip().split()[0] in vocabs:
                ofile.write(line)
