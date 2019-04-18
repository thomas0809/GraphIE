import sys

infile_path = sys.argv[1]

with open(infile_path, 'r') as ifile, open(infile_path+'.indexed', 'w') as ofile:
    sentence = ''
    idx = 1
    for line in ifile:
        if line.strip():
            sentence += '{} {}'.format(idx, line)
            idx += 1
        else:
            ofile.write(sentence+'\n')
            sentence = ''
            idx = 1