def transform(ifile, ofile):
    """
    Transform original CoNLL2003 format to BIO format for the named entity column (last column) only
    :param ifile: input file name (a original CoNLL2003 data file)
    :param ofile: output file name
    """
    with open(ifile, 'r') as reader, open(ofile, 'w') as writer:
        prev = 'O'
        line_idx = 1
        for line in reader:
            line = line.strip()
            if len(line) == 0:
                line_idx = 1
                prev = 'O'
                writer.write('\n')
                continue

            tokens = line.split()
            # print tokens
            label = tokens[-1]
            if label != 'O' and label != prev:
                if prev == 'O':
                    label = 'B-' + label[2:]
                elif label[2:] != prev[2:]:
                    label = 'B-' + label[2:]
                else:
                    label = label
            tokens.insert(0, str(line_idx))
            writer.write(" ".join(tokens[:-1]) + " " + label)
            writer.write('\n')
            prev = tokens[-1]
            line_idx += 1


transform("../data/conll2003/eng.train", "../data/conll2003/eng.train.bio.conll")
transform("../data/conll2003/eng.testa", "../data/conll2003/eng.dev.bio.conll")
transform("../data/conll2003/eng.testb", "../data/conll2003/eng.test.bio.conll")