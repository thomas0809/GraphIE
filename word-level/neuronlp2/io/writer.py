__author__ = 'max'
from .Constants import PAD_ID_WORD, DEFAULT_VALUE


class CoNLL03Writer(object):
    def __init__(self, word_alphabet, char_alphabet, ner_alphabet):
        self.__source_file = None
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__ner_alphabet = ner_alphabet

    def start(self, file_path, mode='w'):
        self.__source_file = open(file_path, mode)

    def close(self):
        self.__source_file.close()

    def write(self, word, predictions, targets, lengths):
        batch_size, _ = word.shape
        for i in range(batch_size):
            for j in range(lengths[i]):
                w = self.__word_alphabet.get_instance(word[i, j])
                tgt = self.__ner_alphabet.get_instance(targets[i, j])
                pred = self.__ner_alphabet.get_instance(predictions[i, j])
                self.__source_file.write('%d %s %s %s\n' % (j + 1, w, tgt, pred))
            self.__source_file.write('\n')

