__author__ = 'jindi'


class Sentence(object):
    def __init__(self, words, word_ids, char_seqs, char_id_seqs):
        self.words = words
        self.word_ids = word_ids
        self.char_seqs = char_seqs
        self.char_id_seqs = char_id_seqs

    def length(self):
        return len(self.words)


class DependencyInstance(object):
    def __init__(self, sentence, postags, pos_ids, heads, types, type_ids):
        self.sentence = sentence
        self.postags = postags
        self.pos_ids = pos_ids
        self.heads = heads
        self.types = types
        self.type_ids = type_ids

    def length(self):
        return self.sentence.length()


class NERInstance(object):
    def __init__(self, sentence, pos_tags, pos_ids, chunk_tags, chunk_ids, ner_tags, ner_ids):
        self.sentence = sentence
        self.pos_tags = pos_tags
        self.pos_ids = pos_ids
        self.chunk_tags = chunk_tags
        self.chunk_ids = chunk_ids
        self.ner_tags = ner_tags
        self.ner_ids = ner_ids

    def length(self):
        return self.sentence.length()


class GraphInstance(object):
    def __init__(self, word_ids, chars, feat_ids, posi, adjs,
                 ner_ids, n_sent, sent_len, word_len,
                 doc_n_words, n_node, words_en):

        self.word_ids = word_ids
        self.chars = chars
        self.feat_ids = feat_ids
        self.posi = posi
        self.adjs = adjs
        self.ner_ids = ner_ids

        self.n_sent = n_sent
        self.sent_len = sent_len
        self.word_len = word_len
        self.doc_n_words = doc_n_words
        self.n_node = n_node

        self.words_en = words_en

    def length(self):
        return (self.n_sent, self.sent_len, self.doc_n_words)
