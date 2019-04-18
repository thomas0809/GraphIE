from collections import defaultdict, Counter
import numpy as np

def majority_rule(words, predictions, lengths):
    '''
    This function takes in the prediction tags. And identify the
    inconsistency of the 'tags' for the same 'word'. Use the
    majority rule to decide a consistent tag.
    :param words: a list of sents of words
    :param predictions: a list of sents of tags
    :param lengths: a list of sent_lens
    :return: a modified list of sents of tags
    '''

    batch_size, _ = words.shape
    assert isinstance(predictions, np.ndarray)
    processed_pred = np.copy(predictions)

    # get word2taglist
    word2taglist = defaultdict(list)
    for i in range(batch_size):
        for j in range(lengths[i] - 1):  # -1 is for "</s>"
            pred = predictions[i, j]
            w = words[i, j]

            word2taglist[w] += [((i, j), pred)]

    # use majority to replace all
    king_tag = {}
    for w, taglist in word2taglist.items():
        tags = [inst[-1] for inst in taglist]
        tags_cnt = Counter(tags)
        king_tag[w] = tags_cnt.most_common(1)[0][0]

    # apply majority rule
    for i in range(batch_size):
        for j in range(lengths[i] - 1):  # -1 is for "</s>"
            pred = processed_pred[i, j]
            w = words[i, j]
            processed_pred[i, j] = king_tag[w]

    return processed_pred



