import numpy as np
from collections import Counter
from functools import cache


split_ngram = None


def set_tokenization(tokenization):
    global split_ngram
    if tokenization == 'char':
        split_ngram = split_char_ngram
    elif tokenization == 'word':
        split_ngram = split_word_ngram
    else:
        assert False


@cache
def ngram_counter(n, sent):
    ngram_list = split_ngram(n, sent)
    return dict(Counter(ngram_list))


def make_dx_xlen(xs_dat):
    lst = [[
        len(split_ngram(1, x))
        for x in xs]
        for xs in xs_dat]
    return np.array(lst, dtype = int)


@cache
def split_char_ngram(n, sent):
    return [sent[i : i + n] for i in range(len(sent) - n + 1)]


@cache
def split_word_ngram(n, sent):
    sent = sent.split()
    return [' '.join(sent[i : i + n]) for i in range(len(sent) - n + 1)]

