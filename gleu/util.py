import random
import numpy as np
from functools import cache


@cache
def log_brevity_penalty(rlen, hlen):
    if (rlen, hlen) == (0, 0):
        return 0.0
    elif hlen == 0:
        return -np.inf
    elif rlen < hlen:
        return 0.0
    else:
        return 1.0 - rlen / hlen


def argmax(xs):
    return max(enumerate(xs), key = lambda x: x[1])[0]


def make_id_rindex(num_iter, d, r, fix = False):
    if fix:
        # comatible with python 2's randint() in original GLEU+ implementation
        id_rindex = []
        for i in range(num_iter):
            random.seed(i * 101, version = 1)
            id_rindex.append([
                int(random.random() * r)
                for _ in range(d)])
        id_rindex = np.array(id_rindex, dtype = int)
    else:
        id_rindex = np.random.randint(r, size = (num_iter, d))
    return id_rindex

