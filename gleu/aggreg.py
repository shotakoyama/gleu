from functools import cache
from .count import ngram_counter
from .ngram import NgramStat


def make_dhrn_accum(max_n, s_dat, rs_dat, hs_dat):
    lst = [
        make_hrn_accum(max_n, s, rs, hs)
        for s, rs, hs
        in zip(s_dat, rs_dat, hs_dat)]
    return lst


def make_hrn_accum(max_n, s, rs, hs):
    lst = [
        make_rn_accum(max_n, s, rs, h)
        for h
        in hs]
    return lst


def make_hdrn_accum(max_n, s_dat, rs_dat, h_dats):
    lst = [
        make_drn_accum(max_n, s_dat, rs_dat, h_dat)
        for h_dat
        in h_dats]
    return lst


def make_drn_accum(max_n, s_dat, rs_dat, h_dat):
    lst = [
        make_rn_accum(max_n, s, rs, h)
        for s, rs, h
        in zip(s_dat, rs_dat, h_dat)]
    return lst


def make_rn_accum(max_n, s, rs, h):
    lst = [
        make_n_accum(max_n, s, r, h)
        for r
        in rs]
    return lst


@cache
def make_n_accum(max_n, s, r, h):
    lst = [
        make_accum(n, s, r, h)
        for n
        in range(1, max_n + 1)]
    return lst


def make_accum(n, s, r, h):
    s_cnt = ngram_counter(n, s)
    r_cnt = ngram_counter(n, r)
    h_cnt = ngram_counter(n, h)
    accum = NgramStat(s_cnt, r_cnt, h_cnt).accumlate()
    return accum

