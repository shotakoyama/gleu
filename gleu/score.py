import numpy as np
from .util import (
        log_brevity_penalty,
        argmax)
from .accum import Accumulator
from .verbose import NVerbose


def drn_accum_to_d_rmax(drn_accum, dr_rlen, d_hlen):
    dr_nvb = drn_accum_to_dr_nverbose(drn_accum, dr_rlen, d_hlen)
    d_rmax = [argmax(r_nvb) for r_nvb in dr_nvb]
    return d_rmax


def drn_accum_to_dr_nverbose(drn_accum, dr_rlen, d_hlen):
    dr_nvb = [[
        NVerbose(n_accum, dr_rlen[d, r], d_hlen[d])
        for r, n_accum
        in enumerate(rn_accum)]
        for d, rn_accum
        in enumerate(drn_accum)]
    return dr_nvb


def drn_accum_to_n_accum(drn_accum, d_rindex):
    dn_accum = [
        rn_accum[r]
        for rn_accum, r
        in zip(drn_accum, d_rindex)]
    n_accum = dn_accum_to_n_accum(dn_accum)
    return n_accum


def dn_accum_to_n_accum(dn_accum):
    nd_accum = list(zip(*dn_accum))
    return [d_accum_to_accum(d_accum) for d_accum in nd_accum]


def d_accum_to_accum(d_accum):
    return sum([accum.rectify() for accum in d_accum], start = Accumulator())


def rindex_to_rhlen(dr_rlen, d_hlen, d_rindex):
    rlen = sum([dr_rlen[d, r] for d, r in enumerate(d_rindex)])
    hlen = sum([d_hlen[d] for d, _ in enumerate(d_rindex)])
    return rlen, hlen


def rn_accum_to_gleu(rn_accum, rlens, hlen, select_max = False):
    gs = [
        n_accum_to_gleu(n_accum, rlen, hlen)
        for n_accum, rlen
        in zip(rn_accum, rlens)]

    if select_max:
        return max(gs)
    else:
        return np.mean(gs)


def n_accum_to_gleu(n_accum, rlen, hlen):
    logps = n_accum_to_logps(n_accum)
    loggmeanp = logps.mean()
    logbp = log_brevity_penalty(rlen, hlen)
    gleu = np.exp(logbp + loggmeanp)
    return gleu


def n_accum_to_logps(n_accum):
    ps = [accum.to_p() for accum in n_accum]
    with np.errstate(divide = 'ignore'):
        logps = np.log(ps)
    return logps

