import numpy as np
from .util import (
        log_brevity_penalty,
        argmax)
from .accum import DataAccumulator
from .verbose import SentNVerbose


def drn_sent_accum_to_d_rmax(drn_sacc, dr_rlen, d_hlen):
    dr_snvb = drn_sent_accum_to_dr_sent_nverbose(drn_sacc, dr_rlen, d_hlen)
    d_rmax = [argmax(r_snvb) for r_snvb in dr_snvb]
    return d_rmax


def drn_sent_accum_to_dr_sent_nverbose(drn_sacc, drrlen, dhlen):
    dr_snvb = [[
        SentNVerbose(n_sacc, drrlen[d, r], dhlen[d])
        for r, n_sacc
        in enumerate(rn_sacc)]
        for d, rn_sacc
        in enumerate(drn_sacc)]
    return dr_snvb


def drn_sent_accum_to_n_data_accum(drn_sacc, d_rindex):
    dn_sacc = [
        rn_sacc[r]
        for rn_sacc, r
        in zip(drn_sacc, d_rindex)]
    n_dacc = dn_sent_accum_to_n_data_accum(dn_sacc)
    return n_dacc


def dn_sent_accum_to_n_data_accum(dn_sacc):
    nd_sacc = list(zip(*dn_sacc))
    return [d_sent_accum_to_data_accum(d_sacc) for d_sacc in nd_sacc]


def d_sent_accum_to_data_accum(d_sacc):
    return sum([sacc.to_data_accum() for sacc in d_sacc], start = DataAccumulator())


def rindex_to_rhlen(dr_rlen, d_hlen, d_rindex):
    rlen = sum([dr_rlen[d, r] for d, r in enumerate(d_rindex)])
    hlen = sum([d_hlen[d] for d, _ in enumerate(d_rindex)])
    return rlen, hlen


def rn_sent_accum_to_gleu(rn_sacc, rlens, hlen, select_max = False):
    gs = [
        n_accum_to_gleu(n_accum, rlen, hlen)
        for n_accum, rlen
        in zip(rn_sacc, rlens)]
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

