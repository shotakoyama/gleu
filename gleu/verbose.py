import numpy as np
from .util import log_brevity_penalty
from .result import round_half_up


class NVerbose:

    def __init__(self, n_accum, rlen, hlen):
        self.header = [
            '', 'match', 'penal',
            'numer', 'denom',
            'p', 'bp', 'gleu']
        self.n = len(n_accum)

        self.matchs = [accum.match for accum in n_accum]
        self.penals = [accum.penal for accum in n_accum]

        n_accum = [accum.rectify() for accum in n_accum]
        self.numers = [accum.match - accum.penal for accum in n_accum]
        self.denoms = [accum.denom for accum in n_accum]
        self.ps = [accum.p() for accum in n_accum]

        with np.errstate(divide = 'ignore'):
            self.logps = np.log(self.ps)
        self.logbp = log_brevity_penalty(rlen, hlen)
        self.bp = np.exp(self.logbp)
        self.gleus = np.exp(self.logbp + self.logps)
        self.loggmeanp = self.logps.mean()
        self.p = np.exp(self.loggmeanp)
        self.gleu = np.exp(self.logbp + self.loggmeanp)

    def iter_row(self, digit):
        for n in range(self.n):
            lst = [
                n + 1,
                self.matchs[n],
                self.penals[n],
                self.numers[n],
                self.denoms[n],
                round_half_up(100 * self.ps[n], digit),
                round_half_up(100 * self.bp, digit),
                round_half_up(100 * self.gleus[n], digit)]
            yield lst

    def total_row(self, digit):
        lst = [
            'total',
            sum(self.matchs),
            sum(self.penals),
            sum(self.numers),
            sum(self.denoms),
            round_half_up(100 * self.p, digit),
            round_half_up(100 * self.bp, digit),
            round_half_up(100 * self.gleu, digit)]
        return lst

    def __lt__(self, other):
        return (self.gleu, list(self.gleus)[::-1]) < (other.gleu, list(other.gleus)[::-1])

