import numpy as np
from .util import log_brevity_penalty
from .result import round_half_up


class NVerbose:

    def set_gleu(self, rlen, hlen):
        with np.errstate(divide = 'ignore'):
            self.logps = np.log(self.ps)
        self.logbp = log_brevity_penalty(rlen, hlen)
        self.bp = np.exp(self.logbp)
        self.gleus = np.exp(self.logbp + self.logps)
        self.loggmeanp = self.logps.mean()
        self.p = np.exp(self.loggmeanp)
        self.gleu = np.exp(self.logbp + self.loggmeanp)

    def __lt__(self, other):
        return (self.gleu, list(self.gleus)[::-1]) < (other.gleu, list(other.gleus)[::-1])


class CorpusNVerbose(NVerbose):

    def __init__(self, n_dacc, rlen, hlen):
        self.header = ['', 'numer', 'denom', 'p', 'bp', 'gleu']
        self.n = len(n_dacc)
        self.numers = [dacc.numer for dacc in n_dacc]
        self.denoms = [dacc.denom for dacc in n_dacc]
        self.ps = [dacc.to_p() for dacc in n_dacc]
        self.set_gleu(rlen, hlen)

    def iter_row(self, digit):
        for n in range(self.n):
            lst = [
                n + 1,
                self.numers[n],
                self.denoms[n],
                round_half_up(100 * self.ps[n], digit),
                round_half_up(100 * self.bp, digit),
                round_half_up(100 * self.gleus[n], digit)]
            yield lst

    def total_row(self, digit):
        lst = [
            'total',
            sum(self.numers),
            sum(self.denoms),
            round_half_up(100 * self.p, digit),
            round_half_up(100 * self.bp, digit),
            round_half_up(100 * self.gleu, digit)]
        return lst


class SentNVerbose(NVerbose):

    def __init__(self, n_sacc, rlen, hlen):
        self.header = ['', 'match', 'penal', 'numer', 'denom', 'p', 'bp', 'gleu']
        self.n = len(n_sacc)
        self.matchs = [sacc.match for sacc in n_sacc]
        self.penals = [sacc.penal for sacc in n_sacc]
        self.numers = [max(0, sacc.match - sacc.penal) for sacc in n_sacc]
        self.denoms = [sacc.denom for sacc in n_sacc]
        self.ps = [sacc.to_p() for sacc in n_sacc]
        self.set_gleu(rlen, hlen)

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

