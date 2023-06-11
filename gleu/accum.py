import dataclasses
from functools import cache


@dataclasses.dataclass
class SentAccumulator:
    match: int = 0
    penal: int = 0
    denom: int = 0

    def __add__(self, other):
        match = self.match + other.match
        penal = self.penal + other.penal
        denom = self.denom + other.denom
        return type(self)(match, penal, denom)

    def to_p(self):
        if self.denom == 0:
            p = 1.0
        else:
            numer = max(0, self.match - self.penal)
            p = numer / self.denom
        return p

    def to_data_accum(self):
        numer = max(0, self.match - self.penal)
        return DataAccumulator(numer, self.denom)

    @classmethod
    @cache
    def from_count(cls, s, r, h):
        sdiff = (0 if r > 0 else s)
        match = min(r, h)
        penal = min(sdiff, h)
        return cls(match, penal, h)


@dataclasses.dataclass
class DataAccumulator:
    numer: int = 0
    denom: int = 0

    def __add__(self, other):
        numer = self.numer + other.numer
        denom = self.denom + other.denom
        return type(self)(numer, denom)

    def to_p(self):
        if self.denom == 0:
            p = 1.0
        else:
            p = self.numer / self.denom
        return p

