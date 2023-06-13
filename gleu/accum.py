import dataclasses


@dataclasses.dataclass
class Accumulator:
    match: int = 0
    penal: int = 0
    denom: int = 0

    def __add__(self, other):
        match = self.match + other.match
        penal = self.penal + other.penal
        denom = self.denom + other.denom
        return type(self)(match, penal, denom)

    def p(self):
        if self.denom == 0:
            p = 1.0
        else:
            numer = self.match - self.penal
            assert numer >= 0, self
            p = numer / self.denom
        return p

    def rectify(self):
        penal = min(self.match, self.penal)
        return Accumulator(self.match, penal, self.denom)

