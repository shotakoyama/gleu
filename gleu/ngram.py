import dataclasses
from functools import cache
from .accum import Accumulator


@dataclasses.dataclass
class NgramStat:
    src_dict: dict
    ref_dict: dict
    hyp_dict: dict

    def __getitem__(self, key):
        s = self.src_dict.get(key, 0)
        r = self.ref_dict.get(key, 0)
        h = self.hyp_dict.get(key, 0)
        return s, r, h

    def accumlate(self):
        lst = [
            make_accum(*self[key])
            for key
            in set(self.hyp_dict)]
        return sum(lst, start = Accumulator())


@cache
def make_accum(s, r, h):
    sdiff = (0 if r > 0 else s)
    match = min(r, h)
    penal = min(sdiff, h)
    return Accumulator(match, penal, h)

