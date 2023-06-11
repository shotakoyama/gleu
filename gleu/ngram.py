import dataclasses
from .accum import SentAccumulator


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
            SentAccumulator.from_count(*self[key])
            for key
            in set(self.hyp_dict)]
        return sum(lst, start = SentAccumulator())

