# shuffler.py
from typing import List, Dict, Optional
import copy
import numpy as np

class Shuffler:

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def anonymize_and_shuffle(self, msgs: List[dict]) -> List[dict]:

        stripped: List[dict] = []
        for m in msgs:
            mm = copy.deepcopy(m)
        
            for k in ("client_id", "sender", "ip", "uid"):
                if k in mm:
                    mm.pop(k)
      
            stripped.append(mm)
        self.rng.shuffle(stripped)
        return stripped
