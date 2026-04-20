import numpy as np
from dataclasses import dataclass

from src.objective_edge import ROTATIONS
from src.compatibility_edge import seam_similarity

SIDES = ["top", "right", "bottom", "left"]
OPP = {"top":"bottom", "bottom":"top", "left":"right", "right":"left"}

@dataclass(frozen=True)
class Match:
    i: int
    ai: int
    si: str
    j: int
    aj: int
    sj: str
    sim: float

def best_match_for_side(edges, i, ai, si, lam=0.001):
    """
    Βρίσκει το καλύτερο (j,aj) για ταίριασμα της πλευράς si του (i,ai) με την απέναντι πλευρά του υποψηφίου.
    Επιτρέπουμε μόνο γεωμετρικά συμβατές απέναντι πλευρές:
      right ταιριάζει με left, top με bottom, κτλ.
    """
    target_sj = OPP[si]
    best = None  # (sim, j, aj)
    for j in range(len(edges)):
        if j == i:
            continue
        for aj in ROTATIONS:
            sim = seam_similarity(edges[i][ai][si], edges[j][aj][target_sj], lam=lam)
            if best is None or sim > best[0]:
                best = (sim, j, aj)
    return best[0], best[1], best[2], target_sj

def compute_all_best(edges, lam=0.001):
    """
    Για κάθε προσανατολισμένο κομμάτι (i,ai) και κάθε πλευρά si, αποθηκεύει το καλύτερο ταίριασμα.
    key = (i,ai,si) -> (sim, j, aj, sj)
    """
    best = {}
    for i in range(len(edges)):
        for ai in ROTATIONS:
            for si in SIDES:
                sim, j, aj, sj = best_match_for_side(edges, i, ai, si, lam=lam)
                best[(i, ai, si)] = (sim, j, aj, sj)
    return best

def mutual_best_matches(best_dict):
    """
    Κρατά μόνο αμοιβαία καλύτερες ακμές.
    """
    mbm = []
    for key, val in best_dict.items():
        i, ai, si = key
        sim, j, aj, sj = val

        back = best_dict.get((j, aj, sj), None)
        if back is None:
            continue
        sim2, i2, ai2, si2 = back
        if i2 == i and ai2 == ai and si2 == si:
            mbm.append(Match(i, ai, si, j, aj, sj, sim))
    return mbm
