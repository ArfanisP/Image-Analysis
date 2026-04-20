import numpy as np
from dataclasses import dataclass

from src.features.color import side_descriptor_color
from src.compatibility import chi2_distance, similarity_from_distance

@dataclass
class GreedyResult:
    grid: np.ndarray         # (P,Q) δείκτες κομματιών
    score: float

def build_side_descs(tiles, wb=8, bins=16):
    """
    Προϋπολογισμός περιγραφέων χρώματος για κάθε κομμάτι και πλευρά.
    descs[i][side] -> διάνυσμα
    """
    sides = ["top", "right", "bottom", "left"]
    descs = []
    for t in tiles:
        d = {}
        for s in sides:
            d[s] = side_descriptor_color(t, s, wb=wb, bins=bins)
        descs.append(d)
    return descs

def side_sim(descs, i, side_i, j, side_j, lam=10.0):
    d = chi2_distance(descs[i][side_i], descs[j][side_j])
    return similarity_from_distance(d, lam=lam)

def greedy_fill_grid(tiles, P, Q, wb=8, bins=16, lam=10.0, seed_piece=0):
    """
    Πολύ απλό greedy:
    - τοποθετεί το seed_piece στο (0,0)
    - γεμίζει την πρώτη σειρά με τα καλύτερα ταιριάσματα δεξιάς-αριστερής πλευράς
    - γεμίζει τις επόμενες σειρές με τα καλύτερα ταιριάσματα πάνω-κάτω πλευράς ως προς το πάνω κομμάτι
    Αυτό είναι baseline· δεν είναι βέλτιστο αλλά δίνει λειτουργική ανακατασκευή.
    """
    N = len(tiles)
    assert N == P * Q

    descs = build_side_descs(tiles, wb=wb, bins=bins)

    grid = -np.ones((P, Q), dtype=int)
    used = set()

    grid[0, 0] = seed_piece
    used.add(seed_piece)

    total_score = 0.0

    # Γέμισμα πρώτης σειράς με ταιριάσματα δεξιάς-αριστερής πλευράς
    for c in range(1, Q):
        left_piece = grid[0, c-1]
        best_j, best_sim = None, -1.0
        for j in range(N):
            if j in used:
                continue
            sim = side_sim(descs, left_piece, "right", j, "left", lam=lam)
            if sim > best_sim:
                best_sim = sim
                best_j = j
        grid[0, c] = best_j
        used.add(best_j)
        total_score += best_sim

    # Γέμισμα υπόλοιπων σειρών
    for r in range(1, P):
        for c in range(Q):
            above_piece = grid[r-1, c]
            best_j, best_sim = None, -1.0
            for j in range(N):
                if j in used:
                    continue
                sim = side_sim(descs, above_piece, "bottom", j, "top", lam=lam)
                # προαιρετικά εφαρμόζουμε και περιορισμό αριστερού γείτονα αν c>0
                if c > 0:
                    left_piece = grid[r, c-1]
                    sim2 = side_sim(descs, left_piece, "right", j, "left", lam=lam)
                    sim = 0.5 * (sim + sim2)
                if sim > best_sim:
                    best_sim = sim
                    best_j = j
            grid[r, c] = best_j
            used.add(best_j)
            total_score += best_sim

    return GreedyResult(grid=grid, score=total_score)
