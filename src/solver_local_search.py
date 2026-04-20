import numpy as np
from dataclasses import dataclass

from src.objective_edge import total_objective, ROTATIONS

@dataclass
class LocalSearchResult:
    grid_piece: np.ndarray
    grid_rot: np.ndarray
    score: float

def local_search(edges, grid_piece, grid_rot, P, Q, lam=0.001, iters=2000, seed=0):
    """
    Τοπική βελτιστοποίηση (hill-climbing):
    - τυχαία ανταλλαγή δύο κελιών
    - τυχαία περιστροφή ενός κελιού
    Κρατάμε την κίνηση αν βελτιώνει τον αντικειμενικό σκοπό.
    """
    rng = np.random.default_rng(seed)

    gp = grid_piece.copy()
    gr = grid_rot.copy()

    best_score = total_objective(edges, gp, gr, P, Q, lam=lam)

    for _ in range(iters):
        move_type = rng.integers(0, 2)  # 0 ανταλλαγή, 1 περιστροφή

        if move_type == 0:
            # ανταλλαγή δύο θέσεων
            r1 = int(rng.integers(0, P)); c1 = int(rng.integers(0, Q))
            r2 = int(rng.integers(0, P)); c2 = int(rng.integers(0, Q))
            if r1 == r2 and c1 == c2:
                continue

            gp[r1, c1], gp[r2, c2] = gp[r2, c2], gp[r1, c1]
            gr[r1, c1], gr[r2, c2] = gr[r2, c2], gr[r1, c1]

            new_score = total_objective(edges, gp, gr, P, Q, lam=lam)
            if new_score >= best_score:
                best_score = new_score
            else:
                # επαναφορά
                gp[r1, c1], gp[r2, c2] = gp[r2, c2], gp[r1, c1]
                gr[r1, c1], gr[r2, c2] = gr[r2, c2], gr[r1, c1]

        else:
            # περιστροφή μίας θέσης
            r = int(rng.integers(0, P)); c = int(rng.integers(0, Q))
            old = int(gr[r, c])
            new = int(rng.choice([a for a in ROTATIONS if a != old]))
            gr[r, c] = new

            new_score = total_objective(edges, gp, gr, P, Q, lam=lam)
            if new_score >= best_score:
                best_score = new_score
            else:
                gr[r, c] = old

    return LocalSearchResult(gp, gr, best_score)
