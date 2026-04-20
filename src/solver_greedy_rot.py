import numpy as np
from dataclasses import dataclass

from src.puzzle_gen import rotate_tile
from src.features.color import side_descriptor_color
from src.compatibility import chi2_distance, similarity_from_distance

ROTATIONS = [0, 90, 180, 270]
SIDES = ["top", "right", "bottom", "left"]

@dataclass
class GreedyRotResult:
    grid_piece: np.ndarray   # (P,Q) δείκτες κομματιών
    grid_rot: np.ndarray     # (P,Q) γωνίες περιστροφής (δεξιόστροφα)
    score: float

def precompute_descs_all_angles(tiles, wb=8, bins=16):
    """
    descs[i][angle][side] -> διάνυσμα περιγραφέα
    Περιστρέφουμε πραγματικά τα pixel για απλότητα (πιο αργό αλλά σωστό και απλό).
    """
    descs = []
    for tile in tiles:
        per_piece = {}
        for ang in ROTATIONS:
            t_rot = rotate_tile(tile, ang)
            per_side = {}
            for s in SIDES:
                per_side[s] = side_descriptor_color(t_rot, s, wb=wb, bins=bins)
            per_piece[ang] = per_side
        descs.append(per_piece)
    return descs

def sim_side(descs, i, ang_i, side_i, j, ang_j, side_j, lam=10.0):
    d = chi2_distance(descs[i][ang_i][side_i], descs[j][ang_j][side_j])
    return similarity_from_distance(d, lam=lam)

def greedy_fill_grid_with_rotations(tiles, P, Q, wb=8, bins=16, lam=10.0, seed_piece=0, seed_angle=0):
    """
    Greedy γέμισμα:
    - τοποθετεί seed στο (0,0) με seed_angle
    - γεμίζει γραμμή-γραμμή
    - σε κάθε κελί επιλέγει το αχρησιμοποίητο (piece, angle) που ταιριάζει καλύτερα με ήδη τοποθετημένους γείτονες:
      πάνω (κάτω-πάνω πλευρές) και αριστερά (δεξιά-αριστερή πλευρά).
    """
    N = len(tiles)
    assert N == P * Q

    descs = precompute_descs_all_angles(tiles, wb=wb, bins=bins)

    grid_piece = -np.ones((P, Q), dtype=int)
    grid_rot = np.zeros((P, Q), dtype=int)
    used = set()

    grid_piece[0, 0] = seed_piece
    grid_rot[0, 0] = seed_angle
    used.add(seed_piece)

    total_score = 0.0

    for r in range(P):
        for c in range(Q):
            if r == 0 and c == 0:
                continue

            best = None  # (score, j, ang_j)

            for j in range(N):
                if j in used:
                    continue
                for ang_j in ROTATIONS:
                    score = 0.0
                    cnt = 0

                    # περιορισμός με τον αριστερό γείτονα
                    if c > 0:
                        left_piece = grid_piece[r, c-1]
                        left_ang = grid_rot[r, c-1]
                        s = sim_side(descs, left_piece, left_ang, "right", j, ang_j, "left", lam=lam)
                        score += s
                        cnt += 1

                    # περιορισμός με τον πάνω γείτονα
                    if r > 0:
                        up_piece = grid_piece[r-1, c]
                        up_ang = grid_rot[r-1, c]
                        s = sim_side(descs, up_piece, up_ang, "bottom", j, ang_j, "top", lam=lam)
                        score += s
                        cnt += 1

                    if cnt > 0:
                        score /= cnt

                    if (best is None) or (score > best[0]):
                        best = (score, j, ang_j)

            best_score, best_j, best_ang = best
            grid_piece[r, c] = best_j
            grid_rot[r, c] = best_ang
            used.add(best_j)
            total_score += best_score

    return GreedyRotResult(grid_piece=grid_piece, grid_rot=grid_rot, score=total_score)

def render_solution(tiles, grid_piece, grid_rot, P, Q):
    """
    Δημιουργεί την ανακατασκευασμένη εικόνα περιστρέφοντας κάθε τοποθετημένο πλακίδιο με τη γωνία που επιλέχθηκε.
    """
    tile_h, tile_w = tiles[0].shape[:2]
    canvas = np.zeros((P * tile_h, Q * tile_w, 3), dtype=tiles[0].dtype)

    for r in range(P):
        for c in range(Q):
            idx = int(grid_piece[r, c])
            ang = int(grid_rot[r, c])
            t = rotate_tile(tiles[idx], ang)
            canvas[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w] = t

    return canvas
