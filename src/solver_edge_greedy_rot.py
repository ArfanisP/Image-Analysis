import numpy as np
from dataclasses import dataclass

from src.puzzle_gen import rotate_tile
from src.features.edge import get_border_pixels
from src.compatibility_edge import seam_similarity

ROTATIONS = [0, 90, 180, 270]

@dataclass
class EdgeGreedyResult:
    grid_piece: np.ndarray
    grid_rot: np.ndarray
    score: float

def precompute_edges(tiles, wb=1):
    descs = []
    sides = ["top", "right", "bottom", "left"]
    for tile in tiles:
        per_piece = {}
        for ang in ROTATIONS:
            t = rotate_tile(tile, ang)
            per_side = {}
            for s in sides:
                per_side[s] = get_border_pixels(t, s, wb=wb)
            per_piece[ang] = per_side
        descs.append(per_piece)
    return descs

def greedy_edge_solver(tiles, P, Q, wb=1, lam=0.001, seed_piece=0):
    N = len(tiles)
    edges = precompute_edges(tiles, wb=wb)

    grid_piece = -np.ones((P, Q), dtype=int)
    grid_rot = np.zeros((P, Q), dtype=int)
    used = set()

    grid_piece[0, 0] = seed_piece
    grid_rot[0, 0] = 0
    used.add(seed_piece)

    total_score = 0.0

    for r in range(P):
        for c in range(Q):
            if r == 0 and c == 0:
                continue

            best = None

            for j in range(N):
                if j in used:
                    continue
                for ang in ROTATIONS:
                    score = 0
                    cnt = 0

                    if c > 0:
                        left_piece = grid_piece[r, c-1]
                        left_ang = grid_rot[r, c-1]
                        s = seam_similarity(
                            edges[left_piece][left_ang]["right"],
                            edges[j][ang]["left"],
                            lam=lam
                        )
                        score += s
                        cnt += 1

                    if r > 0:
                        up_piece = grid_piece[r-1, c]
                        up_ang = grid_rot[r-1, c]
                        s = seam_similarity(
                            edges[up_piece][up_ang]["bottom"],
                            edges[j][ang]["top"],
                            lam=lam
                        )
                        score += s
                        cnt += 1

                    if cnt > 0:
                        score /= cnt

                    if (best is None) or (score > best[0]):
                        best = (score, j, ang)

            best_score, best_j, best_ang = best
            grid_piece[r, c] = best_j
            grid_rot[r, c] = best_ang
            used.add(best_j)
            total_score += best_score

    return EdgeGreedyResult(grid_piece, grid_rot, total_score)