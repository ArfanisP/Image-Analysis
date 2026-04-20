import numpy as np
from src.compatibility_edge import seam_similarity
from src.features.edge import get_border_pixels
from src.puzzle_gen import rotate_tile

SIDES = ["top", "right", "bottom", "left"]
ROTATIONS = [0, 90, 180, 270]

def precompute_edges(tiles, wb=1):
    """
    edges[piece][angle][side] -> pixel περιγράμματος (float32)
    """
    edges = []
    for tile in tiles:
        per_piece = {}
        for ang in ROTATIONS:
            t = rotate_tile(tile, ang)
            per_side = {s: get_border_pixels(t, s, wb=wb) for s in SIDES}
            per_piece[ang] = per_side
        edges.append(per_piece)
    return edges

def total_objective(edges, grid_piece, grid_rot, P, Q, lam=0.001):
    """
    Άθροισμα ομοιοτήτων ραφής για όλες τις δεξιές και κάτω γειτνιάσεις.
    """
    score = 0.0
    for r in range(P):
        for c in range(Q):
            i = int(grid_piece[r, c])
            ai = int(grid_rot[r, c])
            if c + 1 < Q:
                j = int(grid_piece[r, c+1])
                aj = int(grid_rot[r, c+1])
                score += seam_similarity(edges[i][ai]["right"], edges[j][aj]["left"], lam=lam)
            if r + 1 < P:
                j = int(grid_piece[r+1, c])
                aj = int(grid_rot[r+1, c])
                score += seam_similarity(edges[i][ai]["bottom"], edges[j][aj]["top"], lam=lam)
    return float(score)
