import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.puzzle_gen import make_puzzle
from src.features.color import side_descriptor_color
from src.compatibility import chi2_distance, similarity_from_distance
from src.geometry import ROTATIONS, opposite_side

def best_matches_for_piece(tiles_obs, idx_i: int, side_i: str, wb: int = 8, bins: int = 16, topk: int = 5):
    """
    Για ένα σταθερό observed κομμάτι i και μία σταθερή πλευρά (π.χ. 'right'),
    βρίσκει το καλύτερα ταιριαστό κομμάτι j στην απέναντι πλευρά (π.χ. το 'left' του j),
    επιτρέποντας περιστροφή του j.
    Επιστρέφει λίστα από (sim, j, angle).
    """
    di = side_descriptor_color(tiles_obs[idx_i], side_i, wb, bins)

    target_side_j = opposite_side(side_i)

    scores = []
    for j in range(len(tiles_obs)):
        if j == idx_i:
            continue
        best_sim = -1.0
        best_angle = 0

        # δοκιμή όλων των περιστροφών (brute force) για το υποψήφιο j
        for ang in ROTATIONS:
            # Αντί να περιστρέφουμε πραγματικά τα pixel του πλακιδίου εδώ, μπορούμε απλά
            # να συγκρίνουμε τον σωστό side descriptor υπό περιστροφή με αντιστοίχιση πλευρών.
            # Για απλότητα τώρα, υπολογίζουμε descriptor απευθείας στην observed πλευρά:
            # (Το observed πλακίδιο είναι ήδη περιστραμμένο κατά τη δημιουργία του παζλ.)
            dj = side_descriptor_color(tiles_obs[j], target_side_j, wb, bins)
            d = chi2_distance(di, dj)
            sim = similarity_from_distance(d, lam=10.0)
            if sim > best_sim:
                best_sim = sim
                best_angle = ang

        scores.append((best_sim, j, best_angle))

    scores.sort(reverse=True, key=lambda x: x[0])
    return scores[:topk]

def show_tile(tile, title):
    plt.figure()
    plt.title(title)
    plt.imshow(tile)
    plt.axis("off")

def main():
    img = data.astronaut()
    img_c, tiles_obs, P, Q, gt_pos, perm, gt_rot = make_puzzle(img, tile_size=64, seed=42, allow_rotation=True)

    idx_i = 0
    side_i = "right"

    top = best_matches_for_piece(tiles_obs, idx_i, side_i, wb=8, bins=16, topk=5)

    show_tile(tiles_obs[idx_i], f"Query piece i={idx_i} side={side_i}")
    for rank, (sim, j, ang) in enumerate(top, start=1):
        show_tile(tiles_obs[j], f"Rank {rank}: j={j} sim={sim:.4f}")

    plt.show()

if __name__ == "__main__":
    main()
