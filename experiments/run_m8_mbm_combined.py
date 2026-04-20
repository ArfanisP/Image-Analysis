import matplotlib.pyplot as plt
from skimage import data
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.puzzle_gen import make_puzzle
from src.objective_edge import precompute_edges, ROTATIONS
from src.features.color import side_descriptor_color
from src.compatibility_combined import combined_similarity
from src.solver_mbm import build_adjacency, largest_component_nodes, layout_component, embed_in_grid
from src.solver_greedy_rot import render_solution
from src.eval import placement_accuracy, neighbor_accuracy, rotation_accuracy

SIDES = ["top", "right", "bottom", "left"]
OPP = {"top":"bottom", "bottom":"top", "left":"right", "right":"left"}

def precompute_color_descs(tiles, wb=8, bins=16):
    descs = []
    for t in tiles:
        per_piece = {}
        for ang in ROTATIONS:
            # μπορούμε να επαναχρησιμοποιήσουμε το rotate_tile από το puzzle_gen αν χρειάζεται, αλλά για απλότητα:
            # ο υπολογισμός ιστογράμματος επιτόπου αργότερα είναι οκ· παρ' όλα αυτά, κάνουμε προσωρινή αποθήκευση ανά γωνία:
            # έτσι κι αλλιώς θα περιστρέψουμε με render_solution αργότερα· πιο εύκολο είναι να υπολογιστεί τώρα με rotate_tile:
            from src.puzzle_gen import rotate_tile
            tr = rotate_tile(t, ang)
            per_side = {s: side_descriptor_color(tr, s, wb=wb, bins=bins) for s in SIDES}
            per_piece[ang] = per_side
        descs.append(per_piece)
    return descs

def compute_all_best_combined(edges, cols, lam_edge=0.001, lam_col=10.0, w_edge=0.8, w_col=0.2):
    best = {}
    for i in range(len(edges)):
        for ai in ROTATIONS:
            for si in SIDES:
                sj = OPP[si]
                best_val = None
                for j in range(len(edges)):
                    if j == i:
                        continue
                    for aj in ROTATIONS:
                        sim = combined_similarity(
                            edges[i][ai][si], edges[j][aj][sj],
                            cols[i][ai][si], cols[j][aj][sj],
                            lam_edge=lam_edge, lam_col=lam_col, w_edge=w_edge, w_col=w_col
                        )
                        if best_val is None or sim > best_val[0]:
                            best_val = (sim, j, aj, sj)
                best[(i, ai, si)] = best_val
    return best

def mutual_best(best_dict):
    matches = []
    for (i, ai, si), (sim, j, aj, sj) in best_dict.items():
        back = best_dict.get((j, aj, sj), None)
        if back is None:
            continue
        sim2, i2, ai2, si2 = back
        if i2 == i and ai2 == ai and si2 == si:
            matches.append((i, ai, si, j, aj, sj, sim))
    return matches

def main():
    img = data.astronaut()
    img_c, tiles_obs, P, Q, gt_pos, perm, gt_rot = make_puzzle(
        img, tile_size=64, seed=42, allow_rotation=True
    )

    edges = precompute_edges(tiles_obs, wb=1)
    cols = precompute_color_descs(tiles_obs, wb=8, bins=16)

    best = compute_all_best_combined(edges, cols, w_edge=0.85, w_col=0.15)
    matches = mutual_best(best)

    print(f"Mutual-best edges (combined): {len(matches)}")

    # μετατροπή σε αντικείμενα συμβατά με τη γειτνίαση του solver_mbm
    from collections import namedtuple
    M = namedtuple("M", ["i","ai","si","j","aj","sj","sim"])
    matches_obj = [M(*m) for m in matches]

    adj = build_adjacency(matches_obj)
    comp = largest_component_nodes(adj)
    root = comp[0]
    free_place, conflicts = layout_component(adj, root)

    grid_piece, grid_rot = embed_in_grid(len(tiles_obs), P, Q, free_place, fallback_angles={})

    pa = placement_accuracy(grid_piece, perm, gt_pos, P, Q)
    na = neighbor_accuracy(grid_piece, perm, gt_pos, P, Q)
    ra = rotation_accuracy(grid_piece, grid_rot, perm, gt_rot)

    print("MBM COMBINED")
    print(f"Placement: {pa:.6f}")
    print(f"Neighbor : {na:.6f}")
    print(f"Rotation : {ra:.6f}")

    recon_img = render_solution(tiles_obs, grid_piece, grid_rot, P, Q)
    plt.imshow(recon_img)
    plt.title("MBM (edge + color) reconstruction")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
