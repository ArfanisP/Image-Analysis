import os
import csv
import matplotlib.pyplot as plt
from skimage import data
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.puzzle_gen import make_puzzle, render_grid
from src.solver_greedy import greedy_fill_grid
from src.solver_greedy_rot import greedy_fill_grid_with_rotations, render_solution
from src.solver_edge_greedy_rot import greedy_edge_solver
from src.objective_edge import precompute_edges, ROTATIONS
from src.features.color import side_descriptor_color
from src.compatibility_combined import combined_similarity
from src.solver_mbm import build_adjacency, largest_component_nodes, layout_component, embed_in_grid
from src.eval import placement_accuracy, neighbor_accuracy, rotation_accuracy


SIDES = ["top", "right", "bottom", "left"]
OPP = {"top":"bottom", "bottom":"top", "left":"right", "right":"left"}


def save_image(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.imsave(path, img)


def precompute_color_descs(tiles, wb=8, bins=16):
    from src.puzzle_gen import rotate_tile
    descs = []
    for t in tiles:
        per_piece = {}
        for ang in ROTATIONS:
            tr = rotate_tile(t, ang)
            per_side = {s: side_descriptor_color(tr, s, wb=wb, bins=bins) for s in SIDES}
            per_piece[ang] = per_side
        descs.append(per_piece)
    return descs


def compute_all_best_combined(edges, cols, lam_edge=0.001, lam_col=10.0, w_edge=0.85, w_col=0.15):
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


def run():
    img = data.astronaut()
    img_c, tiles_obs, P, Q, gt_pos, perm, gt_rot = make_puzzle(
        img, tile_size=64, seed=42, allow_rotation=True
    )
    shuffled_img = render_grid(tiles_obs, P, Q)

    results = []

    # 1) Greedy μόνο με χρώμα + περιστροφές (αδύναμη)
    res1 = greedy_fill_grid_with_rotations(tiles_obs, P, Q, wb=8, bins=16, lam=10.0)
    recon1 = render_solution(tiles_obs, res1.grid_piece, res1.grid_rot, P, Q)
    results.append(("color_greedy_rot", res1.grid_piece, res1.grid_rot, recon1))

    # 2) Greedy μόνο με ακμές + περιστροφές
    res2 = greedy_edge_solver(tiles_obs, P, Q, wb=1, lam=0.001)
    recon2 = render_solution(tiles_obs, res2.grid_piece, res2.grid_rot, P, Q)
    results.append(("edge_greedy_rot", res2.grid_piece, res2.grid_rot, recon2))

    # 3) MBM - καλύτερο
    edges = precompute_edges(tiles_obs, wb=1)
    cols = precompute_color_descs(tiles_obs, wb=8, bins=16)
    best = compute_all_best_combined(edges, cols, w_edge=0.85, w_col=0.15)
    matches = mutual_best(best)

    from collections import namedtuple
    M = namedtuple("M", ["i","ai","si","j","aj","sj","sim"])
    matches_obj = [M(*m) for m in matches]

    adj = build_adjacency(matches_obj)
    comp = largest_component_nodes(adj)
    root = comp[0]
    free_place, _ = layout_component(adj, root)
    gp, gr = embed_in_grid(len(tiles_obs), P, Q, free_place, fallback_angles={})
    recon3 = render_solution(tiles_obs, gp, gr, P, Q)
    results.append(("mbm_edge_color", gp, gr, recon3))

    # Αποθήκευση και μετρικές
    os.makedirs("outputs", exist_ok=True)
    save_image("outputs/shuffled.png", shuffled_img)
    save_image("outputs/original.png", img_c)

    rows = []
    for name, gp, gr, recon in results:
        pa = placement_accuracy(gp, perm, gt_pos, P, Q)
        na = neighbor_accuracy(gp, perm, gt_pos, P, Q)
        ra = rotation_accuracy(gp, gr, perm, gt_rot)

        rows.append([name, P, Q, pa, na, ra])
        save_image(f"outputs/{name}.png", recon)

    with open("outputs/metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "P", "Q", "placement", "neighbor", "rotation"])
        w.writerows(rows)

    print("Saved outputs/*.png and outputs/metrics.csv")
    for r in rows:
        print(r)


if __name__ == "__main__":
    run()
