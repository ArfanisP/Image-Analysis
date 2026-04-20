import matplotlib.pyplot as plt
from skimage import data
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.puzzle_gen import make_puzzle
from src.objective_edge import precompute_edges
from src.mbm_graph import compute_all_best, mutual_best_matches
from src.solver_mbm import build_adjacency, largest_component_nodes, layout_component, embed_in_grid
from src.solver_greedy_rot import render_solution
from src.eval import placement_accuracy, neighbor_accuracy, rotation_accuracy

def main():
    img = data.astronaut()
    img_c, tiles_obs, P, Q, gt_pos, perm, gt_rot = make_puzzle(
        img, tile_size=64, seed=42, allow_rotation=True
    )

    edges = precompute_edges(tiles_obs, wb=1)

    best = compute_all_best(edges, lam=0.001)
    matches = mutual_best_matches(best)

    print(f"Mutual-best edges: {len(matches)}")

    adj = build_adjacency(matches)
    comp = largest_component_nodes(adj)
    if not comp:
        print("No component found; try different lam or wb.")
        return

    root = comp[0]
    free_place, conflicts = layout_component(adj, root)
    print(f"Component size: {len(free_place)} | conflicts: {conflicts}")

    # fallback γωνία = 0 για κομμάτια που δεν ανήκουν στη συνιστώσα (απλό)
    fallback = {}

    grid_piece, grid_rot = embed_in_grid(len(tiles_obs), P, Q, free_place, fallback)

    pa = placement_accuracy(grid_piece, perm, gt_pos, P, Q)
    na = neighbor_accuracy(grid_piece, perm, gt_pos, P, Q)
    ra = rotation_accuracy(grid_piece, grid_rot, perm, gt_rot)

    print("MBM BASELINE")
    print(f"Placement: {pa:.6f}")
    print(f"Neighbor : {na:.6f}")
    print(f"Rotation : {ra:.6f}")

    recon_img = render_solution(tiles_obs, grid_piece, grid_rot, P, Q)
    plt.imshow(recon_img)
    plt.title("MBM component-based reconstruction")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
