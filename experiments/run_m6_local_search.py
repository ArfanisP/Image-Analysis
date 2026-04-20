import matplotlib.pyplot as plt
from skimage import data
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.puzzle_gen import make_puzzle
from src.solver_edge_greedy_rot import greedy_edge_solver
from src.solver_greedy_rot import render_solution
from src.objective_edge import precompute_edges
from src.solver_local_search import local_search
from src.eval import placement_accuracy, neighbor_accuracy, rotation_accuracy

def main():
    img = data.astronaut()
    img_c, tiles_obs, P, Q, gt_pos, perm, gt_rot = make_puzzle(
        img, tile_size=64, seed=42, allow_rotation=True
    )

    # αρχική λύση από greedy ακμών
    init = greedy_edge_solver(tiles_obs, P, Q, wb=1, lam=0.001)

    edges = precompute_edges(tiles_obs, wb=1)

    improved = local_search(
        edges,
        init.grid_piece,
        init.grid_rot,
        P, Q,
        lam=0.001,
        iters=5000,
        seed=0
    )

    pa = placement_accuracy(improved.grid_piece, perm, gt_pos, P, Q)
    na = neighbor_accuracy(improved.grid_piece, perm, gt_pos, P, Q)
    ra = rotation_accuracy(improved.grid_piece, improved.grid_rot, perm, gt_rot)

    print("AFTER LOCAL SEARCH")
    print(f"Placement: {pa:.6f}")
    print(f"Neighbor : {na:.6f}")
    print(f"Rotation : {ra:.6f}")
    print(f"Objective: {improved.score:.2f}")

    recon_img = render_solution(tiles_obs, improved.grid_piece, improved.grid_rot, P, Q)
    plt.imshow(recon_img)
    plt.title("Edge + Local Search reconstruction")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
