import matplotlib.pyplot as plt
from skimage import data
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.puzzle_gen import make_puzzle, render_grid
from src.solver_edge_greedy_rot import greedy_edge_solver
from src.solver_greedy_rot import render_solution
from src.eval import placement_accuracy, neighbor_accuracy, rotation_accuracy

def main():
    img = data.astronaut()

    img_c, tiles_obs, P, Q, gt_pos, perm, gt_rot = make_puzzle(
        img, tile_size=64, seed=42, allow_rotation=True
    )

    res = greedy_edge_solver(tiles_obs, P, Q, wb=1, lam=0.001)

    pa = placement_accuracy(res.grid_piece, perm, gt_pos, P, Q)
    na = neighbor_accuracy(res.grid_piece, perm, gt_pos, P, Q)
    ra = rotation_accuracy(res.grid_piece, res.grid_rot, perm, gt_rot)

    print("EDGE SOLVER RESULTS")
    print("Placement:", pa)
    print("Neighbor :", na)
    print("Rotation :", ra)

    recon_img = render_solution(tiles_obs, res.grid_piece, res.grid_rot, P, Q)

    plt.imshow(recon_img)
    plt.title("Edge-based reconstruction")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
