import matplotlib.pyplot as plt
from skimage import data
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.puzzle_gen import make_puzzle, render_grid
from src.solver_greedy_rot import greedy_fill_grid_with_rotations, render_solution
from src.eval import placement_accuracy, neighbor_accuracy, rotation_accuracy

def main():
    img = data.astronaut()

    img_c, tiles_obs, P, Q, gt_pos, perm, gt_rot = make_puzzle(
        img, tile_size=64, seed=42, allow_rotation=True
    )

    res = greedy_fill_grid_with_rotations(
        tiles_obs, P, Q, wb=8, bins=16, lam=10.0, seed_piece=0, seed_angle=0
    )

    # Μετρικές
    pa = placement_accuracy(res.grid_piece, perm, gt_pos, P, Q)
    na = neighbor_accuracy(res.grid_piece, perm, gt_pos, P, Q)
    ra = rotation_accuracy(res.grid_piece, res.grid_rot, perm, gt_rot)

    print(f"P×Q = {P}×{Q}")
    print(f"Placement accuracy: {pa:.4f}")
    print(f"Neighbor accuracy:  {na:.4f}")
    print(f"Rotation accuracy:  {ra:.4f}")
    print(f"Greedy score:       {res.score:.2f}")

    # Οπτικοποίηση
    shuffled_img = render_grid(tiles_obs, P, Q)
    recon_img = render_solution(tiles_obs, res.grid_piece, res.grid_rot, P, Q)

    plt.figure()
    plt.title("Shuffled + Rotated puzzle")
    plt.imshow(shuffled_img)
    plt.axis("off")

    plt.figure()
    plt.title("Reconstruction (Greedy + rotations)")
    plt.imshow(recon_img)
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    main()
