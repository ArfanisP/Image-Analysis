import matplotlib.pyplot as plt
from skimage import data
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.puzzle_gen import make_puzzle, render_grid
from src.solver_greedy_rot import greedy_fill_grid_with_rotations, render_solution

def main():
    img = data.astronaut()

    # ΤΩΡΑ: allow_rotation=True (η πραγματική ρύθμιση)
    img_c, tiles_obs, P, Q, gt_pos, perm, gt_rot = make_puzzle(
        img, tile_size=64, seed=42, allow_rotation=True
    )

    res = greedy_fill_grid_with_rotations(
        tiles_obs, P, Q, wb=8, bins=16, lam=10.0, seed_piece=0, seed_angle=0
    )

    shuffled_img = render_grid(tiles_obs, P, Q)
    recon_img = render_solution(tiles_obs, res.grid_piece, res.grid_rot, P, Q)

    plt.figure()
    plt.title(f"Original (cropped) P×Q={P}×{Q}")
    plt.imshow(img_c)
    plt.axis("off")

    plt.figure()
    plt.title("Shuffled + Rotated puzzle")
    plt.imshow(shuffled_img)
    plt.axis("off")

    plt.figure()
    plt.title(f"Reconstruction (Greedy + rotations) | score={res.score:.2f}")
    plt.imshow(recon_img)
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    main()
