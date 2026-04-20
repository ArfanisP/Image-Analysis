import matplotlib.pyplot as plt
from skimage import data
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.puzzle_gen import make_puzzle, render_grid
from src.solver_greedy import greedy_fill_grid

def main():
    img = data.astronaut()

    # ΣΗΜΑΝΤΙΚΟ: allow_rotation=False για τη γραμμή βάσης M3a
    img_c, tiles_obs, P, Q, gt_pos, perm, gt_rot = make_puzzle(
        img, tile_size=64, seed=42, allow_rotation=False
    )

    # Επίλυση
    res = greedy_fill_grid(tiles_obs, P, Q, wb=8, bins=16, lam=10.0, seed_piece=0)

    recon_tiles = [tiles_obs[idx] for idx in res.grid.flatten().tolist()]
    recon_img = render_grid(recon_tiles, P, Q)

    # Εμφάνιση
    plt.figure()
    plt.title(f"Original (cropped) P×Q={P}×{Q}")
    plt.imshow(img_c)
    plt.axis("off")

    plt.figure()
    plt.title(f"Shuffled (no rotation) Puzzle")
    plt.imshow(render_grid(tiles_obs, P, Q))
    plt.axis("off")

    plt.figure()
    plt.title(f"Reconstruction (Greedy, no rotations) | score={res.score:.2f}")
    plt.imshow(recon_img)
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    main()
