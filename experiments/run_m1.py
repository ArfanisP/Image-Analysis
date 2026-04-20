import os
import glob
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from skimage import data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.puzzle_gen import make_puzzle, render_grid

def load_image_any():
    # 1) αν υπάρχει οποιαδήποτε εικόνα στο data/images, φόρτωσε την πρώτη
    paths = sorted(glob.glob("data/images/*.*"))
    if paths:
        img_bgr = cv2.imread(paths[0], cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb, f"file: {os.path.basename(paths[0])}"

    # 2) αλλιώς χρησιμοποίησε την ενσωματωμένη
    return data.astronaut(), "skimage.data.astronaut"

def main():
    img, name = load_image_any()

    img_c, tiles_obs, P, Q, gt_pos, perm, gt_rot = make_puzzle(
        img, tile_size=64, seed=42, allow_rotation=True
    )

    shuffled_img = render_grid(tiles_obs, P, Q)

    plt.figure()
    plt.title(f"Original (cropped) [{name}] | P×Q={P}×{Q}")
    plt.imshow(img_c)
    plt.axis("off")

    plt.figure()
    plt.title("Shuffled + Rotated puzzle")
    plt.imshow(shuffled_img)
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    main()
