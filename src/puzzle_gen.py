import numpy as np
import cv2

ROTATIONS = [0, 90, 180, 270]

def center_crop_to_multiple(img: np.ndarray, tile_size: int) -> np.ndarray:
    """Κεντρικό crop της εικόνας ώστε τα H και W να είναι πολλαπλάσια του tile_size."""
    h, w = img.shape[:2]
    hc = (h // tile_size) * tile_size
    wc = (w // tile_size) * tile_size
    y0 = (h - hc) // 2
    x0 = (w - wc) // 2
    return img[y0:y0+hc, x0:x0+wc].copy()

def split_into_tiles(img: np.ndarray, tile_size: int):
    """Επιστρέφει λίστα πλακιδίων και διαστάσεις πλέγματος (P,Q)."""
    h, w = img.shape[:2]
    assert h % tile_size == 0 and w % tile_size == 0
    P = h // tile_size
    Q = w // tile_size

    tiles = []
    for r in range(P):
        for c in range(Q):
            y0 = r * tile_size
            x0 = c * tile_size
            tile = img[y0:y0+tile_size, x0:x0+tile_size].copy()
            tiles.append(tile)
    return tiles, P, Q

def rotate_tile(tile: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return tile
    if angle == 90:
        return cv2.rotate(tile, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(tile, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(tile, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("angle must be in {0,90,180,270}")

def make_puzzle(img: np.ndarray, tile_size: int = 64, seed: int = 0, allow_rotation: bool = True):
    """
    Δημιουργεί ανακατεμένα (+ προαιρετικά περιστραμμένα) πλακίδια.
    Επιστρέφει:
      tiles_obs: λίστα observed πλακιδίων μετά από shuffle+rotation
      P,Q: διαστάσεις πλέγματος
      gt_pos: λίστα από (r,c) για κάθε αρχικό δείκτη πλακιδίου k
      perm: μετάθεση που εφαρμόστηκε (observed index -> original index)
      gt_rot: γωνία περιστροφής που εφαρμόστηκε σε κάθε observed πλακίδιο (ευθυγραμμισμένη με tiles_obs)
    """
    rng = np.random.default_rng(seed)

    img_c = center_crop_to_multiple(img, tile_size)
    tiles, P, Q = split_into_tiles(img_c, tile_size)

    N = len(tiles)
    gt_pos = [(k // Q, k % Q) for k in range(N)]  # αρχικό k -> (r,c)

    # Ανακάτεμα
    perm = rng.permutation(N)  # observed δείκτης i αντιστοιχεί στο αρχικό perm[i]
    tiles_shuffled = [tiles[perm[i]] for i in range(N)]

    # Περιστροφή
    gt_rot = []
    tiles_obs = []
    for t in tiles_shuffled:
        angle = int(rng.choice(ROTATIONS)) if allow_rotation else 0
        gt_rot.append(angle)
        tiles_obs.append(rotate_tile(t, angle))

    return img_c, tiles_obs, P, Q, gt_pos, perm, gt_rot

def render_grid(tiles: list[np.ndarray], P: int, Q: int) -> np.ndarray:
    """Συνθέτει τα πλακίδια σε μία ενιαία εικόνα πλέγματος (διατήρηση BGR/RGB)."""
    tile_h, tile_w = tiles[0].shape[:2]
    canvas = np.zeros((P * tile_h, Q * tile_w, 3), dtype=tiles[0].dtype)
    idx = 0
    for r in range(P):
        for c in range(Q):
            canvas[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w] = tiles[idx]
            idx += 1
    return canvas
