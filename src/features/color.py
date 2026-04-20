import numpy as np

SIDES = ["top", "right", "bottom", "left"]

def extract_border(tile: np.ndarray, side: str, wb: int) -> np.ndarray:
    """
    Εξάγει λωρίδα περιγράμματος πλάτους wb pixel από ένα πλακίδιο.
    tile: (H,W,3) uint8
    επιστρέφει: λωρίδα περιγράμματος ως πίνακα εικόνας
    """
    H, W = tile.shape[:2]
    if wb < 1 or wb > min(H, W):
        raise ValueError("wb must be in [1, min(H,W)]")

    if side == "top":
        return tile[:wb, :, :]
    if side == "bottom":
        return tile[H-wb:H, :, :]
    if side == "left":
        return tile[:, :wb, :]
    if side == "right":
        return tile[:, W-wb:W, :]
    raise ValueError(f"Unknown side: {side}")

def color_histogram_rgb(patch: np.ndarray, bins: int = 16) -> np.ndarray:
    """
    Απλός περιγραφέας ιστογράμματος RGB.
    Επιστρέφει 1D διάνυσμα μήκους 3*bins, κανονικοποιημένο ως προς L1.
    """
    # το patch είναι uint8 RGB
    desc_parts = []
    for ch in range(3):
        hist, _ = np.histogram(patch[:, :, ch], bins=bins, range=(0, 256))
        hist = hist.astype(np.float32)
        desc_parts.append(hist)
    desc = np.concatenate(desc_parts, axis=0)
    s = desc.sum()
    if s > 0:
        desc /= s
    return desc

def side_descriptor_color(tile: np.ndarray, side: str, wb: int, bins: int = 16) -> np.ndarray:
    border = extract_border(tile, side, wb)
    return color_histogram_rgb(border, bins=bins)
