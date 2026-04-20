import numpy as np

def get_border_pixels(tile: np.ndarray, side: str, wb: int = 1) -> np.ndarray:
    """
    Επιστρέφει ακατέργαστα pixel περιγράμματος (όχι ιστόγραμμα).
    Το wb είναι συνήθως 1–3 για ταίριασμα ραφής.
    """
    H, W = tile.shape[:2]

    if side == "top":
        return tile[:wb, :, :].astype(np.float32)
    if side == "bottom":
        return tile[H-wb:H, :, :].astype(np.float32)
    if side == "left":
        return tile[:, :wb, :].astype(np.float32)
    if side == "right":
        return tile[:, W-wb:W, :].astype(np.float32)

    raise ValueError("invalid side")
