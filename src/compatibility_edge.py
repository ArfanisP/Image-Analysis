import numpy as np

def seam_distance(edge_a: np.ndarray, edge_b: np.ndarray) -> float:
    """
    Διαφορά pixel-προς-pixel κατά μήκος της ραφής.
    Μικρότερη τιμή = καλύτερο ταίριασμα.
    """
    # μετατροπή σε επίπεδη μορφή
    diff = edge_a - edge_b
    return float(np.mean(diff**2))

def seam_similarity(edge_a: np.ndarray, edge_b: np.ndarray, lam: float = 0.001) -> float:
    d = seam_distance(edge_a, edge_b)
    return float(np.exp(-lam * d))
