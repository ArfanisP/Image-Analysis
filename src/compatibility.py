import numpy as np

def chi2_distance(h1: np.ndarray, h2: np.ndarray, eps: float = 1e-10) -> float:
    """
    Απόσταση Chi-square για ιστογράμματα.
    """
    num = (h1 - h2) ** 2
    den = (h1 + h2 + eps)
    return 0.5 * float(np.sum(num / den))

def similarity_from_distance(d: float, lam: float = 10.0) -> float:
    """
    Μετατρέπει απόσταση σε ομοιότητα στο (0,1] με χρήση exp(-lam*d).
    """
    return float(np.exp(-lam * d))
