import numpy as np
from src.compatibility_edge import seam_similarity, seam_distance
from src.compatibility import chi2_distance

def combined_similarity(edge_a, edge_b, hist_a, hist_b, lam_edge=0.001, lam_col=10.0, w_edge=0.8, w_col=0.2):
    """
    Συνδυάζει ομοιότητα ραφής (edge) με ομοιότητα ιστογράμματος χρώματος.
    """
    s_edge = seam_similarity(edge_a, edge_b, lam=lam_edge)  # στο (0,1]
    d_col = chi2_distance(hist_a, hist_b)
    s_col = float(np.exp(-lam_col * d_col))
    return float(w_edge * s_edge + w_col * s_col)
