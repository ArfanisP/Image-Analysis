import numpy as np

def build_gt_grid(P: int, Q: int, gt_pos):
    """
    gt_pos: λίστα από (r,c) για κάθε ΑΡΧΙΚΟ δείκτη πλακιδίου k
    επιστρέφει gt_grid[r,c] = αρχικός δείκτης πλακιδίου σε αυτή τη θέση
    """
    gt_grid = -np.ones((P, Q), dtype=int)
    for k, (r, c) in enumerate(gt_pos):
        gt_grid[r, c] = k
    return gt_grid

def reconstructed_orig_grid(grid_piece: np.ndarray, perm: np.ndarray):
    """
    Το grid_piece περιέχει δείκτες ΠΑΡΑΤΗΡΗΜΕΝΩΝ πλακιδίων.
    Το perm χαρτογραφεί observed_index -> original_index.
    επιστρέφει grid_orig[r,c] = original_index τοποθετημένο στο (r,c)
    """
    P, Q = grid_piece.shape
    grid_orig = -np.ones((P, Q), dtype=int)
    for r in range(P):
        for c in range(Q):
            obs_idx = int(grid_piece[r, c])
            grid_orig[r, c] = int(perm[obs_idx])
    return grid_orig

def placement_accuracy(grid_piece: np.ndarray, perm: np.ndarray, gt_pos, P: int, Q: int) -> float:
    gt_grid = build_gt_grid(P, Q, gt_pos)
    grid_orig = reconstructed_orig_grid(grid_piece, perm)
    correct = (grid_orig == gt_grid).sum()
    return float(correct) / float(P * Q)

def true_neighbor_pairs(P: int, Q: int, gt_grid: np.ndarray):
    """
    Επιστρέφει σύνολο από μη διατεταγμένα ζεύγη {min,max} που παριστάνουν αληθινούς γείτονες στο GT.
    Μετράμε δεξιούς και κάτω γείτονες για αποφυγή διπλομετρήσεων.
    """
    pairs = set()
    for r in range(P):
        for c in range(Q):
            a = int(gt_grid[r, c])
            if c + 1 < Q:
                b = int(gt_grid[r, c+1])
                pairs.add((min(a, b), max(a, b)))
            if r + 1 < P:
                b = int(gt_grid[r+1, c])
                pairs.add((min(a, b), max(a, b)))
    return pairs

def reconstructed_neighbor_pairs(P: int, Q: int, grid_orig: np.ndarray):
    """
    Ίδια ιδέα, αλλά από το ανακατασκευασμένο πλέγμα ΑΡΧΙΚΩΝ δεικτών.
    """
    pairs = set()
    for r in range(P):
        for c in range(Q):
            a = int(grid_orig[r, c])
            if c + 1 < Q:
                b = int(grid_orig[r, c+1])
                pairs.add((min(a, b), max(a, b)))
            if r + 1 < P:
                b = int(grid_orig[r+1, c])
                pairs.add((min(a, b), max(a, b)))
    return pairs

def neighbor_accuracy(grid_piece: np.ndarray, perm: np.ndarray, gt_pos, P: int, Q: int) -> float:
    gt_grid = build_gt_grid(P, Q, gt_pos)
    gt_pairs = true_neighbor_pairs(P, Q, gt_grid)

    grid_orig = reconstructed_orig_grid(grid_piece, perm)
    rec_pairs = reconstructed_neighbor_pairs(P, Q, grid_orig)

    recovered = len(gt_pairs.intersection(rec_pairs))
    return float(recovered) / float(len(gt_pairs))

def rotation_accuracy(grid_piece: np.ndarray, grid_rot: np.ndarray, perm: np.ndarray, gt_rot_obs: list[int]) -> float:
    """
    gt_rot_obs: λίστα ευθυγραμμισμένη με δείκτες OBSERVED (ίδια σειρά με το tiles_obs),
                γωνία που εφαρμόστηκε κατά τη δημιουργία του παζλ.

    Ο solver αποθηκεύει grid_rot[r,c] = επιπλέον δεξιόστροφη περιστροφή που ΕΦΑΡΜΟΖΟΥΜΕ στο OBSERVED πλακίδιο κατά την τοποθέτηση.
    Η καθαρή περιστροφή ως προς το αρχικό είναι (gt_rot_obs[obs] + grid_rot[r,c]) mod 360.
    Σωστός προσανατολισμός σημαίνει net == 0.
    """
    P, Q = grid_piece.shape
    correct = 0
    total = P * Q
    for r in range(P):
        for c in range(Q):
            obs = int(grid_piece[r, c])
            ang = int(grid_rot[r, c]) % 360
            net = (int(gt_rot_obs[obs]) + ang) % 360
            if net == 0:
                correct += 1
    return float(correct) / float(total)
