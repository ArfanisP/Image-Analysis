import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass

SIDES = ["top", "right", "bottom", "left"]
DELTA = {"top":(-1,0), "bottom":(1,0), "left":(0,-1), "right":(0,1)}

@dataclass
class MBMLayout:
    # αντιστοίχιση από δείκτη κομματιού -> (r,c,angle) σε ελεύθερο σύστημα συντεταγμένων
    placement: dict
    # λίστα συγκρούσεων για αποσφαλμάτωση
    conflicts: int

def build_adjacency(matches):
    """
    Δημιουργεί κατευθυνόμενη γειτνίαση:
    (i,ai) --si--> (j,aj) με μετατόπιση delta.
    """
    adj = defaultdict(list)
    for m in matches:
        dr, dc = DELTA[m.si]
        adj[(m.i, m.ai)].append(((m.j, m.aj), (dr, dc), m.sim))
        # και η αντίστροφη σχέση που προκύπτει:
        dr2, dc2 = (-dr, -dc)
        adj[(m.j, m.aj)].append(((m.i, m.ai), (dr2, dc2), m.sim))
    return adj

def largest_component_nodes(adj):
    """
    Βρίσκει τους κόμβους της μεγαλύτερης συνεκτικής συνιστώσας στην ακατεύθυντη εκδοχή.
    """
    nodes = set(adj.keys())
    seen = set()
    best_comp = []

    for n in list(nodes):
        if n in seen:
            continue
        q = deque([n])
        comp = []
        seen.add(n)
        while q:
            u = q.popleft()
            comp.append(u)
            for v, _, _ in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        if len(comp) > len(best_comp):
            best_comp = comp
    return best_comp

def layout_component(adj, root):
    """
    Ανάθεση ακέραιων μετατοπίσεων (r,c) στους κόμβους με BFS.
    """
    placement = {root: (0,0)}
    q = deque([root])
    conflicts = 0

    while q:
        u = q.popleft()
        ur, uc = placement[u]
        for v, (dr, dc), _ in adj[u]:
            vr, vc = ur + dr, uc + dc
            if v not in placement:
                placement[v] = (vr, vc)
                q.append(v)
            else:
                if placement[v] != (vr, vc):
                    conflicts += 1
    return placement, conflicts

def embed_in_grid(tiles_count, P, Q, placement, fallback_angles):
    """
    Μετατρέπει ελεύθερες συντεταγμένες σε πραγματικούς δείκτες πλέγματος.
    Κεντράρουμε τη συνιστώσα στο πλέγμα P×Q και μετά γεμίζουμε τα κενά με τα υπόλοιπα κομμάτια.
    """
    grid_piece = -np.ones((P,Q), dtype=int)
    grid_rot = np.zeros((P,Q), dtype=int)

    # επιλέγουμε μόνο έναν προσανατολισμό ανά κομμάτι (αν υπάρχουν πολλοί, κρατάμε τον πρώτο)
    per_piece = {}
    for (i, ai), (r, c) in placement.items():
        if i not in per_piece:
            per_piece[i] = (ai, r, c)

    coords = [(r,c) for (_, r, c) in per_piece.values()]
    rs = [r for r,c in coords]; cs = [c for r,c in coords]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)

    comp_h = rmax - rmin + 1
    comp_w = cmax - cmin + 1

    # μετατόπιση κεντραρίσματος
    off_r = (P - comp_h)//2 - rmin
    off_c = (Q - comp_w)//2 - cmin

    used = set()

    for i, (ai, r, c) in per_piece.items():
        rr = r + off_r
        cc = c + off_c
        if 0 <= rr < P and 0 <= cc < Q and grid_piece[rr,cc] == -1:
            grid_piece[rr,cc] = i
            grid_rot[rr,cc] = ai
            used.add(i)

    # συμπλήρωση των υπολοίπων με αχρησιμοποίητα κομμάτια
    unused = [i for i in range(tiles_count) if i not in used]
    k = 0
    for r in range(P):
        for c in range(Q):
            if grid_piece[r,c] == -1:
                i = unused[k]; k += 1
                grid_piece[r,c] = i
                grid_rot[r,c] = fallback_angles.get(i, 0)

    return grid_piece, grid_rot
