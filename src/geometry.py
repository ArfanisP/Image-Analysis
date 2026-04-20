ROTATIONS = [0, 90, 180, 270]
SIDES = ["top", "right", "bottom", "left"]

# Για πλακίδιο περιστραμμένο δεξιόστροφα:
# το new_top    προέρχεται από το old_left
# το new_right  προέρχεται από το old_top
# το new_bottom προέρχεται από το old_right
# το new_left   προέρχεται από το old_bottom
ROT_CW_SIDE_MAP = {
    0:   {"top":"top", "right":"right", "bottom":"bottom", "left":"left"},
    90:  {"top":"left", "right":"top", "bottom":"right", "left":"bottom"},
    180: {"top":"bottom", "right":"left", "bottom":"top", "left":"right"},
    270: {"top":"right", "right":"bottom", "bottom":"left", "left":"top"},
}

def rotated_side(original_side: str, angle: int) -> str:
    """
    Αν περιστρέψεις το πλακίδιο κατά 'angle' δεξιόστροφα, ποια αρχική πλευρά γίνεται 'original_side' στο περιστραμμένο πλακίδιο;
    Εδώ χρησιμοποιούμε ROT_CW_SIDE_MAP[angle][new_side] = old_side.
    """
    return ROT_CW_SIDE_MAP[angle][original_side]

def opposite_side(side: str) -> str:
    return {"top":"bottom", "bottom":"top", "left":"right", "right":"left"}[side]
