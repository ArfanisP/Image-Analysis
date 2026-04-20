[README.md](https://github.com/user-attachments/files/26905383/README.md)


# Image Jigsaw Reconstruction  
Course Project – Ανάλυση Εικόνας

## Περιγραφή
Το παρόν project υλοποιεί ένα σύστημα ανακατασκευής εικόνων από τμηματοποιημένα κομμάτια (jigsaw puzzle reconstruction).  
Η αρχική εικόνα χωρίζεται σε ίσα tiles, τα οποία στη συνέχεια ανακατεύονται και περιστρέφονται. Στόχος είναι η εκτίμηση της σωστής θέσης και του προσανατολισμού κάθε κομματιού, ώστε να ανακατασκευαστεί η αρχική εικόνα χρησιμοποιώντας οπτικά χαρακτηριστικά.

Η υλοποίηση βασίζεται σε:
- χαρακτηριστικά χρώματος
- σύγκριση ακμών (edge seam matching)
- συνδυασμό χαρακτηριστικών
- αλγόριθμο ανακατασκευής βασισμένο σε mutual best matches (MBM)

---

## Δομή Project

```
src/
    puzzle_gen.py          # δημιουργία puzzle (split, shuffle, rotations)
    features/
        color.py
        edge.py
    compatibility/
    solver/
        greedy solvers
        MBM solver
    eval.py                # metrics αξιολόγησης

experiments/
    run_all.py             # κύριο script εκτέλεσης πειραμάτων

outputs/
    original.png
    shuffled.png
    reconstructions
    metrics.csv
```

---

## Απαιτήσεις

Python 3.x

Libraries:
```
numpy
opencv-python
matplotlib
scikit-image
```

Εγκατάσταση:
```bash
pip install numpy opencv-python matplotlib scikit-image
```

---

## Εκτέλεση

Τρέξε το κύριο script:

```bash
python experiments/run_all.py
```

Το πρόγραμμα:

1. Δημιουργεί puzzle από εικόνα (split, shuffle, rotations)
2. Εφαρμόζει διαφορετικές μεθόδους ανακατασκευής
3. Υπολογίζει metrics αξιολόγησης
4. Αποθηκεύει αποτελέσματα στον φάκελο `outputs/`

---

## Υλοποιημένες Μέθοδοι

- Color histogram baseline
- Edge seam matching
- Συνδυασμός edge + color
- Greedy reconstruction
- MBM (Mutual Best Matches) component-based reconstruction

---

## Metrics Αξιολόγησης

Υπολογίζονται:

- Placement accuracy
- Neighbor accuracy
- Rotation accuracy

Τα αποτελέσματα αποθηκεύονται στο:

```
outputs/metrics.csv
```

---

## Ενδεικτικά Αποτελέσματα

| Method | Placement | Neighbor | Rotation |
|---|---|---|---|
| color_greedy_rot | 0.016 | 0.304 | 0.094 |
| edge_greedy_rot | 0.016 | 0.554 | 0.156 |
| mbm_edge_color | **0.891** | **0.884** | **0.891** |

Η συνδυασμένη μέθοδος edge + color με MBM reconstruction προσφέρει σημαντικά καλύτερη ανακατασκευή της εικόνας.

---

## Παραγόμενα Αποτελέσματα

Στον φάκελο `outputs/` δημιουργούνται:

- αρχική εικόνα
- shuffled puzzle
- reconstructed εικόνες για κάθε μέθοδο
- αρχείο metrics.csv με τα αποτελέσματα

---

## Στόχος Εργασίας

Η εργασία στοχεύει στην ανάπτυξη και αξιολόγηση ενός συστήματος ανακατασκευής εικόνας από τμηματοποιημένα κομμάτια, με χρήση κλασικών τεχνικών ανάλυσης εικόνας και σύγκριση διαφορετικών στρατηγικών ανακατασκευής.
