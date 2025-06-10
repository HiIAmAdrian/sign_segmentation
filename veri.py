import pickle
import numpy as np
from pathlib import Path

FINAL_ANNOTATION_FILE = Path("./final_combined_data_for_training_ALL_SIGNERS/annotations_bio_final_combined.pkl")
MODEL_SAVE_DIR = Path("./trained_models_final")  # Pentru consistență în rularea locală a snippet-ului

with open(FINAL_ANNOTATION_FILE, 'rb') as f:
    annotations = pickle.load(f)
y_test_raw = annotations['test']

print(f"Număr de secvențe în y_test_raw: {len(y_test_raw)}")

has_b_or_i = False
total_frames_test = 0
b_counts_test = 0
i_counts_test = 0
o_counts_test = 0

for i, seq in enumerate(y_test_raw):
    total_frames_test += len(seq)
    unique_labels, counts = np.unique(seq, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))

    b_count = label_counts.get(1, 0)  # 1 este LABEL_B
    i_count = label_counts.get(2, 0)  # 2 este LABEL_I
    o_count = label_counts.get(0, 0)  # 0 este LABEL_O

    b_counts_test += b_count
    i_counts_test += i_count
    o_counts_test += o_count

    if b_count > 0 or i_count > 0:
        has_b_or_i = True
        print(f"  Secvența de test {i} (lungime {len(seq)}) CONȚINE B/I: {label_counts}")
    # else:
    # print(f"  Secvența de test {i} (lungime {len(seq)}) conține doar O: {label_counts}")

if not has_b_or_i:
    print("\n!!! ALERTĂ: Nicio secvență din y_test_raw nu conține etichete B sau I !!!")
else:
    print("\nCel puțin o secvență din y_test_raw conține etichete B sau I.")

print(f"\nDistribuția totală a etichetelor în y_test_raw (cadre non-padate):")
print(f"  Total cadre: {total_frames_test}")
print(
    f"  Număr de O: {o_counts_test} ({(o_counts_test / total_frames_test) * 100 if total_frames_test > 0 else 0:.2f}%)")
print(
    f"  Număr de B: {b_counts_test} ({(b_counts_test / total_frames_test) * 100 if total_frames_test > 0 else 0:.2f}%)")
print(
    f"  Număr de I: {i_counts_test} ({(i_counts_test / total_frames_test) * 100 if total_frames_test > 0 else 0:.2f}%)")

# Verifică și setul de validare, pentru context
y_val_raw = annotations['val']
has_b_or_i_val = False
for seq_val in y_val_raw:
    if np.any(seq_val == 1) or np.any(seq_val == 2):
        has_b_or_i_val = True
        break
print(f"\nSetul de validare conține B/I: {has_b_or_i_val}")

# Verifică și setul de antrenament
y_train_raw = annotations['train']
has_b_or_i_train = False
for seq_train in y_train_raw:
    if np.any(seq_train == 1) or np.any(seq_train == 2):
        has_b_or_i_train = True
        break
print(f"Setul de antrenament conține B/I: {has_b_or_i_train}")