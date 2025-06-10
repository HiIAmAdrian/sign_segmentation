import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# --- Configuration ---
# !!! MODIFICĂ ACEASTĂ CALE PENTRU A SE POTRIVI CU FIȘIERUL TĂU !!!
PKL_FILE_PATH = Path("./final_combined_data_for_training_ALL_SIGNERS/all_data_final_features_ts_facial.pkl")

# Pentru a salva ca CSV, va trebui să alegi UN set de date (e.g., X_train) și O secvență din acel set.
# Specifică indexul secvenței din setul ales pe care vrei să o salvezi ca CSV.
SET_TO_INSPECT_AND_SAVE_AS_CSV = 'X_train' # Poate fi 'X_train', 'X_val', sau 'X_test'
SEQUENCE_INDEX_TO_SAVE_AS_CSV = 0          # Indexul secvenței din lista aleasă (e.g., prima secvență)
OUTPUT_CSV_PATH_FOR_SEQUENCE = Path(f"./final_combined_data_for_training_ALL_SIGNERS/inspected_sequence_{SET_TO_INSPECT_AND_SAVE_AS_CSV}_{SEQUENCE_INDEX_TO_SAVE_AS_CSV}.csv")

N_ROWS_TO_DISPLAY = 5 # Pentru head/tail în consolă

# --- Main Inspection Logic ---
def inspect_final_pkl(pkl_path, set_to_inspect='X_train', sequence_index_for_csv=0, csv_output_path=None, display_rows=5):
    print(f"--- Inspecting Final Combined PKL File: {pkl_path} ---")

    if not pkl_path.exists():
        print(f"Error: PKL file not found at '{pkl_path}'")
        return

    try:
        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)
        print(f"Successfully loaded data from '{pkl_path}'.")
    except Exception as e:
        print(f"Error loading PKL file '{pkl_path}': {e}")
        return

    if not isinstance(data_dict, dict):
        print(f"Error: Loaded object is not a dictionary as expected. Type: {type(data_dict)}")
        return

    print("\n--- Dictionary Keys ---")
    print(list(data_dict.keys()))

    feature_names = data_dict.get('feature_names')
    if feature_names:
        print(f"\n--- Feature Names (Total: {len(feature_names)}) ---")
        print(f"First 5: {feature_names[:5]}")
        print(f"Last 5: {feature_names[-5:]}")
    else:
        print("\nWarning: 'feature_names' key not found in the dictionary.")

    for data_key in ['X_train', 'X_val', 'X_test']:
        print(f"\n--- Inspecting Data for: {data_key} ---")
        if data_key in data_dict:
            sequence_list = data_dict[data_key]
            ids_list_key = data_key.replace('X_', '') + '_ids' # e.g. 'train_ids'
            ids_list = data_dict.get(ids_list_key, [])

            print(f"Number of sequences: {len(sequence_list)}")
            if sequence_list:
                # Inspectă prima secvență ca exemplu
                first_sequence_np = sequence_list[0]
                if isinstance(first_sequence_np, np.ndarray):
                    print(f"  Shape of the first sequence ({ids_list[0].get('filename', 'N/A') if ids_list else 'N/A'}): {first_sequence_np.shape}")
                    if first_sequence_np.shape[0] > 0 and feature_names and first_sequence_np.shape[1] != len(feature_names):
                        print(f"    WARNING: Feature count mismatch! Sequence has {first_sequence_np.shape[1]} features, expected {len(feature_names)}")

                    # Afișează head/tail pentru prima secvență
                    if first_sequence_np.shape[0] >= display_rows:
                        # Pentru a afișa ca DataFrame pentru lizibilitate
                        temp_df_display = pd.DataFrame(first_sequence_np, columns=feature_names if feature_names else None)
                        print(f"    Head of the first sequence:\n{temp_df_display.head(display_rows)}")
                        print(f"    Tail of the first sequence:\n{temp_df_display.tail(display_rows)}")
                    else:
                        print(f"    First sequence (too short to show head/tail distinctively):\n{first_sequence_np}")

                    # Verifică NaNs/Infs în prima secvență
                    if np.isnan(first_sequence_np).any():
                        print(f"    WARNING: NaNs found in the first sequence of {data_key}!")
                    if np.isinf(first_sequence_np).any():
                        print(f"    WARNING: Infinite values found in the first sequence of {data_key}!")

                else:
                    print(f"  First item in {data_key} is not a NumPy array. Type: {type(first_sequence_np)}")
            else:
                print(f"  No sequences in {data_key}.")
        else:
            print(f"  Key '{data_key}' not found in the dictionary.")

    # Salvează o secvență specifică ca CSV (dacă este specificat)
    if csv_output_path and set_to_inspect in data_dict:
        sequences_to_check = data_dict[set_to_inspect]
        if sequences_to_check and 0 <= sequence_index_for_csv < len(sequences_to_check):
            selected_sequence_np = sequences_to_check[sequence_index_for_csv]
            if isinstance(selected_sequence_np, np.ndarray):
                try:
                    # Creează un DataFrame din array-ul NumPy pentru a-l salva ca CSV
                    # Folosește feature_names dacă sunt disponibile
                    df_to_save = pd.DataFrame(selected_sequence_np, columns=feature_names if feature_names else None)
                    df_to_save.to_csv(csv_output_path, index_label='frame_index_in_sequence')
                    print(f"\nSelected sequence ({set_to_inspect}[{sequence_index_for_csv}]) saved to CSV: {csv_output_path}")
                except Exception as e:
                    print(f"\nError saving selected sequence to CSV '{csv_output_path}': {e}")
            else:
                print(f"\nSelected item at {set_to_inspect}[{sequence_index_for_csv}] is not a NumPy array. Cannot save as CSV.")
        else:
            print(f"\nCannot save CSV: Invalid set '{set_to_inspect}' or sequence index {sequence_index_for_csv}.")

if __name__ == "__main__":
    inspect_final_pkl(PKL_FILE_PATH,
                      set_to_inspect=SET_TO_INSPECT_AND_SAVE_AS_CSV,
                      sequence_index_for_csv=SEQUENCE_INDEX_TO_SAVE_AS_CSV,
                      csv_output_path=OUTPUT_CSV_PATH_FOR_SEQUENCE,
                      display_rows=N_ROWS_TO_DISPLAY)

    # Pentru a inspecta o altă secvență, poți rula din nou cu valori diferite:
    # inspect_final_pkl(PKL_FILE_PATH, set_to_inspect='X_val', sequence_index_for_csv=0, csv_output_path=Path("./inspected_X_val_0.csv"))