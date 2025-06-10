import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import re
from sklearn.preprocessing import StandardScaler
import traceback

# --- Configuration ---
TESLASUIT_PROCESSED_DATA_DIR = Path("./processed_combined_data_all_participants_TESLASUIT_DF_trimmed")
TESLASUIT_PKL_FILE = TESLASUIT_PROCESSED_DATA_DIR / "combined_all_participants_sequences_DF.pkl"

FACIAL_DATA_BASE_DIR = Path("./process_face_data")
SIGNER_FACIAL_FOLDERS = {
    "catalin": FACIAL_DATA_BASE_DIR / "final_facial_data_processed_catalin",
    "marinela": FACIAL_DATA_BASE_DIR / "final_facial_data_processed_marinela",
}

FINAL_OUTPUT_DIR = Path("./final_combined_data_for_training_ALL_SIGNERS")
FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DATA_PKL = FINAL_OUTPUT_DIR / "all_data_final_features_ts_facial.pkl"
FINAL_SCALER_PKL = FINAL_OUTPUT_DIR / "final_features_ts_facial_scaler.pkl"

FACIAL_MERGE_TOLERANCE_MS = 30


# --- Helper Functions ---
def extract_signer_and_sentence_id(teslasuit_filename_str, participant_name_from_id_dict=None):
    filename = Path(teslasuit_filename_str).stem.lower()
    signer_name, sentence_id_num = None, None

    if participant_name_from_id_dict:
        for known_signer_key in SIGNER_FACIAL_FOLDERS.keys():
            if known_signer_key.lower() in participant_name_from_id_dict.lower():
                signer_name = known_signer_key.lower()
                break

    if signer_name is None:
        for known_signer_key in SIGNER_FACIAL_FOLDERS.keys():
            if known_signer_key.lower() in filename:
                signer_name = known_signer_key.lower()
                break

    match = re.search(r'(?:sentence|propozitia)_(\d+)', filename)
    if match:
        sentence_id_num = int(match.group(1))
    return signer_name, sentence_id_num


def get_facial_pkl_path(signer_name, sentence_id_num):
    if signer_name not in SIGNER_FACIAL_FOLDERS: return None
    facial_pkl_filename = f"sentence_{int(sentence_id_num):03d}_facial_processed_bagts.pkl"
    return SIGNER_FACIAL_FOLDERS[signer_name] / facial_pkl_filename


def filter_none_and_empty_items(X_list, ids_list):
    filtered_X, filtered_ids = [], []
    removed_count = 0
    if not X_list:  # Dacă X_list e None sau goală
        return [], [], 0

    if len(X_list) != len(ids_list):
        print(
            f"    --> filter_none_and_empty_items ERROR: X_list length ({len(X_list)}) != ids_list length ({len(ids_list)})")
        # Decide ce să returnezi. Returnarea nemodificată poate propaga eroarea.
        # Poate e mai bine să returnezi liste goale dacă nu se potrivesc.
        return [], [], len(X_list)  # Sau X_list, ids_list, len(X_list) dacă vrei să încerci să continui

    for i, item_data in enumerate(X_list):
        is_valid = False
        if isinstance(item_data, pd.DataFrame):
            is_valid = not item_data.empty
        elif isinstance(item_data, np.ndarray):  # Folosit pentru datele scalate
            is_valid = item_data is not None and item_data.size > 0  # Verifică și None explicit

        if is_valid:
            filtered_X.append(item_data)
            filtered_ids.append(ids_list[i])
        else:
            removed_count += 1
    if removed_count > 0: print(f"    --> filter_none_and_empty_items removed {removed_count} sequences.")
    return filtered_X, filtered_ids, removed_count


# --- Main Processing Logic ---
print("--- Starting Final Data Combination (TeslaSuit + Facial) ---")
if not TESLASUIT_PKL_FILE.exists(): print(f"FATAL: TeslaSuit PKL not found at {TESLASUIT_PKL_FILE}."); exit()
try:
    with open(TESLASUIT_PKL_FILE, 'rb') as f:
        ts_data = pickle.load(f)
    X_train_ts_df_list = ts_data.get('X_train_df', ts_data.get('X_train', []))
    X_val_ts_df_list = ts_data.get('X_val_df', ts_data.get('X_val', []))
    X_test_ts_df_list = ts_data.get('X_test_df', ts_data.get('X_test', []))

    train_ids_ts = ts_data.get('train_ids', [])
    val_ids_ts = ts_data.get('val_ids', [])
    test_ids_ts = ts_data.get('test_ids', [])
    # feature_names_ts = ts_data.get('feature_names', []) # Nu este folosit direct în combinare

    if not X_train_ts_df_list and not X_val_ts_df_list and not X_test_ts_df_list:
        if not train_ids_ts and not val_ids_ts and not test_ids_ts:  # Verifică și ID-urile
            print(
                f"FATAL: No TeslaSuit data sequences or IDs found in PKL ({TESLASUIT_PKL_FILE}). Check keys like 'X_train_df' and 'train_ids'.");
            exit()
        else:  # Avem ID-uri dar nu date, ceea ce e ciudat
            print(
                f"WARNING: TeslaSuit data sequences (e.g. X_train_df) seem empty in PKL, but IDs exist. Proceeding cautiously.")

    print(
        f"TeslaSuit data loaded: {len(X_train_ts_df_list)} train DF, {len(X_val_ts_df_list)} val DF, {len(X_test_ts_df_list)} test DF sequences.")
    print(
        f"TeslaSuit IDs loaded: {len(train_ids_ts)} train IDs, {len(val_ids_ts)} val IDs, {len(test_ids_ts)} test IDs.")

except Exception as e:
    print(f"FATAL: Error loading TeslaSuit PKL ({TESLASUIT_PKL_FILE}): {e}");
    traceback.print_exc();
    exit()

final_feature_names_list = None
TIMESTAMP_COL_FOR_MERGE = 'normalized_timestamp_us'

datasets_to_process = {
    "train": (X_train_ts_df_list, train_ids_ts, []),  # Listele goale pentru output vor fi populate
    "val": (X_val_ts_df_list, val_ids_ts, []),
    "test": (X_test_ts_df_list, test_ids_ts, []),
}

for split_name, (X_ts_list_current_split, ids_ts_list_current_split,
                 X_final_df_list_output_ref) in datasets_to_process.items():
    print(f"\n--- Processing '{split_name}' set ---")

    if not X_ts_list_current_split and not ids_ts_list_current_split:  # Dacă ambele sunt goale
        print(f"  No TeslaSuit data or IDs for '{split_name}' set. Skipping.")
        # Nu este nevoie să actualizăm datasets_to_process[split_name][2] deoarece este deja []
        continue

    if len(X_ts_list_current_split) != len(ids_ts_list_current_split):
        print(
            f"  CRITICAL WARNING: Mismatch in lengths for '{split_name}': {len(X_ts_list_current_split)} data items vs {len(ids_ts_list_current_split)} IDs. This will likely cause errors or data loss for this split.")
        # Sărim peste acest split pentru a evita erori necontrolate
        continue

    temp_X_final_for_split = []
    temp_ids_final_for_split = []

    for i, df_teslasuit_current in enumerate(X_ts_list_current_split):
        current_id_dict = ids_ts_list_current_split[i]
        ts_filename_identifier = current_id_dict['filename']
        participant_name_key_from_ts = current_id_dict.get('participant')

        # print(f"  Processing TS: {ts_filename_identifier} (Participant ID: {participant_name_key_from_ts})") # Verbose

        if df_teslasuit_current is None or df_teslasuit_current.empty:
            # print(f"    Skipping empty/None TeslaSuit DataFrame for {ts_filename_identifier}.")
            continue

        if not isinstance(df_teslasuit_current.index, pd.TimedeltaIndex):
            # print(f"    Skipping TeslaSuit DataFrame for {ts_filename_identifier} due to non-TimedeltaIndex.")
            continue

        signer_name, sentence_id = extract_signer_and_sentence_id(ts_filename_identifier, participant_name_key_from_ts)
        if signer_name is None or sentence_id is None:
            # print(f"    Could not extract signer/sentence_id for {ts_filename_identifier}. Skipping.")
            continue

        facial_pkl_path = get_facial_pkl_path(signer_name, sentence_id)
        df_facial_current = pd.DataFrame()
        if facial_pkl_path and facial_pkl_path.exists():
            try:
                with open(facial_pkl_path, 'rb') as f:
                    loaded_facial_data = pickle.load(f)
                    if isinstance(loaded_facial_data, pd.DataFrame):
                        df_facial_current = loaded_facial_data

                if not df_facial_current.empty:
                    if not isinstance(df_facial_current.index, pd.TimedeltaIndex):
                        if pd.api.types.is_numeric_dtype(df_facial_current.index):
                            df_facial_current.index = pd.to_timedelta(df_facial_current.index, unit='us',
                                                                      errors='coerce')
                            df_facial_current.dropna(subset=[df_facial_current.index.name or 'index'], inplace=True)
                        else:
                            df_facial_current = pd.DataFrame()
                    if not df_facial_current.empty and isinstance(df_facial_current.index, pd.TimedeltaIndex):
                        df_facial_current = df_facial_current.sort_index()
                    elif not isinstance(df_facial_current.index, pd.TimedeltaIndex):  # Dacă tot nu e Timedelta
                        df_facial_current = pd.DataFrame()
            except Exception as e:
                # print(f"    Error loading or processing facial PKL {facial_pkl_path.name}: {e}")
                df_facial_current = pd.DataFrame()
                # else:
            # print(f"    No facial data found for {signer_name}, sentence {sentence_id} (Path: {facial_pkl_path}).")

        ts_index_name = df_teslasuit_current.index.name if df_teslasuit_current.index.name is not None else 'index'
        df_ts_reset = df_teslasuit_current.reset_index().rename(columns={ts_index_name: TIMESTAMP_COL_FOR_MERGE})
        df_final_aligned = df_ts_reset.copy()

        if not df_facial_current.empty:
            fc_index_name = df_facial_current.index.name if df_facial_current.index.name is not None else 'index'
            df_fc_reset = df_facial_current.reset_index().rename(columns={fc_index_name: TIMESTAMP_COL_FOR_MERGE})

            if TIMESTAMP_COL_FOR_MERGE not in df_ts_reset.columns or TIMESTAMP_COL_FOR_MERGE not in df_fc_reset.columns:
                # print(f"    Timestamp column for merge missing in TS or FC reset DFs for {ts_filename_identifier}. Using TS data only.")
                # df_final_aligned este deja df_ts_reset.copy()
                pass  # Continuă doar cu TS
            else:
                df_ts_reset_sorted = df_ts_reset.sort_values(TIMESTAMP_COL_FOR_MERGE)
                df_fc_reset_sorted = df_fc_reset.sort_values(TIMESTAMP_COL_FOR_MERGE)
                sfx_ts, sfx_fc = '_ts', '_fc'

                df_final_aligned_temp = pd.merge_asof(df_ts_reset_sorted, df_fc_reset_sorted,
                                                      on=TIMESTAMP_COL_FOR_MERGE, direction='nearest',
                                                      tolerance=pd.Timedelta(
                                                          microseconds=FACIAL_MERGE_TOLERANCE_MS * 1000),
                                                      suffixes=(sfx_ts, sfx_fc))

                cols_to_rename_fc = {f"{col}{sfx_fc}": col for col in df_facial_current.columns if
                                     f"{col}{sfx_fc}" in df_final_aligned_temp.columns}
                df_final_aligned_temp.rename(columns=cols_to_rename_fc, inplace=True)
                df_final_aligned = df_final_aligned_temp

            if TIMESTAMP_COL_FOR_MERGE in df_final_aligned.columns:
                df_final_aligned = df_final_aligned.set_index(TIMESTAMP_COL_FOR_MERGE).sort_index()
            else:
                # print(f"    Timestamp column '{TIMESTAMP_COL_FOR_MERGE}' lost after merge for {ts_filename_identifier}. Using original TS data.")
                df_final_aligned = df_teslasuit_current.copy()
                if not isinstance(df_final_aligned.index, pd.TimedeltaIndex):
                    # print(f"    Fallback TS data for {ts_filename_identifier} also has problematic index. Skipping sequence.")
                    continue

            fc_cols_in_final = [col for col in df_facial_current.columns if col in df_final_aligned.columns]
            if fc_cols_in_final and df_final_aligned[fc_cols_in_final].isnull().values.any():
                df_final_aligned[fc_cols_in_final] = df_final_aligned[fc_cols_in_final].interpolate(
                    method='time').ffill().bfill().fillna(0)

        else:
            if TIMESTAMP_COL_FOR_MERGE in df_final_aligned.columns:
                df_final_aligned = df_final_aligned.set_index(TIMESTAMP_COL_FOR_MERGE).sort_index()
            else:
                # print(f"    Timestamp column '{TIMESTAMP_COL_FOR_MERGE}' missing in TS-only data for {ts_filename_identifier}. Using original TS data.")
                df_final_aligned = df_teslasuit_current.copy()
                if not isinstance(df_final_aligned.index, pd.TimedeltaIndex):
                    # print(f"    Fallback TS data for {ts_filename_identifier} has problematic index. Skipping sequence.")
                    continue

        current_cols_in_df = list(df_final_aligned.columns)
        if final_feature_names_list is None:
            if not df_final_aligned.empty:
                final_feature_names_list = sorted(current_cols_in_df)
                # print(f"    Established feature list with {len(final_feature_names_list)} features from {ts_filename_identifier}.")
            else:
                # print(f"    Skipping {ts_filename_identifier} as it's empty and cannot establish feature list.")
                continue
        else:
            new_cols_found_in_current = [col for col in current_cols_in_df if col not in final_feature_names_list]
            if new_cols_found_in_current:
                final_feature_names_list.extend(new_cols_found_in_current)
                final_feature_names_list = sorted(list(set(final_feature_names_list)))

        if final_feature_names_list is not None and not df_final_aligned.empty:
            df_final_aligned = df_final_aligned.reindex(columns=final_feature_names_list, fill_value=0.0)
        elif df_final_aligned.empty and final_feature_names_list is not None:
            df_final_aligned = pd.DataFrame(columns=final_feature_names_list, index=df_final_aligned.index,
                                            dtype=float).fillna(0.0)

        if df_final_aligned.empty:
            # print(f"    Resulting DataFrame is empty for {ts_filename_identifier} after feature alignment. Skipping.")
            continue

        X_final_df_list_output_ref.append(df_final_aligned)  # Adaugă direct la lista de output a split-ului
        temp_ids_final_for_split.append(current_id_dict)  # Colectează ID-urile care au avut succes

    # Actualizează ID-urile pentru split-ul curent cu cele care au fost procesate cu succes
    datasets_to_process[split_name] = (None, temp_ids_final_for_split, X_final_df_list_output_ref)

# Extrage listele finale după procesarea tuturor split-urilor
X_train_final_df_list_unscaled_clean = datasets_to_process["train"][2]
train_ids_final = datasets_to_process["train"][1]

X_val_final_df_list_unscaled_clean = datasets_to_process["val"][2]
val_ids_final = datasets_to_process["val"][1]

X_test_final_df_list_unscaled_clean = datasets_to_process["test"][2]
test_ids_final = datasets_to_process["test"][1]

print("\n--- Final Data Checks and Normalization ---")
if final_feature_names_list is None:
    temp_feature_list_candidates = []
    for df_list_candidate in [X_train_final_df_list_unscaled_clean, X_val_final_df_list_unscaled_clean,
                              X_test_final_df_list_unscaled_clean]:
        if df_list_candidate:  # Verifică dacă lista nu e goală
            for df_item_candidate in df_list_candidate:  # Iterează prin DF-urile din listă
                if df_item_candidate is not None and not df_item_candidate.empty:
                    temp_feature_list_candidates.extend(df_item_candidate.columns)
                    break  # E suficient un DF valid pentru a lua coloanele
    if temp_feature_list_candidates:
        final_feature_names_list = sorted(list(set(temp_feature_list_candidates)))
        print(
            f"  Established final_feature_names_list from available data with {len(final_feature_names_list)} features as it was None.")
    else:
        print("FATAL: final_feature_names_list is None and cannot be derived from any processed data. Exiting.");
        exit()

final_scaler = None
if not X_train_final_df_list_unscaled_clean:
    print("WARNING: No data in training split (X_train_final_df_list_unscaled_clean is empty). Scaler cannot be fit.")
else:
    final_scaler = StandardScaler()
    num_expected_features = len(final_feature_names_list)

    all_train_dfs_for_scaler_values_list = []
    for df_train_item in X_train_final_df_list_unscaled_clean:
        if df_train_item is not None and not df_train_item.empty:
            df_reordered = df_train_item.reindex(columns=final_feature_names_list, fill_value=0.0)
            if df_reordered.shape[1] == num_expected_features:
                all_train_dfs_for_scaler_values_list.append(df_reordered.values)
            # else: # Nu ar trebui să se întâmple
            # print(f"  WARNING: Training DataFrame skipped for scaler due to unexpected feature mismatch after reindex.")
    if not all_train_dfs_for_scaler_values_list:
        print("FATAL: No valid DFs/values for scaler after attempting feature alignment. Scaler not fit.");
        final_scaler = None  # Resetează scaler-ul
    else:
        concatenated_train_values_for_scaler = np.concatenate(all_train_dfs_for_scaler_values_list, axis=0)
        if np.isnan(concatenated_train_values_for_scaler).any() or np.isinf(concatenated_train_values_for_scaler).any():
            # print("  Warning: NaNs or Infs found in data for scaler. Replacing with 0.")
            concatenated_train_values_for_scaler = np.nan_to_num(concatenated_train_values_for_scaler, nan=0.0,
                                                                 posinf=0.0, neginf=0.0)
        if concatenated_train_values_for_scaler.shape[0] == 0:
            print("FATAL: Concatenated data for scaler has 0 rows. Scaler not fit.");
            final_scaler = None  # Resetează scaler-ul
        else:
            final_scaler.fit(concatenated_train_values_for_scaler)
            print("  Scaler fitted on training data.")


def scale_final_df_sequence(df_seq, scaler, expected_features_names_list):
    if df_seq is None or df_seq.empty: return None

    df_reordered_for_scaling = df_seq.reindex(columns=expected_features_names_list, fill_value=0.0)
    values = df_reordered_for_scaling.values
    if np.isnan(values).any() or np.isinf(values).any():
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    if scaler is None:
        return values
    try:
        if values.shape[0] == 0: return np.array([])
        return scaler.transform(values)
    except Exception as e:
        # print(f"    scale_final_df_sequence: Scaling error: {e}. Returning None.");
        return None


X_train_final_scaled = [scale_final_df_sequence(df, final_scaler, final_feature_names_list) for df in
                        X_train_final_df_list_unscaled_clean]
X_val_final_scaled = [scale_final_df_sequence(df, final_scaler, final_feature_names_list) for df in
                      X_val_final_df_list_unscaled_clean]
X_test_final_scaled = [scale_final_df_sequence(df, final_scaler, final_feature_names_list) for df in
                       X_test_final_df_list_unscaled_clean]

X_train_final_scaled_clean, train_ids_final_clean, _ = filter_none_and_empty_items(X_train_final_scaled,
                                                                                   train_ids_final)
X_val_final_scaled_clean, val_ids_final_clean, _ = filter_none_and_empty_items(X_val_final_scaled, val_ids_final)
X_test_final_scaled_clean, test_ids_final_clean, _ = filter_none_and_empty_items(X_test_final_scaled, test_ids_final)


def filter_df_list_by_ids(df_list_original, ids_original, ids_clean_target):
    if not ids_clean_target: return []
    if not df_list_original or not ids_original: return []

    # Creează un map din ID-urile originale către DataFrame-urile lor
    # Folosește un tuplu (participant, filename) ca cheie pentru a gestiona duplicatele de nume de fișier între participanți
    ids_original_map = {}
    for i_orig, id_orig_dict in enumerate(ids_original):
        if i_orig < len(df_list_original):  # Asigură-te că indexul e valid
            key = (id_orig_dict.get('participant', 'UnknownParticipant'), id_orig_dict['filename'])
            ids_original_map[key] = df_list_original[i_orig]
        # else: Ar fi o nepotrivire de lungime între df_list_original și ids_original

    filtered_df_list = []
    for id_clean_dict in ids_clean_target:
        key_clean = (id_clean_dict.get('participant', 'UnknownParticipant'), id_clean_dict['filename'])
        if key_clean in ids_original_map:
            filtered_df_list.append(ids_original_map[key_clean])
        # else:
        # print(f"  WARNING: Could not find original DataFrame for cleaned ID key: {key_clean} when filtering unscaled DFs.")
        # Ar trebui să adaugi un None sau să gestionezi altfel dacă acest caz apare frecvent
    return filtered_df_list


X_train_df_indexed_clean = filter_df_list_by_ids(X_train_final_df_list_unscaled_clean, train_ids_final,
                                                 train_ids_final_clean)
X_val_df_indexed_clean = filter_df_list_by_ids(X_val_final_df_list_unscaled_clean, val_ids_final, val_ids_final_clean)
X_test_df_indexed_clean = filter_df_list_by_ids(X_test_final_df_list_unscaled_clean, test_ids_final,
                                                test_ids_final_clean)

print(f"\nSaving final combined data to: {FINAL_DATA_PKL}")
print(
    f"  Train sequences: Scaled={len(X_train_final_scaled_clean)}, Unscaled DFs={len(X_train_df_indexed_clean)}, IDs={len(train_ids_final_clean)}")
print(
    f"  Val sequences:   Scaled={len(X_val_final_scaled_clean)}, Unscaled DFs={len(X_val_df_indexed_clean)}, IDs={len(val_ids_final_clean)}")
print(
    f"  Test sequences:  Scaled={len(X_test_final_scaled_clean)}, Unscaled DFs={len(X_test_df_indexed_clean)}, IDs={len(test_ids_final_clean)}")

# Verificare finală de consistență a lungimilor înainte de salvare
if not (len(X_train_final_scaled_clean) == len(X_train_df_indexed_clean) == len(train_ids_final_clean) and \
        len(X_val_final_scaled_clean) == len(X_val_df_indexed_clean) == len(val_ids_final_clean) and \
        len(X_test_final_scaled_clean) == len(X_test_df_indexed_clean) == len(test_ids_final_clean)):
    print("CRITICAL ERROR: Mismatch in final list lengths before saving. Data might be inconsistent.")
    # Aici ai putea decide să nu salvezi sau să investighezi mai departe.

final_data_to_save = {
    'X_train': X_train_final_scaled_clean, 'X_val': X_val_final_scaled_clean, 'X_test': X_test_final_scaled_clean,
    'X_train_df_indexed': X_train_df_indexed_clean,
    'X_val_df_indexed': X_val_df_indexed_clean,
    'X_test_df_indexed': X_test_df_indexed_clean,
    'train_ids': train_ids_final_clean, 'val_ids': val_ids_final_clean, 'test_ids': test_ids_final_clean,
    'feature_names': final_feature_names_list
}
try:
    with open(FINAL_DATA_PKL, 'wb') as f:
        pickle.dump(final_data_to_save, f)
except Exception as e:
    print(f"Error saving final data PKL: {e}")

if final_scaler:
    print(f"Saving final scaler to: {FINAL_SCALER_PKL}")
    try:
        with open(FINAL_SCALER_PKL, 'wb') as f:
            pickle.dump(final_scaler, f)
    except Exception as e:
        print(f"Error saving final scaler PKL: {e}")
else:
    print("Final scaler was not trained or available. Scaler PKL not saved.")

print("\n--- Final Combination Script Finished ---")