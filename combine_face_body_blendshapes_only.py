import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import re
from sklearn.preprocessing import StandardScaler
import traceback

TESLASUIT_PROCESSED_DATA_DIR = Path("./processed_combined_data_all_participants_TESLASUIT_DF_trimmed")
TESLASUIT_PKL_FILE = TESLASUIT_PROCESSED_DATA_DIR / "combined_all_participants_sequences_DF.pkl"

FACIAL_DATA_BASE_DIR = Path("./process_face_data")
SIGNER_FACIAL_FOLDERS = {
    "p1": FACIAL_DATA_BASE_DIR / "final_facial_data_processed_p1",
    "p2": FACIAL_DATA_BASE_DIR / "final_facial_data_processed_p2",
}

FINAL_OUTPUT_DIR = Path("./final_combined_data_for_training_ALL_SIGNERS_BLENDSHAPES_ONLY")
FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DATA_PKL = FINAL_OUTPUT_DIR / "all_data_final_features_ts_blendshapes.pkl"
FINAL_SCALER_PKL = FINAL_OUTPUT_DIR / "final_features_ts_blendshapes_scaler.pkl"

FACIAL_MERGE_TOLERANCE_MS = 30


def is_blendshape_column(col_name):
    col_lower = col_name.lower()
    return "blendshape" in col_lower or \
        col_lower.endswith(".value") or \
        col_lower.startswith("eye") or \
        col_lower.startswith("mouth") or \
        col_lower.startswith("brow") or \
        col_lower.startswith("jaw") or \
        col_lower.startswith("cheek") or \
        col_lower.startswith("nose")


def extract_signer_and_sentence_id(teslasuit_filename_str, participant_name_from_id_dict=None):
    filename = Path(teslasuit_filename_str).stem.lower()
    signer_name, sentence_id_num = None, None
    if participant_name_from_id_dict:
        for known_signer_key in SIGNER_FACIAL_FOLDERS.keys():
            if known_signer_key.lower() in participant_name_from_id_dict.lower():
                signer_name = known_signer_key.lower();
                break
    if signer_name is None:
        for known_signer_key in SIGNER_FACIAL_FOLDERS.keys():
            if known_signer_key.lower() in filename:
                signer_name = known_signer_key.lower();
                break
    match = re.search(r'(?:sentence|propozitia)_(\d+)', filename)
    if match: sentence_id_num = int(match.group(1))
    return signer_name, sentence_id_num


def get_facial_pkl_path(signer_name, sentence_id_num):
    if signer_name not in SIGNER_FACIAL_FOLDERS: return None
    facial_pkl_filename = f"sentence_{int(sentence_id_num):03d}_facial_processed_bagts.pkl"
    return SIGNER_FACIAL_FOLDERS[signer_name] / facial_pkl_filename


def filter_none_and_empty_items(X_list, ids_list):
    filtered_X, filtered_ids, removed_count = [], [], 0
    if not X_list: return [], [], 0
    if len(X_list) != len(ids_list):
        print(f"    Filter ERROR: X len ({len(X_list)}) != ids len ({len(ids_list)})")
        return [], [], len(X_list)
    for i, item_data in enumerate(X_list):
        is_valid = (isinstance(item_data, pd.DataFrame) and not item_data.empty) or \
                   (isinstance(item_data, np.ndarray) and item_data is not None and item_data.size > 0)
        if is_valid:
            filtered_X.append(item_data);
            filtered_ids.append(ids_list[i])
        else:
            removed_count += 1
    if removed_count > 0: print(f"    Filter removed {removed_count} sequences.")
    return filtered_X, filtered_ids, removed_count


print("--- Starting Final Data Combination (TeslaSuit + BLENDSHAPES Only) ---")
if not TESLASUIT_PKL_FILE.exists(): print(f"FATAL: TeslaSuit PKL not found at {TESLASUIT_PKL_FILE}."); exit()
try:
    with open(TESLASUIT_PKL_FILE, 'rb') as f:
        ts_data = pickle.load(f)
    X_train_ts_df_list = ts_data.get('X_train_df', ts_data.get('X_train', []))
    X_val_ts_df_list = ts_data.get('X_val_df', ts_data.get('X_val', []))
    X_test_ts_df_list = ts_data.get('X_test_df', ts_data.get('X_test', []))
    train_ids_ts, val_ids_ts, test_ids_ts = ts_data.get('train_ids', []), ts_data.get('val_ids', []), ts_data.get(
        'test_ids', [])
    if not any([X_train_ts_df_list, X_val_ts_df_list, X_test_ts_df_list]) and not any(
            [train_ids_ts, val_ids_ts, test_ids_ts]):
        print(f"FATAL: No TeslaSuit data or IDs in PKL ({TESLASUIT_PKL_FILE}).");
        exit()
    print(
        f"TeslaSuit data: {len(X_train_ts_df_list)} train, {len(X_val_ts_df_list)} val, {len(X_test_ts_df_list)} test DFs.")
    print(f"TeslaSuit IDs: {len(train_ids_ts)} train, {len(val_ids_ts)} val, {len(test_ids_ts)} test IDs.")
except Exception as e:
    print(f"FATAL loading TeslaSuit PKL: {e}"); traceback.print_exc(); exit()

final_feature_names_list = None
TIMESTAMP_COL_FOR_MERGE = 'normalized_timestamp_us'

datasets_to_process = {
    "train": (X_train_ts_df_list, train_ids_ts, []),
    "val": (X_val_ts_df_list, val_ids_ts, []),
    "test": (X_test_ts_df_list, test_ids_ts, []),
}

for split_name, (X_ts_list_current_split, ids_ts_list_current_split,
                 X_final_df_list_output_ref) in datasets_to_process.items():
    print(f"\n--- Processing '{split_name}' set ---")
    if not X_ts_list_current_split and not ids_ts_list_current_split:
        print(f"  No data/IDs for '{split_name}'. Skipping.");
        continue
    if len(X_ts_list_current_split) != len(ids_ts_list_current_split):
        print(f"  CRITICAL WARNING: Mismatch data/IDs for '{split_name}'. Skipping.");
        continue

    temp_ids_final_for_split = []

    for i, df_teslasuit_current in enumerate(X_ts_list_current_split):
        current_id_dict = ids_ts_list_current_split[i]
        ts_filename_identifier = current_id_dict['filename']
        participant_name_key_from_ts = current_id_dict.get('participant')

        print(f"  Processing TS: {ts_filename_identifier} (Participant: {participant_name_key_from_ts})")

        if df_teslasuit_current is None or df_teslasuit_current.empty: continue
        if not isinstance(df_teslasuit_current.index, pd.TimedeltaIndex): continue

        signer_name, sentence_id = extract_signer_and_sentence_id(ts_filename_identifier, participant_name_key_from_ts)
        if signer_name is None or sentence_id is None: continue

        facial_pkl_path = get_facial_pkl_path(signer_name, sentence_id)
        df_facial_all_features = pd.DataFrame()
        if facial_pkl_path and facial_pkl_path.exists():
            try:
                with open(facial_pkl_path, 'rb') as f:
                    loaded_facial_data = pickle.load(f)
                    if isinstance(loaded_facial_data, pd.DataFrame):
                        df_facial_all_features = loaded_facial_data

                if not df_facial_all_features.empty:
                    blendshape_cols = [col for col in df_facial_all_features.columns if is_blendshape_column(col)]
                    if not blendshape_cols:
                        print(
                            f"    DEBUG: No blendshape columns identified in facial data for {ts_filename_identifier} using defined pattern.")
                        df_facial_current = pd.DataFrame()
                    else:
                        print(
                            f"    DEBUG: Identified {len(blendshape_cols)} blendshape columns for {ts_filename_identifier}.")
                        df_facial_current = df_facial_all_features[
                            blendshape_cols].copy()

                        if not isinstance(df_facial_current.index, pd.TimedeltaIndex):
                            if pd.api.types.is_numeric_dtype(df_facial_current.index):
                                df_facial_current.index = pd.to_timedelta(df_facial_current.index, unit='us',
                                                                          errors='coerce')
                                df_facial_current.dropna(subset=[df_facial_current.index.name or 'index'], inplace=True)
                            else:
                                df_facial_current = pd.DataFrame()
                        if not df_facial_current.empty and isinstance(df_facial_current.index, pd.TimedeltaIndex):
                            df_facial_current = df_facial_current.sort_index()
                        elif not isinstance(df_facial_current.index, pd.TimedeltaIndex):
                            df_facial_current = pd.DataFrame()
                else:
                    df_facial_current = pd.DataFrame()

            except Exception as e:
                print(f"    Error loading/processing facial PKL {facial_pkl_path.name}: {e}")
                df_facial_current = pd.DataFrame()
        else:
            df_facial_current = pd.DataFrame()

        if df_facial_current.empty:
            print(f"    DEBUG: For {ts_filename_identifier}, df_facial_current (blendshapes) is EMPTY before merge.")
        else:
            print(
                f"    DEBUG: For {ts_filename_identifier}, df_facial_current (blendshapes) has {df_facial_current.shape[1]} columns.")

        ts_index_name = df_teslasuit_current.index.name or 'index'
        df_ts_reset = df_teslasuit_current.reset_index().rename(columns={ts_index_name: TIMESTAMP_COL_FOR_MERGE})
        df_final_aligned = df_ts_reset.copy()

        if not df_facial_current.empty:
            fc_index_name = df_facial_current.index.name or 'index'
            df_fc_reset = df_facial_current.reset_index().rename(columns={fc_index_name: TIMESTAMP_COL_FOR_MERGE})

            if TIMESTAMP_COL_FOR_MERGE in df_ts_reset.columns and TIMESTAMP_COL_FOR_MERGE in df_fc_reset.columns:
                df_ts_reset_sorted = df_ts_reset.sort_values(TIMESTAMP_COL_FOR_MERGE)
                df_fc_reset_sorted = df_fc_reset.sort_values(TIMESTAMP_COL_FOR_MERGE)

                df_final_aligned_temp = pd.merge_asof(df_ts_reset_sorted, df_fc_reset_sorted,
                                                      on=TIMESTAMP_COL_FOR_MERGE, direction='nearest',
                                                      tolerance=pd.Timedelta(
                                                          microseconds=FACIAL_MERGE_TOLERANCE_MS * 1000),
                                                      suffixes=('_ts',
                                                                '_fc'))

                cols_to_rename_fc = {f"{col}_fc": col for col in df_facial_current.columns if
                                     f"{col}_fc" in df_final_aligned_temp.columns and col not in df_ts_reset_sorted.columns}
                df_final_aligned_temp.rename(columns=cols_to_rename_fc, inplace=True)
                df_final_aligned = df_final_aligned_temp

            if TIMESTAMP_COL_FOR_MERGE in df_final_aligned.columns:
                df_final_aligned = df_final_aligned.set_index(TIMESTAMP_COL_FOR_MERGE).sort_index()
            else:
                df_final_aligned = df_teslasuit_current.copy()
                if not isinstance(df_final_aligned.index, pd.TimedeltaIndex): continue

            fc_cols_in_final = [col for col in df_facial_current.columns if col in df_final_aligned.columns]
            if fc_cols_in_final and df_final_aligned[fc_cols_in_final].isnull().values.any():
                df_final_aligned[fc_cols_in_final] = df_final_aligned[fc_cols_in_final].interpolate(
                    method='time').ffill().bfill().fillna(0)

        else:
            if TIMESTAMP_COL_FOR_MERGE in df_final_aligned.columns:
                df_final_aligned = df_final_aligned.set_index(TIMESTAMP_COL_FOR_MERGE).sort_index()
            elif isinstance(df_teslasuit_current.index,
                            pd.TimedeltaIndex):
                df_final_aligned = df_teslasuit_current.copy()
            else:
                continue

        current_cols_in_df = list(df_final_aligned.columns)
        if final_feature_names_list is None:
            if not df_final_aligned.empty:
                final_feature_names_list = sorted(current_cols_in_df)
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

        if df_final_aligned.empty and final_feature_names_list:
            df_final_aligned = pd.DataFrame(columns=final_feature_names_list, dtype=float)

        if df_final_aligned.empty and not final_feature_names_list:
            print(
                f"    Resulting DataFrame is empty for {ts_filename_identifier} and no global feature list yet. Skipping.")
            continue
        elif df_final_aligned.empty and final_feature_names_list:
            print(f"    Resulting DataFrame is empty for {ts_filename_identifier} but appending structured empty DF.")

        X_final_df_list_output_ref.append(df_final_aligned)
        temp_ids_final_for_split.append(current_id_dict)

    datasets_to_process[split_name] = (None, temp_ids_final_for_split, X_final_df_list_output_ref)

X_train_final_df_list_unscaled = datasets_to_process["train"][2]
train_ids_final = datasets_to_process["train"][1]
X_val_final_df_list_unscaled = datasets_to_process["val"][2]
val_ids_final = datasets_to_process["val"][1]
X_test_final_df_list_unscaled = datasets_to_process["test"][2]
test_ids_final = datasets_to_process["test"][1]

X_train_final_df_list_unscaled_clean, train_ids_final_clean, _ = filter_none_and_empty_items(
    X_train_final_df_list_unscaled, train_ids_final)
X_val_final_df_list_unscaled_clean, val_ids_final_clean, _ = filter_none_and_empty_items(X_val_final_df_list_unscaled,
                                                                                         val_ids_final)
X_test_final_df_list_unscaled_clean, test_ids_final_clean, _ = filter_none_and_empty_items(
    X_test_final_df_list_unscaled, test_ids_final)

print("\n--- Final Data Checks and Normalization ---")
if final_feature_names_list is None:
    temp_feature_list_candidates = []
    for df_list_cand in [X_train_final_df_list_unscaled_clean, X_val_final_df_list_unscaled_clean,
                         X_test_final_df_list_unscaled_clean]:
        if df_list_cand:
            for df_item_cand in df_list_cand:
                if df_item_cand is not None and not df_item_cand.empty:
                    temp_feature_list_candidates.extend(df_item_cand.columns);
                    break
    if temp_feature_list_candidates:
        final_feature_names_list = sorted(list(set(temp_feature_list_candidates)))
    else:
        print("FATAL: final_feature_names_list is None and cannot be derived. Exiting."); exit()
print(f"  Final global feature list has {len(final_feature_names_list)} features before scaler.")

final_scaler = None
if not X_train_final_df_list_unscaled_clean:
    print("WARNING: No data in training split. Scaler cannot be fit.")
else:
    final_scaler = StandardScaler()
    all_train_values_for_scaler = []
    for df_train_item in X_train_final_df_list_unscaled_clean:
        if df_train_item is not None and not df_train_item.empty:
            df_reordered = df_train_item.reindex(columns=final_feature_names_list, fill_value=0.0)
            all_train_values_for_scaler.append(df_reordered.values)

    if not all_train_values_for_scaler:
        print("FATAL: No valid DFs for scaler. Scaler not fit."); final_scaler = None
    else:
        concatenated_train_values = np.concatenate(all_train_values_for_scaler, axis=0)
        if np.isnan(concatenated_train_values).any() or np.isinf(concatenated_train_values).any():
            concatenated_train_values = np.nan_to_num(concatenated_train_values, nan=0.0, posinf=0.0, neginf=0.0)
        if concatenated_train_values.shape[0] == 0:
            print("FATAL: Concatenated data for scaler has 0 rows."); final_scaler = None
        else:
            final_scaler.fit(concatenated_train_values); print("  Scaler fitted on training data.")


def scale_final_df_sequence(df_seq, scaler, expected_features_names):
    if df_seq is None or df_seq.empty: return None
    if not expected_features_names: return None
    df_reordered = df_seq.reindex(columns=expected_features_names, fill_value=0.0)
    values = df_reordered.values
    if np.isnan(values).any() or np.isinf(values).any():
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if scaler is None: return values
    try:
        return scaler.transform(values) if values.shape[0] > 0 else np.array([])
    except:
        return None


def filter_df_list_by_ids(df_list_original, ids_original, ids_clean_target):
    """
    Filtrează o listă de DataFrames (df_list_original) pe baza unei liste "curate" de ID-uri (ids_clean_target).
    Se potrivește pe baza combinației (participant, filename) din dicționarele de ID-uri.
    """
    if not ids_clean_target:
        return []
    if not df_list_original or not ids_original:
        return []
    if len(df_list_original) != len(ids_original):
        print(
            f"    WARNING (filter_df_list_by_ids): Length mismatch between df_list_original ({len(df_list_original)}) and ids_original ({len(ids_original)}). Results might be incorrect.")
        return []

    ids_original_map_to_df = {}
    for i_orig, id_orig_dict in enumerate(ids_original):
        if i_orig < len(df_list_original):
            participant_key = id_orig_dict.get('participant', 'UnknownParticipant_Orig')
            filename_key = id_orig_dict.get('filename', f'UnknownFile_Orig_{i_orig}')
            global_key = (participant_key, filename_key)
            if global_key in ids_original_map_to_df:
                print(
                    f"    WARNING (filter_df_list_by_ids): Duplicate global key '{global_key}' found in ids_original. Overwriting.")
            ids_original_map_to_df[global_key] = df_list_original[i_orig]
        else:
            print(
                f"    ERROR (filter_df_list_by_ids): Index out of bounds for df_list_original at original index {i_orig}.")

    filtered_df_list = []
    for id_clean_dict in ids_clean_target:
        participant_key_clean = id_clean_dict.get('participant', 'UnknownParticipant_Clean')
        filename_key_clean = id_clean_dict.get('filename', 'UnknownFile_Clean_Target')
        global_key_clean = (participant_key_clean, filename_key_clean)

        if global_key_clean in ids_original_map_to_df:
            filtered_df_list.append(ids_original_map_to_df[global_key_clean])
        else:
            print(
                f"    WARNING (filter_df_list_by_ids): Could not find original DataFrame for cleaned ID key: {global_key_clean}.")
            filtered_df_list.append(None)

    if len(filtered_df_list) != len(ids_clean_target):
        print(
            f"    WARNING (filter_df_list_by_ids): Final filtered_df_list length ({len(filtered_df_list)}) does not match ids_clean_target length ({len(ids_clean_target)}). This may happen if some IDs were not found.")

    return filtered_df_list


X_train_final_scaled = [scale_final_df_sequence(df, final_scaler, final_feature_names_list) for df in
                        X_train_final_df_list_unscaled_clean]
X_val_final_scaled = [scale_final_df_sequence(df, final_scaler, final_feature_names_list) for df in
                      X_val_final_df_list_unscaled_clean]
X_test_final_scaled = [scale_final_df_sequence(df, final_scaler, final_feature_names_list) for df in
                       X_test_final_df_list_unscaled_clean]

X_train_final_scaled_clean, train_ids_final_clean_after_scale, _ = filter_none_and_empty_items(X_train_final_scaled,
                                                                                               train_ids_final_clean)
X_val_final_scaled_clean, val_ids_final_clean_after_scale, _ = filter_none_and_empty_items(X_val_final_scaled,
                                                                                           val_ids_final_clean)
X_test_final_scaled_clean, test_ids_final_clean_after_scale, _ = filter_none_and_empty_items(X_test_final_scaled,
                                                                                             test_ids_final_clean)

X_train_df_indexed_clean_final = filter_df_list_by_ids(X_train_final_df_list_unscaled_clean, train_ids_final_clean,
                                                       train_ids_final_clean_after_scale)
X_val_df_indexed_clean_final = filter_df_list_by_ids(X_val_final_df_list_unscaled_clean, val_ids_final_clean,
                                                     val_ids_final_clean_after_scale)
X_test_df_indexed_clean_final = filter_df_list_by_ids(X_test_final_df_list_unscaled_clean, test_ids_final_clean,
                                                      test_ids_final_clean_after_scale)

print(f"\nSaving final combined data (TS + Blendshapes) to: {FINAL_DATA_PKL}")
print(
    f"  Train: Scaled={len(X_train_final_scaled_clean)}, Unscaled={len(X_train_df_indexed_clean_final)}, IDs={len(train_ids_final_clean_after_scale)}")

final_data_to_save = {
    'X_train': X_train_final_scaled_clean, 'X_val': X_val_final_scaled_clean, 'X_test': X_test_final_scaled_clean,
    'X_train_df_indexed': X_train_df_indexed_clean_final,
    'X_val_df_indexed': X_val_df_indexed_clean_final,
    'X_test_df_indexed': X_test_df_indexed_clean_final,
    'train_ids': train_ids_final_clean_after_scale,
    'val_ids': val_ids_final_clean_after_scale,
    'test_ids': test_ids_final_clean_after_scale,
    'feature_names': final_feature_names_list
}
try:
    with open(FINAL_DATA_PKL, 'wb') as f:
        pickle.dump(final_data_to_save, f)
    print(f"Saved final_feature_names_list with {len(final_feature_names_list)} features (TS + Blendshapes).")
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
    print("Final scaler was not trained/available. Scaler PKL not saved.")

print("\n--- Final Combination (TS + Blendshapes Only) Script Finished ---")