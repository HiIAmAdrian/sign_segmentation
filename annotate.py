import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import traceback
import re

# --- Configuration ---
FINAL_COMBINED_DATA_DIR = Path("./final_combined_data_for_training_ALL_SIGNERS")
FINAL_COMBINED_PKL = FINAL_COMBINED_DATA_DIR / "all_data_final_features_ts_facial.pkl"

PARTICIPANT_INFO_CONFIG = {
    "participant_catalin_data": {
        "base_dir_path": Path("D:/SegmentationThesis/output_realsense60fps+tesla Catalin"),
        "trim_sec": 1.0
    },
    "participant_marinela_data": {
        "base_dir_path": Path("D:/SegmentationThesis/output_realsense60fps+tesla Marinela"),
        "trim_sec": 0.3
    },
}
EAF_SUBDIR_NAME = "elan_annotations"
# ORIGINAL_SUIT_CSV_SUBDIR_NAME nu mai este necesar pentru logica principală de mapare, dar poate fi păstrat dacă e util altundeva
OUTPUT_ANNOTATION_BIO_PKL = FINAL_COMBINED_DATA_DIR / "annotations_bio_final_combined.pkl"

TARGET_TIER_ID = "default"
SYNC_OFFSET_MS = 0
LABEL_O, LABEL_B, LABEL_I = 0, 1, 2


def parse_eaf(eaf_path, target_tier_id="default"):
    segments = []
    try:
        tree = ET.parse(eaf_path)
        root = tree.getroot()
        time_slots = {slot.attrib['TIME_SLOT_ID']: int(slot.attrib['TIME_VALUE']) for slot in
                      root.findall(".//TIME_SLOT")}
        tier_found = False
        for tier in root.findall(f".//TIER"):
            if tier.attrib.get('TIER_ID') == target_tier_id:
                tier_found = True
                for ann in tier.findall(".//ALIGNABLE_ANNOTATION"):
                    t1, t2 = ann.attrib['TIME_SLOT_REF1'], ann.attrib['TIME_SLOT_REF2']
                    if t1 in time_slots and t2 in time_slots:
                        segments.append((time_slots[t1], time_slots[t2]))
                break
        if not tier_found:
            # print(f"    DEBUG: Tier '{target_tier_id}' not found in {eaf_path.name}.")
            return []
        segments.sort()
        return segments
    except ET.ParseError as pe:
        print(f"    WARNING: XML ParseError for EAF {eaf_path.name}: {pe}")
        return None
    except Exception as e:
        print(f"    WARNING: General error parsing EAF {eaf_path.name}: {e}")
        return None


# get_original_mocap_timestamps_trimmed nu mai este necesară pentru această logică

print(f"Loading final combined data from: {FINAL_COMBINED_PKL}")
if not FINAL_COMBINED_PKL.exists():
    print(f"FATAL: Final combined PKL file not found at {FINAL_COMBINED_PKL}.");
    exit()
try:
    with open(FINAL_COMBINED_PKL, 'rb') as f:
        loaded_final_data = pickle.load(f)
except Exception as e:
    print(f"FATAL: Error loading final combined PKL: {e}"); traceback.print_exc(); exit()

X_train_df_indexed = loaded_final_data.get('X_train_df_indexed', [])
train_ids_final = loaded_final_data.get('train_ids', [])
X_val_df_indexed = loaded_final_data.get('X_val_df_indexed', [])
val_ids_final = loaded_final_data.get('val_ids', [])
X_test_df_indexed = loaded_final_data.get('X_test_df_indexed', [])
test_ids_final = loaded_final_data.get('test_ids', [])

y_splits_bio = {'train': [], 'val': [], 'test': []}
processed_eaf_count = 0
error_details_eaf_processing = []

datasets_for_annotation = {
    "train": (X_train_df_indexed, train_ids_final),
    "val": (X_val_df_indexed, val_ids_final),
    "test": (X_test_df_indexed, test_ids_final),
}
print(f"\n--- Processing EAF annotations for final combined data ---")

for split_name, (X_df_list_current_split, ids_list_current_split) in datasets_for_annotation.items():
    print(f"\n  -- Annotating '{split_name}' set ({len(ids_list_current_split)} sequences) --")
    if not X_df_list_current_split or not ids_list_current_split: continue
    if len(X_df_list_current_split) != len(ids_list_current_split):
        print(f"    CRITICAL WARNING: Mismatch lengths for '{split_name}'. Skipping.");
        continue

    for i, df_final_combined_seq in enumerate(X_df_list_current_split):
        current_id_dict = ids_list_current_split[i]
        original_ts_csv_filename = current_id_dict['filename']
        participant_key_raw_from_id = current_id_dict.get('participant')
        global_processing_key = f"{participant_key_raw_from_id}::{original_ts_csv_filename}"

        if df_final_combined_seq is None or df_final_combined_seq.empty:
            error_details_eaf_processing.append({'k': global_processing_key, 'r': "DF empty", 's': split_name})
            y_splits_bio[split_name].append(np.array([], dtype=int));
            continue
        if not isinstance(df_final_combined_seq.index, pd.TimedeltaIndex):
            error_details_eaf_processing.append({'k': global_processing_key, 'r': "No TimedeltaIndex", 's': split_name})
            y_splits_bio[split_name].append(np.full(df_final_combined_seq.shape[0], LABEL_O, dtype=int));
            continue

        num_frames_final_seq = df_final_combined_seq.shape[0]
        final_timedelta_index = df_final_combined_seq.index

        participant_config_key = None
        if participant_key_raw_from_id:
            if "catalin" in participant_key_raw_from_id.lower():
                participant_config_key = "participant_catalin_data"
            elif "marinela" in participant_key_raw_from_id.lower():
                participant_config_key = "participant_marinela_data"

        if not participant_config_key or participant_config_key not in PARTICIPANT_INFO_CONFIG:
            error_details_eaf_processing.append(
                {'k': global_processing_key, 'r': f"No PConfig for '{participant_key_raw_from_id}'", 's': split_name})
            y_splits_bio[split_name].append(np.full(num_frames_final_seq, LABEL_O, dtype=int));
            continue

        p_conf = PARTICIPANT_INFO_CONFIG[participant_config_key]
        p_base_dir, trim_sec_for_this_participant = p_conf["base_dir_path"], p_conf["trim_sec"]

        eaf_match = re.match(r"(sentence_\d+_ts)_suit_mocap\.csv", original_ts_csv_filename, re.IGNORECASE)
        if not eaf_match:
            error_details_eaf_processing.append(
                {'k': global_processing_key, 'r': f"No base name from {original_ts_csv_filename}", 's': split_name})
            y_splits_bio[split_name].append(np.full(num_frames_final_seq, LABEL_O, dtype=int));
            continue
        eaf_base = eaf_match.group(1).replace('_ts', '')
        eaf_path = p_base_dir / (EAF_SUBDIR_NAME or "") / f"{eaf_base}_realsense.eaf"

        if not eaf_path.exists():
            error_details_eaf_processing.append(
                {'k': global_processing_key, 'r': "EAF not found", 'p': str(eaf_path), 's': split_name})
            y_splits_bio[split_name].append(np.full(num_frames_final_seq, LABEL_O, dtype=int));
            continue

        video_segments_ms = parse_eaf(eaf_path, TARGET_TIER_ID)
        if video_segments_ms is None:
            error_details_eaf_processing.append(
                {'k': global_processing_key, 'r': f"EAF parse error: {eaf_path.name}", 's': split_name})
            y_splits_bio[split_name].append(np.full(num_frames_final_seq, LABEL_O, dtype=int));
            continue

        labels_bio = np.full(num_frames_final_seq, LABEL_O, dtype=int)
        trim_offset_us = trim_sec_for_this_participant * 1_000_000

        if video_segments_ms:
            # print(f"    DEBUG Key: {global_processing_key} (Frames: {num_frames_final_seq})")
            for seg_idx, (video_start_ms, video_end_ms) in enumerate(video_segments_ms):
                eaf_start_sync_us = (video_start_ms + SYNC_OFFSET_MS) * 1000
                eaf_end_sync_us = (video_end_ms + SYNC_OFFSET_MS) * 1000

                target_start_for_lookup_us = max(0, eaf_start_sync_us - trim_offset_us)
                target_end_for_lookup_us = max(0, eaf_end_sync_us - trim_offset_us)

                if target_end_for_lookup_us <= target_start_for_lookup_us:
                    continue

                start_frame_idx = final_timedelta_index.searchsorted(
                    pd.Timedelta(microseconds=target_start_for_lookup_us), side='left')
                end_frame_idx = final_timedelta_index.searchsorted(pd.Timedelta(microseconds=target_end_for_lookup_us),
                                                                   side='right') - 1

                actual_sf = max(0, min(start_frame_idx, num_frames_final_seq - 1))
                actual_ef = max(0, min(end_frame_idx, num_frames_final_seq - 1))

                if actual_sf <= actual_ef:
                    labels_bio[actual_sf] = LABEL_B
                    if actual_sf < actual_ef:
                        labels_bio[actual_sf + 1: actual_ef + 1] = LABEL_I

        y_splits_bio[split_name].append(labels_bio)
        processed_eaf_count += 1

print(f"\n--- Summary of Annotation ---")
print(f"Successfully generated annotations for {processed_eaf_count} sequences.")
num_errors_recorded = len(error_details_eaf_processing)
if num_errors_recorded > 0:
    print(f"Skipped or encountered errors for {num_errors_recorded} sequences. Details:")
    for err in error_details_eaf_processing: print(
        f"  - Key: {err.get('k')}, Split: {err.get('s')}, Reason: {err.get('r')}")
total_labels_assigned = sum(len(v) for v in y_splits_bio.values())
print(f"Total B-I-O label arrays assigned: {total_labels_assigned}")

lengths_match_final = True;
error_sequences_final_check = []
total_ids_from_combined_pkl_check = len(train_ids_final) + len(val_ids_final) + len(test_ids_final)
for split_name in ['train', 'val', 'test']:
    X_df_list, Y_labels_list, ids_list = datasets_for_annotation[split_name][0], y_splits_bio[split_name], \
    datasets_for_annotation[split_name][1]
    if len(X_df_list) != len(Y_labels_list):
        print(f"FATAL COUNT MISMATCH split '{split_name}': X {len(X_df_list)}, Y {len(Y_labels_list)}")
        lengths_match_final = False;
        continue
    for k in range(len(X_df_list)):
        df_x, arr_y, id_dict_k = X_df_list[k], Y_labels_list[k], ids_list[k]
        key_display = f"{id_dict_k.get('participant', '?')}::{id_dict_k.get('filename', '?')}"
        if df_x is None or arr_y is None:
            lengths_match_final = False;
            error_sequences_final_check.append(f"{key_display}_None");
            continue
        if df_x.shape[0] != arr_y.shape[0]:
            print(f"FATAL LENGTH MISMATCH '{key_display}' (Split {split_name}): X {df_x.shape[0]}, Y {arr_y.shape[0]}")
            lengths_match_final = False;
            error_sequences_final_check.append(key_display + "_LenMismatch")
if lengths_match_final and total_labels_assigned > 0:
    if total_labels_assigned != total_ids_from_combined_pkl_check:
        print(
            f"Warning: Total assigned labels ({total_labels_assigned}) != total PKL IDs ({total_ids_from_combined_pkl_check}).")
    print(f"\nSaving final B-I-O annotation lists to: {OUTPUT_ANNOTATION_BIO_PKL}")
    try:
        with open(OUTPUT_ANNOTATION_BIO_PKL, 'wb') as f:
            pickle.dump(y_splits_bio, f)
        print("B-I-O annotation lists saved successfully.")
    except Exception as e:
        print(f"Error saving final B-I-O PKL: {e}")
else:
    print(f"\nAnnotations not saved. FinalMatch: {lengths_match_final}, LabelsAssigned: {total_labels_assigned}")
    if error_sequences_final_check: print("Issues:", sorted(list(set(error_sequences_final_check))))
print("\n--- Annotation for Final Combined Data Finished ---")