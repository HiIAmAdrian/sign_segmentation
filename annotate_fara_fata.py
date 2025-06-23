import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import traceback
import re

INPUT_FEATURES_DATA_DIR = Path("./final_data_ts_gloves_only")
FEATURES_PKL_FOR_ANNOTATION = INPUT_FEATURES_DATA_DIR / "all_data_features_ts_gloves_only.pkl"
OUTPUT_ANNOTATION_BIO_PKL = INPUT_FEATURES_DATA_DIR / "annotations_bio_ts_gloves_only.pkl"

PARTICIPANT_INFO_CONFIG = {
    "participant_p1_data": {
        "base_dir_path": Path("D:/SegmentationThesis/output_realsense60fps+tesla p1"),
    },
    "participant_p2_data": {
        "base_dir_path": Path("D:/SegmentationThesis/output_realsense60fps+tesla p2"),
    },
}
EAF_SUBDIR_NAME = "elan_annotations"
TARGET_TIER_ID = "default"
SYNC_OFFSET_MS = 0
LABEL_O, LABEL_B, LABEL_I = 0, 1, 2


def parse_eaf(eaf_path, target_tier_id="default"):
    segments = []
    try:
        tree = ET.parse(eaf_path);
        root = tree.getroot()
        time_slots = {slot.attrib['TIME_SLOT_ID']: int(slot.attrib['TIME_VALUE']) for slot in
                      root.findall(".//TIME_SLOT")}
        tier_found = False
        for tier in root.findall(f".//TIER"):
            if tier.attrib.get('TIER_ID') == target_tier_id:
                tier_found = True
                for ann in tier.findall(".//ALIGNABLE_ANNOTATION"):
                    t1, t2 = ann.attrib['TIME_SLOT_REF1'], ann.attrib['TIME_SLOT_REF2']
                    if t1 in time_slots and t2 in time_slots: segments.append((time_slots[t1], time_slots[t2]))
                break
        if not tier_found:
            return []
        segments.sort();
        return segments
    except ET.ParseError as pe:
        print(f"    WARNING: XML ParseError for EAF {eaf_path.name}: {pe}")
        return None
    except Exception as e:
        print(f"    WARNING: General error parsing EAF {eaf_path.name}: {e}")
        return None


print(f"Loading features data from: {FEATURES_PKL_FOR_ANNOTATION}")
if not FEATURES_PKL_FOR_ANNOTATION.exists(): print(f"FATAL: PKL not found: {FEATURES_PKL_FOR_ANNOTATION}."); exit()
try:
    with open(FEATURES_PKL_FOR_ANNOTATION, 'rb') as f:
        loaded_features_data = pickle.load(f)
except Exception as e:
    print(f"FATAL: Error loading PKL: {e}"); traceback.print_exc(); exit()

X_train_df_indexed = loaded_features_data.get('X_train_df_indexed', [])
train_ids = loaded_features_data.get('train_ids', [])
X_val_df_indexed = loaded_features_data.get('X_val_df_indexed', [])
val_ids = loaded_features_data.get('val_ids', [])
X_test_df_indexed = loaded_features_data.get('X_test_df_indexed', [])
test_ids = loaded_features_data.get('test_ids', [])

y_splits_bio = {'train': [], 'val': [], 'test': []}
processed_count = 0;
error_details = []

datasets_for_annotation = {
    "train": (X_train_df_indexed, train_ids),
    "val": (X_val_df_indexed, val_ids),
    "test": (X_test_df_indexed, test_ids),
}
print(f"\n--- Processing EAF annotations for TS+Gloves data ---")

for split_name, (X_df_list, ids_list) in datasets_for_annotation.items():
    print(f"\n  -- Annotating '{split_name}' set ({len(ids_list)} sequences) --")
    if not X_df_list or not ids_list or len(X_df_list) != len(ids_list):
        print(f"    Skipping '{split_name}' due to empty data/IDs or length mismatch.");
        continue

    for i, df_seq in enumerate(X_df_list):
        current_id_dict = ids_list[i]
        original_mocap_filename = current_id_dict['filename']
        participant_name_from_id_raw = current_id_dict.get('participant')
        trim_seconds_applied = current_id_dict.get('trim_seconds_applied', 0.0)
        if trim_seconds_applied is None: trim_seconds_applied = 0.0

        global_key_log = f"{participant_name_from_id_raw}::{original_mocap_filename}"

        if df_seq is None or df_seq.empty or not isinstance(df_seq.index, pd.TimedeltaIndex):
            error_details.append({'k': global_key_log, 'r': "DF empty or no TimedeltaIndex", 's': split_name})
            y_splits_bio[split_name].append(
                np.array([], dtype=int) if df_seq is None else np.full(df_seq.shape[0], LABEL_O, dtype=int))
            continue

        num_frames_seq = df_seq.shape[0]
        seq_timedelta_index = df_seq.index

        simplified_participant_name_for_config = None
        if participant_name_from_id_raw:
            if "p1" in participant_name_from_id_raw.lower():
                simplified_participant_name_for_config = "p1"
            elif "p2" in participant_name_from_id_raw.lower():
                simplified_participant_name_for_config = "p2"

        participant_config_key_found = None
        if simplified_participant_name_for_config:
            expected_config_key = f"participant_{simplified_participant_name_for_config}_data"
            if expected_config_key in PARTICIPANT_INFO_CONFIG:
                participant_config_key_found = expected_config_key

        if not participant_config_key_found:
            error_details.append({'k': global_key_log,
                                  'r': f"No PConfig for '{participant_name_from_id_raw}' (simplified: '{simplified_participant_name_for_config}')",
                                  's': split_name})
            y_splits_bio[split_name].append(np.full(num_frames_seq, LABEL_O, dtype=int));
            continue

        p_conf = PARTICIPANT_INFO_CONFIG[participant_config_key_found]
        p_base_dir = p_conf["base_dir_path"]

        eaf_match = re.match(r"(sentence_\d+_ts)_suit_mocap\.csv", original_mocap_filename, re.IGNORECASE)
        if not eaf_match:
            error_details.append(
                {'k': global_key_log, 'r': f"No base name from {original_mocap_filename}", 's': split_name})
            y_splits_bio[split_name].append(np.full(num_frames_seq, LABEL_O, dtype=int));
            continue

        eaf_base_name_for_file = eaf_match.group(1).replace('_ts', '')
        eaf_path = p_base_dir / (EAF_SUBDIR_NAME or "") / f"{eaf_base_name_for_file}_realsense.eaf"

        if not eaf_path.exists():
            error_details.append({'k': global_key_log, 'r': "EAF not found", 'p': str(eaf_path), 's': split_name})
            y_splits_bio[split_name].append(np.full(num_frames_seq, LABEL_O, dtype=int));
            continue

        video_segments_ms = parse_eaf(eaf_path, TARGET_TIER_ID)
        if video_segments_ms is None:
            error_details.append({'k': global_key_log, 'r': f"EAF parse error: {eaf_path.name}", 's': split_name})
            y_splits_bio[split_name].append(np.full(num_frames_seq, LABEL_O, dtype=int));
            continue

        labels_bio = np.full(num_frames_seq, LABEL_O, dtype=int)
        trim_offset_us_for_seq = trim_seconds_applied * 1_000_000

        if video_segments_ms:
            for seg_idx, (video_start_ms, video_end_ms) in enumerate(video_segments_ms):
                eaf_start_sync_us = (video_start_ms + SYNC_OFFSET_MS) * 1000
                eaf_end_sync_us = (video_end_ms + SYNC_OFFSET_MS) * 1000
                target_start_lookup_us = max(0, eaf_start_sync_us - trim_offset_us_for_seq)
                target_end_lookup_us = max(0, eaf_end_sync_us - trim_offset_us_for_seq)
                if target_end_lookup_us <= target_start_lookup_us: continue
                start_frame_idx = seq_timedelta_index.searchsorted(pd.Timedelta(microseconds=target_start_lookup_us),
                                                                   side='left')
                end_frame_idx = seq_timedelta_index.searchsorted(pd.Timedelta(microseconds=target_end_lookup_us),
                                                                 side='right') - 1
                actual_sf = max(0, min(start_frame_idx, num_frames_seq - 1))
                actual_ef = max(0, min(end_frame_idx, num_frames_seq - 1))
                if actual_sf <= actual_ef:
                    labels_bio[actual_sf] = LABEL_B
                    if actual_sf < actual_ef:
                        labels_bio[actual_sf + 1: actual_ef + 1] = LABEL_I

        y_splits_bio[split_name].append(labels_bio)
        processed_count += 1

print(f"\n--- Summary of Annotation (TS+Gloves) ---")
print(f"Generated annotations for {processed_count} sequences.")
if error_details: print(f"Errors/Skips for {len(error_details)} sequences. First 5 details:"); [print(f"  - {err}") for
                                                                                                err in
                                                                                                error_details[:5]]
total_labels = sum(len(v) for v_list in y_splits_bio.values() for v in v_list)
total_sequences_with_labels = sum(len(v_list) for v_list in y_splits_bio.values())
print(f"Total B-I-O label arrays assigned: {total_sequences_with_labels}")
if total_sequences_with_labels > 0:
    try:
        with open(OUTPUT_ANNOTATION_BIO_PKL, 'wb') as f:
            pickle.dump(y_splits_bio, f)
        print(f"Saved final B-I-O (TS+Gloves) annotation lists to: {OUTPUT_ANNOTATION_BIO_PKL}")
    except Exception as e:
        print(f"Error saving final B-I-O PKL: {e}")
else:
    print("Annotations not saved as no labels were generated or all sequences had errors.")
print("\n--- Annotation for TS+Gloves Data Finished ---")