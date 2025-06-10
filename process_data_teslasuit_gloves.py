import pandas as pd
import numpy as np
# from sklearn.preprocessing import StandardScaler # Eliminat dacă nu se face scalare aici
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
import traceback
import re

# --- Configuration ---
PARTICIPANT_BASE_DIRS_WITH_TRIM = {
    Path("D:\SegmentationThesis\output_realsense60fps+tesla Catalin"): 1.0,  # Cale -> trim_sec
    Path("D:\SegmentationThesis\output_realsense60fps+tesla Marinela"): 0.3,
}
OUTPUT_DIR = Path("./processed_combined_data_all_participants_TESLASUIT_DF_trimmed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ... (Restul flag-urilor USE_... și Feature List Generation rămân la fel) ...
USE_SUIT_ROTATIONS = True
USE_SUIT_POSITIONS = False
USE_SUIT_HIPS_POSITION = True
USE_SUIT_BIOMECH = True

USE_GLOVE_RIGHT_FINGER_ROTATIONS = True
USE_GLOVE_RIGHT_FINGER_POSITIONS = False
USE_GLOVE_RIGHT_HAND_ROOT = True

USE_GLOVE_LEFT_FINGER_ROTATIONS = True
USE_GLOVE_LEFT_FINGER_POSITIONS = False
USE_GLOVE_LEFT_HAND_ROOT = True

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
RANDOM_STATE = 42
MERGE_TOLERANCE_MS = 30

SUIT_FEATURES_TO_KEEP = []
selected_suit_bone_rotations = []
selected_suit_bone_positions = []
selected_suit_hip_position = []
relevant_biomech_joints = []

upper_body_bones_suit = [
    "hips", "spine", "upper_spine", "neck", "head",
    "left_shoulder", "right_shoulder",
    "left_upper_arm", "right_upper_arm",
    "left_lower_arm", "right_lower_arm"
]

if USE_SUIT_ROTATIONS:
    for bone in upper_body_bones_suit:
        if bone == "left_hand" and USE_GLOVE_LEFT_HAND_ROOT: continue
        if bone == "right_hand" and USE_GLOVE_RIGHT_HAND_ROOT: continue
        selected_suit_bone_rotations.extend([f"{bone}.rotation.{axis}" for axis in ['w', 'x', 'y', 'z']])
    SUIT_FEATURES_TO_KEEP.extend(selected_suit_bone_rotations)

if USE_SUIT_POSITIONS:
    for bone in upper_body_bones_suit:
        if bone == "hips": continue
        if bone == "left_hand" and USE_GLOVE_LEFT_HAND_ROOT: continue
        if bone == "right_hand" and USE_GLOVE_RIGHT_HAND_ROOT: continue
        selected_suit_bone_positions.extend([f"{bone}.position.{axis}" for axis in ['x', 'y', 'z']])
    SUIT_FEATURES_TO_KEEP.extend(selected_suit_bone_positions)

if USE_SUIT_HIPS_POSITION:
    selected_suit_hip_position.extend([f"hips.position.{axis}" for axis in ['x', 'y', 'z']])
    SUIT_FEATURES_TO_KEEP.extend(selected_suit_hip_position)

if USE_SUIT_BIOMECH:
    relevant_biomech_joints = [
        "PelvisTilt", "PelvisList", "PelvisRotation",
        "HipFlexExtR", "HipFlexExtL", "HipAddAbdR", "HipAddAbdL", "HipRotR", "HipRotL",
        "ElbowFlexExtR", "ElbowFlexExtL", "ForearmProSupR", "ForearmProSupL",
        "WristFlexExtR", "WristFlexExtL", "WristDeviationR", "WristDeviationL",
        "LumbarFlexExt", "LumbarLatFlex", "LumbarRot",
        "LowerThoraxFlexExt", "LowerThoraxLatFlex", "LowerThoraxRot",
        "UpperThoraxFlexExt", "UpperThoraxLatFlex", "UpperThoraxRot",
        "ShoulderFlexExtR", "ShoulderFlexExtL", "ShoulderAddAbdR", "ShoulderAddAbdL",
        "ShoulderRotR", "ShoulderRotL"
    ]
    selected_suit_biomech_features_angles = [f"{joint}.angle" for joint in relevant_biomech_joints]
    SUIT_FEATURES_TO_KEEP.extend(selected_suit_biomech_features_angles)

GLOVE_RIGHT_FEATURES_TO_KEEP = []
GLOVE_LEFT_FEATURES_TO_KEEP = []


def get_finger_features_configurable(hand_prefix, use_rotations, use_positions, use_hand_root):
    features = []
    finger_segments = ["thumb", "index", "middle", "ring", "little"]
    phalanges = ["proximal", "intermediate", "distal"]

    if use_rotations:
        for finger in finger_segments:
            for phalanx in phalanges:
                if finger == "thumb" and phalanx == "intermediate":
                    continue
                features.extend([f"{hand_prefix}_{finger}_{phalanx}.rotation.{axis}" for axis in ['w', 'x', 'y', 'z']])

    if use_positions:
        for finger in finger_segments:
            for phalanx in phalanges:
                if finger == "thumb" and phalanx == "intermediate": continue
                features.extend([f"{hand_prefix}_{finger}_{phalanx}.position.{axis}" for axis in ['x', 'y', 'z']])

    if use_hand_root:
        features.extend([f"{hand_prefix}_hand.rotation.{axis}" for axis in ['w', 'x', 'y', 'z']])
        features.extend([f"{hand_prefix}_hand.position.{axis}" for axis in ['x', 'y', 'z']])

    return sorted(list(set(features)))


if USE_GLOVE_RIGHT_FINGER_ROTATIONS or USE_GLOVE_RIGHT_FINGER_POSITIONS or USE_GLOVE_RIGHT_HAND_ROOT:
    GLOVE_RIGHT_FEATURES_TO_KEEP = get_finger_features_configurable(
        "right", USE_GLOVE_RIGHT_FINGER_ROTATIONS, USE_GLOVE_RIGHT_FINGER_POSITIONS, USE_GLOVE_RIGHT_HAND_ROOT
    )

if USE_GLOVE_LEFT_FINGER_ROTATIONS or USE_GLOVE_LEFT_FINGER_POSITIONS or USE_GLOVE_LEFT_HAND_ROOT:
    GLOVE_LEFT_FEATURES_TO_KEEP = get_finger_features_configurable(
        "left", USE_GLOVE_LEFT_FINGER_ROTATIONS, USE_GLOVE_LEFT_FINGER_POSITIONS, USE_GLOVE_LEFT_HAND_ROOT
    )

SUIT_FEATURES_TO_KEEP = sorted(list(set(SUIT_FEATURES_TO_KEEP)))


def get_related_files_for_participant(suit_file_path, participant_dir):
    base_name_match = re.match(r"(sentence_\d+_ts)_suit_mocap\.csv", suit_file_path.name, re.IGNORECASE)
    if not base_name_match: return None, None
    base_name_prefix = base_name_match.group(1)
    glove_r_file = participant_dir / f"{base_name_prefix}_glove_R_mocap.csv"
    glove_l_file = participant_dir / f"{base_name_prefix}_glove_L_mocap.csv"
    return glove_r_file, glove_l_file


def load_and_prepare_df(csv_path, selected_features, filename_for_log, trim_start_sec=0.0):
    timestamp_col_name = 'frame_timestamp_us'
    try:
        df = pd.read_csv(csv_path)
        if df.empty or timestamp_col_name not in df.columns:
            return None, []

        df[timestamp_col_name] = pd.to_numeric(df[timestamp_col_name], errors='coerce')
        df.dropna(subset=[timestamp_col_name], inplace=True)
        if df.empty:
            return None, []

        if trim_start_sec > 0:
            trim_start_us = trim_start_sec * 1_000_000
            df = df[df[timestamp_col_name] >= trim_start_us].copy()  # .copy() explicit
            if df.empty:
                return None, []

        actual_selected_features = [f for f in selected_features if f in df.columns]
        cols_to_load_final = [timestamp_col_name] + actual_selected_features
        df = df[cols_to_load_final]

        df[timestamp_col_name] = pd.to_timedelta(df[timestamp_col_name], unit='us', errors='coerce')
        df.dropna(subset=[timestamp_col_name], inplace=True)
        if df.empty:
            return None, []

        df = df.sort_values(by=timestamp_col_name)
        if not df.empty:
            first_ts = df[timestamp_col_name].iloc[0]
            df[timestamp_col_name] = df[timestamp_col_name] - first_ts
        else:
            return None, []

        df = df.set_index(timestamp_col_name)
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep='first')]

        for col in actual_selected_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        return df, actual_selected_features

    except FileNotFoundError:
        return None, []
    except Exception as e:
        return None, []


def calculate_biomech_derivatives(df, biomech_angle_joints_list):
    if df.empty or not isinstance(df.index, pd.TimedeltaIndex):
        return df

    df_out = df.copy()

    delta_time_sec_series = df_out.index.to_series().diff().dt.total_seconds()
    min_time_delta = 1e-7

    delta_time_sec_processed = np.where(delta_time_sec_series < min_time_delta, np.nan, delta_time_sec_series)
    delta_time_sec_processed = pd.Series(delta_time_sec_processed, index=df_out.index)

    delta_time_sec_processed = delta_time_sec_processed.ffill().bfill()
    delta_time_sec_processed.fillna(1.0, inplace=True)

    new_columns_data = {}

    for joint_base_name in biomech_angle_joints_list:
        angle_col = f"{joint_base_name}.angle"
        vel_col = f"{joint_base_name}.angular_v"
        acc_col = f"{joint_base_name}.angular_acc"

        if angle_col in df_out.columns:
            velocity_series = df_out[angle_col].diff() / delta_time_sec_processed
            new_columns_data[vel_col] = velocity_series.fillna(0)

            acceleration_series = new_columns_data[vel_col].diff() / delta_time_sec_processed
            new_columns_data[acc_col] = acceleration_series.fillna(0)
        else:
            new_columns_data[vel_col] = pd.Series(0.0, index=df_out.index, name=vel_col)
            new_columns_data[acc_col] = pd.Series(0.0, index=df_out.index, name=acc_col)

    if new_columns_data:
        new_df_part = pd.DataFrame(new_columns_data, index=df_out.index)
        if not new_df_part.empty:
            df_out = df_out.assign(**new_df_part)

    return df_out


all_processed_dfs_temp = []
file_identifiers_temp = []
collected_all_feature_names = set()

for participant_base_dir, trim_seconds_for_participant in PARTICIPANT_BASE_DIRS_WITH_TRIM.items():
    participant_name_str = participant_base_dir.name
    print(f"\n--- Processing Participant: {participant_name_str} (Trim: {trim_seconds_for_participant}s) ---")
    if not participant_base_dir.exists():
        print(f"  Directory not found: {participant_base_dir}. Skipping.")
        continue

    suit_csv_files_participant = sorted(list(participant_base_dir.glob("sentence_*_ts_suit_mocap.csv")))
    if not suit_csv_files_participant:
        print(f"  No suit CSV files found for participant {participant_name_str}.")
        continue

    processed_count_participant, skipped_count_participant = 0, 0

    for suit_csv_path in suit_csv_files_participant:
        glove_r_path, glove_l_path = get_related_files_for_participant(suit_csv_path, participant_base_dir)

        if not (glove_r_path and glove_l_path and glove_r_path.exists() and glove_l_path.exists()):
            skipped_count_participant += 1
            continue

        try:
            df_suit, _ = load_and_prepare_df(suit_csv_path, SUIT_FEATURES_TO_KEEP, suit_csv_path.name,
                                             trim_seconds_for_participant)
            if df_suit is None or df_suit.empty:
                skipped_count_participant += 1;
                continue

            if USE_SUIT_BIOMECH and relevant_biomech_joints:
                df_suit = calculate_biomech_derivatives(df_suit, relevant_biomech_joints)

            df_glove_r, _ = load_and_prepare_df(glove_r_path, GLOVE_RIGHT_FEATURES_TO_KEEP, glove_r_path.name,
                                                trim_seconds_for_participant)
            if df_glove_r is None or df_glove_r.empty:
                df_glove_r = pd.DataFrame(
                    index=df_suit.index if not df_suit.empty else pd.Index([], dtype='timedelta64[ns]',
                                                                           name='frame_timestamp_us'))

            df_glove_l, _ = load_and_prepare_df(glove_l_path, GLOVE_LEFT_FEATURES_TO_KEEP, glove_l_path.name,
                                                trim_seconds_for_participant)
            if df_glove_l is None or df_glove_l.empty:
                df_glove_l = pd.DataFrame(
                    index=df_suit.index if not df_suit.empty else pd.Index([], dtype='timedelta64[ns]',
                                                                           name='frame_timestamp_us'))

            merge_col_name = 'timestamp_for_merge'
            df_s_reset = df_suit.reset_index().rename(columns={df_suit.index.name or 'index': merge_col_name})
            df_gr_reset = df_glove_r.reset_index().rename(columns={df_glove_r.index.name or 'index': merge_col_name})
            df_gl_reset = df_glove_l.reset_index().rename(columns={df_glove_l.index.name or 'index': merge_col_name})

            if merge_col_name not in df_gr_reset.columns and not df_glove_r.empty:
                df_gr_reset[merge_col_name] = df_glove_r.index.to_series()
            if merge_col_name not in df_gl_reset.columns and not df_glove_l.empty:
                df_gl_reset[merge_col_name] = df_glove_l.index.to_series()

            if df_gr_reset.empty and merge_col_name not in df_gr_reset.columns:
                df_gr_reset = pd.DataFrame({merge_col_name: pd.Series(dtype='timedelta64[ns]')})
            if df_gl_reset.empty and merge_col_name not in df_gl_reset.columns:
                df_gl_reset = pd.DataFrame({merge_col_name: pd.Series(dtype='timedelta64[ns]')})

            df_merged_sr = pd.merge_asof(df_s_reset.sort_values(merge_col_name),
                                         df_gr_reset.sort_values(merge_col_name),
                                         on=merge_col_name,
                                         direction='nearest',
                                         tolerance=pd.Timedelta(microseconds=MERGE_TOLERANCE_MS * 1000),
                                         suffixes=('_s', '_gr'))

            df_combined_temp = pd.merge_asof(df_merged_sr.sort_values(merge_col_name),
                                             df_gl_reset.sort_values(merge_col_name),
                                             on=merge_col_name,
                                             direction='nearest',
                                             tolerance=pd.Timedelta(microseconds=MERGE_TOLERANCE_MS * 1000),
                                             suffixes=('_sr', '_gl'))

            if merge_col_name in df_combined_temp.columns:
                df_combined = df_combined_temp.set_index(merge_col_name).sort_index()
                df_combined.index.name = 'frame_timestamp_us'
            else:
                skipped_count_participant += 1;
                continue

            cols_to_interpolate_gloves = []
            for col_list in [GLOVE_RIGHT_FEATURES_TO_KEEP, GLOVE_LEFT_FEATURES_TO_KEEP]:
                for feat_name in col_list:
                    if feat_name in df_combined.columns:
                        cols_to_interpolate_gloves.append(feat_name)

            if cols_to_interpolate_gloves and df_combined[cols_to_interpolate_gloves].isnull().values.any():
                df_combined[cols_to_interpolate_gloves] = df_combined[cols_to_interpolate_gloves].interpolate(
                    method='time').ffill().bfill().fillna(0)

            if df_combined.empty:
                skipped_count_participant += 1;
                continue

            collected_all_feature_names.update(df_combined.columns)

            all_processed_dfs_temp.append(df_combined)
            file_identifiers_temp.append({'filename': suit_csv_path.name, 'participant': participant_name_str})
            processed_count_participant += 1

        except Exception as e:
            skipped_count_participant += 1

    print(
        f"  Participant {participant_name_str}: Processed {processed_count_participant}, Skipped {skipped_count_participant} triplete.")

final_combined_feature_names = sorted(list(collected_all_feature_names))
print(f"\n--- Standardizing all DataFrames to {len(final_combined_feature_names)} features ---")

all_combined_dataframes_standardized = []
if final_combined_feature_names:
    for df_orig in all_processed_dfs_temp:
        df_std = df_orig.reindex(columns=final_combined_feature_names, fill_value=0.0)
        all_combined_dataframes_standardized.append(df_std)
else:
    all_combined_dataframes_standardized = list(all_processed_dfs_temp)

print(f"\n--- Consolidating and Deduplicating Processed Data ---")
unique_data_map = {}
for i, identifier_dict in enumerate(file_identifiers_temp):
    participant_name_for_key = identifier_dict['participant']
    original_filename_for_key = identifier_dict['filename']
    global_unique_key = f"{participant_name_for_key}::{original_filename_for_key}"

    if global_unique_key in unique_data_map:
        print(
            f"  WARNING: Duplicate global key '{global_unique_key}' encountered. Keeping the last processed instance.")

    unique_data_map[global_unique_key] = {
        'df': all_combined_dataframes_standardized[i],
        'id': identifier_dict
    }

all_combined_dataframes_global = []
file_identifiers_global = []
sorted_global_unique_keys = sorted(list(unique_data_map.keys()))

for g_key in sorted_global_unique_keys:
    data_entry = unique_data_map[g_key]
    all_combined_dataframes_global.append(data_entry['df'])
    file_identifiers_global.append(data_entry['id'])

print(f"Total unique sequences after deduplication: {len(all_combined_dataframes_global)}")

print(f"\n--- Global Post-processing ---")
if not all_combined_dataframes_global:
    print("Error: No data available after deduplication. Exiting.");
    exit()
if not final_combined_feature_names:
    if all_combined_dataframes_global and not all_combined_dataframes_global[0].empty:
        final_combined_feature_names = sorted(list(all_combined_dataframes_global[0].columns))
    else:
        print("Error: No feature names collected or derivable. Exiting.");
        exit()

indices_global = list(range(len(all_combined_dataframes_global)))
num_total_samples = len(indices_global)

# Recalculare target-uri pentru split
train_n_target = num_total_samples
test_n_target = 0
if TEST_SIZE > 0 and num_total_samples > 0:
    test_n_target = max(1, int(round(num_total_samples * TEST_SIZE)))
train_n_target = max(0, train_n_target - test_n_target)  # Ce rămâne pentru train+val

val_n_target = 0
if VALIDATION_SIZE > 0 and train_n_target > 0:  # Dacă mai e ceva pentru train+val
    # VALIDATION_SIZE e procent din totalul inițial
    potential_val_n = max(1, int(round(num_total_samples * VALIDATION_SIZE)))
    val_n_target = min(potential_val_n, train_n_target)  # Nu poate fi mai mare decât ce a rămas
train_n_target = max(0, train_n_target - val_n_target)  # Ce rămâne efectiv pentru train

# Ajustări finale pentru a asigura că train are cel puțin 1 dacă e posibil
if train_n_target < 1 and num_total_samples > 0:
    if val_n_target > 0:  # Încearcă să iei din validare
        val_n_target -= 1
        train_n_target += 1
    elif test_n_target > 0 and (num_total_samples - (test_n_target - 1)) >= 1:  # Încearcă să iei din test
        test_n_target -= 1
        train_n_target += 1
    # Dacă train_n_target tot e < 1, înseamnă că num_total_samples e prea mic (0 sau 1)

# Asigură că suma nu depășește totalul și că nu sunt negative
if train_n_target + val_n_target + test_n_target > num_total_samples:
    # Prioritizează test, apoi val. Ajustează train.
    train_n_target = num_total_samples - val_n_target - test_n_target
if train_n_target < 0: train_n_target = 0
if val_n_target < 0: val_n_target = 0
if test_n_target < 0: test_n_target = 0

# Condiție minimă pentru a continua
min_req_samples = 0
if train_n_target > 0: min_req_samples += 1
if val_n_target > 0: min_req_samples += 1
if test_n_target > 0: min_req_samples += 1

if num_total_samples < min_req_samples and num_total_samples > 0:  # Dacă avem mai puțin decât suma componentelor dorite
    # Această logică poate deveni complexă. O simplificare ar fi:
    # Dacă num_total_samples e 1, totul e train.
    # Dacă e 2, și vrem test, 1 test, 1 train. Dacă vrem val, 1 val, 1 train.
    # Dacă e 3, și vrem toate, 1, 1, 1.
    # Pentru scopul actual, vom lăsa verificarea de mai jos să prindă erori.
    print(
        f"Warning: Number of samples ({num_total_samples}) might be too small for desired splits (T:{train_n_target}, V:{val_n_target}, Te:{test_n_target}).")

if train_n_target == 0 and num_total_samples > 0:
    print(
        f"Error: Training set size is 0 with {num_total_samples} total samples. Check split logic/percentages. Exiting.")
    exit()

print(
    f"Calculated split sizes: Total={num_total_samples}, Train={train_n_target}, Val={val_n_target}, Test={test_n_target}")

train_idx, val_idx, test_idx = [], [], []
if num_total_samples > 0:
    if test_n_target > 0:
        test_split_fraction = test_n_target / num_total_samples
        # Asigură-te că fracția e validă (între 0 și 1)
        test_split_fraction = max(0.0, min(test_split_fraction, 1.0))
        if test_split_fraction > 0 and test_split_fraction < 1.0:  # Doar dacă e un split real
            train_val_indices, test_idx_temp = train_test_split(indices_global, test_size=test_split_fraction,
                                                                random_state=RANDOM_STATE, shuffle=True)
            test_idx.extend(test_idx_temp)
        elif test_split_fraction == 1.0:  # Totul merge la test
            test_idx.extend(indices_global)
            train_val_indices = []
        else:  # test_split_fraction == 0.0
            train_val_indices = list(indices_global)
    else:
        train_val_indices = list(indices_global)

    if val_n_target > 0 and len(train_val_indices) > 0:
        if len(train_val_indices) <= val_n_target:
            val_idx_temp = list(train_val_indices)
            train_idx_temp = []
        else:
            val_split_fraction = val_n_target / len(train_val_indices)
            val_split_fraction = max(0.0, min(val_split_fraction, 1.0))
            if val_split_fraction > 0 and val_split_fraction < 1.0:
                train_idx_temp, val_idx_temp = train_test_split(train_val_indices, test_size=val_split_fraction,
                                                                random_state=RANDOM_STATE, shuffle=True)
            elif val_split_fraction == 1.0:
                val_idx_temp = list(train_val_indices)
                train_idx_temp = []
            else:  # val_split_fraction == 0.0
                train_idx_temp = list(train_val_indices)
                val_idx_temp = []

        val_idx.extend(val_idx_temp)
        train_idx.extend(train_idx_temp)
    elif len(train_val_indices) > 0:
        train_idx.extend(train_val_indices)

X_train_df_raw = [all_combined_dataframes_global[i] for i in train_idx] if train_idx else []
X_val_df_raw = [all_combined_dataframes_global[i] for i in val_idx] if val_idx else []
X_test_df_raw = [all_combined_dataframes_global[i] for i in test_idx] if test_idx else []
train_ids = [file_identifiers_global[i] for i in train_idx] if train_idx else []
val_ids = [file_identifiers_global[i] for i in val_idx] if val_idx else []
test_ids = [file_identifiers_global[i] for i in test_idx] if test_idx else []

if len(train_ids) + len(val_ids) + len(test_ids) != num_total_samples and num_total_samples > 0:
    print(
        f"CRITICAL ERROR: Sum of ID list lengths ({len(train_ids) + len(val_ids) + len(test_ids)}) does not match total unique sequences ({num_total_samples}) after split!")

print(
    f"Global data split (DataFrames): Train={len(X_train_df_raw)}, Val={len(X_val_df_raw)}, Test={len(X_test_df_raw)}")

processed_data_path_global = OUTPUT_DIR / "combined_all_participants_sequences_DF.pkl"
data_to_save_global = {
    'X_train_df': X_train_df_raw,
    'X_val_df': X_val_df_raw,
    'X_test_df': X_test_df_raw,
    'train_ids': train_ids,
    'val_ids': val_ids,
    'test_ids': test_ids,
    'feature_names': final_combined_feature_names
}
with open(processed_data_path_global, 'wb') as f: pickle.dump(data_to_save_global, f)
print(f"\nSaved final processed DataFrames for all participants to: {processed_data_path_global}")
print("\n--- TeslaSuit Data Processing (DataFrame Output with Trim) Finished ---")