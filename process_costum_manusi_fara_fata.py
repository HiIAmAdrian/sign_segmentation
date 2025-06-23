import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
import traceback
import re

PARTICIPANT_BASE_DIRS_WITH_TRIM = {
    Path("D:\SegmentationThesis\output_realsense60fps+tesla p1"): 1.0,
    Path("D:\SegmentationThesis\output_realsense60fps+tesla p2"): 0.3,
}
OUTPUT_DATA_DIR = Path("./final_data_ts_gloves_only")
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
FINAL_FEATURES_PKL_PATH = OUTPUT_DATA_DIR / "all_data_features_ts_gloves_only.pkl"
FINAL_SCALER_PKL_PATH = OUTPUT_DATA_DIR / "ts_gloves_only_scaler.pkl"

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
upper_body_bones_suit = ["hips", "spine", "upper_spine", "neck", "head", "left_shoulder", "right_shoulder",
                         "left_upper_arm", "right_upper_arm", "left_lower_arm", "right_lower_arm"]
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
    relevant_biomech_joints = ["PelvisTilt", "PelvisList", "PelvisRotation", "HipFlexExtR", "HipFlexExtL", "HipAddAbdR",
                               "HipAddAbdL", "HipRotR", "HipRotL", "ElbowFlexExtR", "ElbowFlexExtL", "ForearmProSupR",
                               "ForearmProSupL", "WristFlexExtR", "WristFlexExtL", "WristDeviationR", "WristDeviationL",
                               "LumbarFlexExt", "LumbarLatFlex", "LumbarRot", "LowerThoraxFlexExt",
                               "LowerThoraxLatFlex", "LowerThoraxRot", "UpperThoraxFlexExt", "UpperThoraxLatFlex",
                               "UpperThoraxRot", "ShoulderFlexExtR", "ShoulderFlexExtL", "ShoulderAddAbdR",
                               "ShoulderAddAbdL", "ShoulderRotR", "ShoulderRotL"]
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
                if finger == "thumb" and phalanx == "intermediate": continue
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
    GLOVE_RIGHT_FEATURES_TO_KEEP = get_finger_features_configurable("right", USE_GLOVE_RIGHT_FINGER_ROTATIONS,
                                                                    USE_GLOVE_RIGHT_FINGER_POSITIONS,
                                                                    USE_GLOVE_RIGHT_HAND_ROOT)
if USE_GLOVE_LEFT_FINGER_ROTATIONS or USE_GLOVE_LEFT_FINGER_POSITIONS or USE_GLOVE_LEFT_HAND_ROOT:
    GLOVE_LEFT_FEATURES_TO_KEEP = get_finger_features_configurable("left", USE_GLOVE_LEFT_FINGER_ROTATIONS,
                                                                   USE_GLOVE_LEFT_FINGER_POSITIONS,
                                                                   USE_GLOVE_LEFT_HAND_ROOT)
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
        if df.empty or timestamp_col_name not in df.columns: return None, []
        df[timestamp_col_name] = pd.to_numeric(df[timestamp_col_name], errors='coerce')
        df.dropna(subset=[timestamp_col_name], inplace=True)
        if df.empty: return None, []
        if trim_start_sec > 0:
            df = df[df[timestamp_col_name] >= trim_start_sec * 1_000_000].copy()
            if df.empty: return None, []
        actual_selected_features = [f for f in selected_features if f in df.columns]
        df = df[[timestamp_col_name] + actual_selected_features]
        df[timestamp_col_name] = pd.to_timedelta(df[timestamp_col_name], unit='us', errors='coerce')
        df.dropna(subset=[timestamp_col_name], inplace=True)
        if df.empty: return None, []
        df = df.sort_values(by=timestamp_col_name)
        if not df.empty:
            df[timestamp_col_name] = df[timestamp_col_name] - df[timestamp_col_name].iloc[0]
        else:
            return None, []
        df = df.set_index(timestamp_col_name)
        if df.index.has_duplicates: df = df[~df.index.duplicated(keep='first')]
        for col in actual_selected_features:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df, actual_selected_features
    except Exception:
        return None, []


def calculate_biomech_derivatives(df, biomech_angle_joints_list):
    if df.empty or not isinstance(df.index, pd.TimedeltaIndex): return df
    df_out = df.copy()
    delta_time_sec_series = df_out.index.to_series().diff().dt.total_seconds()
    delta_time_sec_processed = pd.Series(np.where(delta_time_sec_series < 1e-7, np.nan, delta_time_sec_series),
                                         index=df_out.index).ffill().bfill().fillna(1.0)
    new_cols_data = {}
    for joint_base_name in biomech_angle_joints_list:
        angle_col, vel_col, acc_col = f"{joint_base_name}.angle", f"{joint_base_name}.angular_v", f"{joint_base_name}.angular_acc"
        if angle_col in df_out.columns:
            new_cols_data[vel_col] = (df_out[angle_col].diff() / delta_time_sec_processed).fillna(0)
            new_cols_data[acc_col] = (new_cols_data[vel_col].diff() / delta_time_sec_processed).fillna(0)
        else:
            new_cols_data[vel_col] = pd.Series(0.0, index=df_out.index, name=vel_col)
            new_cols_data[acc_col] = pd.Series(0.0, index=df_out.index, name=acc_col)
    if new_cols_data: df_out = df_out.assign(**pd.DataFrame(new_cols_data, index=df_out.index))
    return df_out


all_processed_dfs_temp = []
file_identifiers_temp = []
collected_all_feature_names = set()

for participant_base_dir, trim_seconds_for_participant in PARTICIPANT_BASE_DIRS_WITH_TRIM.items():
    participant_name_str = participant_base_dir.name
    print(f"\n--- Processing Participant: {participant_name_str} (Trim: {trim_seconds_for_participant}s) ---")
    if not participant_base_dir.exists(): print(f"  Dir not found: {participant_base_dir}. Skip."); continue
    suit_csv_files = sorted(list(participant_base_dir.glob("sentence_*_ts_suit_mocap.csv")))
    if not suit_csv_files: print(f"  No suit CSVs for {participant_name_str}."); continue

    p_proc, p_skip = 0, 0
    for suit_csv_path in suit_csv_files:
        glove_r_path, glove_l_path = get_related_files_for_participant(suit_csv_path, participant_base_dir)
        if not (glove_r_path and glove_l_path and glove_r_path.exists() and glove_l_path.exists()):
            p_skip += 1;
            continue
        try:
            df_suit, _ = load_and_prepare_df(suit_csv_path, SUIT_FEATURES_TO_KEEP, suit_csv_path.name,
                                             trim_seconds_for_participant)
            if df_suit is None or df_suit.empty: p_skip += 1; continue
            if USE_SUIT_BIOMECH and relevant_biomech_joints:
                df_suit = calculate_biomech_derivatives(df_suit, relevant_biomech_joints)

            df_glove_r, _ = load_and_prepare_df(glove_r_path, GLOVE_RIGHT_FEATURES_TO_KEEP, glove_r_path.name,
                                                trim_seconds_for_participant)
            if df_glove_r is None or df_glove_r.empty: df_glove_r = pd.DataFrame(index=df_suit.index)
            df_glove_l, _ = load_and_prepare_df(glove_l_path, GLOVE_LEFT_FEATURES_TO_KEEP, glove_l_path.name,
                                                trim_seconds_for_participant)
            if df_glove_l is None or df_glove_l.empty: df_glove_l = pd.DataFrame(index=df_suit.index)

            merge_col = 'ts_merge'
            df_s_r = df_suit.reset_index().rename(columns={df_suit.index.name or 'index': merge_col})
            df_gr_r = df_glove_r.reset_index().rename(columns={df_glove_r.index.name or 'index': merge_col})
            df_gl_r = df_glove_l.reset_index().rename(columns={df_glove_l.index.name or 'index': merge_col})

            for df_reset in [df_gr_r, df_gl_r]:
                if df_reset.empty and merge_col not in df_reset.columns and not df_suit.empty:
                    df_reset[merge_col] = pd.Series(df_suit.index, name=merge_col)

            df_m_sr = pd.merge_asof(df_s_r.sort_values(merge_col), df_gr_r.sort_values(merge_col), on=merge_col,
                                    direction='nearest', tolerance=pd.Timedelta(microseconds=MERGE_TOLERANCE_MS * 1000))
            df_comb = pd.merge_asof(df_m_sr.sort_values(merge_col), df_gl_r.sort_values(merge_col), on=merge_col,
                                    direction='nearest', tolerance=pd.Timedelta(microseconds=MERGE_TOLERANCE_MS * 1000))

            if merge_col in df_comb.columns:
                df_comb = df_comb.set_index(merge_col).sort_index(); df_comb.index.name = 'frame_timestamp_us'
            else:
                p_skip += 1; continue

            glove_cols_to_interp = [f for f_list in [GLOVE_RIGHT_FEATURES_TO_KEEP, GLOVE_LEFT_FEATURES_TO_KEEP] for f in
                                    f_list if f in df_comb.columns]
            if glove_cols_to_interp and df_comb[glove_cols_to_interp].isnull().values.any():
                df_comb[glove_cols_to_interp] = df_comb[glove_cols_to_interp].interpolate(
                    method='time').ffill().bfill().fillna(0)

            if df_comb.empty: p_skip += 1; continue

            collected_all_feature_names.update(df_comb.columns)
            all_processed_dfs_temp.append(df_comb)
            file_identifiers_temp.append({'filename': suit_csv_path.name,
                                          'participant': participant_name_str,
                                          'trim_seconds_applied': trim_seconds_for_participant})
            p_proc += 1
        except Exception as e_file:
            p_skip += 1
    print(f"  Participant {participant_name_str}: Processed {p_proc}, Skipped {p_skip} triplets.")

final_combined_feature_names = sorted(list(collected_all_feature_names))
print(f"\n--- Standardizing all DFs to {len(final_combined_feature_names)} features ---")
all_combined_dfs_std = [df.reindex(columns=final_combined_feature_names, fill_value=0.0) for df in
                        all_processed_dfs_temp] if final_combined_feature_names else list(all_processed_dfs_temp)

unique_data_map = {}
for i, identifier_dict in enumerate(file_identifiers_temp):
    global_key = f"{identifier_dict['participant']}::{identifier_dict['filename']}"
    if global_key in unique_data_map: print(f"  WARNING: Duplicate global key '{global_key}'. Keeping last.")
    unique_data_map[global_key] = {'df': all_combined_dfs_std[i], 'id': identifier_dict}
all_combined_dataframes_global_df_unscaled = [unique_data_map[g_key]['df'] for g_key in sorted(unique_data_map.keys())]
file_identifiers_global = [unique_data_map[g_key]['id'] for g_key in sorted(unique_data_map.keys())]
print(f"Total unique sequences after deduplication: {len(all_combined_dataframes_global_df_unscaled)}")

if not all_combined_dataframes_global_df_unscaled: print("Error: No data after deduplication. Exiting."); exit()
if not final_combined_feature_names:
    if not all_combined_dataframes_global_df_unscaled[0].empty:
        final_combined_feature_names = sorted(list(all_combined_dataframes_global_df_unscaled[0].columns))
    else:
        print("Error: No feature names. Exiting."); exit()

indices_global = list(range(len(all_combined_dataframes_global_df_unscaled)))
if not indices_global: print("No data to split."); exit()

temp_train_val_indices, test_idx = train_test_split(indices_global, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                                                    shuffle=True) if TEST_SIZE > 0 else (list(indices_global), [])

val_split_from_train_val = 0
if VALIDATION_SIZE > 0 and len(temp_train_val_indices) > 0:
    abs_val_count = int(round(len(indices_global) * VALIDATION_SIZE))
    if len(temp_train_val_indices) > abs_val_count and abs_val_count > 0:
        val_split_from_train_val = abs_val_count / len(temp_train_val_indices)
    elif len(
            temp_train_val_indices) > 0 and abs_val_count > 0:
        val_split_from_train_val = (len(temp_train_val_indices) - 1) / len(temp_train_val_indices) if len(
            temp_train_val_indices) > 1 else 0

if val_split_from_train_val > 0 and val_split_from_train_val < 1 and len(temp_train_val_indices) > 1:
    train_idx, val_idx = train_test_split(temp_train_val_indices, test_size=val_split_from_train_val,
                                          random_state=RANDOM_STATE, shuffle=True)
elif len(temp_train_val_indices) > 0 and val_split_from_train_val == 0:
    train_idx = list(temp_train_val_indices)
    val_idx = []
else:
    val_idx = list(temp_train_val_indices)
    train_idx = []

if not train_idx and val_idx:
    train_idx = [val_idx.pop(0)]
elif not train_idx and test_idx and not val_idx:
    if len(test_idx) > 1:
        train_idx = [test_idx.pop(0)]
    elif len(indices_global) == 1:
        train_idx = list(indices_global)
        test_idx = []
        val_idx = []

X_train_df_unscaled = [all_combined_dataframes_global_df_unscaled[i] for i in train_idx]
X_val_df_unscaled = [all_combined_dataframes_global_df_unscaled[i] for i in val_idx]
X_test_df_unscaled = [all_combined_dataframes_global_df_unscaled[i] for i in test_idx]
train_ids = [file_identifiers_global[i] for i in train_idx]
val_ids = [file_identifiers_global[i] for i in val_idx]
test_ids = [file_identifiers_global[i] for i in test_idx]

print(
    f"Data split (DFs): Train={len(X_train_df_unscaled)}, Val={len(X_val_df_unscaled)}, Test={len(X_test_df_unscaled)}")

scaler = StandardScaler()
if X_train_df_unscaled:
    train_dfs_for_scaling = [df.reindex(columns=final_combined_feature_names, fill_value=0.0).values for df in
                             X_train_df_unscaled if not df.empty]
    if train_dfs_for_scaling:
        concatenated_train_data = np.concatenate(train_dfs_for_scaling, axis=0)
        scaler.fit(concatenated_train_data)
        print("Scaler fitted on training data (TS+Gloves only).")

        X_train_scaled = [scaler.transform(
            df.reindex(columns=final_combined_feature_names, fill_value=0.0).values) if not df.empty else np.array([])
                          for df in X_train_df_unscaled]
        X_val_scaled = [scaler.transform(
            df.reindex(columns=final_combined_feature_names, fill_value=0.0).values) if not df.empty else np.array([])
                        for df in X_val_df_unscaled]
        X_test_scaled = [scaler.transform(
            df.reindex(columns=final_combined_feature_names, fill_value=0.0).values) if not df.empty else np.array([])
                         for df in X_test_df_unscaled]
    else:
        print("Warning: No data in X_train_df_unscaled to fit scaler. Scaled data will be empty.")
        X_train_scaled, X_val_scaled, X_test_scaled = [], [], []
        scaler = None
else:
    print("Warning: X_train_df_unscaled is empty. Scaler not fitted. Scaled data will be empty.")
    X_train_scaled, X_val_scaled, X_test_scaled = [], [], []
    scaler = None

data_to_save = {
    'X_train': X_train_scaled,
    'X_val': X_val_scaled,
    'X_test': X_test_scaled,
    'X_train_df_indexed': X_train_df_unscaled,
    'X_val_df_indexed': X_val_df_unscaled,
    'X_test_df_indexed': X_test_df_unscaled,
    'train_ids': train_ids,
    'val_ids': val_ids,
    'test_ids': test_ids,
    'feature_names': final_combined_feature_names
}

with open(FINAL_FEATURES_PKL_PATH, 'wb') as f:
    pickle.dump(data_to_save, f)
print(f"\nSaved final processed data (TS+Gloves only) to: {FINAL_FEATURES_PKL_PATH}")

if scaler is not None:
    with open(FINAL_SCALER_PKL_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler (TS+Gloves only) to: {FINAL_SCALER_PKL_PATH}")
else:
    print(f"Scaler was not saved as it was not fitted.")

print("\n--- TeslaSuit & Gloves Data Processing (for TS+Gloves only training) Finished ---")