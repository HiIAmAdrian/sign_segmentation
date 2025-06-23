import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import traceback
import re
from sklearn.model_selection import train_test_split

DFLT_MERGE_TOLERANCE_MS = 30

DFLT_USE_SUIT_ROTATIONS = True
DFLT_USE_SUIT_POSITIONS = False
DFLT_USE_SUIT_HIPS_POSITION = True
DFLT_USE_SUIT_BIOMECH = True

DFLT_USE_GLOVE_RIGHT_FINGER_ROTATIONS = True
DFLT_USE_GLOVE_RIGHT_FINGER_POSITIONS = False
DFLT_USE_GLOVE_RIGHT_HAND_ROOT = True

DFLT_USE_GLOVE_LEFT_FINGER_ROTATIONS = True
DFLT_USE_GLOVE_LEFT_FINGER_POSITIONS = False
DFLT_USE_GLOVE_LEFT_HAND_ROOT = True

DFLT_RELEVANT_BIOMECH_JOINTS = [
    "PelvisTilt", "PelvisList", "PelvisRotation", "HipFlexExtR", "HipFlexExtL", "HipAddAbdR", "HipAddAbdL", "HipRotR",
    "HipRotL",
    "ElbowFlexExtR", "ElbowFlexExtL", "ForearmProSupR", "ForearmProSupL",
    "WristFlexExtR", "WristFlexExtL", "WristDeviationR", "WristDeviationL",
    "LumbarFlexExt", "LumbarLatFlex", "LumbarRot", "LowerThoraxFlexExt", "LowerThoraxLatFlex", "LowerThoraxRot",
    "UpperThoraxFlexExt", "UpperThoraxLatFlex", "UpperThoraxRot",
    "ShoulderFlexExtR", "ShoulderFlexExtL", "ShoulderAddAbdR", "ShoulderAddAbdL", "ShoulderRotR", "ShoulderRotL"
]
DFLT_UPPER_BODY_BONES_SUIT = [
    "hips", "spine", "upper_spine", "neck", "head", "left_shoulder", "right_shoulder",
    "left_upper_arm", "right_upper_arm", "left_lower_arm", "right_lower_arm"
]


def get_related_files_for_participant(suit_file_path: Path, participant_dir: Path):
    """
    Găsește fișierele CSV pentru mănușa dreaptă și stângă pe baza numelui fișierului CSV al costumului.
    Presupune o convenție de numire de tipul:
    sentence_XXX_ts_suit_mocap.csv
    sentence_XXX_ts_glove_R_mocap.csv
    sentence_XXX_ts_glove_L_mocap.csv
    toate în același participant_dir.

    Args:
        suit_file_path (Path): Calea către fișierul CSV al costumului.
        participant_dir (Path): Directorul participantului unde se află toate fișierele.

    Returns:
        tuple: (Path_glove_r, Path_glove_l) sau (None, None) dacă nu se potrivesc.
    """
    base_name_match = re.match(r"(sentence_\d+_ts)_suit_mocap\.csv", suit_file_path.name, re.IGNORECASE)
    if not base_name_match:
        # print(f"    FUNC_TS_HELPER: Could not match base name pattern for suit file {suit_file_path.name}")
        return None, None

    base_name_prefix = base_name_match.group(1)

    glove_r_file = participant_dir / f"{base_name_prefix}_glove_R_mocap.csv"
    glove_l_file = participant_dir / f"{base_name_prefix}_glove_L_mocap.csv"

    return glove_r_file, glove_l_file

def _generate_feature_lists(
        use_suit_rotations, use_suit_positions, use_suit_hips_position, use_suit_biomech,
        use_glove_left_hand_root, use_glove_right_hand_root,
        use_glove_right_finger_rotations, use_glove_right_finger_positions,
        use_glove_left_finger_rotations, use_glove_left_finger_positions,
        upper_body_bones, relevant_biomech_angle_joints
):
    suit_features_to_keep = []
    if use_suit_rotations:
        for bone in upper_body_bones:
            if bone == "left_hand" and use_glove_left_hand_root: continue
            if bone == "right_hand" and use_glove_right_hand_root: continue
            suit_features_to_keep.extend([f"{bone}.rotation.{axis}" for axis in ['w', 'x', 'y', 'z']])
    if use_suit_positions:
        for bone in upper_body_bones:
            if bone == "hips": continue
            if bone == "left_hand" and use_glove_left_hand_root: continue
            if bone == "right_hand" and use_glove_right_hand_root: continue
            suit_features_to_keep.extend([f"{bone}.position.{axis}" for axis in ['x', 'y', 'z']])
    if use_suit_hips_position:
        suit_features_to_keep.extend([f"hips.position.{axis}" for axis in ['x', 'y', 'z']])
    if use_suit_biomech:
        suit_features_to_keep.extend([f"{joint}.angle" for joint in relevant_biomech_angle_joints])

    glove_right_features = []
    if use_glove_right_finger_rotations or use_glove_right_finger_positions or use_glove_right_hand_root:
        glove_right_features = _get_glove_finger_features_configurable(
            "right", use_glove_right_finger_rotations, use_glove_right_finger_positions, use_glove_right_hand_root)

    glove_left_features = []
    if use_glove_left_finger_rotations or use_glove_left_finger_positions or use_glove_left_hand_root:
        glove_left_features = _get_glove_finger_features_configurable(
            "left", use_glove_left_finger_rotations, use_glove_left_finger_positions, use_glove_left_hand_root)

    return sorted(list(set(suit_features_to_keep))), glove_right_features, glove_left_features


def _get_glove_finger_features_configurable(hand_prefix, use_rotations, use_positions, use_hand_root):
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


def _load_and_prepare_df_internal(csv_path_or_stream, selected_features, filename_for_log, trim_start_sec=0.0):
    timestamp_col_name = 'frame_timestamp_us'
    try:
        if isinstance(csv_path_or_stream, (str, Path)):
            df = pd.read_csv(csv_path_or_stream)
        else:
            df = pd.read_csv(csv_path_or_stream)

        if df.empty or timestamp_col_name not in df.columns: return None, []
        df[timestamp_col_name] = pd.to_numeric(df[timestamp_col_name], errors='coerce')
        df.dropna(subset=[timestamp_col_name], inplace=True)
        if df.empty: return None, []
        if trim_start_sec > 0:
            df = df[df[timestamp_col_name] >= trim_start_sec * 1_000_000].copy()
            if df.empty: return None, []
        actual_selected_features = [f for f in selected_features if f in df.columns]
        if not actual_selected_features and selected_features:
            print(
                f"    FUNC_TS: Warning - No selected features found in {filename_for_log} from list: {selected_features[:5]}...")
        cols_to_load_final = [timestamp_col_name] + actual_selected_features
        df = df[cols_to_load_final]
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
    except FileNotFoundError:
        print(f"    FUNC_TS: File not found {csv_path_or_stream}"); return None, []
    except Exception as e:
        print(f"    FUNC_TS: Error L&P {filename_for_log}: {e}"); return None, []


def _calculate_biomech_derivatives_internal(df, biomech_angle_joints_list):
    if df.empty or not isinstance(df.index, pd.TimedeltaIndex): return df
    df_out = df.copy()
    delta_time_sec_series = df_out.index.to_series().diff().dt.total_seconds()
    min_time_delta = 1e-7
    delta_time_sec_processed = np.where(delta_time_sec_series < min_time_delta, np.nan, delta_time_sec_series)
    delta_time_sec_processed = pd.Series(delta_time_sec_processed, index=df_out.index).ffill().bfill()
    delta_time_sec_processed.fillna(1.0, inplace=True)
    new_columns_data = {}
    for joint_base_name in biomech_angle_joints_list:
        angle_col = f"{joint_base_name}.angle";
        vel_col = f"{joint_base_name}.angular_v";
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
        if not new_df_part.empty: df_out = df_out.assign(**new_df_part)
    return df_out


def process_single_teslasuit_triplet(
        suit_file_path_or_stream,
        glove_r_file_path_or_stream,
        glove_l_file_path_or_stream,
        trim_sec=0.0,
        merge_tolerance_ms=DFLT_MERGE_TOLERANCE_MS,
        use_suit_rotations=DFLT_USE_SUIT_ROTATIONS,
        use_suit_positions=DFLT_USE_SUIT_POSITIONS,
        use_suit_hips_position=DFLT_USE_SUIT_HIPS_POSITION,
        use_suit_biomech=DFLT_USE_SUIT_BIOMECH,
        use_glove_left_hand_root=DFLT_USE_GLOVE_LEFT_HAND_ROOT,
        use_glove_right_hand_root=DFLT_USE_GLOVE_RIGHT_HAND_ROOT,
        use_glove_right_finger_rotations=DFLT_USE_GLOVE_RIGHT_FINGER_ROTATIONS,
        use_glove_right_finger_positions=DFLT_USE_GLOVE_RIGHT_FINGER_POSITIONS,
        use_glove_left_finger_rotations=DFLT_USE_GLOVE_LEFT_FINGER_ROTATIONS,
        use_glove_left_finger_positions=DFLT_USE_GLOVE_LEFT_FINGER_POSITIONS,
        upper_body_bones=DFLT_UPPER_BODY_BONES_SUIT,
        relevant_biomech_angle_joints=DFLT_RELEVANT_BIOMECH_JOINTS
):
    """
    Procesează un triplet de fișiere TeslaSuit (suit, glove_r, glove_l) și returnează
    un DataFrame combinat, cu TimedeltaIndex normalizat numit 'normalized_timestamp_us'.
    Fișierele pot fi căi sau stream-uri (ex: din request.files).
    """
    print(f"  FUNC_TS: Processing TeslaSuit triplet (trim: {trim_sec}s)...")

    cfg_suit_features, cfg_glove_r_features, cfg_glove_l_features = _generate_feature_lists(
        use_suit_rotations, use_suit_positions, use_suit_hips_position, use_suit_biomech,
        use_glove_left_hand_root, use_glove_right_hand_root,
        use_glove_right_finger_rotations, use_glove_right_finger_positions,
        use_glove_left_finger_rotations, use_glove_left_finger_positions,
        upper_body_bones, relevant_biomech_angle_joints
    )

    df_suit, _ = _load_and_prepare_df_internal(suit_file_path_or_stream, cfg_suit_features, "suit_file", trim_sec)
    if df_suit is None or df_suit.empty:
        print("    FUNC_TS: ERROR - Suit data could not be loaded or is empty.")
        return pd.DataFrame(), []

    if use_suit_biomech and relevant_biomech_angle_joints:
        df_suit = _calculate_biomech_derivatives_internal(df_suit, relevant_biomech_angle_joints)

    df_glove_r, _ = _load_and_prepare_df_internal(glove_r_file_path_or_stream, cfg_glove_r_features, "glove_r_file",
                                                  trim_sec)
    if df_glove_r is None: df_glove_r = pd.DataFrame(index=df_suit.index)

    df_glove_l, _ = _load_and_prepare_df_internal(glove_l_file_path_or_stream, cfg_glove_l_features, "glove_l_file",
                                                  trim_sec)
    if df_glove_l is None: df_glove_l = pd.DataFrame(index=df_suit.index)

    merge_col = 'normalized_timestamp_us'
    df_s_reset = df_suit.reset_index().rename(columns={df_suit.index.name or 'index': merge_col})
    df_gr_reset = df_glove_r.reset_index().rename(columns={df_glove_r.index.name or 'index': merge_col})
    df_gl_reset = df_glove_l.reset_index().rename(columns={df_glove_l.index.name or 'index': merge_col})

    for df_r in [df_gr_reset, df_gl_reset]:
        if merge_col not in df_r.columns: df_r[merge_col] = pd.Series(dtype='timedelta64[ns]')

    df_merged = pd.merge_asof(df_s_reset.sort_values(merge_col), df_gr_reset.sort_values(merge_col),
                              on=merge_col, direction='nearest',
                              tolerance=pd.Timedelta(microseconds=merge_tolerance_ms * 1000),
                              suffixes=('_s_dup', '_gr_dup'))

    df_combined = pd.merge_asof(df_merged.sort_values(merge_col), df_gl_reset.sort_values(merge_col),
                                on=merge_col, direction='nearest',
                                tolerance=pd.Timedelta(microseconds=merge_tolerance_ms * 1000),
                                suffixes=('',
                                          '_gl_dup'))


    final_columns_ordered = []
    final_columns_ordered.extend(c for c in df_suit.columns if c in df_combined.columns)

    for glove_feat_list, suffix_check in [(cfg_glove_r_features, '_gr_dup'), (cfg_glove_l_features, '_gl_dup')]:
        for feat_name in glove_feat_list:
            suffixed_name = f"{feat_name}{suffix_check}"
            if suffixed_name in df_combined.columns:
                if feat_name not in df_combined.columns:
                    df_combined.rename(columns={suffixed_name: feat_name}, inplace=True)
                    final_columns_ordered.append(feat_name)
                else:
                    final_columns_ordered.append(suffixed_name)
                    print(
                        f"    FUNC_TS: Warning - Feature '{feat_name}' already exists. Keeping suffixed version '{suffixed_name}'.")
            elif feat_name in df_combined.columns:
                final_columns_ordered.append(feat_name)

    df_combined = df_combined.set_index(merge_col).sort_index()

    cols_to_interpolate = [col for col_list in [cfg_glove_r_features, cfg_glove_l_features] for col in col_list if
                           col in df_combined.columns]
    if cols_to_interpolate and df_combined[cols_to_interpolate].isnull().values.any():
        df_combined[cols_to_interpolate] = df_combined[cols_to_interpolate].interpolate(
            method='time').ffill().bfill().fillna(0)

    current_feature_names = sorted(list(set(c for c in df_combined.columns if not c.endswith('_dup'))))

    df_final = df_combined[current_feature_names].copy()

    print(f"  FUNC_TS: TeslaSuit triplet processed. Shape: {df_final.shape}, Features: {len(current_feature_names)}")
    return df_final, current_feature_names


if __name__ == "__main__":
    print("--- Running TeslaSuit Data Processing (Standalone Batch Mode) ---")
    PARTICIPANT_BASE_DIRS_WITH_TRIM_STANDALONE = {
        Path("D:/SegmentationThesis/output_realsense60fps+tesla p1"): 1.0,
        Path("D:/SegmentationThesis/output_realsense60fps+tesla p2"): 0.3,
    }
    OUTPUT_DIR_STANDALONE = Path("./processed_combined_data_all_participants_TESLASUIT_DF_trimmed_FUNC")
    OUTPUT_DIR_STANDALONE.mkdir(parents=True, exist_ok=True)

    TEST_SIZE_STANDALONE = 0.15
    VALIDATION_SIZE_STANDALONE = 0.15
    RANDOM_STATE_STANDALONE = 42

    all_dfs_temp_standalone = []
    all_ids_temp_standalone = []

    collected_feature_names_standalone = set()

    for participant_dir, trim_val in PARTICIPANT_BASE_DIRS_WITH_TRIM_STANDALONE.items():
        participant_name = participant_dir.name
        print(f"\n  Processing Participant (standalone): {participant_name} (Trim: {trim_val}s)")

        suit_files = sorted(list(participant_dir.glob("sentence_*_ts_suit_mocap.csv")))
        for suit_f_path in suit_files:
            glove_r_f_path, glove_l_f_path = get_related_files_for_participant(suit_f_path,
                                                                               participant_dir)
            if not (glove_r_f_path and glove_l_f_path and glove_r_f_path.exists() and glove_l_f_path.exists()):
                print(f"    Skipping {suit_f_path.name}, missing glove files.")
                continue

            try:
                df_processed, features_in_df = process_single_teslasuit_triplet(
                    suit_f_path, glove_r_f_path, glove_l_f_path, trim_sec=trim_val
                )
                if not df_processed.empty:
                    all_dfs_temp_standalone.append(df_processed)
                    all_ids_temp_standalone.append({'filename': suit_f_path.name, 'participant': participant_name})
                    collected_feature_names_standalone.update(features_in_df)
            except Exception as e_proc:
                print(f"    ERROR processing triplet {suit_f_path.name}: {e_proc}")
                traceback.print_exc()

    if not all_dfs_temp_standalone:
        print("No data processed in standalone mode. Exiting.")
        exit()

    final_feature_names_batch = sorted(list(collected_feature_names_standalone))
    print(f"\n  Standalone: Final global feature list established with {len(final_feature_names_batch)} features.")

    all_dfs_standardized_standalone = []
    for df_item in all_dfs_temp_standalone:
        df_reindexed = df_item.reindex(columns=final_feature_names_batch, fill_value=0.0)
        all_dfs_standardized_standalone.append(df_reindexed)


    indices = list(range(len(all_dfs_standardized_standalone)))
    if len(indices) < 3: print(f"Error: Only {len(indices)} sequences. Need >= 3 for split."); exit()

    num_total = len(indices)
    ts_n = max(1, int(round(num_total * TEST_SIZE_STANDALONE)))
    remaining_after_ts = num_total - ts_n
    val_n = 0
    if remaining_after_ts > 0:
        val_n = max(1, int(round(remaining_after_ts * (
                    VALIDATION_SIZE_STANDALONE / (1 - TEST_SIZE_STANDALONE if TEST_SIZE_STANDALONE < 1 else 1)))))
        if remaining_after_ts - val_n < 1: val_n = max(0, remaining_after_ts - 1)

    tr_n = num_total - ts_n - val_n
    if tr_n < 0: tr_n = 0
    if tr_n + val_n + ts_n != num_total: tr_n = num_total - val_n - ts_n

    print(f"  Standalone split sizes: Total={num_total}, Train={tr_n}, Val={val_n}, Test={ts_n}")

    train_val_idx, test_idx = train_test_split(indices, test_size=ts_n, random_state=RANDOM_STATE_STANDALONE,
                                               shuffle=True) if ts_n > 0 else (list(indices), [])
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_n / len(train_val_idx) if len(
        train_val_idx) > 0 and val_n > 0 else 0.0, random_state=RANDOM_STATE_STANDALONE,
                                          shuffle=True) if val_n > 0 and len(train_val_idx) > 0 else (
        list(train_val_idx), [])

    X_train_df = [all_dfs_standardized_standalone[i] for i in train_idx]
    X_val_df = [all_dfs_standardized_standalone[i] for i in val_idx]
    X_test_df = [all_dfs_standardized_standalone[i] for i in test_idx]
    train_ids_list = [all_ids_temp_standalone[i] for i in train_idx]
    val_ids_list = [all_ids_temp_standalone[i] for i in val_idx]
    test_ids_list = [all_ids_temp_standalone[i] for i in test_idx]

    print(f"  Standalone split: Train={len(X_train_df)}, Val={len(X_val_df)}, Test={len(X_test_df)}")

    output_pkl_path = OUTPUT_DIR_STANDALONE / "combined_all_participants_sequences_DF_FUNC.pkl"
    data_to_save = {
        'X_train_df': X_train_df, 'X_val_df': X_val_df, 'X_test_df': X_test_df,
        'train_ids': train_ids_list, 'val_ids': val_ids_list, 'test_ids': test_ids_list,
        'feature_names': final_feature_names_batch
    }
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f"\n  Standalone: Saved processed DataFrames to: {output_pkl_path}")
    print("--- TeslaSuit Data Processing (Standalone Batch Mode) Finished ---")