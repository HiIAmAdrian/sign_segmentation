import traceback
from pathlib import Path
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tempfile
import shutil
import time

try:
    from mediapipe_face_landmark_functional import run_landmark_extraction_for_bag as process_bag_for_landmarks_external, \
    VisionRunningMode
    from mediapipe_face_landmark_functional import create_landmarker_options_for_landmarks_functional as create_landmark_opts_for_landmarks_external
    from extract_blendshapes_functional import run_blendshape_extraction_for_bag as process_bag_for_blendshapes_external
    from extract_blendshapes_functional import \
        create_landmarker_options_for_blendshapes_functional as create_blendshape_opts_external

    from smooth_functional import apply_oneeuro_to_single_file as smooth_landmarks_file_external


    import normalize_face_data_with_meanshape_functional as normalize_script_module
    from add_timestamp_prepare_for_training_functional import (
        run_add_timestamps_and_combine_facial as ats_run_add_timestamps_and_combine_facial_external,
        extract_timestamps_from_bag as ats_extract_timestamps_external,
        pivot_landmarks as ats_pivot_landmarks_external,
        pivot_blendshapes as ats_pivot_blendshapes_external
    )

except ImportError as e:
    print(f"WARNING: Could not import one or more preprocessing script functions: {e}")
    print("         Backend will rely on internal implementations or fail if functions are called.")
    process_bag_for_landmarks_external = None
    create_landmark_opts_for_landmarks_external = None
    process_bag_for_blendshapes_external = None
    create_blendshape_opts_external = None
    smooth_landmarks_file_external = None
    normalize_script_module = None
    ats_extract_timestamps_external = None
    ats_pivot_landmarks_external = None
    ats_pivot_blendshapes_external = None

try:
    from infer_logic_functional import (
        run_inference_for_sequence_data,
        scale_data_for_inference
    )
except ImportError as e:
    print(f"FATAL: Could not import from infer_logic_functional.py: {e}")
    print("       Ensure infer_logic_functional.py is in the correct location and has no import errors itself.")
    exit()

app = Flask(__name__)
CORS(app)

MODEL_PATH_BILSTM = Path("./trained_models_final/bilstm_best_final.keras")
MODEL_PATH_BIGRU = Path("./trained_models_final/bigru_best_final.keras")
SCALER_PATH = Path("./final_data_ts_gloves_only/ts_gloves_only_scaler.pkl")
TRAINING_CONFIG_PKL = Path("./final_data_ts_gloves_only/all_data_features_ts_gloves.pkl")

MEDIAPIPE_LANDMARK_MODEL_FILE = Path('./model/face_landmarker_v2_with_blendshapes.task')
MEAN_SHAPE_FILE_FOR_NORMALIZATION = Path("./P1_face_mean_shape/mean_face_shape_478_cleaned.npy")

TS_UPPER_BODY_BONES_SUIT = [
    "hips", "spine", "upper_spine", "neck", "head",
    "left_shoulder", "right_shoulder",
    "left_upper_arm", "right_upper_arm",
    "left_lower_arm", "right_lower_arm"
]

TS_RELEVANT_BIOMECH_JOINTS = [
    "PelvisTilt", "PelvisList", "PelvisRotation", "HipFlexExtR", "HipFlexExtL", "HipAddAbdR", "HipAddAbdL", "HipRotR", "HipRotL",
    "ElbowFlexExtR", "ElbowFlexExtL", "ForearmProSupR", "ForearmProSupL",
    "WristFlexExtR", "WristFlexExtL", "WristDeviationR", "WristDeviationL",
    "LumbarFlexExt", "LumbarLatFlex", "LumbarRot", "LowerThoraxFlexExt", "LowerThoraxLatFlex", "LowerThoraxRot",
    "UpperThoraxFlexExt", "UpperThoraxLatFlex", "UpperThoraxRot",
    "ShoulderFlexExtR", "ShoulderFlexExtL", "ShoulderAddAbdR", "ShoulderAddAbdL", "ShoulderRotR", "ShoulderRotL"
]

FACIAL_MERGE_TOLERANCE_MS = 30
TS_MERGE_TOLERANCE_MS = 30
TS_USE_SUIT_ROTATIONS = True
TS_USE_SUIT_POSITIONS = False
TS_USE_SUIT_HIPS_POSITION = True
TS_USE_SUIT_BIOMECH = True
TS_USE_GLOVE_LEFT_HAND_ROOT = True
TS_USE_GLOVE_RIGHT_HAND_ROOT = True

TS_GLOVE_RIGHT_FEATURES_TO_KEEP = []
TS_GLOVE_LEFT_FEATURES_TO_KEEP = []
TS_SUIT_FEATURES_TO_KEEP = []

model_bilstm, model_bigru, scaler, FEATURE_NAMES, NUM_FEATURES, MODEL_EXPECTED_LEN = None, None, None, [], 0, None

DFLT_TS_USE_GLOVE_RIGHT_FINGER_ROTATIONS = True
DFLT_TS_USE_GLOVE_RIGHT_FINGER_POSITIONS = False
DFLT_TS_USE_GLOVE_RIGHT_HAND_ROOT = True
DFLT_TS_USE_GLOVE_LEFT_FINGER_ROTATIONS = True
DFLT_TS_USE_GLOVE_LEFT_FINGER_POSITIONS = False
DFLT_TS_USE_GLOVE_LEFT_HAND_ROOT = True



def generate_glove_feature_list_for_backend(
        hand_prefix: str,
        use_finger_rotations: bool,
        use_finger_positions: bool,
        use_hand_root: bool
) -> list[str]:
    """Generates a list of glove features based on configuration flags."""
    features = []
    finger_segments = ["thumb", "index", "middle", "ring", "little"]
    phalanges = ["proximal", "intermediate", "distal"]

    if use_finger_rotations:
        for finger in finger_segments:
            for phalanx in phalanges:
                if finger == "thumb" and phalanx == "intermediate":
                    continue
                features.extend([f"{hand_prefix}_{finger}_{phalanx}.rotation.{axis}" for axis in ['w', 'x', 'y', 'z']])

    if use_finger_positions:
        for finger in finger_segments:
            for phalanx in phalanges:
                if finger == "thumb" and phalanx == "intermediate": continue
                features.extend([f"{hand_prefix}_{finger}_{phalanx}.position.{axis}" for axis in ['x', 'y', 'z']])

    if use_hand_root:
        features.extend([f"{hand_prefix}_hand.rotation.{axis}" for axis in ['w', 'x', 'y', 'z']])
        features.extend([f"{hand_prefix}_hand.position.{axis}" for axis in ['x', 'y', 'z']])

    return sorted(list(set(features)))


def populate_ts_feature_lists():
    """Populează listele globale de caracteristici TeslaSuit pe baza flag-urilor globale."""
    global TS_SUIT_FEATURES_TO_KEEP, TS_GLOVE_LEFT_FEATURES_TO_KEEP, TS_GLOVE_RIGHT_FEATURES_TO_KEEP

    temp_suit_features = []
    if TS_USE_SUIT_ROTATIONS:
        for bone in TS_UPPER_BODY_BONES_SUIT:
            if bone == "left_hand" and DFLT_TS_USE_GLOVE_LEFT_HAND_ROOT: continue
            if bone == "right_hand" and DFLT_TS_USE_GLOVE_RIGHT_HAND_ROOT: continue
            temp_suit_features.extend([f"{bone}.rotation.{axis}" for axis in ['w', 'x', 'y', 'z']])
    if TS_USE_SUIT_POSITIONS:
        for bone in TS_UPPER_BODY_BONES_SUIT:
            if bone == "hips": continue
            if bone == "left_hand" and DFLT_TS_USE_GLOVE_LEFT_HAND_ROOT: continue
            if bone == "right_hand" and DFLT_TS_USE_GLOVE_RIGHT_HAND_ROOT: continue
            temp_suit_features.extend([f"{bone}.position.{axis}" for axis in ['x', 'y', 'z']])
    if TS_USE_SUIT_HIPS_POSITION:
        temp_suit_features.extend([f"hips.position.{axis}" for axis in ['x', 'y', 'z']])
    if TS_USE_SUIT_BIOMECH:
        temp_suit_features.extend(
            [f"{joint}.angle" for joint in TS_RELEVANT_BIOMECH_JOINTS])

    TS_SUIT_FEATURES_TO_KEEP = sorted(list(set(temp_suit_features)))
    print(f"  Backend Config: Generated {len(TS_SUIT_FEATURES_TO_KEEP)} SUIT features to keep.")

    if DFLT_TS_USE_GLOVE_RIGHT_FINGER_ROTATIONS or DFLT_TS_USE_GLOVE_RIGHT_FINGER_POSITIONS or DFLT_TS_USE_GLOVE_RIGHT_HAND_ROOT:
        TS_GLOVE_RIGHT_FEATURES_TO_KEEP = generate_glove_feature_list_for_backend(
            "right",
            DFLT_TS_USE_GLOVE_RIGHT_FINGER_ROTATIONS,
            DFLT_TS_USE_GLOVE_RIGHT_FINGER_POSITIONS,
            DFLT_TS_USE_GLOVE_RIGHT_HAND_ROOT
        )
    else:
        TS_GLOVE_RIGHT_FEATURES_TO_KEEP = []
    print(f"  Backend Config: Generated {len(TS_GLOVE_RIGHT_FEATURES_TO_KEEP)} RIGHT GLOVE features to keep.")

    if DFLT_TS_USE_GLOVE_LEFT_FINGER_ROTATIONS or DFLT_TS_USE_GLOVE_LEFT_FINGER_POSITIONS or DFLT_TS_USE_GLOVE_LEFT_HAND_ROOT:
        TS_GLOVE_LEFT_FEATURES_TO_KEEP = generate_glove_feature_list_for_backend(
            "left",
            DFLT_TS_USE_GLOVE_LEFT_FINGER_ROTATIONS,
            DFLT_TS_USE_GLOVE_LEFT_FINGER_POSITIONS,
            DFLT_TS_USE_GLOVE_LEFT_HAND_ROOT
        )
    else:
        TS_GLOVE_LEFT_FEATURES_TO_KEEP = []
    print(f"  Backend Config: Generated {len(TS_GLOVE_LEFT_FEATURES_TO_KEEP)} LEFT GLOVE features to keep.")


def load_global_components():
    global model_bilstm, model_bigru, scaler, FEATURE_NAMES, NUM_FEATURES, MODEL_EXPECTED_LEN

    populate_ts_feature_lists()

    try:
        print(f"Loading feature names and config from: {TRAINING_CONFIG_PKL}")
        if not TRAINING_CONFIG_PKL.exists():
            raise FileNotFoundError(f"Training config PKL not found: {TRAINING_CONFIG_PKL}")
        with open(TRAINING_CONFIG_PKL, 'rb') as f:
            data_info = pickle.load(f)
            FEATURE_NAMES = data_info['feature_names']
            NUM_FEATURES = len(FEATURE_NAMES)
            print(f"Loaded {NUM_FEATURES} feature names (expected by the trained model).")

        print(f"Loading scaler from: {SCALER_PATH}")
        if not SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler PKL not found: {SCALER_PATH}")
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        if scaler.n_features_in_ != NUM_FEATURES:
            print(
                f"WARNING: Scaler expects {scaler.n_features_in_} features, but loaded feature_names list has {NUM_FEATURES}.")
        print("Scaler loaded.")

        print(f"Loading BiLSTM model from: {MODEL_PATH_BILSTM}")
        model_bilstm = tf.keras.models.load_model(MODEL_PATH_BILSTM, compile=False)
        print("BiLSTM Model loaded.")
        MODEL_EXPECTED_LEN = model_bilstm.input_shape[1]
        print(f"BiLSTM expects input length: {MODEL_EXPECTED_LEN}, features: {model_bilstm.input_shape[2]}")
        if model_bilstm.input_shape[2] != NUM_FEATURES:
            print(
                f"FATAL ERROR: BiLSTM feature count mismatch! Model: {model_bilstm.input_shape[2]}, Data from PKL: {NUM_FEATURES}")
            exit()

        print(f"Loading BiGRU model from: {MODEL_PATH_BIGRU}")
        model_bigru = tf.keras.models.load_model(MODEL_PATH_BIGRU, compile=False)
        print("BiGRU Model loaded.")
        if model_bigru.input_shape[1] != MODEL_EXPECTED_LEN or model_bigru.input_shape[2] != NUM_FEATURES:
            print(
                f"WARNING: BiGRU shape mismatch! Input: {model_bigru.input_shape}, Expected len: {MODEL_EXPECTED_LEN}, Expected feat: {NUM_FEATURES}")

    except Exception as e:
        print(f"FATAL ERROR during global model/scaler loading: {e}")
        traceback.print_exc()
        exit()

def adapted_load_and_prepare_df(file_path, selected_features, filename_for_log, trim_start_sec=0.0):
    try:
        if not Path(file_path).exists(): raise FileNotFoundError(f"{filename_for_log} not found at {file_path}")
        df = pd.read_csv(file_path)
        if 'frame_timestamp_us' not in df.columns:
            if 'frame_timestamp' in df.columns:
                df.rename(columns={'frame_timestamp': 'frame_timestamp_us'}, inplace=True)
                df['frame_timestamp_us'] = df['frame_timestamp_us'] * 1000
            else:
                raise ValueError(f"'frame_timestamp_us' or 'frame_timestamp' missing in {filename_for_log}")

        df['frame_timestamp_us'] = pd.to_numeric(df['frame_timestamp_us'], errors='coerce')
        df.dropna(subset=['frame_timestamp_us'], inplace=True)
        if trim_start_sec > 0:
            df = df[df['frame_timestamp_us'] >= trim_start_sec * 1_000_000].copy()

        df['frame_timestamp_us'] = pd.to_timedelta(df['frame_timestamp_us'], unit='us')
        df = df.sort_values('frame_timestamp_us')
        if not df.empty:
            df['frame_timestamp_us'] = df['frame_timestamp_us'] - df['frame_timestamp_us'].iloc[0]
        df = df.set_index('frame_timestamp_us')

        actual_features = [f for f in selected_features if f in df.columns]
        missing_in_df = [f for f in selected_features if f not in df.columns]
        if missing_in_df: print(
            f"Warning: Features {missing_in_df} not in {filename_for_log}, will be filled with 0 by reindex later.")

        df_to_return = df[actual_features].copy()
        for col in actual_features:
            df_to_return[col] = pd.to_numeric(df_to_return[col], errors='coerce').fillna(0)

        return df_to_return, actual_features
    except Exception as e:
        print(f"Error in adapted_load_and_prepare_df for {filename_for_log}: {e}")
        return pd.DataFrame(), []


def adapted_calculate_biomech_derivatives(df, biomech_angle_joints_list):
    if df.empty or not isinstance(df.index, pd.TimedeltaIndex): return df
    df_out = df.copy()
    delta_time_sec_series = df_out.index.to_series().diff().dt.total_seconds()
    min_time_delta = 1e-7
    delta_time_sec_processed = np.where(delta_time_sec_series < min_time_delta, np.nan, delta_time_sec_series)
    delta_time_sec_processed = pd.Series(delta_time_sec_processed, index=df_out.index)
    delta_time_sec_processed = delta_time_sec_processed.ffill().bfill()
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


def process_teslasuit_for_demo_backend(suit_fpath, glove_r_fpath, glove_l_fpath, trim_sec_ts=0.0):
    print("  Processing TeslaSuit CSVs...")
    df_suit, _ = adapted_load_and_prepare_df(suit_fpath, TS_SUIT_FEATURES_TO_KEEP, "suit_file_uploaded", trim_sec_ts)
    if df_suit.empty: raise ValueError("Suit data processing failed or resulted in empty DataFrame.")

    if TS_USE_SUIT_BIOMECH and TS_RELEVANT_BIOMECH_JOINTS:
        df_suit = adapted_calculate_biomech_derivatives(df_suit, TS_RELEVANT_BIOMECH_JOINTS)

    df_glove_r, _ = adapted_load_and_prepare_df(glove_r_fpath, TS_GLOVE_RIGHT_FEATURES_TO_KEEP, "glove_r_file_uploaded",
                                                trim_sec_ts)
    if df_glove_r.empty: df_glove_r = pd.DataFrame(index=df_suit.index)

    df_glove_l, _ = adapted_load_and_prepare_df(glove_l_fpath, TS_GLOVE_LEFT_FEATURES_TO_KEEP, "glove_l_file_uploaded",
                                                trim_sec_ts)
    if df_glove_l.empty: df_glove_l = pd.DataFrame(index=df_suit.index)

    merge_col = 'normalized_timestamp_us'
    df_s_reset = df_suit.reset_index().rename(columns={df_suit.index.name or 'index': merge_col})
    df_gr_reset = df_glove_r.reset_index().rename(columns={df_glove_r.index.name or 'index': merge_col})
    df_gl_reset = df_glove_l.reset_index().rename(columns={df_glove_l.index.name or 'index': merge_col})

    for df_reset in [df_gr_reset, df_gl_reset]:
        if merge_col not in df_reset.columns: df_reset[merge_col] = pd.Series(dtype='timedelta64[ns]')

    df_merged = pd.merge_asof(df_s_reset.sort_values(merge_col), df_gr_reset.sort_values(merge_col),
                              on=merge_col, direction='nearest',
                              tolerance=pd.Timedelta(microseconds=TS_MERGE_TOLERANCE_MS * 1000),
                              suffixes=('_left', '_gr'))
    df_merged = pd.merge_asof(df_merged.sort_values(merge_col), df_gl_reset.sort_values(merge_col),
                              on=merge_col, direction='nearest',
                              tolerance=pd.Timedelta(microseconds=TS_MERGE_TOLERANCE_MS * 1000), suffixes=('', '_gl'))

    df_ts_combined = df_merged.set_index(merge_col).sort_index()

    cols_to_interpolate_ts = [col for col_list in [TS_GLOVE_RIGHT_FEATURES_TO_KEEP, TS_GLOVE_LEFT_FEATURES_TO_KEEP] for
                              col in col_list if col in df_ts_combined.columns]
    if cols_to_interpolate_ts and df_ts_combined[cols_to_interpolate_ts].isnull().values.any():
        df_ts_combined[cols_to_interpolate_ts] = df_ts_combined[cols_to_interpolate_ts].interpolate(
            method='time').ffill().bfill().fillna(0)

    print(f"  TeslaSuit data processed, shape: {df_ts_combined.shape}")
    return df_ts_combined


def process_facial_for_demo_backend(bag_file_path, temp_output_base, sentence_id_str="001", trim_sec_facial=0.0):
    print(f"  Processing facial data from BAG: {bag_file_path.name}...")
    if not MEDIAPIPE_LANDMARK_MODEL_FILE.exists():
        raise FileNotFoundError(f"MediaPipe model file not found at: {MEDIAPIPE_LANDMARK_MODEL_FILE}")
    if normalize_script_module and not MEAN_SHAPE_FILE_FOR_NORMALIZATION.exists():
        raise FileNotFoundError(f"Mean shape file not found at: {MEAN_SHAPE_FILE_FOR_NORMALIZATION}")

    run_temp_dir = temp_output_base / f"facial_run_{sentence_id_str}_{int(time.time())}"
    temp_landmarks_csv_dir = run_temp_dir / "landmarks_csv"
    temp_blendshapes_csv_dir = run_temp_dir / "blendshapes_csv"
    temp_landmarks_smoothed_dir = run_temp_dir / "landmarks_smoothed"
    temp_landmarks_normalized_dir = run_temp_dir / "landmarks_normalized"

    for d_path in [temp_landmarks_csv_dir, temp_blendshapes_csv_dir, temp_landmarks_smoothed_dir,
                   temp_landmarks_normalized_dir]:
        d_path.mkdir(parents=True, exist_ok=True)

    if process_bag_for_landmarks_external and create_landmark_opts_for_landmarks_external:
        print(f"    Extracting landmarks...")
        lm_opts = create_landmark_opts_for_landmarks_external(
            VisionRunningMode.VIDEO,
            str(MEDIAPIPE_LANDMARK_MODEL_FILE)
        )
        process_bag_for_landmarks_external(
            bag_file_path_str=str(bag_file_path),
            output_csv_dir_str=str(temp_landmarks_csv_dir),
            sentence_num_int=int(sentence_id_str),
            landmarker_model_file_path_str=str(MEDIAPIPE_LANDMARK_MODEL_FILE),
            trim_duration_sec=trim_sec_facial
        )
        landmarks_csv_path = temp_landmarks_csv_dir / f"sentence_{int(sentence_id_str):03d}_mediapipe_landmarks_py.csv"
        if not landmarks_csv_path.exists(): raise FileNotFoundError(f"Landmark CSV not created: {landmarks_csv_path}")
    else:
        raise RuntimeError(
            "Landmark processing functions (process_bag_for_landmarks_external or create_landmark_opts_for_landmarks_external) not imported correctly.")

    print("--- Inline Sanity Check for Blendshape Options (process_facial_for_demo_backend) ---")
    try:
        import mediapipe as mp
        TestBaseOptions_inline = mp.tasks.BaseOptions
        TestFaceLandmarker_inline = mp.tasks.vision.FaceLandmarker
        TestFaceLandmarkerOptions_inline = mp.tasks.vision.FaceLandmarkerOptions
        TestVisionRunningMode_inline = mp.tasks.vision.RunningMode

        model_path_for_inline_test = str(MEDIAPIPE_LANDMARK_MODEL_FILE)
        print(f"Inline Sanity Check: Using model path: {model_path_for_inline_test}")

        test_options_inline = TestFaceLandmarkerOptions_inline(
            base_options=TestBaseOptions_inline(model_asset_path=model_path_for_inline_test),
            running_mode=TestVisionRunningMode_inline.VIDEO,
            output_face_blendshapes=True
        )
        print(f"Inline Sanity Check: Options object created: {test_options_inline}")

        with TestFaceLandmarker_inline.create_from_options(test_options_inline) as landmarker_inline_test:
            print("Inline Sanity Check: FaceLandmarker with blendshapes created successfully!")
        print("--- Inline Sanity Check Passed ---")
    except Exception as e_inline_sanity:
        print(f"Inline Sanity Check FAILED: {e_inline_sanity}")
        traceback.print_exc()

    if process_bag_for_blendshapes_external and create_blendshape_opts_external:
        print(f"    Extracting blendshapes...")
        bs_opts = create_blendshape_opts_external(
            VisionRunningMode.VIDEO,
            str(MEDIAPIPE_LANDMARK_MODEL_FILE)
        )
        process_bag_for_blendshapes_external(str(bag_file_path), str(temp_blendshapes_csv_dir),
                                             int(sentence_id_str), bs_opts, trim_duration_sec=trim_sec_facial)
        process_bag_for_blendshapes_external(
            bag_file_path_str=str(bag_file_path),
            output_csv_dir_str=str(temp_blendshapes_csv_dir),
            sentence_num_int=int(sentence_id_str),
            landmarker_model_file_path_str=str(MEDIAPIPE_LANDMARK_MODEL_FILE),
            trim_duration_sec=trim_sec_facial
        )
        blendshapes_csv_path = temp_blendshapes_csv_dir / f"sentence_{int(sentence_id_str):03d}_mediapipe_blendshapes.csv"
        if not blendshapes_csv_path.exists(): print(f"    Warning: Blendshape CSV not created: {blendshapes_csv_path}")
    else:
        raise RuntimeError("Blendshape processing functions not imported.")

    smoothed_landmarks_csv_path = temp_landmarks_smoothed_dir / f"sentence_{int(sentence_id_str):03d}_mediapipe_landmarks_py_oneeuro_smoothed.csv"
    if smooth_landmarks_file_external:
        print(f"    Smoothing landmarks...")
        smooth_landmarks_file_external(str(landmarks_csv_path),
                                       str(smoothed_landmarks_csv_path))
        if not smoothed_landmarks_csv_path.exists(): raise FileNotFoundError(
            f"Smoothed landmark CSV not created: {smoothed_landmarks_csv_path}")
        print("    DEBUG: Landmark smoothing function not imported, using raw landmarks as smoothed.")
        shutil.copy(str(landmarks_csv_path), str(smoothed_landmarks_csv_path))

    normalized_landmarks_csv_path = smoothed_landmarks_csv_path
    if normalize_script_module:
        print(f"    Normalizing & filling landmarks...")
        normalize_script_module.MEAN_SHAPE_FILE = str(MEAN_SHAPE_FILE_FOR_NORMALIZATION)
        normalize_script_module.INPUT_CSV_DIR = str(temp_landmarks_smoothed_dir)
        normalize_script_module.OUTPUT_CSV_DIR = str(temp_landmarks_normalized_dir)
        normalize_script_module.INPUT_CSV_PATTERN = Path(smoothed_landmarks_csv_path).name

        normalize_script_module.process_all_csvs_ransac_pass2()

        temp_normalized_path = temp_landmarks_normalized_dir / Path(smoothed_landmarks_csv_path).name.replace(".csv",
                                                                                                              "_filled_ransac.csv")
        if temp_normalized_path.exists():
            normalized_landmarks_csv_path = temp_normalized_path
        else:
            print(
                f"    Warning: Normalized landmark CSV not created at {temp_normalized_path}. Using smoothed version.")
    else:
        print("    DEBUG: Landmark normalization script/module not imported. Using smoothed landmarks.")

    print(f"    Adding timestamps and combining facial features...")
    if not (ats_extract_timestamps_external and ats_pivot_landmarks_external and ats_pivot_blendshapes_external):
        raise RuntimeError("Timestamping/Pivoting functions for facial data not imported.")

    realsense_timestamps_ms = ats_extract_timestamps_external(bag_file_path, trim_sec=trim_sec_facial)
    if realsense_timestamps_ms is None or len(realsense_timestamps_ms) == 0:
        raise ValueError(f"No timestamps extracted from BAG for final facial: {bag_file_path.name}")

    df_landmarks_long = pd.read_csv(normalized_landmarks_csv_path)
    df_landmarks_long['frame_id'] = df_landmarks_long['frame_id'].astype(int)
    max_lm_frame_id = df_landmarks_long['frame_id'].max() if not df_landmarks_long.empty else -1
    if max_lm_frame_id >= len(realsense_timestamps_ms):
        print(
            f"    Warning: Max landmark frame_id {max_lm_frame_id} >= num timestamps {len(realsense_timestamps_ms)}. Truncating landmarks.")
        df_landmarks_long = df_landmarks_long[df_landmarks_long['frame_id'] < len(realsense_timestamps_ms)]

    df_landmarks_wide = ats_pivot_landmarks_external(df_landmarks_long)
    if df_landmarks_wide.empty: raise ValueError("Pivoted landmarks empty (final step).")

    valid_lm_indices = df_landmarks_wide.index[df_landmarks_wide.index < len(realsense_timestamps_ms)]
    df_landmarks_wide = df_landmarks_wide.loc[valid_lm_indices]
    if df_landmarks_wide.empty: raise ValueError("Landmarks empty after timestamp index validation (final step).")

    timestamps_for_lm_frames = realsense_timestamps_ms[df_landmarks_wide.index.astype(int)]
    df_landmarks_wide['realsense_timestamp_ms'] = timestamps_for_lm_frames
    df_landmarks_wide.set_index('realsense_timestamp_ms', inplace=True)

    df_blendshapes_wide = pd.DataFrame()
    if blendshapes_csv_path.exists():
        df_blendshapes_long = pd.read_csv(blendshapes_csv_path)
        if not df_blendshapes_long.empty:
            df_blendshapes_long['frame_id'] = df_blendshapes_long['frame_id'].astype(int)
            max_bs_frame_id = df_blendshapes_long['frame_id'].max() if not df_blendshapes_long.empty else -1
            if max_bs_frame_id >= len(realsense_timestamps_ms):
                print(
                    f"    Warning: Max blendshape frame_id {max_bs_frame_id} >= num timestamps {len(realsense_timestamps_ms)}. Truncating blendshapes.")
                df_blendshapes_long = df_blendshapes_long[
                    df_blendshapes_long['frame_id'] < len(realsense_timestamps_ms)]

            df_blendshapes_wide_temp = ats_pivot_blendshapes_external(df_blendshapes_long)
            if not df_blendshapes_wide_temp.empty:
                valid_bs_indices = df_blendshapes_wide_temp.index[
                    df_blendshapes_wide_temp.index < len(realsense_timestamps_ms)]
                df_blendshapes_wide_temp = df_blendshapes_wide_temp.loc[valid_bs_indices]
                if not df_blendshapes_wide_temp.empty:
                    timestamps_for_bs_frames = realsense_timestamps_ms[df_blendshapes_wide_temp.index.astype(int)]
                    df_blendshapes_wide_temp['realsense_timestamp_ms'] = timestamps_for_bs_frames
                    df_blendshapes_wide = df_blendshapes_wide_temp.set_index('realsense_timestamp_ms')

    if not df_blendshapes_wide.empty:
        df_facial_combined_final = pd.merge(df_landmarks_wide, df_blendshapes_wide,
                                            left_index=True, right_index=True, how='outer', suffixes=('_lm', '_bs'))
        df_facial_combined_final.index.name = 'realsense_timestamp_ms'
        df_facial_combined_final = df_facial_combined_final.ffill().bfill().fillna(0)
    else:
        df_facial_combined_final = df_landmarks_wide.copy()
        if df_facial_combined_final.index.name != 'realsense_timestamp_ms':
            df_facial_combined_final.index.name = 'realsense_timestamp_ms'

    if df_facial_combined_final.empty: raise ValueError(
        "Final combined facial data is empty before timestamp normalization.")

    df_facial_combined_final = df_facial_combined_final.reset_index()
    df_facial_combined_final['realsense_timestamp_ms'] = pd.to_timedelta(
        df_facial_combined_final['realsense_timestamp_ms'], unit='ms', errors='coerce')
    df_facial_combined_final.dropna(subset=['realsense_timestamp_ms'], inplace=True)
    if df_facial_combined_final.empty: raise ValueError("Facial data empty after final timestamp conversion.")

    df_facial_combined_final = df_facial_combined_final.set_index('realsense_timestamp_ms').sort_index()
    if not df_facial_combined_final.empty:
        first_ts_final = df_facial_combined_final.index.min()
        df_facial_combined_final.index = df_facial_combined_final.index - first_ts_final
    df_facial_combined_final = df_facial_combined_final.sort_index()
    df_facial_combined_final.index.name = 'normalized_timestamp_us'

    print(f"  Final facial data processed for demo, shape: {df_facial_combined_final.shape}")
    return df_facial_combined_final


def combine_ts_and_facial_for_demo(df_ts, df_facial, feature_names_from_training):
    print(
        f"  Combining TS ({df_ts.shape if df_ts is not None else 'None'}) and Facial ({df_facial.shape if df_facial is not None else 'None'}) for demo...")
    if df_ts is None or df_ts.empty: raise ValueError("TeslaSuit DataFrame is empty for combination.")

    TIMESTAMP_COL_FOR_MERGE = 'normalized_timestamp_us'

    if df_ts.index.name != TIMESTAMP_COL_FOR_MERGE:
        df_ts = df_ts.rename_axis(TIMESTAMP_COL_FOR_MERGE)

    df_ts_reset = df_ts.reset_index()

    if df_facial is None or df_facial.empty:
        print("    Facial DataFrame is empty, proceeding with TeslaSuit data only.")
        df_final_aligned = df_ts_reset.set_index(TIMESTAMP_COL_FOR_MERGE).sort_index()
    else:
        if df_facial.index.name != TIMESTAMP_COL_FOR_MERGE:
            df_facial = df_facial.rename_axis(TIMESTAMP_COL_FOR_MERGE)
        df_fc_reset = df_facial.reset_index()

        df_final_aligned = pd.merge_asof(
            df_ts_reset.sort_values(TIMESTAMP_COL_FOR_MERGE),
            df_fc_reset.sort_values(TIMESTAMP_COL_FOR_MERGE),
            on=TIMESTAMP_COL_FOR_MERGE,
            direction='nearest',
            tolerance=pd.Timedelta(microseconds=FACIAL_MERGE_TOLERANCE_MS * 1000),
            suffixes=('_ts_dup', '_fc_dup')
        )
        df_final_aligned = df_final_aligned.set_index(TIMESTAMP_COL_FOR_MERGE).sort_index()

        facial_cols_in_final = [col for col in df_facial.columns if col in df_final_aligned.columns]
        if facial_cols_in_final and df_final_aligned[facial_cols_in_final].isnull().values.any():
            df_final_aligned[facial_cols_in_final] = df_final_aligned[facial_cols_in_final].interpolate(
                method='time').ffill().bfill().fillna(0)

    df_final_for_model = df_final_aligned.reindex(columns=feature_names_from_training, fill_value=0.0)
    print(f"  Combined data for model, shape: {df_final_for_model.shape}")
    return df_final_for_model


@app.route('/segment_pipeline', methods=['POST'])
def segment_full_pipeline_endpoint():
    print("\nReceived request for FULL PIPELINE segmentation...")
    required_files = ['suit_file', 'glove_right_file', 'glove_left_file', 'bag_file']
    if not all(k in request.files for k in required_files):
        missing = [k for k in required_files if k not in request.files]
        return make_response(jsonify({"error": f"Missing required files: {', '.join(missing)}"}), 400)

    temp_dir_obj = tempfile.TemporaryDirectory(prefix="sl_demo_")
    temp_dir = Path(temp_dir_obj.name)
    print(f"Created temporary directory: {temp_dir}")

    try:
        suit_file_form = request.files['suit_file']
        glove_r_file_form = request.files['glove_right_file']
        glove_l_file_form = request.files['glove_left_file']
        bag_file_form = request.files['bag_file']

        suit_path = temp_dir / "uploaded_suit.csv"
        glove_r_path = temp_dir / "uploaded_glove_r.csv"
        glove_l_path = temp_dir / "uploaded_glove_l.csv"
        bag_file_path = temp_dir / bag_file_form.filename

        suit_file_form.save(suit_path)
        glove_r_file_form.save(glove_r_path)
        glove_l_file_form.save(glove_l_path)
        bag_file_form.save(bag_file_path)

        print("Step 1: Processing TeslaSuit data...")
        df_teslasuit_processed = process_teslasuit_for_demo_backend(suit_path, glove_r_path, glove_l_path,
                                                                    trim_sec_ts=1)
        if df_teslasuit_processed.empty:
            raise ValueError("TeslaSuit data processing resulted in an empty DataFrame.")

        print("Step 2: Processing Facial data from BAG file...")
        df_facial_processed = process_facial_for_demo_backend(bag_file_path, temp_dir, sentence_id_str="001",
                                                              trim_sec_facial=1)
        if df_facial_processed.empty:
            print(
                "Warning: Facial data processing resulted in an empty DataFrame. Proceeding with TS data only for features.")

        print("Step 3: Combining TeslaSuit and Facial features...")
        df_final_features = combine_ts_and_facial_for_demo(df_teslasuit_processed, df_facial_processed, FEATURE_NAMES)
        if df_final_features.empty:
            raise ValueError("Combined feature DataFrame is empty.")

        final_timestamps_ms = (df_final_features.index.total_seconds() * 1000).to_numpy()
        sequence_values_np = df_final_features.values

        print("Step 4: Scaling features...")
        sequence_values_np = df_final_features.values
        if np.isnan(sequence_values_np).any() or np.isinf(sequence_values_np).any():
            sequence_values_np = np.nan_to_num(sequence_values_np, nan=0.0, posinf=0.0, neginf=0.0)

        scaled_features_np = scale_data_for_inference(sequence_values_np, scaler)

        print("Step 5: Running inference...")

        original_td_index_for_segments = df_final_features.index

        segments_bilstm = run_inference_for_sequence_data(
            model_to_use=model_bilstm,
            feature_sequence_scaled_np=scaled_features_np,
            original_timedelta_index=original_td_index_for_segments,
            model_max_len=MODEL_EXPECTED_LEN,
            num_model_output_classes=model_bilstm.output_shape[-1],
        )

        segments_bigru = run_inference_for_sequence_data(
            model_to_use=model_bigru,
            feature_sequence_scaled_np=scaled_features_np,
            original_timedelta_index=original_td_index_for_segments,
            model_max_len=MODEL_EXPECTED_LEN,
            num_model_output_classes=model_bigru.output_shape[-1]
        )

        response_data = {
            "bilstm_segments": segments_bilstm,
            "bigru_segments": segments_bigru,
            "message": "Processing successful",
            "num_frames_processed": scaled_features_np.shape[0],
            "num_features_final": scaled_features_np.shape[1]
        }
        return jsonify(response_data)

    except FileNotFoundError as e_fnf:
        print(f"File Not Found Error during full pipeline: {e_fnf}")
        traceback.print_exc()
        return make_response(jsonify({"error": f"A required file or resource was not found: {e_fnf}"}), 500)
    except ValueError as e_val:
        print(f"Value Error during full pipeline: {e_val}")
        traceback.print_exc()
        return make_response(jsonify({"error": f"Data processing error: {e_val}"}), 400)
    except RuntimeError as e_rt:
        print(f"Runtime Error during full pipeline: {e_rt}")
        traceback.print_exc()
        return make_response(jsonify({"error": f"Runtime error during processing: {e_rt}"}), 500)
    except Exception as e:
        print(f"Unexpected Server Error during full pipeline: {e}")
        traceback.print_exc()
        return make_response(jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500)
    finally:
        if temp_dir_obj and temp_dir.exists():
            print(f"Cleaning up temporary directory: {temp_dir}")
            # shutil.rmtree(temp_dir) # Comentează pentru debug, decomentează pentru producție
            print(f"  (Cleanup skipped for debugging, path: {temp_dir})")
        elif temp_dir_obj:
            temp_dir_obj.cleanup()


if __name__ == '__main__':
    load_global_components()
    if scaler is None or model_bilstm is None or model_bigru is None or not FEATURE_NAMES:
        print("Exiting: Critical components (Scaler, Models, Feature Names) failed to load.")
        exit()
    print("\n--- Starting Flask Server for Full Pipeline Demo ---")
    app.run(host='0.0.0.0', port=5000, debug=True)