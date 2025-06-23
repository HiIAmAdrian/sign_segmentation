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


TRIM_SECONDS_APPLIED_TO_INPUT = 1.0
process_bag_for_landmarks_external = None
create_landmark_opts_for_landmarks_external = None
process_bag_for_blendshapes_external = None
create_blendshape_opts_external = None
smooth_landmarks_file_external = None
normalize_script_module = None
ats_extract_timestamps_external = None
ats_pivot_landmarks_external = None
ats_pivot_blendshapes_external = None
ats_run_add_timestamps_and_combine_facial_external = None

print("INFO: Facial processing modules are intentionally disabled for feature extraction in this version.")

try:
    from infer_logic_functional_fara_fata import (
        run_inference_for_sequence_data,
        scale_data_for_inference
    )
except ImportError as e:
    print(f"FATAL: Could not import from 'infer_logic_functional.py' (sau 'infer_logic_functional_fara_fata.py'): {e}")
    print("       Ensure the inference logic Python file is in the correct location and has no import errors itself.")
    exit()

app = Flask(__name__)
CORS(app)


MODEL_PATH_BILSTM = Path("./trained_models_final/bilstm_best_final.keras")
MODEL_PATH_BIGRU = Path("./trained_models_final/bigru_best_final.keras")
SCALER_PATH = Path("./final_data_ts_gloves_only/ts_gloves_only_scaler.pkl")
TRAINING_CONFIG_PKL = Path("./final_data_ts_gloves_only/all_data_features_ts_gloves_only.pkl")

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

        expected_scaler_features = -1
        if hasattr(scaler, 'n_features_in_'):
            expected_scaler_features = scaler.n_features_in_
        elif hasattr(scaler, 'mean_') and scaler.mean_ is not None:
            expected_scaler_features = scaler.mean_.shape[-1]

        if expected_scaler_features != -1 and expected_scaler_features != NUM_FEATURES:
            print(
                f"WARNING: Scaler expects {expected_scaler_features} features, but loaded feature_names list has {NUM_FEATURES}.")
        elif expected_scaler_features == -1:
            print(f"WARNING: Could not determine expected feature count from scaler object of type {type(scaler)}. Cannot verify feature count.")
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
        if isinstance(e, ValueError) and ("'frame_timestamp_us' or 'frame_timestamp' missing" in str(e)):
            raise
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
    print(f"  Processing TeslaSuit CSVs (trim_sec_ts: {trim_sec_ts})...")
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


def combine_ts_and_facial_for_demo(df_ts, df_facial, feature_names_from_training):
    print(
        f"  Combining TS ({df_ts.shape if df_ts is not None else 'None'}) and Facial (None) for demo...")
    if df_ts is None or df_ts.empty: raise ValueError("TeslaSuit DataFrame is empty for combination.")

    TIMESTAMP_COL_FOR_MERGE = 'normalized_timestamp_us'

    if df_ts.index.name != TIMESTAMP_COL_FOR_MERGE:
        df_ts = df_ts.rename_axis(TIMESTAMP_COL_FOR_MERGE)

    df_ts_reset = df_ts.reset_index()

    if df_facial is None or df_facial.empty:
        print("    Facial DataFrame is None/empty, proceeding with TeslaSuit data only.")
        df_final_aligned = df_ts_reset.set_index(TIMESTAMP_COL_FOR_MERGE).sort_index()
    else:
        print("    WARNING: Facial DataFrame is unexpectedly not empty. Merging (this may not be intended).")
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
    print(f"  Combined data for model (TS only, facial features zeroed if in training config), shape: {df_final_for_model.shape}")
    return df_final_for_model


@app.route('/segment_pipeline', methods=['POST'])
def segment_full_pipeline_endpoint():
    print("\nReceived request for TESLASUIT-ONLY (but BAG file accepted) PIPELINE segmentation...")
    required_files = ['suit_file', 'glove_right_file', 'glove_left_file', 'bag_file']
    if not all(k in request.files for k in required_files):
        missing = [k for k in required_files if k not in request.files]
        return make_response(jsonify({"error": f"Missing required files: {', '.join(missing)}"}), 400)

    temp_dir_obj = tempfile.TemporaryDirectory(prefix="sl_demo_ts_bag_")
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
        print(f"  Saved BAG file to: {bag_file_path} (will not be used for feature extraction)")


        print("Step 1: Processing TeslaSuit data...")
        df_teslasuit_processed = process_teslasuit_for_demo_backend(
            suit_path, glove_r_path, glove_l_path,
            trim_sec_ts=TRIM_SECONDS_APPLIED_TO_INPUT
        )
        if df_teslasuit_processed.empty:
            raise ValueError("TeslaSuit data processing resulted in an empty DataFrame.")

        print("Step 2: Facial data processing (SKIPPED as BAG file is not used for features).")
        df_facial_processed = None

        print("Step 3: Preparing TeslaSuit features for model...")
        df_final_features = combine_ts_and_facial_for_demo(df_teslasuit_processed, df_facial_processed, FEATURE_NAMES)
        if df_final_features.empty:
            raise ValueError("Combined feature DataFrame is empty (after TS processing).")

        print("Step 4: Scaling features...")
        sequence_values_np = df_final_features.values
        if np.isnan(sequence_values_np).any() or np.isinf(sequence_values_np).any():
            sequence_values_np = np.nan_to_num(sequence_values_np, nan=0.0, posinf=0.0, neginf=0.0)

        scaled_features_np = scale_data_for_inference(sequence_values_np, scaler)

        print("Step 5: Running inference...")
        original_td_index_for_segments = df_final_features.index

        trim_offset_ms_for_inference = int(TRIM_SECONDS_APPLIED_TO_INPUT * 1000)

        segments_bilstm = run_inference_for_sequence_data(
            model_to_use=model_bilstm,
            feature_sequence_scaled_np=scaled_features_np,
            original_timedelta_index=original_td_index_for_segments,
            model_max_len=MODEL_EXPECTED_LEN,
            num_model_output_classes=model_bilstm.output_shape[-1],
            initial_trim_offset_ms=trim_offset_ms_for_inference
        )

        segments_bigru = run_inference_for_sequence_data(
            model_to_use=model_bigru,
            feature_sequence_scaled_np=scaled_features_np,
            original_timedelta_index=original_td_index_for_segments,
            model_max_len=MODEL_EXPECTED_LEN,
            num_model_output_classes=model_bigru.output_shape[-1],
            initial_trim_offset_ms=trim_offset_ms_for_inference
        )

        response_data = {
            "bilstm_segments": segments_bilstm,
            "bigru_segments": segments_bigru,
            "message": "Processing successful (TeslaSuit data only; BAG file received but not used for features)",
            "num_frames_processed": scaled_features_np.shape[0],
            "num_features_final": scaled_features_np.shape[1],
            "trim_applied_input_ms": trim_offset_ms_for_inference
        }
        return jsonify(response_data)

    except FileNotFoundError as e_fnf:
        print(f"File Not Found Error during pipeline: {e_fnf}")
        traceback.print_exc()
        return make_response(jsonify({"error": f"A required file or resource was not found: {e_fnf}"}), 500)
    except ValueError as e_val:
        print(f"Value Error during pipeline: {e_val}")
        traceback.print_exc()
        return make_response(jsonify({"error": f"Data processing error: {e_val}"}), 400)
    except RuntimeError as e_rt:
        print(f"Runtime Error during pipeline: {e_rt}")
        traceback.print_exc()
        return make_response(jsonify({"error": f"Runtime error during processing: {e_rt}"}), 500)
    except Exception as e:
        print(f"Unexpected Server Error during pipeline: {e}")
        traceback.print_exc()
        return make_response(jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500)
    finally:
        if temp_dir_obj and temp_dir.exists():
            print(f"Cleaning up temporary directory: {temp_dir}")
            # shutil.rmtree(temp_dir) # Uncomment for production
            print(f"  (Cleanup skipped for debugging, path: {temp_dir})")
        elif temp_dir_obj:
            temp_dir_obj.cleanup()


if __name__ == '__main__':
    load_global_components()
    if scaler is None or model_bilstm is None or model_bigru is None or not FEATURE_NAMES:
        print("Exiting: Critical components (Scaler, Models, Feature Names) failed to load.")
        exit()
    print("\n--- Starting Flask Server for TeslaSuit-Only (BAG accepted but not used for features) Pipeline Demo ---")
    print(f"--- Input data trim setting: {TRIM_SECONDS_APPLIED_TO_INPUT} seconds ---")
    app.run(host='0.0.0.0', port=5000, debug=True)