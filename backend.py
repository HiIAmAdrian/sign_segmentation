import flask
from flask import request, jsonify, make_response
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import io  # To read files from memory
import traceback
from flask_cors import CORS

from process_data_teslasuit_gloves import GLOVE_RIGHT_DATA_DIR, GLOVE_LEFT_DATA_DIR

# --- Configuration ---
# Paths relative to where this script is run
PROCESSED_DATA_DIR = Path("./processed_combined_data_both_gloves")  # For loading feature names
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "combined_interpolated_processed_sequences.pkl"  # To get feature names
SCALER_FILE = PROCESSED_DATA_DIR / "combined_interpolated_scaler.pkl"
MODEL_SAVE_DIR = Path("./trained_models")
# !!! Paths for BOTH models !!!
MODEL_FILE_BILSTM = MODEL_SAVE_DIR / "bilstm_best.keras"
MODEL_FILE_BIGRU = MODEL_SAVE_DIR / "bigru_best.keras"

# Preprocessing Parameters (should match training)
MERGE_TOLERANCE_MS = 30
PADDING_TYPE = 'post'
LABEL_O = 0
LABEL_I = 1
NUM_CLASSES = 2  # O and I

# --- Feature Selection Configuration Flags ---
# !!! MUST MATCH THE FLAGS USED FOR TRAINING THE LOADED MODELS !!!
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
# --- (End of Flags Section) ---


# --- Pre-load Scaler, Models, and Feature Names ---
model_bilstm = None
model_bigru = None
scaler = None
FEATURE_NAMES = []
NUM_FEATURES = 0
MODEL_EXPECTED_LEN = None  # Will store expected length (should be same for both)

try:
    print(f"Loading feature names from: {PROCESSED_DATA_FILE}")
    with open(PROCESSED_DATA_FILE, 'rb') as f:
        data_info = pickle.load(f)
        if 'feature_names' in data_info:
            FEATURE_NAMES = data_info['feature_names']
            NUM_FEATURES = len(FEATURE_NAMES)
            print(f"Loaded {NUM_FEATURES} feature names.")
        else:
            raise ValueError("feature_names not found in processed data pkl file.")

    print(f"Loading scaler from: {SCALER_FILE}")
    with open(SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)
    print("Scaler loaded.")

    # --- Load BiLSTM Model ---
    print(f"Loading BiLSTM model from: {MODEL_FILE_BILSTM}")
    if not MODEL_FILE_BILSTM.exists():
        raise FileNotFoundError(f"BiLSTM model file not found: {MODEL_FILE_BILSTM}")
    model_bilstm = tf.keras.models.load_model(MODEL_FILE_BILSTM)
    print("BiLSTM Model loaded.")
    try:
        expected_len_bilstm = model_bilstm.input_shape[1]
        print(f" -> BiLSTM expects input length: {expected_len_bilstm}")
    except Exception:
        expected_len_bilstm = None

    # --- Load BiGRU Model ---
    print(f"Loading BiGRU model from: {MODEL_FILE_BIGRU}")
    if not MODEL_FILE_BIGRU.exists():
        raise FileNotFoundError(f"BiGRU model file not found: {MODEL_FILE_BIGRU}")
    model_bigru = tf.keras.models.load_model(MODEL_FILE_BIGRU)
    print("BiGRU Model loaded.")
    try:
        expected_len_bigru = model_bigru.input_shape[1]
        print(f" -> BiGRU expects input length: {expected_len_bigru}")
    except Exception:
        expected_len_bigru = None

    # --- Verify Model Compatibility ---
    if expected_len_bilstm != expected_len_bigru:
        # This could happen if one expects None and the other a fixed length, which is okay.
        # But if both are fixed and different, it's an issue.
        if expected_len_bilstm is not None and expected_len_bigru is not None:
            raise ValueError(
                f"Model input length mismatch! BiLSTM expects {expected_len_bilstm}, BiGRU expects {expected_len_bigru}.")
        else:
            print("Warning: Models have different input length expectations (one fixed, one variable). Proceeding.")
            # Use None if either is None, otherwise use the fixed length if one is fixed
            MODEL_EXPECTED_LEN = None if expected_len_bilstm is None or expected_len_bigru is None else expected_len_bilstm
    else:
        MODEL_EXPECTED_LEN = expected_len_bilstm  # They are the same

    print(f"Using effective model input length: {MODEL_EXPECTED_LEN}")


except FileNotFoundError as e:
    print(f"FATAL ERROR: Could not find required file: {e}")
    exit()
except Exception as e:
    print(f"FATAL ERROR: Failed to load scaler/model/features: {e}")
    traceback.print_exc()
    exit()

# --- Feature List Generation (Reuse logic) ---
print("--- Feature Selection Settings ---")

# 1. Generate SUIT Feature List based on flags
SUIT_FEATURES_TO_KEEP = []
selected_suit_bone_rotations = []
selected_suit_bone_positions = []
selected_suit_hip_position = []
selected_suit_biomech_features = []

# Define bones whose data might come from the suit
upper_body_bones_suit = [
    "hips", "spine", "upper_spine", "neck", "head",
    "left_shoulder", "right_shoulder",
    "left_upper_arm", "right_upper_arm",
    "left_lower_arm", "right_lower_arm",
    # "left_hand", "right_hand"  # Initially include hands, may be skipped below
]

if USE_SUIT_ROTATIONS:
    print("Including SUIT: Upper Body Rotations")
    for bone in upper_body_bones_suit:
        # Skip hand rotations if they are explicitly taken from gloves
        if bone == "left_hand" and USE_GLOVE_LEFT_HAND_ROOT:
            continue
        if bone == "right_hand" and USE_GLOVE_RIGHT_HAND_ROOT:
            continue
        selected_suit_bone_rotations.extend([f"{bone}.rotation.{axis}" for axis in ['w', 'x', 'y', 'z']])
    SUIT_FEATURES_TO_KEEP.extend(selected_suit_bone_rotations)
else:
    print("Excluding SUIT: Upper Body Rotations")

if USE_SUIT_POSITIONS:
    print("Including SUIT: Upper Body Positions (excluding Hips)")
    for bone in upper_body_bones_suit:
        # Always skip Hips position here, handle separately
        if bone == "hips":
            continue
        # Skip hand positions if they are explicitly taken from gloves
        if bone == "left_hand" and USE_GLOVE_LEFT_HAND_ROOT:
            continue
        if bone == "right_hand" and USE_GLOVE_RIGHT_HAND_ROOT:
            continue
        selected_suit_bone_positions.extend([f"{bone}.position.{axis}" for axis in ['x', 'y', 'z']])
    SUIT_FEATURES_TO_KEEP.extend(selected_suit_bone_positions)
else:
    print("Excluding SUIT: Upper Body Positions")

if USE_SUIT_HIPS_POSITION:
    print("Including SUIT: Hips Position")
    selected_suit_hip_position.extend([f"hips.position.{axis}" for axis in ['x', 'y', 'z']])
    SUIT_FEATURES_TO_KEEP.extend(selected_suit_hip_position)
else:
    print("Excluding SUIT: Hips Position")

if USE_SUIT_BIOMECH:
    print("Including SUIT: Biomechanical Angles/Vel/Acc")
    relevant_biomech_joints = [
        "PelvisTilt", "PelvisList", "PelvisRotation",
        "HipFlexExtR", "HipFlexExtL", "HipAddAbdR", "HipAddAbdL", "HipRotR", "HipRotL",
        "ElbowFlexExtR", "ElbowFlexExtL",
        "ForearmProSupR", "ForearmProSupL",
        "WristFlexExtR", "WristFlexExtL", "WristDeviationR", "WristDeviationL",
        "LumbarFlexExt", "LumbarLatFlex", "LumbarRot",
        "LowerThoraxFlexExt", "LowerThoraxLatFlex", "LowerThoraxRot",
        "UpperThoraxFlexExt", "UpperThoraxLatFlex", "UpperThoraxRot",
        "ShoulderFlexExtR", "ShoulderFlexExtL", "ShoulderAddAbdR", "ShoulderAddAbdL", "ShoulderRotR", "ShoulderRotL"
    ]
    selected_suit_biomech_features = [
        f"{joint}.{measure}"
        for joint in relevant_biomech_joints
        for measure in ["angle", "angular_v", "angular_acc"]
    ]
    SUIT_FEATURES_TO_KEEP.extend(selected_suit_biomech_features)
else:
    print("Excluding SUIT: Biomechanical Angles/Vel/Acc")

print(f"Total SUIT features selected: {len(SUIT_FEATURES_TO_KEEP)}")


# 2. Generate GLOVE Feature Lists based on flags
def get_finger_features_configurable(hand_prefix, glove_dir,
                                     use_finger_rotations, use_finger_positions, use_hand_root):
    """Generates list of finger features based on config flags."""
    features = []
    fingers = ["thumb", "index", "middle", "ring", "little"]
    phalanges = ["proximal", "intermediate", "distal"]

    # Check available columns from a sample file
    try:
        sample_file = next(glove_dir.glob("*.csv"))
        sample_cols = pd.read_csv(sample_file, nrows=0).columns
    except (StopIteration, FileNotFoundError):
        print(f"Warning: No sample glove file found in {glove_dir} to check structure.")
        sample_cols = []

    # Generate Finger Joint Features
    for finger in fingers:
        current_phalanges = phalanges
        thumb_intermediate_check_rot = f"{hand_prefix}_{finger}_intermediate.rotation.w"
        thumb_intermediate_check_pos = f"{hand_prefix}_{finger}_intermediate.position.x"
        # Check if intermediate thumb is missing based on either rotation or position columns
        if finger == "thumb" and (thumb_intermediate_check_rot not in sample_cols and
                                  thumb_intermediate_check_pos not in sample_cols):
            current_phalanges = ["proximal", "distal"]

        for phalanx in current_phalanges:
            if use_finger_rotations:
                rot_features = [f"{hand_prefix}_{finger}_{phalanx}.rotation.{axis}" for axis in ['w', 'x', 'y', 'z']]
                # Only add if columns actually exist in sample file
                features.extend([fea for fea in rot_features if fea in sample_cols])
            if use_finger_positions:
                pos_features = [f"{hand_prefix}_{finger}_{phalanx}.position.{axis}" for axis in ['x', 'y', 'z']]
                # Only add if columns actually exist in sample file
                features.extend([fea for fea in pos_features if fea in sample_cols])

    # Generate Hand Root Features (if requested and available)
    if use_hand_root:
        hand_rot_features = [f"{hand_prefix}_hand.rotation.{axis}" for axis in ['w', 'x', 'y', 'z']]
        hand_pos_features = [f"{hand_prefix}_hand.position.{axis}" for axis in ['x', 'y', 'z']]
        # Add if columns exist in sample file
        features.extend([fea for fea in hand_rot_features if fea in sample_cols])
        features.extend([fea for fea in hand_pos_features if fea in sample_cols])

    return features


# Generate Right Glove Features
GLOVE_RIGHT_FEATURES_TO_KEEP = []
print("\n--- Right Glove ---")
if USE_GLOVE_RIGHT_FINGER_ROTATIONS or USE_GLOVE_RIGHT_FINGER_POSITIONS or USE_GLOVE_RIGHT_HAND_ROOT:
    GLOVE_RIGHT_FEATURES_TO_KEEP = get_finger_features_configurable(
        "right", GLOVE_RIGHT_DATA_DIR,
        USE_GLOVE_RIGHT_FINGER_ROTATIONS,
        USE_GLOVE_RIGHT_FINGER_POSITIONS,
        USE_GLOVE_RIGHT_HAND_ROOT
    )
    print(f"Including RIGHT GLOVE features ({len(GLOVE_RIGHT_FEATURES_TO_KEEP)}):")
    if USE_GLOVE_RIGHT_FINGER_ROTATIONS: print("  - Finger Rotations")
    if USE_GLOVE_RIGHT_FINGER_POSITIONS: print("  - Finger Positions")
    if USE_GLOVE_RIGHT_HAND_ROOT: print("  - Hand Root (Position/Rotation)")
else:
    print("Excluding ALL RIGHT GLOVE features.")

# Generate Left Glove Features
GLOVE_LEFT_FEATURES_TO_KEEP = []
print("\n--- Left Glove ---")
if USE_GLOVE_LEFT_FINGER_ROTATIONS or USE_GLOVE_LEFT_FINGER_POSITIONS or USE_GLOVE_LEFT_HAND_ROOT:
    GLOVE_LEFT_FEATURES_TO_KEEP = get_finger_features_configurable(
        "left", GLOVE_LEFT_DATA_DIR,
        USE_GLOVE_LEFT_FINGER_ROTATIONS,
        USE_GLOVE_LEFT_FINGER_POSITIONS,
        USE_GLOVE_LEFT_HAND_ROOT
    )
    print(f"Including LEFT GLOVE features ({len(GLOVE_LEFT_FEATURES_TO_KEEP)}):")
    if USE_GLOVE_LEFT_FINGER_ROTATIONS: print("  - Finger Rotations")
    if USE_GLOVE_LEFT_FINGER_POSITIONS: print("  - Finger Positions")
    if USE_GLOVE_LEFT_HAND_ROOT: print("  - Hand Root (Position/Rotation)")
else:
    print("Excluding ALL LEFT GLOVE features.")

# Remove duplicates just in case (shouldn't happen with this logic, but safe)
SUIT_FEATURES_TO_KEEP = sorted(list(set(SUIT_FEATURES_TO_KEEP)))
GLOVE_RIGHT_FEATURES_TO_KEEP = sorted(list(set(GLOVE_RIGHT_FEATURES_TO_KEEP)))
GLOVE_LEFT_FEATURES_TO_KEEP = sorted(list(set(GLOVE_LEFT_FEATURES_TO_KEEP)))


# --- End of Feature List Generation ---


# --- Helper Functions ---
# (check_features_exist, load_and_prepare_df_from_stream, extract_segments_from_labels remain the same)
def check_features_exist(df, feature_list, filename):
    present_features = [f for f in feature_list if f in df.columns]
    missing = [f for f in feature_list if f not in df.columns]
    if missing: print(f"  --> Warning: Missing features in {filename}: {missing}")
    return present_features


def load_and_prepare_df_from_stream(file_stream, selected_features, filename):
    try:
        df = pd.read_csv(file_stream)
        if 'frame_timestamp' not in df.columns: raise ValueError(f"'frame_timestamp' missing in {filename}")
        cols_to_keep = ['frame_timestamp'] + selected_features
        actual_features_with_ts = check_features_exist(df, cols_to_keep, filename)
        actual_features = [f for f in actual_features_with_ts if f != 'frame_timestamp']
        if not actual_features: raise ValueError(f"No selected features found in {filename}")
        df_selected = df[actual_features_with_ts].copy()
        try:
            df_selected['frame_timestamp'] = pd.to_timedelta(df_selected['frame_timestamp'], unit='ms', errors='coerce')
        except ValueError:
            df_selected['frame_timestamp'] = pd.to_numeric(df_selected['frame_timestamp'], errors='coerce')
            df_selected['frame_timestamp'] = pd.to_timedelta(df_selected['frame_timestamp'], unit='ms', errors='coerce')
        df_selected.dropna(subset=['frame_timestamp'], inplace=True)
        if df_selected.empty: raise ValueError(f"No valid timestamps found in {filename} after conversion.")
        for col in actual_features: df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')
        if df_selected[actual_features].isnull().any().any():
            print(f"  --> Warning: NaNs found in raw data of {filename}. Applying ffill/bfill.")
            df_selected[actual_features] = df_selected[actual_features].ffill().bfill().fillna(0)
        df_selected = df_selected.sort_values(by='frame_timestamp').set_index('frame_timestamp')
        if df_selected.index.has_duplicates:
            dup_count = df_selected.index.duplicated().sum()
            print(f"  --> Warning: Found {dup_count} duplicate timestamps in {filename}. Keeping first entry.")
            df_selected = df_selected[~df_selected.index.duplicated(keep='first')]
        return df_selected, actual_features
    except Exception as e:
        raise ValueError(f"Failed to process file {filename}: {e}") from e


def extract_segments_from_labels(labels, timestamps_ms):
    segments = []
    start_frame = None
    n_frames = len(labels)
    if n_frames == 0 or timestamps_ms is None or len(timestamps_ms) != n_frames:
        print(
            f"Warning: Cannot extract segments. Label length ({n_frames}) or timestamp length ({len(timestamps_ms) if timestamps_ms is not None else 'None'}) mismatch or empty.")
        return []  # Return empty list if lengths mismatch

    for i in range(n_frames):
        current_label = labels[i]
        if current_label == LABEL_I and start_frame is None:
            start_frame = i
        elif current_label == LABEL_O and start_frame is not None:
            end_frame = i - 1
            start_ms = int(timestamps_ms[start_frame])
            end_ms = int(timestamps_ms[end_frame])
            segments.append({"start_ms": start_ms, "end_ms": end_ms})
            start_frame = None
        elif current_label == LABEL_I and i == n_frames - 1 and start_frame is not None:
            end_frame = i
            start_ms = int(timestamps_ms[start_frame])
            end_ms = int(timestamps_ms[end_frame])
            segments.append({"start_ms": start_ms, "end_ms": end_ms})
    return segments


# --- Flask App ---
app = flask.Flask(__name__)
CORS(app)


@app.route('/segment', methods=['POST'])
def segment_mocap_data():
    print("\nReceived segmentation request...")
    # --- 1. Get Uploaded Files ---
    if 'suit_file' not in request.files or \
            'glove_right_file' not in request.files or \
            'glove_left_file' not in request.files:
        return make_response(
            jsonify({"error": "Missing one or more files (expected: suit_file, glove_right_file, glove_left_file)"}),
            400)

    suit_file = request.files['suit_file']
    glove_r_file = request.files['glove_right_file']
    glove_l_file = request.files['glove_left_file']

    if suit_file.filename == '' or glove_r_file.filename == '' or glove_l_file.filename == '':
        return make_response(jsonify({"error": "One or more uploaded files have no filename"}), 400)

    try:
        # --- 2. Load and Prepare DataFrames ---
        df_suit, actual_suit_features = load_and_prepare_df_from_stream(suit_file.stream, SUIT_FEATURES_TO_KEEP,
                                                                        "suit_file")
        original_suit_timestamps_ms = df_suit.index.total_seconds() * 1000  # Keep for mapping

        # --- Load Gloves ---
        try:
            df_glove_r, actual_glove_right_features = load_and_prepare_df_from_stream(glove_r_file.stream,
                                                                                      GLOVE_RIGHT_FEATURES_TO_KEEP,
                                                                                      "glove_right_file")
        except ValueError as e:
            print(f"Warning: Right glove: {e}")
            actual_glove_right_features = []
            df_glove_r = pd.DataFrame(index=pd.TimedeltaIndex([], name='frame_timestamp'))
        try:
            df_glove_l, actual_glove_left_features = load_and_prepare_df_from_stream(glove_l_file.stream,
                                                                                     GLOVE_LEFT_FEATURES_TO_KEEP,
                                                                                     "glove_left_file")
        except ValueError as e:
            print(f"Warning: Left glove: {e}")
            actual_glove_left_features = []
            df_glove_l = pd.DataFrame(index=pd.TimedeltaIndex([], name='frame_timestamp'))

        # --- 3. Align using merge_asof ---
        print("Aligning timestamps...")
        df_suit_reset = df_suit.reset_index()
        df_glove_r_reset = df_glove_r.reset_index()
        df_glove_l_reset = df_glove_l.reset_index()
        df_merged_right = pd.merge_asof(df_suit_reset, df_glove_r_reset, on='frame_timestamp', direction='nearest',
                                        tolerance=pd.Timedelta(milliseconds=MERGE_TOLERANCE_MS), suffixes=('', '_gr'))
        rename_dict_r = {f"{col}_gr": col for col in actual_glove_right_features if
                         f"{col}_gr" in df_merged_right.columns}
        df_merged_right.rename(columns=rename_dict_r, inplace=True)
        df_combined = pd.merge_asof(df_merged_right, df_glove_l_reset, on='frame_timestamp', direction='nearest',
                                    tolerance=pd.Timedelta(milliseconds=MERGE_TOLERANCE_MS), suffixes=('', '_gl'))
        rename_dict_l = {f"{col}_gl": col for col in actual_glove_left_features if f"{col}_gl" in df_combined.columns}
        df_combined.rename(columns=rename_dict_l, inplace=True)

        # --- 4. Interpolate ---
        df_combined.dropna(subset=['frame_timestamp'], inplace=True)
        df_combined = df_combined.drop_duplicates(subset=['frame_timestamp'], keep='first')
        df_combined = df_combined.set_index('frame_timestamp').sort_index()
        features_to_interpolate = actual_glove_right_features + actual_glove_left_features
        features_to_interpolate = [f for f in features_to_interpolate if f in df_combined.columns]
        if features_to_interpolate:
            nan_count_before = df_combined[features_to_interpolate].isnull().sum().sum()
            if nan_count_before > 0:
                print(f"Interpolating {nan_count_before} missing glove data points...")
                df_combined[features_to_interpolate] = df_combined[features_to_interpolate].interpolate(method='time')
                nan_count_after = df_combined[features_to_interpolate].isnull().sum().sum()
                if nan_count_after > 0: df_combined[features_to_interpolate] = df_combined[
                    features_to_interpolate].ffill().bfill().fillna(0)

        # --- 5. Final Feature Selection and Ordering ---
        missing_features = [f for f in FEATURE_NAMES if f not in df_combined.columns]
        if missing_features:
            raise ValueError(f"Input data missing expected features: {missing_features}")
        df_final_features = df_combined[FEATURE_NAMES]
        combined_sequence_np = df_final_features.values
        final_timestamps_ms = (
                df_final_features.index.total_seconds() * 1000).to_numpy()  # Timestamps for output mapping

        # --- 6. Scale Features ---
        print("Scaling features...")
        if np.isnan(combined_sequence_np).any() or np.isinf(combined_sequence_np).any():
            print("Warning: NaNs/Infs found before scaling. Replacing with 0.")
            combined_sequence_np = np.nan_to_num(combined_sequence_np)
        scaled_features = scaler.transform(combined_sequence_np)

        # --- 7. Pad Sequence for Model ---
        current_len = scaled_features.shape[0]
        target_len = MODEL_EXPECTED_LEN if MODEL_EXPECTED_LEN is not None else current_len
        print(f"Padding sequence from {current_len} to {target_len} frames...")
        padded_features = pad_sequences([scaled_features], padding=PADDING_TYPE, dtype='float32', maxlen=target_len)
        # Reshape to (1, target_len, num_features) - already done by pad_sequences with list input

        # --- 8. Predict with BOTH Models ---
        print("Running BiLSTM prediction...")
        predictions_bilstm = model_bilstm.predict(padded_features)  # Shape (1, padded_length, num_classes)
        print("Running BiGRU prediction...")
        predictions_bigru = model_bigru.predict(padded_features)  # Shape (1, padded_length, num_classes)

        # --- 9. Decode Predictions for BOTH Models ---
        # Get predictions for the original length (ignore padding)
        pred_labels_onehot_bilstm = predictions_bilstm[0, :current_len, :]
        pred_labels_onehot_bigru = predictions_bigru[0, :current_len, :]
        # Convert probabilities to class labels (0 or 1)
        predicted_labels_bilstm = np.argmax(pred_labels_onehot_bilstm, axis=-1)
        predicted_labels_bigru = np.argmax(pred_labels_onehot_bigru, axis=-1)
        print(f"Decoded {len(predicted_labels_bilstm)} BiLSTM labels and {len(predicted_labels_bigru)} BiGRU labels.")

        # --- 10. Extract Segments for BOTH Models ---
        print("Extracting segments...")
        segments_bilstm = extract_segments_from_labels(predicted_labels_bilstm, final_timestamps_ms)
        segments_bigru = extract_segments_from_labels(predicted_labels_bigru, final_timestamps_ms)
        print(f"Found {len(segments_bilstm)} BiLSTM segments and {len(segments_bigru)} BiGRU segments.")

        # --- 11. Return JSON Response ---
        response_data = {
            "bilstm_segments": segments_bilstm,
            "bigru_segments": segments_bigru
        }
        return jsonify(response_data)

    except ValueError as e:
        print(f"Data Processing Error: {e}")
        traceback.print_exc()
        return make_response(jsonify({"error": f"Error processing input data: {e}"}), 400)
    except Exception as e:
        print(f"Unexpected Server Error: {e}")
        traceback.print_exc()
        return make_response(jsonify({"error": "An internal server error occurred."}), 500)


# --- Run Flask App ---
if __name__ == '__main__':
    # Make sure models and scaler are loaded before starting the server
    if scaler is None or model_bilstm is None or model_bigru is None or not FEATURE_NAMES:
        print("Exiting: Scaler, Models or Feature Names failed to load.")
        exit()
    print("\n--- Starting Flask Server ---")
    app.run(host='0.0.0.0', port=5000, debug=True)  # Use debug=False for production
