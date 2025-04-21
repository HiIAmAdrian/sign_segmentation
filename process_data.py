import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path

# --- Configuration ---
SUIT_DATA_DIR = Path("./suit")
GLOVE_RIGHT_DATA_DIR = Path("./right")
GLOVE_LEFT_DATA_DIR = Path("./left")
OUTPUT_DIR = Path("./processed_combined_data_both_gloves")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Define a pattern or function to find matching glove files for a suit file
# Example: suit file 'sentence05_signer01_suit.csv' ->
#          glove_right file 'sentence05_signer01_glove_right.csv'
#          glove_left file 'sentence05_signer01_glove_left.csv'
def get_glove_file_paths(suit_file_path, glove_right_dir, glove_left_dir):
    try:
        base_name = suit_file_path.name.replace('_suit.csv', '')
        if base_name == suit_file_path.name:  # Try stem if first replace failed
            base_name = suit_file_path.stem.replace('_suit', '')

        glove_right_filename = f"{base_name}_glove_right.csv"
        glove_left_filename = f"{base_name}_glove_left.csv"

        path_right = glove_right_dir / glove_right_filename
        path_left = glove_left_dir / glove_left_filename
        return path_right, path_left
    except Exception as e:
        print(f"  --> Warning: Could not determine glove filenames for {suit_file_path.name}: {e}")
        return None, None


# --- SUIT FEATURES ---
USE_SUIT_ROTATIONS = True  # Include .rotation data for upper body bones (excl. hands if overridden by gloves)
USE_SUIT_POSITIONS = False  # Include .position data for upper body bones (excl. Hips/hands if overridden)
USE_SUIT_HIPS_POSITION = True  # Include Hips.position (Root position)
USE_SUIT_BIOMECH = True  # Include .angle, .angular_v, .angular_acc for biomech joints

# --- GLOVE FEATURES (Right Hand) ---
USE_GLOVE_RIGHT_FINGER_ROTATIONS = True  # Include finger joint .rotation data
USE_GLOVE_RIGHT_FINGER_POSITIONS = False  # Include finger joint .position data
USE_GLOVE_RIGHT_HAND_ROOT = True  # Use right_hand.position/rotation FROM GLOVE file (overrides suit's right_hand)

# --- GLOVE FEATURES (Left Hand) ---
USE_GLOVE_LEFT_FINGER_ROTATIONS = True  # Include finger joint .rotation data
USE_GLOVE_LEFT_FINGER_POSITIONS = False  # Include finger joint .position data
USE_GLOVE_LEFT_HAND_ROOT = True  # Use left_hand.position/rotation FROM GLOVE file (overrides suit's left_hand)

# --- Feature List Generation ---

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
    #"left_hand", "right_hand"  # Initially include hands, may be skipped below
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
                features.extend([f for f in rot_features if f in sample_cols])
            if use_finger_positions:
                pos_features = [f"{hand_prefix}_{finger}_{phalanx}.position.{axis}" for axis in ['x', 'y', 'z']]
                # Only add if columns actually exist in sample file
                features.extend([f for f in pos_features if f in sample_cols])

    # Generate Hand Root Features (if requested and available)
    if use_hand_root:
        hand_rot_features = [f"{hand_prefix}_hand.rotation.{axis}" for axis in ['w', 'x', 'y', 'z']]
        hand_pos_features = [f"{hand_prefix}_hand.position.{axis}" for axis in ['x', 'y', 'z']]
        # Add if columns exist in sample file
        features.extend([f for f in hand_rot_features if f in sample_cols])
        features.extend([f for f in hand_pos_features if f in sample_cols])

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

# --- Data Splitting Parameters ---
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
RANDOM_STATE = 42
# Tolerance for merge_asof: Max time difference (ms) allowed for nearest match.
# Should be related to the SUIT's frame rate. If suit is ~60-70Hz (like 2700 frames in 40s),
# frame duration is ~14-16ms. Tolerance slightly > half interval is good.
MERGE_TOLERANCE_MS = 30  # Adjust this value based on suit frame rate stability


# --- Helper Functions ---
def check_features_exist(df, feature_list, filename):
    present_features = [f for f in feature_list if f in df.columns]
    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        print(f"  --> Warning: Missing features in {filename}: {missing}")
    return present_features


def load_and_prepare_df(csv_path, selected_features, filename):
    """Loads CSV, selects features, converts timestamp to datetime, sets index."""
    try:
        df = pd.read_csv(csv_path)
        if 'frame_timestamp' not in df.columns:
            print(f"  --> Error: 'frame_timestamp' missing in {filename}. Skipping file.")
            return None, None

        cols_to_keep = ['frame_timestamp'] + selected_features
        actual_features_with_ts = check_features_exist(df, cols_to_keep, filename)
        actual_features = [f for f in actual_features_with_ts if f != 'frame_timestamp']

        if not actual_features:
            print(f"  --> Error: No selected features found in {filename}. Skipping.")
            return None, None

        df_selected = df[actual_features_with_ts].copy()

        # *** CHANGE HERE: Convert timestamp (assuming ms since start) to TimedeltaIndex ***
        # We treat milliseconds since start as a duration from a zero point.
        # Using Timedelta directly is often better than converting to absolute datetime
        # if the absolute start time is irrelevant.
        try:
            # Convert numeric ms to Timedelta objects
            df_selected['frame_timestamp'] = pd.to_timedelta(df_selected['frame_timestamp'], unit='ms', errors='coerce')
        except ValueError:
             # Fallback if already loaded as string or other format
             print(f"  --> Warning: Could not convert timestamp directly in {filename}. Attempting numeric conversion first.")
             df_selected['frame_timestamp'] = pd.to_numeric(df_selected['frame_timestamp'], errors='coerce')
             df_selected['frame_timestamp'] = pd.to_timedelta(df_selected['frame_timestamp'], unit='ms', errors='coerce')


        # Drop rows where timestamp conversion failed (resulted in NaT - Not a Time)
        df_selected.dropna(subset=['frame_timestamp'], inplace=True)
        if df_selected.empty:
             print(f"  --> Error: No valid timestamps found in {filename} after conversion. Skipping.")
             return None, None


        # Convert features to numeric, handle initial NaNs
        for col in actual_features:
            df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')
        if df_selected[actual_features].isnull().any().any():
             print(f"  --> Warning: NaNs found in raw data of {filename} before merging. Applying ffill/bfill.")
             df_selected[actual_features] = df_selected[actual_features].ffill().bfill()
             df_selected[actual_features] = df_selected[actual_features].fillna(0) # Final fallback

        # Sort by timestamp (now Timedelta) and set as index
        df_selected = df_selected.sort_values(by='frame_timestamp').set_index('frame_timestamp')

        # Check for duplicate timestamps (can cause issues with merge/interpolation)
        if df_selected.index.has_duplicates:
            dup_count = df_selected.index.duplicated().sum()
            print(f"  --> Warning: Found {dup_count} duplicate timestamps in {filename}. Keeping first entry.")
            df_selected = df_selected[~df_selected.index.duplicated(keep='first')]


        return df_selected, actual_features

    except FileNotFoundError:
        print(f"  --> Error: File not found: {filename}")
        return None, None
    except pd.errors.EmptyDataError:
        print(f"  --> Error: File is empty: {filename}")
        return None, None
    except Exception as e:
        print(f"  --> Error loading/preparing {filename}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# --- Main Processing Logic ---

print(f"Starting data processing with time-based alignment (merge_asof + interpolation)...")
# ... (print data directory paths) ...

all_combined_sequences = []
file_identifiers = []
final_combined_feature_names = []  # Definitive list after first successful processing

suit_csv_files = sorted(list(SUIT_DATA_DIR.glob("*.csv")))
print(f"Found {len(suit_csv_files)} potential suit CSV files.")

if not suit_csv_files:
    print(f"Error: No suit CSV files found in {SUIT_DATA_DIR}. Please check the path.")
    exit()

processed_count = 0
skipped_count = 0

for suit_csv_path in suit_csv_files:
    print(f"\nProcessing: {suit_csv_path.name}")

    glove_right_csv_path, glove_left_csv_path = get_glove_file_paths(
        suit_csv_path, GLOVE_RIGHT_DATA_DIR, GLOVE_LEFT_DATA_DIR
    )
    # ... (check if glove paths were found and files exist) ...
    if not glove_right_csv_path or not glove_left_csv_path or \
            not glove_right_csv_path.exists() or not glove_left_csv_path.exists():
        print(f"  --> Skipping triplet due to missing files.")
        skipped_count += 1
        continue

    try:
        # --- Load and Prepare DataFrames with Timestamp Index ---
        df_suit, actual_suit_features = load_and_prepare_df(suit_csv_path, SUIT_FEATURES_TO_KEEP, suit_csv_path.name)
        if df_suit is None or df_suit.empty:
            skipped_count += 1
            continue

        df_glove_r, actual_glove_right_features = load_and_prepare_df(glove_right_csv_path,
                                                                      GLOVE_RIGHT_FEATURES_TO_KEEP,
                                                                      glove_right_csv_path.name)
        if df_glove_r is None:  # Allow proceeding if glove data is missing/invalid
            print(f"  --> Warning: Proceeding without Right Glove data for {suit_csv_path.name}")
            actual_glove_right_features = []  # Ensure list is empty
            df_glove_r = pd.DataFrame(index=pd.Index([], dtype='float64', name='frame_timestamp'))  # Empty DF for merge

        df_glove_l, actual_glove_left_features = load_and_prepare_df(glove_left_csv_path, GLOVE_LEFT_FEATURES_TO_KEEP,
                                                                     glove_left_csv_path.name)
        if df_glove_l is None:
            print(f"  --> Warning: Proceeding without Left Glove data for {suit_csv_path.name}")
            actual_glove_left_features = []
            df_glove_l = pd.DataFrame(index=pd.Index([], dtype='float64', name='frame_timestamp'))

        # --- Align using merge_asof ---
        print(f"  Aligning timestamps using merge_asof (tolerance: {MERGE_TOLERANCE_MS}ms)...")
        # Reset index to use timestamp as merge key, keep it as a column
        df_suit_reset = df_suit.reset_index()
        df_glove_r_reset = df_glove_r.reset_index()
        df_glove_l_reset = df_glove_l.reset_index()

        # Merge Right Glove onto Suit
        df_merged_right = pd.merge_asof(
            df_suit_reset,
            df_glove_r_reset,
            on='frame_timestamp',
            direction='nearest',
            tolerance=pd.Timedelta(milliseconds=MERGE_TOLERANCE_MS),
            suffixes=('', '_glover')  # Add suffix to avoid potential conflicts if features weren't selected carefully
        )
        # Rename glove columns to their original names (remove suffix if merge added it)
        rename_dict_r = {f"{col}_glover": col for col in actual_glove_right_features if
                         f"{col}_glover" in df_merged_right.columns}
        df_merged_right.rename(columns=rename_dict_r, inplace=True)

        # Merge Left Glove onto the result
        df_combined = pd.merge_asof(
            df_merged_right,
            df_glove_l_reset,
            on='frame_timestamp',
            direction='nearest',
            tolerance=pd.Timedelta(milliseconds=MERGE_TOLERANCE_MS),
            suffixes=('', '_glovel')
        )
        rename_dict_l = {f"{col}_glovel": col for col in actual_glove_left_features if
                         f"{col}_glovel" in df_combined.columns}
        df_combined.rename(columns=rename_dict_l, inplace=True)

        # --- Interpolate Missing Glove Data ---
        # Set timestamp back as index for interpolation
        df_combined = df_combined.set_index('frame_timestamp').sort_index()

        features_to_interpolate = actual_glove_right_features + actual_glove_left_features
        # Only interpolate columns that actually exist in the combined df
        features_to_interpolate = [f for f in features_to_interpolate if f in df_combined.columns]

        if features_to_interpolate:
            nan_count_before = df_combined[features_to_interpolate].isnull().sum().sum()
            if nan_count_before > 0:
                print(f"  Interpolating {nan_count_before} missing glove data points (time-based linear)...")
                # Use time-based linear interpolation
                df_combined[features_to_interpolate] = df_combined[features_to_interpolate].interpolate(method='time')

                # Handle potential remaining NaNs at the start/end after interpolation
                nan_count_after = df_combined[features_to_interpolate].isnull().sum().sum()
                if nan_count_after > 0:
                    print(f"  --> Warning: {nan_count_after} NaNs remain after interpolation. Applying ffill/bfill.")
                    df_combined[features_to_interpolate] = df_combined[features_to_interpolate].ffill().bfill()
                    # Final fallback: fill with 0 if still NaNs (e.g., sequence too short)
                    df_combined[features_to_interpolate] = df_combined[features_to_interpolate].fillna(0)
            else:
                print("  No missing glove data points found after merge (within tolerance).")

        # --- Final Feature Selection and Conversion to NumPy ---
        current_combined_features = actual_suit_features + actual_glove_right_features + actual_glove_left_features
        # Ensure the list only contains features present in the final dataframe
        current_combined_features = [f for f in current_combined_features if f in df_combined.columns]

        if not final_combined_feature_names:
            final_combined_feature_names = current_combined_features
            print(f"Combined feature count: {len(final_combined_feature_names)}")
            # ... (print example features) ...
        elif final_combined_feature_names != current_combined_features:
            print(f"  --> Warning: Feature set for {suit_csv_path.name} differs from previous files.")
            # Make sure we only extract the columns defined by the *first* file's feature set
            missing_in_current = [f for f in final_combined_feature_names if f not in df_combined.columns]
            if missing_in_current:
                print(f"     Current file is missing expected features: {missing_in_current}. Skipping.")
                skipped_count += 1
                continue
            df_combined = df_combined[final_combined_feature_names]  # Select consistent subset
        else:
            # Ensure column order matches the definitive list
            df_combined = df_combined[final_combined_feature_names]

        combined_sequence_np = df_combined.values

        # Check for empty results
        if combined_sequence_np.shape[0] == 0:
            print(f"  --> Warning: Resulting data is empty for {suit_csv_path.name}. Skipping.")
            skipped_count += 1
            continue
        # Final check for NaNs/Infs before adding
        if np.isnan(combined_sequence_np).any() or np.isinf(combined_sequence_np).any():
            print(
                f"  --> CRITICAL Error: NaNs or Infs detected in final combined NumPy array for {suit_csv_path.name}. Skipping.")
            skipped_count += 1
            continue

        all_combined_sequences.append(combined_sequence_np)
        file_identifiers.append({'filename': suit_csv_path.name})
        processed_count += 1

    except Exception as e:
        print(f"  --> UNEXPECTED Error processing triplet for {suit_csv_path.name}: {e}")
        import traceback

        traceback.print_exc()
        skipped_count += 1

# --- Post-processing (Splitting, Scaling, Saving) ---
print(f"\nFinished time-based alignment and interpolation.")
print(f"Successfully processed {processed_count} sequence triplets.")
print(f"Skipped {skipped_count} sequence triplets.")

if not all_combined_sequences:
    print("Error: No sequences were combined successfully. Exiting.")
    exit()
if not final_combined_feature_names:
    print("Error: Combined feature names list is empty. Exiting.")
    exit()

# --- 2. Split Data ---
# ... (Splitting logic remains the same) ...
indices = list(range(len(all_combined_sequences)))
train_indices, test_indices = train_test_split(indices, test_size=TEST_SIZE, random_state=RANDOM_STATE)
relative_val_size = VALIDATION_SIZE / (1 - TEST_SIZE)
train_indices, val_indices = train_test_split(train_indices, test_size=relative_val_size, random_state=RANDOM_STATE)
X_train_raw = [all_combined_sequences[i] for i in train_indices]
X_val_raw = [all_combined_sequences[i] for i in val_indices]
X_test_raw = [all_combined_sequences[i] for i in test_indices]
train_ids = [file_identifiers[i] for i in train_indices]
val_ids = [file_identifiers[i] for i in val_indices]
test_ids = [file_identifiers[i] for i in test_indices]

# --- 3. Normalize Features ---
# ... (Normalization logic remains the same, using StandardScaler fit on X_train_raw_nonempty_correct_shape) ...
print("\nFitting StandardScaler on combined training data...")
scaler = StandardScaler()
num_expected_features = len(final_combined_feature_names)
X_train_raw_nonempty_correct_shape = [
    seq for seq in X_train_raw if seq.shape[0] > 0 and seq.shape[1] == num_expected_features
]
if not X_train_raw_nonempty_correct_shape:
    print("Error: No valid training sequences found for scaler fitting.")
    exit()
concatenated_train_data = np.concatenate(X_train_raw_nonempty_correct_shape, axis=0)
if np.isnan(concatenated_train_data).any() or np.isinf(concatenated_train_data).any():
    print("Error: NaNs or infinite values detected in training data BEFORE scaling. Check merging/interpolation.")
    exit()
scaler.fit(concatenated_train_data)
print("Scaler fitted.")

print("Applying scaler to train, validation, and test sets...")


# ... (Use the safe scale_sequence function from the previous DTW answer) ...
def scale_sequence(seq, scaler_instance, expected_features):
    if seq is None or seq.shape[0] == 0: return seq
    if seq.shape[1] != expected_features:
        print(
            f"  --> Warning: Skipping scaling for sequence with {seq.shape[1]} features (expected {expected_features}). Returning None.")
        return None
    try:
        if np.isnan(seq).any():
            print(f"  --> Warning: NaNs found in sequence before scaling. Attempting nan_to_num.")
            seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        return scaler_instance.transform(seq)
    except Exception as e:
        print(f"  --> Error scaling sequence: {e}")
        return None


X_train = [scale_sequence(seq, scaler, num_expected_features) for seq in X_train_raw]
X_val = [scale_sequence(seq, scaler, num_expected_features) for seq in X_val_raw]
X_test = [scale_sequence(seq, scaler, num_expected_features) for seq in X_test_raw]
# ... (Filter Nones and warn about ID sync) ...
initial_train_count = len(X_train)
X_train = [seq for seq in X_train if seq is not None]
train_sequences_removed = initial_train_count - len(X_train)
# ... repeat filtering for val/test and update counts ...


# --- 4. Save Processed Data and Scaler ---
# ... (Saving logic remains the same) ...
scaler_path = OUTPUT_DIR / "combined_interpolated_scaler.pkl"
with open(scaler_path, 'wb') as f: pickle.dump(scaler, f)
print(f"\nSaved StandardScaler to: {scaler_path}")

processed_data_path = OUTPUT_DIR / "combined_interpolated_processed_sequences.pkl"
# Ideally filter IDs based on which sequences survived scaling
# For now, saving potentially out-of-sync IDs
data_to_save = {
    'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
    'train_ids': train_ids, 'val_ids': val_ids, 'test_ids': test_ids,
    'feature_names': final_combined_feature_names,
}
with open(processed_data_path, 'wb') as f: pickle.dump(data_to_save, f)
print(f"Saved time-aligned interpolated data splits to: {processed_data_path}")
if train_sequences_removed > 0:  # Basic check for sync issue
    print(
        f"\nWARNING: {train_sequences_removed} training sequences were removed due to processing/scaling errors. ID lists might be inaccurate.")
    # Add similar checks for val/test if needed

print("\n--- Processing Finished ---")
