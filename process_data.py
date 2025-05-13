import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
import traceback # For detailed error printing

# --- Configuration ---
SUIT_DATA_DIR = Path("./suit")
GLOVE_RIGHT_DATA_DIR = Path("./right")
GLOVE_LEFT_DATA_DIR = Path("./left")
OUTPUT_DIR = Path("./processed_combined_data_both_gloves")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Feature Selection Flags ---
USE_SUIT_ROTATIONS = True
USE_SUIT_POSITIONS = False
USE_SUIT_HIPS_POSITION = True
USE_SUIT_BIOMECH = True # Calculate velocity/acceleration if True

USE_GLOVE_RIGHT_FINGER_ROTATIONS = True
USE_GLOVE_RIGHT_FINGER_POSITIONS = False
USE_GLOVE_RIGHT_HAND_ROOT = True

USE_GLOVE_LEFT_FINGER_ROTATIONS = True
USE_GLOVE_LEFT_FINGER_POSITIONS = False
USE_GLOVE_LEFT_HAND_ROOT = True

# --- Data Splitting & Merge Parameters ---
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
RANDOM_STATE = 42
MERGE_TOLERANCE_MS = 30 # Tolerance for merge_asof alignment

# --- Feature List Generation ---

print("--- Feature Selection Settings ---")

# 1. Generate SUIT Feature List based on flags (Defines the *desired* final set)
SUIT_FEATURES_TO_KEEP = []
selected_suit_bone_rotations = []
selected_suit_bone_positions = []
selected_suit_hip_position = []
selected_suit_biomech_features = []
relevant_biomech_joints = [] # Keep track of joints for derivative calc

# Define bones whose data might come from the suit
upper_body_bones_suit = [
    "hips", "spine", "upper_spine", "neck", "head",
    "left_shoulder", "right_shoulder",
    "left_upper_arm", "right_upper_arm",
    "left_lower_arm", "right_lower_arm",
    #"left_hand", "right_hand" # Initially include hands, may be skipped below
]

if USE_SUIT_ROTATIONS:
    print("Including SUIT: Upper Body Rotations")
    for bone in upper_body_bones_suit:
        if bone == "left_hand" and USE_GLOVE_LEFT_HAND_ROOT: continue
        if bone == "right_hand" and USE_GLOVE_RIGHT_HAND_ROOT: continue
        selected_suit_bone_rotations.extend([f"{bone}.rotation.{axis}" for axis in ['w', 'x', 'y', 'z']])
    SUIT_FEATURES_TO_KEEP.extend(selected_suit_bone_rotations)
else:
    print("Excluding SUIT: Upper Body Rotations")

if USE_SUIT_POSITIONS:
    print("Including SUIT: Upper Body Positions (excluding Hips)")
    for bone in upper_body_bones_suit:
        if bone == "hips": continue
        if bone == "left_hand" and USE_GLOVE_LEFT_HAND_ROOT: continue
        if bone == "right_hand" and USE_GLOVE_RIGHT_HAND_ROOT: continue
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
        for measure in ["angle", "angular_v", "angular_acc"] # Include desired calculated features
    ]
    SUIT_FEATURES_TO_KEEP.extend(selected_suit_biomech_features)
else:
    print("Excluding SUIT: Biomechanical Angles/Vel/Acc")

print(f"Total desired SUIT features: {len(SUIT_FEATURES_TO_KEEP)}")

# 2. Generate GLOVE Feature Lists based on flags
def get_finger_features_configurable(hand_prefix, glove_dir,
                                     use_finger_rotations, use_finger_positions, use_hand_root):
    """Generates list of desired finger features based on config flags."""
    features = []
    fingers = ["thumb", "index", "middle", "ring", "little"]
    phalanges = ["proximal", "intermediate", "distal"]
    sample_cols = [] # Get column names later if needed for checking

    # Generate Finger Joint Features
    for finger in fingers:
        current_phalanges = phalanges # Assume all phalanges exist unless thumb intermediate is missing
        # Check if intermediate thumb phalanx exists (based on common naming)
        # We might need to check this against actual file columns later if structure varies
        # For now, assume standard naming for desired features
        if finger == "thumb":
            # Logic to check if intermediate thumb exists could be added here if needed
            pass # Assuming it exists for the desired feature list

        for phalanx in current_phalanges:
            if use_finger_rotations:
                features.extend([f"{hand_prefix}_{finger}_{phalanx}.rotation.{axis}" for axis in ['w', 'x', 'y', 'z']])
            if use_finger_positions:
                features.extend([f"{hand_prefix}_{finger}_{phalanx}.position.{axis}" for axis in ['x', 'y', 'z']])

    # Generate Hand Root Features
    if use_hand_root:
        features.extend([f"{hand_prefix}_hand.rotation.{axis}" for axis in ['w', 'x', 'y', 'z']])
        features.extend([f"{hand_prefix}_hand.position.{axis}" for axis in ['x', 'y', 'z']])

    return sorted(list(set(features))) # Return unique sorted list

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
    print(f"Including desired RIGHT GLOVE features ({len(GLOVE_RIGHT_FEATURES_TO_KEEP)}):")
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
    print(f"Including desired LEFT GLOVE features ({len(GLOVE_LEFT_FEATURES_TO_KEEP)}):")
    if USE_GLOVE_LEFT_FINGER_ROTATIONS: print("  - Finger Rotations")
    if USE_GLOVE_LEFT_FINGER_POSITIONS: print("  - Finger Positions")
    if USE_GLOVE_LEFT_HAND_ROOT: print("  - Hand Root (Position/Rotation)")
else:
    print("Excluding ALL LEFT GLOVE features.")

# Remove duplicates just in case (shouldn't happen with this logic, but safe)
SUIT_FEATURES_TO_KEEP = sorted(list(set(SUIT_FEATURES_TO_KEEP)))
GLOVE_RIGHT_FEATURES_TO_KEEP = sorted(list(set(GLOVE_RIGHT_FEATURES_TO_KEEP)))
GLOVE_LEFT_FEATURES_TO_KEEP = sorted(list(set(GLOVE_LEFT_FEATURES_TO_KEEP)))

# --- Helper Functions ---
def get_glove_file_paths(suit_file_path, glove_right_dir, glove_left_dir):
    """Finds matching glove CSV file paths based on the suit file name."""
    try:
        # Try replacing _suit.csv first
        base_name = suit_file_path.name.replace('_suit.csv', '')
        if base_name == suit_file_path.name: # If no change, try replacing _suit from stem
            base_name = suit_file_path.stem.replace('_suit', '')
        # Handle potential double underscores or other naming variations if needed
        if not base_name or base_name == suit_file_path.stem: # Basic check if extraction failed
            print(f"  --> Warning: Could not reliably extract base name from {suit_file_path.name}")
            return None, None

        glove_right_filename = f"{base_name}_glove_right.csv"
        glove_left_filename = f"{base_name}_glove_left.csv"

        path_right = glove_right_dir / glove_right_filename
        path_left = glove_left_dir / glove_left_filename
        return path_right, path_left
    except Exception as e:
        print(f"  --> Warning: Could not determine glove filenames for {suit_file_path.name}: {e}")
        return None, None

def check_features_exist(df, feature_list, filename):
    """Checks which features from a list are present in a DataFrame's columns."""
    present_features = [f for f in feature_list if f in df.columns]
    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        # This might be expected if _v and _acc aren't in the raw file, suppress warning?
        # Only print if it's not a velocity or acceleration column we intend to calculate
        non_calc_missing = [m for m in missing if not (m.endswith('.angular_v') or m.endswith('.angular_acc'))]
        if non_calc_missing:
            print(f"  --> Warning: Missing expected non-calculated features in {filename}: {non_calc_missing}")
    return present_features

def load_and_prepare_df(csv_path, selected_features, filename):
    """Loads CSV, selects available features, normalizes timestamp, sets index."""
    try:
        df = pd.read_csv(csv_path)
        if 'frame_timestamp' not in df.columns:
            print(f"  --> Error: 'frame_timestamp' missing in {filename}. Skipping file.")
            return None, [] # Return empty list for features

        # Check which of the *requested* features are *actually* in the CSV initially
        cols_to_keep_initial = ['frame_timestamp'] + [f for f in selected_features if f in df.columns]
        actual_features_in_csv = [f for f in cols_to_keep_initial if f != 'frame_timestamp']

        df_selected = df[cols_to_keep_initial].copy()

        # Convert timestamp to Timedelta
        try:
            df_selected['frame_timestamp'] = pd.to_timedelta(df_selected['frame_timestamp'], unit='ms', errors='coerce')
        except (TypeError, ValueError):
            print(f"  --> Warning: Timestamp in {filename} not directly numeric. Attempting conversion via numeric.")
            df_selected['frame_timestamp'] = pd.to_numeric(df_selected['frame_timestamp'], errors='coerce')
            df_selected['frame_timestamp'] = pd.to_timedelta(df_selected['frame_timestamp'], unit='ms', errors='coerce')

        # Drop rows where timestamp conversion failed
        initial_rows = len(df_selected)
        df_selected.dropna(subset=['frame_timestamp'], inplace=True)
        if len(df_selected) < initial_rows:
            print(f"  --> Warning: Dropped {initial_rows - len(df_selected)} rows due to invalid timestamps in {filename}.")

        if df_selected.empty:
            print(f"  --> Error: No valid timestamps found in {filename} after conversion. Skipping.")
            return None, []

        # Convert features to numeric before sorting/normalizing time
        for col in actual_features_in_csv:
            df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')
            # Fill columns that became entirely NaN after coercion
            if df_selected[col].isnull().all():
                print(f"  --> Warning: Feature {col} in {filename} is all NaN after loading/coercion. Filling with 0.")
                df_selected[col].fillna(0, inplace=True)

        # Sort by timestamp *before* normalization
        df_selected = df_selected.sort_values(by='frame_timestamp')

        # --- TIMESTAMP NORMALIZATION ---
        if not df_selected.empty:
            first_timestamp = df_selected['frame_timestamp'].iloc[0]
            # print(f"  Normalizing timestamps for {filename}. First timestamp: {first_timestamp}") # Can be verbose
            df_selected['frame_timestamp'] = df_selected['frame_timestamp'] - first_timestamp
        else:
            print(f"  --> Error: DataFrame empty before timestamp normalization for {filename}. Skipping.")
            return None, []
        # --- END TIMESTAMP NORMALIZATION ---

        # Set the *normalized* timestamp as the index
        df_selected = df_selected.set_index('frame_timestamp')

        # Check for duplicate normalized timestamps
        if df_selected.index.has_duplicates:
            dup_count = df_selected.index.duplicated().sum()
            print(f"  --> Warning: Found {dup_count} duplicate normalized timestamps in {filename}. Keeping first entry.")
            df_selected = df_selected[~df_selected.index.duplicated(keep='first')]

        # Return the dataframe and the list of features successfully loaded *from the CSV*
        return df_selected, actual_features_in_csv

    except FileNotFoundError:
        print(f"  --> Error: File not found: {filename}")
        return None, []
    except pd.errors.EmptyDataError:
        print(f"  --> Error: File is empty: {filename}")
        return None, []
    except Exception as e:
        print(f"  --> Error loading/preparing {filename}: {e}")
        traceback.print_exc()
        return None, []

def calculate_biomech_derivatives(df, joints_list):
    """Calculates angular velocity and acceleration using numerical differentiation."""
    if df.empty or not isinstance(df.index, pd.TimedeltaIndex):
        print("  --> Warning: Cannot calculate derivatives on empty or non-Timedelta indexed DataFrame.")
        return df

    print("  Calculating biomechanical velocity and acceleration...")
    # Calculate time difference between consecutive rows in seconds
    delta_time_sec = df.index.to_series().diff().dt.total_seconds()
    # Replace 0 or very small time differences with NaN to avoid division errors
    delta_time_sec = delta_time_sec.replace(0, np.nan)
    min_time_delta = 1e-7 # Heuristic threshold
    delta_time_sec[delta_time_sec < min_time_delta] = np.nan

    calculated_vel_cols = []
    calculated_acc_cols = []

    for joint in joints_list:
        angle_col = f"{joint}.angle"
        vel_col = f"{joint}.angular_v"
        acc_col = f"{joint}.angular_acc"

        if angle_col in df.columns:
            # Calculate Velocity (V = dAngle / dt)
            delta_angle = df[angle_col].diff()
            df[vel_col] = delta_angle / delta_time_sec
            calculated_vel_cols.append(vel_col)

            # Calculate Acceleration (A = dV / dt) using the just calculated velocity
            delta_velocity = df[vel_col].diff()
            df[acc_col] = delta_velocity / delta_time_sec
            calculated_acc_cols.append(acc_col)
        else:
            # If angle column is missing, ensure V and Acc columns don't exist or are NaN/0
            # print(f"    Angle column {angle_col} missing, cannot calculate derivatives.") # Can be verbose
            if vel_col in df.columns: df[vel_col] = 0.0
            if acc_col in df.columns: df[acc_col] = 0.0

    # Fill NaNs resulting from diff() and division by NaN (from delta_time_sec)
    cols_to_fill = calculated_vel_cols + calculated_acc_cols
    for col in cols_to_fill:
        if col in df.columns: # Check if column was created
            nan_count = df[col].isnull().sum()
            if nan_count > 0:
                # print(f"    Filling {nan_count} NaNs in calculated column {col} with 0.")
                df[col].fillna(0, inplace=True) # Fill with 0 is simplest

    print("  Finished calculating derivatives.")
    return df

def scale_sequence(seq, scaler_instance, expected_features):
    """Safely scales a sequence, handling potential errors and shape mismatches."""
    if seq is None or seq.shape[0] == 0: return None # Return None for empty sequences
    if seq.shape[1] != expected_features:
        print(
            f"  --> Warning: Skipping scaling for sequence with {seq.shape[1]} features (expected {expected_features}). Returning None.")
        return None
    try:
        # Check for NaNs/Infs *before* scaling
        if np.isnan(seq).any() or np.isinf(seq).any():
            print(f"  --> Warning: NaNs/Infs found in sequence before scaling. Attempting nan_to_num.")
            seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0) # Simple replacement

        scaled_seq = scaler_instance.transform(seq)

        # Double check for NaNs/Infs *after* scaling
        if np.isnan(scaled_seq).any() or np.isinf(scaled_seq).any():
            print(f"  --> CRITICAL Warning: NaNs/Infs found *after* scaling. Returning None.")
            return None
        return scaled_seq
    except ValueError as ve:
        if "Input contains NaN" in str(ve):
            print(f"  --> Error during scaling (ValueError: Input contains NaN). Sequence shape: {seq.shape}. Returning None.")
            return None
        else:
            print(f"  --> Error scaling sequence (ValueError): {ve}. Returning None.")
            return None
    except Exception as e:
        print(f"  --> Error scaling sequence: {e}")
        traceback.print_exc()
        return None

def filter_none_sequences(X_scaled, X_raw, ids):
    """Filters out None entries from scaled data and corresponding raw/id lists."""
    filtered_X = []
    filtered_ids = []
    removed_count = 0
    original_indices_kept = [] # Keep track of original indices that passed

    for i, seq in enumerate(X_scaled):
        if seq is not None:
            filtered_X.append(seq)
            filtered_ids.append(ids[i])
            original_indices_kept.append(i) # Store the original index if kept
        else:
            removed_count += 1
            # Optionally print which raw sequence (e.g., by its original index or ID) was removed
            # print(f"  Removed sequence corresponding to original ID: {ids[i]['filename']} due to scaling error.")

    return filtered_X, filtered_ids, removed_count


# --- Main Processing Logic ---

print(f"\nStarting data processing...")
print(f"Suit data dir: {SUIT_DATA_DIR.resolve()}")
print(f"Glove R data dir: {GLOVE_RIGHT_DATA_DIR.resolve()}")
print(f"Glove L data dir: {GLOVE_LEFT_DATA_DIR.resolve()}")
print(f"Output dir: {OUTPUT_DIR.resolve()}")

all_combined_sequences = []
file_identifiers = []
final_combined_feature_names = []  # Definitive list after first successful processing

# Define the *complete* desired feature set based on flags
# These lists were generated earlier based on flags
FINAL_SUIT_FEATURES = SUIT_FEATURES_TO_KEEP[:]
FINAL_GLOVE_R_FEATURES = GLOVE_RIGHT_FEATURES_TO_KEEP[:]
FINAL_GLOVE_L_FEATURES = GLOVE_LEFT_FEATURES_TO_KEEP[:]

suit_csv_files = sorted(list(SUIT_DATA_DIR.glob("*.csv")))
print(f"\nFound {len(suit_csv_files)} potential suit CSV files.")

if not suit_csv_files:
    print(f"Error: No suit CSV files found in {SUIT_DATA_DIR}. Please check the path.")
    exit()

processed_count = 0
skipped_count = 0

for suit_csv_path in suit_csv_files:
    print(f"\nProcessing Triplet for: {suit_csv_path.name}")

    glove_right_csv_path, glove_left_csv_path = get_glove_file_paths(
        suit_csv_path, GLOVE_RIGHT_DATA_DIR, GLOVE_LEFT_DATA_DIR
    )

    if not glove_right_csv_path or not glove_left_csv_path :
        print(f"  --> Skipping: Could not determine glove file paths.")
        skipped_count += 1
        continue
    if not glove_right_csv_path.exists():
        print(f"  --> Skipping: Right glove file not found: {glove_right_csv_path.name}")
        skipped_count += 1
        continue
    if not glove_left_csv_path.exists():
        print(f"  --> Skipping: Left glove file not found: {glove_left_csv_path.name}")
        skipped_count += 1
        continue

    try:
        # --- Load and Prepare DataFrames (Includes Timestamp Normalization) ---
        # Pass the *full* desired feature list; function checks what's available
        df_suit, actual_suit_features_in_csv = load_and_prepare_df(suit_csv_path, FINAL_SUIT_FEATURES, suit_csv_path.name)
        if df_suit is None or df_suit.empty:
            print(f"  --> Skipping: Failed to load or prepare suit data.")
            skipped_count += 1
            continue

        # --- Calculate Derivatives for Suit Data ---
        actual_suit_features = actual_suit_features_in_csv[:] # Start with loaded features
        if USE_SUIT_BIOMECH:
            angle_cols_present = [f for f in actual_suit_features_in_csv if f.endswith('.angle')]
            if angle_cols_present:
                df_suit = calculate_biomech_derivatives(df_suit, relevant_biomech_joints)
                # Update the list of features *now available* in the suit df
                actual_suit_features = [f for f in FINAL_SUIT_FEATURES if f in df_suit.columns]
            else:
                print("  --> Skipping derivative calculation: No '.angle' columns loaded from suit CSV.")
                # Keep actual_suit_features as just those loaded from CSV


        # --- Load Gloves (Includes Timestamp Normalization) ---
        df_glove_r, actual_glove_right_features_in_csv = load_and_prepare_df(glove_right_csv_path,
                                                                             FINAL_GLOVE_R_FEATURES,
                                                                             glove_right_csv_path.name)
        actual_glove_right_features = [] # Default to empty list
        if df_glove_r is None:
            print(f"  --> Warning: Could not load Right Glove data for {suit_csv_path.name}. Proceeding without.")
            df_glove_r = pd.DataFrame(index=pd.Index([], dtype='timedelta64[ns]', name='frame_timestamp')) # Empty DF for merge
        else:
            actual_glove_right_features = [f for f in FINAL_GLOVE_R_FEATURES if f in df_glove_r.columns]


        df_glove_l, actual_glove_left_features_in_csv = load_and_prepare_df(glove_left_csv_path,
                                                                            FINAL_GLOVE_L_FEATURES,
                                                                            glove_left_csv_path.name)
        actual_glove_left_features = [] # Default to empty list
        if df_glove_l is None:
            print(f"  --> Warning: Could not load Left Glove data for {suit_csv_path.name}. Proceeding without.")
            df_glove_l = pd.DataFrame(index=pd.Index([], dtype='timedelta64[ns]', name='frame_timestamp')) # Empty DF for merge
        else:
            actual_glove_left_features = [f for f in FINAL_GLOVE_L_FEATURES if f in df_glove_l.columns]


        # --- Align using merge_asof ---
        print(f"  Aligning timestamps using merge_asof (tolerance: {MERGE_TOLERANCE_MS}ms)...")
        df_suit_reset = df_suit.reset_index()
        df_glove_r_reset = df_glove_r.reset_index()
        df_glove_l_reset = df_glove_l.reset_index()

        # Merge Right Glove onto Suit
        df_merged_right = pd.merge_asof(
            df_suit_reset.sort_values('frame_timestamp'),
            df_glove_r_reset.sort_values('frame_timestamp'),
            on='frame_timestamp',
            direction='nearest',
            tolerance=pd.Timedelta(milliseconds=MERGE_TOLERANCE_MS),
            suffixes=('', '_glover') # Suffix helps identify potentially conflicting columns if any slip through selection
        )
        # Clean up suffixes if they were added to intended glove columns
        rename_dict_r = {f"{col}_glover": col for col in actual_glove_right_features if f"{col}_glover" in df_merged_right.columns}
        df_merged_right.rename(columns=rename_dict_r, inplace=True)

        # Merge Left Glove onto the result
        df_combined = pd.merge_asof(
            df_merged_right.sort_values('frame_timestamp'),
            df_glove_l_reset.sort_values('frame_timestamp'),
            on='frame_timestamp',
            direction='nearest',
            tolerance=pd.Timedelta(milliseconds=MERGE_TOLERANCE_MS),
            suffixes=('', '_glovel')
        )
        # Clean up suffixes if they were added to intended glove columns
        rename_dict_l = {f"{col}_glovel": col for col in actual_glove_left_features if f"{col}_glovel" in df_combined.columns}
        df_combined.rename(columns=rename_dict_l, inplace=True)

        # --- Interpolate Missing Glove Data ---
        df_combined = df_combined.set_index('frame_timestamp').sort_index()
        features_to_interpolate = actual_glove_right_features + actual_glove_left_features
        # Only interpolate columns that actually exist in the combined df
        features_to_interpolate = [f for f in features_to_interpolate if f in df_combined.columns]

        if features_to_interpolate:
            nan_count_before = df_combined[features_to_interpolate].isnull().sum().sum()
            if nan_count_before > 0:
                print(f"  Interpolating {nan_count_before} missing glove data points (time-based linear)...")
                df_combined[features_to_interpolate] = df_combined[features_to_interpolate].interpolate(method='time')
                nan_count_after = df_combined[features_to_interpolate].isnull().sum().sum()
                if nan_count_after > 0:
                    print(f"  --> Warning: {nan_count_after} NaNs remain after interpolation. Applying ffill/bfill.")
                    df_combined[features_to_interpolate] = df_combined[features_to_interpolate].ffill().bfill()
                    # Final fallback: fill with 0 if still NaNs (e.g., sequence too short or only one data point)
                    df_combined[features_to_interpolate] = df_combined[features_to_interpolate].fillna(0)
            # else: # Can be verbose
            #     print("  No missing glove data points found after merge (within tolerance).")


        # --- Final Feature Selection and Conversion to NumPy ---
        # Combine all features that are *actually present* after loading/calculation/merging
        current_actual_combined_features = actual_suit_features + actual_glove_right_features + actual_glove_left_features
        current_actual_combined_features = sorted(list(set([f for f in current_actual_combined_features if f in df_combined.columns])))

        # Check if the combined df is empty *before* selecting columns
        if df_combined.empty:
            print(f"  --> Warning: Combined dataframe is empty after merging/interpolation for {suit_csv_path.name}. Skipping.")
            skipped_count += 1
            continue

        # Set the definitive feature list on the first successful processing
        if not final_combined_feature_names:
            final_combined_feature_names = current_actual_combined_features
            print(f"Established final combined feature set ({len(final_combined_feature_names)} features).")
            # print(f"Example Features: {final_combined_feature_names[:5]} ... {final_combined_feature_names[-5:]}")
        elif final_combined_feature_names != current_actual_combined_features:
            print(f"  --> Warning: Feature set for {suit_csv_path.name} ({len(current_actual_combined_features)}) differs from established set ({len(final_combined_feature_names)}). Aligning...")
            # Align current dataframe to the established feature set
            missing_expected = [f for f in final_combined_feature_names if f not in df_combined.columns]
            extra_found = [f for f in df_combined.columns if f not in final_combined_feature_names]
            if missing_expected:
                print(f"     Adding missing expected columns as 0: {missing_expected}")
                for col in missing_expected: df_combined[col] = 0.0
            if extra_found:
                print(f"     Dropping extra unexpected columns: {extra_found}")
                # Drop columns inplace or reassign
                df_combined = df_combined.drop(columns=extra_found)

            # Ensure column order matches the definitive list
            df_combined = df_combined[final_combined_feature_names]
        else:
            # Ensure column order matches if feature sets were identical
            df_combined = df_combined[final_combined_feature_names]


        # Convert to NumPy
        combined_sequence_np = df_combined.values

        # Final check for NaNs/Infs before adding
        if np.isnan(combined_sequence_np).any() or np.isinf(combined_sequence_np).any():
            print(f"  --> CRITICAL Error: NaNs or Infs detected in final combined NumPy array for {suit_csv_path.name}. Attempting fillna(0).")
            df_filled = df_combined.fillna(0) # Try filling the DataFrame again
            combined_sequence_np = df_filled.values
            if np.isnan(combined_sequence_np).any() or np.isinf(combined_sequence_np).any():
                print(f"  --> CRITICAL Error: NaNs/Infs persist after fillna(0). Skipping {suit_csv_path.name}.")
                skipped_count += 1
                continue

        # Append results
        all_combined_sequences.append(combined_sequence_np)
        file_identifiers.append({'filename': suit_csv_path.name}) # Store identifier
        processed_count += 1

    except Exception as e:
        print(f"  --> UNEXPECTED Error processing triplet for {suit_csv_path.name}: {e}")
        traceback.print_exc()
        skipped_count += 1

# --- Post-processing (Splitting, Scaling, Saving) ---
print(f"\nFinished processing loop.")
print(f"Successfully processed {processed_count} sequence triplets.")
print(f"Skipped {skipped_count} sequence triplets.")

if not all_combined_sequences:
    print("Error: No sequences were combined successfully. Exiting.")
    exit()
if not final_combined_feature_names:
    print("Error: Combined feature names list is empty (no sequences processed successfully?). Exiting.")
    exit()

# --- 2. Split Data ---
print("\nSplitting data into training, validation, and test sets...")
indices = list(range(len(all_combined_sequences)))

# Robust splitting for small datasets
min_sequences_for_split = 3 # Need at least one for each set
if len(indices) < min_sequences_for_split:
    print(f"Error: Only {len(indices)} processed sequences. Need at least {min_sequences_for_split} for train/val/test split. Exiting.")
    exit()

test_indices_count = max(1, int(len(indices) * TEST_SIZE))
val_indices_count = max(1, int(len(indices) * VALIDATION_SIZE))
train_indices_count = len(indices) - test_indices_count - val_indices_count

if train_indices_count < 1:
    print(f"Error: Calculated train set size is {train_indices_count}. Adjust TEST_SIZE/VALIDATION_SIZE.")
    exit()

# First split off test set
train_val_indices, test_indices = train_test_split(indices, test_size=test_indices_count, random_state=RANDOM_STATE, shuffle=True)

# Then split remaining into train and validation
# Calculate validation size relative to the remaining train_val set
remaining_count = len(train_val_indices)
if remaining_count < 2: # Need at least 1 for train and 1 for val after test split
    print(f"Error: Not enough data left ({remaining_count}) to split into train and validation after taking test set. Adjust sizes.")
    # Decide: Maybe put all remaining into train? Or exit?
    exit()
relative_val_size_calc = val_indices_count / remaining_count
if relative_val_size_calc >= 1.0: # Ensure some data left for training
    relative_val_size_calc = max(0.0, 1.0 - (1 / remaining_count)) # Leave at least 1 sample for training
    print(f"Warning: Adjusting relative validation split size to {relative_val_size_calc:.2f} to ensure training data remains.")

train_indices, val_indices = train_test_split(train_val_indices, test_size=relative_val_size_calc, random_state=RANDOM_STATE, shuffle=True)

# Create the raw data splits
X_train_raw = [all_combined_sequences[i] for i in train_indices]
X_val_raw = [all_combined_sequences[i] for i in val_indices]
X_test_raw = [all_combined_sequences[i] for i in test_indices]
train_ids = [file_identifiers[i] for i in train_indices]
val_ids = [file_identifiers[i] for i in val_indices]
test_ids = [file_identifiers[i] for i in test_indices]
print(f"Data split: Train={len(X_train_raw)}, Validation={len(X_val_raw)}, Test={len(X_test_raw)}")

# --- 3. Normalize Features ---
print("\nFitting StandardScaler on combined training data...")
scaler = StandardScaler()
num_expected_features = len(final_combined_feature_names)

# Filter training data for scaler fitting (non-empty and correct feature dimension)
X_train_raw_nonempty_correct_shape = [
    seq for seq in X_train_raw if seq is not None and seq.shape[0] > 0 and seq.shape[1] == num_expected_features
]
if not X_train_raw_nonempty_correct_shape:
    print("Error: No valid training sequences found for scaler fitting (check processing steps).")
    exit()

# Concatenate valid training data for fitting
concatenated_train_data = np.concatenate(X_train_raw_nonempty_correct_shape, axis=0)
if np.isnan(concatenated_train_data).any() or np.isinf(concatenated_train_data).any():
    print("Error: NaNs or infinite values detected in concatenated training data BEFORE scaling. Check processing steps.")
    # Add debug here: which file(s) might be causing this?
    exit()

scaler.fit(concatenated_train_data)
print("Scaler fitted.")
print("Applying scaler to train, validation, and test sets...")

# Apply scaling using the safe helper function
X_train_scaled = [scale_sequence(seq, scaler, num_expected_features) for seq in X_train_raw]
X_val_scaled = [scale_sequence(seq, scaler, num_expected_features) for seq in X_val_raw]
X_test_scaled = [scale_sequence(seq, scaler, num_expected_features) for seq in X_test_raw]

# Filter out None values introduced by scaling errors and update counts/IDs
X_train, train_ids, train_removed = filter_none_sequences(X_train_scaled, X_train_raw, train_ids)
X_val, val_ids, val_removed = filter_none_sequences(X_val_scaled, X_val_raw, val_ids)
X_test, test_ids, test_removed = filter_none_sequences(X_test_scaled, X_test_raw, test_ids)

total_removed = train_removed + val_removed + test_removed
if total_removed > 0:
    print(f"\nWARNING: A total of {total_removed} sequences were removed due to errors during scaling.")
    print(f"Final counts after scaling: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")


# --- 4. Save Processed Data and Scaler ---
scaler_path = OUTPUT_DIR / "combined_interpolated_scaler.pkl"
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"\nSaved StandardScaler to: {scaler_path}")

processed_data_path = OUTPUT_DIR / "combined_interpolated_processed_sequences.pkl"
# Save the filtered data and corresponding IDs
data_to_save = {
    'X_train': X_train,
    'X_val': X_val,
    'X_test': X_test,
    'train_ids': train_ids, # These IDs now match the filtered X_train
    'val_ids': val_ids,     # These IDs now match the filtered X_val
    'test_ids': test_ids,   # These IDs now match the filtered X_test
    'feature_names': final_combined_feature_names, # The list of features in the scaled data
}
with open(processed_data_path, 'wb') as f:
    pickle.dump(data_to_save, f)
print(f"Saved final processed data splits to: {processed_data_path}")


print("\n--- Processing Finished ---")