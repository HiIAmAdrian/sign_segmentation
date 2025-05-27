import pandas as pd
import numpy as np
import os
import glob  # To process all smoothed files if desired

# --- Configuration ---
# Input can be a single smoothed CSV or a pattern for all smoothed CSVs
INPUT_CSV_PATTERN_MODEL_FITTING = "output_landmarks_csv_catalin_smoothed_vX/*.csv"  # Process ALL smoothed files
# Or for a single file:
# INPUT_CSV_PATH_MODEL_FITTING = "output_landmarks_csv_catalin_smoothed_vX/sentence_001_mediapipe_landmarks_py_oneeuro_smoothed.csv"

OUTPUT_MODEL_FITTED_CSV_DIR = "output_landmarks_csv_catalin_model_fitted_vX"

MEDIAPIPE_LANDMARK_COUNT = 478  # Or 468
MEAN_SHAPE_3D_FILE = "mean_face_shape_478.npy"  # Path to your saved mean shape

MIN_VALID_POINTS_FOR_FIT = 15  # Minimum number of valid points in a frame to attempt fitting


# --- Helper Functions (rigid_transform_3d_kabsch and apply_transform are the same as in create_mean_shape.py) ---

def rigid_transform_3d_kabsch(A, B):
    assert A.shape == B.shape
    if A.shape[0] == 0: return np.eye(3), np.zeros(3)
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B.T - R @ centroid_A.T
    return R, t


def apply_transform(points, R, t):
    return (R @ points.T).T + t


def load_mean_shape(filepath):
    if not os.path.exists(filepath):
        print(f"ERROR: Mean shape file not found: {filepath}")
        # Create a dummy if not found, for testing, but this is not recommended for real use.
        print("CRITICAL ERROR: Mean shape file missing. Cannot proceed with model fitting.")
        return None  # Indicate failure
    return np.load(filepath)


# --- Main Processing Function ---
def fit_model_to_single_csv(input_csv_path, output_csv_path, mean_shape):
    print(f"Fitting model: {os.path.basename(input_csv_path)} -> {os.path.basename(output_csv_path)}")

    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"  Error reading {input_csv_path}: {e}")
        return

    if df.empty:
        print(f"  Input CSV {input_csv_path} is empty.")
        return

    fitted_rows = []
    grouped_by_frame = df.groupby('frame_id')
    unique_frame_ids = sorted(df['frame_id'].unique())

    last_R, last_t = np.eye(3), np.zeros(3)  # Initialize default pose
    pose_is_initialized = False  # Track if we have a valid pose from a previous frame

    for frame_id in unique_frame_ids:
        group = grouped_by_frame.get_group(frame_id)
        group = group.sort_values(by='landmark_id')

        observed_landmarks_xyz = np.zeros((MEDIAPIPE_LANDMARK_COUNT, 3))
        for _, row in group.iterrows():
            lm_id = int(row['landmark_id'])
            if 0 <= lm_id < MEDIAPIPE_LANDMARK_COUNT:
                observed_landmarks_xyz[lm_id, :] = [row['x_cam'], row['y_cam'], row['z_cam']]

        valid_observed_mask = ~np.all(np.isclose(observed_landmarks_xyz, 0.0), axis=1)

        # For alignment, use only points that are valid in BOTH observed and mean_shape
        # (though mean_shape should ideally have all points defined away from origin)
        valid_mean_shape_mask = ~np.all(np.isclose(mean_shape, 0.0), axis=1)  # Should be all True
        common_valid_mask_for_fitting = valid_observed_mask & valid_mean_shape_mask

        observed_points_for_fitting = observed_landmarks_xyz[common_valid_mask_for_fitting, :]
        mean_shape_points_for_fitting = mean_shape[common_valid_mask_for_fitting, :]

        current_R, current_t = np.eye(3), np.zeros(3)  # Default if fit fails

        if observed_points_for_fitting.shape[0] >= MIN_VALID_POINTS_FOR_FIT:
            try:
                R_fit, t_fit = rigid_transform_3d_kabsch(mean_shape_points_for_fitting, observed_points_for_fitting)
                current_R, current_t = R_fit, t_fit
                last_R, last_t = R_fit, t_fit  # Update last known good pose
                pose_is_initialized = True
            except Exception as e_fit:
                # print(f"  Frame {frame_id}: Rigid transform fit failed: {e_fit}. Using last known pose if available.")
                if pose_is_initialized:
                    current_R, current_t = last_R, last_t
                # Else, current_R, current_t remain identity/zero
        elif pose_is_initialized:  # Not enough points, use last good pose
            # print(f"  Frame {frame_id}: Not enough valid points ({observed_points_for_fitting.shape[0]}), using last known pose.")
            current_R, current_t = last_R, last_t
        # else: Not enough points and no prior pose, use default identity/zero

        # Transform the *entire* mean_shape using the estimated (or last known) pose
        model_posed_landmarks = apply_transform(mean_shape, current_R, current_t)

        # --- Decision Logic for Final Output ---
        final_output_landmarks = np.zeros_like(observed_landmarks_xyz)
        for lm_idx in range(MEDIAPIPE_LANDMARK_COUNT):
            if valid_observed_mask[lm_idx]:  # If the input (smoothed) landmark was valid
                final_output_landmarks[lm_idx, :] = observed_landmarks_xyz[lm_idx, :]
            else:  # Input landmark was (0,0,0), fill with posed model landmark
                final_output_landmarks[lm_idx, :] = model_posed_landmarks[lm_idx, :]

        sentence_id = df['sentence_id'].iloc[0]  # Assuming constant for the file
        for lm_idx_out in range(MEDIAPIPE_LANDMARK_COUNT):
            fitted_rows.append({
                'sentence_id': sentence_id,
                'frame_id': frame_id,
                'landmark_id': lm_idx_out,
                'x_cam': final_output_landmarks[lm_idx_out, 0],
                'y_cam': final_output_landmarks[lm_idx_out, 1],
                'z_cam': final_output_landmarks[lm_idx_out, 2]
            })

    if fitted_rows:
        df_fitted = pd.DataFrame(fitted_rows)
        df_fitted.to_csv(output_csv_path, index=False, float_format='%.6f')
        print(f"  Saved model-fitted data to {os.path.basename(output_csv_path)}")


# --- Main Execution ---
if __name__ == "__main__":
    mean_3d_shape = load_mean_shape(MEAN_SHAPE_3D_FILE)
    if mean_3d_shape is None:
        print("Exiting due to missing mean shape file.")
        exit()

    if not os.path.exists(OUTPUT_MODEL_FITTED_CSV_DIR):
        os.makedirs(OUTPUT_MODEL_FITTED_CSV_DIR, exist_ok=True)

    # Option 1: Process all files matching a pattern
    input_csv_files = sorted(glob.glob(INPUT_CSV_PATTERN_MODEL_FITTING))
    if not input_csv_files:
        print(f"No input CSVs found for model fitting matching: {INPUT_CSV_PATTERN_MODEL_FITTING}")
        exit()

    print(f"Found {len(input_csv_files)} CSV files for model fitting.")
    for input_path in input_csv_files:
        base = os.path.basename(input_path)
        name, ext = os.path.splitext(base)
        # Ensure output name is distinct, e.g., append "_model_fitted" before the original extension part if it was complex
        # For example, if input is "sentence_001_py_oneeuro_smoothed.csv"
        # Output becomes "sentence_001_py_oneeuro_smoothed_model_fitted.csv"
        output_path = os.path.join(OUTPUT_MODEL_FITTED_CSV_DIR, f"{name}_model_fitted{ext}")
        fit_model_to_single_csv(input_path, output_path, mean_3d_shape)

    # Option 2: Process a single predefined file (uncomment below and comment above glob loop)
    # if os.path.exists(INPUT_CSV_PATH_MODEL_FITTING):
    #     output_single_path = os.path.join(OUTPUT_MODEL_FITTED_CSV_DIR,
    #                                       os.path.basename(INPUT_CSV_PATH_MODEL_FITTING).replace(".csv", "_model_fitted.csv"))
    #     fit_model_to_single_csv(INPUT_CSV_PATH_MODEL_FITTING, output_single_path, mean_3d_shape)
    # else:
    #     print(f"ERROR: Input CSV not found: {INPUT_CSV_PATH_MODEL_FITTING}")

    print("\nModel fitting process complete.")