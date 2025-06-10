import pandas as pd
import numpy as np
import os
import glob
from OneEuroFilter import OneEuroFilter

# --- Configuration ---
INPUT_CSV_PATTERN = "output_landmarks_csv_marinela/sentence_*_mediapipe_landmarks_py.csv"
OUTPUT_SMOOTHED_CSV_DIR = "output_landmarks_csv_marinela_smoothed"  # New output dir

FILTER_MIN_CUTOFF = 0.7
FILTER_BETA = 0.007
FILTER_DERIVATIVE_CUTOFF = 1.0

MEDIAPIPE_LANDMARK_COUNT = 478
EFFECTIVE_FPS = 60


def apply_oneeuro_to_file(input_csv_path, output_csv_path):
    print(f"Processing and smoothing: {os.path.basename(input_csv_path)} -> {os.path.basename(output_csv_path)}")

    try:
        df = pd.read_csv(input_csv_path)
    except pd.errors.EmptyDataError:
        print(f"  Warning: Input CSV file {input_csv_path} is empty. Skipping.")
        return
    except Exception as e:
        print(f"  Error reading CSV {input_csv_path}: {e}. Skipping.")
        return

    if df.empty or not all(
            col in df.columns for col in ['sentence_id', 'frame_id', 'landmark_id', 'x_cam', 'y_cam', 'z_cam']):
        print(f"  Warning: CSV file {input_csv_path} is empty or has missing columns. Skipping.")
        return

    smoothed_data_rows = []
    grouped_by_frame = df.groupby('frame_id')

    landmark_filters = {}  # Key: landmark_id, Value: [filter_x, filter_y, filter_z]

    unique_frame_ids = sorted(df['frame_id'].unique())
    if not unique_frame_ids:
        print(f"  No frame_ids found in {input_csv_path}. Skipping.")
        return

    for frame_id in unique_frame_ids:
        group = grouped_by_frame.get_group(frame_id)
        group = group.sort_values(by='landmark_id')

        current_timestamp = frame_id / EFFECTIVE_FPS

        frame_landmarks_xyz_original = np.zeros((MEDIAPIPE_LANDMARK_COUNT, 3))
        for _, row in group.iterrows():
            lm_id = int(row['landmark_id'])
            if 0 <= lm_id < MEDIAPIPE_LANDMARK_COUNT:
                frame_landmarks_xyz_original[lm_id, 0] = row['x_cam']
                frame_landmarks_xyz_original[lm_id, 1] = row['y_cam']
                frame_landmarks_xyz_original[lm_id, 2] = row['z_cam']

        smoothed_frame_landmarks_xyz = np.copy(frame_landmarks_xyz_original)

        for lm_idx in range(MEDIAPIPE_LANDMARK_COUNT):
            original_x = frame_landmarks_xyz_original[lm_idx, 0]
            original_y = frame_landmarks_xyz_original[lm_idx, 1]
            original_z = frame_landmarks_xyz_original[lm_idx, 2]

            is_invalid_point = np.isclose(original_x, 0.0) and \
                               np.isclose(original_y, 0.0) and \
                               np.isclose(original_z, 0.0)

            if is_invalid_point:
                # If the point is invalid (0,0,0), output (0,0,0).
                # If filters existed for this landmark, remove them so they are re-initialized
                # when the next valid point for this landmark appears.
                if lm_idx in landmark_filters:
                    del landmark_filters[lm_idx]
                smoothed_frame_landmarks_xyz[lm_idx, :] = 0.0
            else:
                # Valid point, apply filtering
                if lm_idx not in landmark_filters:
                    # First time seeing this landmark_id as valid, or first valid after reset.
                    # Initialize new filters. The first call will seed them.
                    config = {
                        'freq': EFFECTIVE_FPS,
                        'mincutoff': FILTER_MIN_CUTOFF,
                        'beta': FILTER_BETA,
                        'dcutoff': FILTER_DERIVATIVE_CUTOFF
                    }
                    filter_x = OneEuroFilter(**config)
                    filter_y = OneEuroFilter(**config)
                    filter_z = OneEuroFilter(**config)
                    landmark_filters[lm_idx] = [filter_x, filter_y, filter_z]

                    # First call seeds the filter and returns the original value
                    sx = filter_x(original_x, current_timestamp)
                    sy = filter_y(original_y, current_timestamp)
                    sz = filter_z(original_z, current_timestamp)
                else:  # Filter exists for this landmark, and it's a valid point
                    filter_x, filter_y, filter_z = landmark_filters[lm_idx]
                    sx = filter_x(original_x, current_timestamp)
                    sy = filter_y(original_y, current_timestamp)
                    sz = filter_z(original_z, current_timestamp)

                # The filter should return the value itself if it's the first call,
                # or the filtered value. It shouldn't return None if the input 'x' is not None.
                smoothed_frame_landmarks_xyz[lm_idx, 0] = sx
                smoothed_frame_landmarks_xyz[lm_idx, 1] = sy
                smoothed_frame_landmarks_xyz[lm_idx, 2] = sz

        sentence_id = group['sentence_id'].iloc[0] if not group.empty else df['sentence_id'].iloc[0]
        for lm_idx_out in range(MEDIAPIPE_LANDMARK_COUNT):
            smoothed_data_rows.append({
                'sentence_id': sentence_id,
                'frame_id': frame_id,
                'landmark_id': lm_idx_out,
                'x_cam': smoothed_frame_landmarks_xyz[lm_idx_out, 0],
                'y_cam': smoothed_frame_landmarks_xyz[lm_idx_out, 1],
                'z_cam': smoothed_frame_landmarks_xyz[lm_idx_out, 2]
            })

    if smoothed_data_rows:
        df_smoothed = pd.DataFrame(smoothed_data_rows)
        df_smoothed.to_csv(output_csv_path, index=False, float_format='%.6f')
        print(f"  Successfully saved smoothed data to {os.path.basename(output_csv_path)}")
    else:
        print(f"  No data to smooth or write for {os.path.basename(input_csv_path)}")


def main():
    # ... (main function remains the same as the previous version) ...
    if not os.path.exists(OUTPUT_SMOOTHED_CSV_DIR):
        os.makedirs(OUTPUT_SMOOTHED_CSV_DIR, exist_ok=True)
        print(f"Created output directory for smoothed CSVs: {OUTPUT_SMOOTHED_CSV_DIR}")

    csv_files = sorted(glob.glob(INPUT_CSV_PATTERN))
    if not csv_files:
        print(f"No input CSV files found matching pattern: {INPUT_CSV_PATTERN}")
        print(f"Please check the path: {os.path.abspath(INPUT_CSV_PATTERN)}")
        return

    print(f"Found {len(csv_files)} CSV files to process for smoothing.")

    for input_csv_file_path in csv_files:
        base_name = os.path.basename(input_csv_file_path)
        name_part, ext_part = os.path.splitext(base_name)
        output_csv_file_path = os.path.join(OUTPUT_SMOOTHED_CSV_DIR, f"{name_part}_oneeuro_smoothed{ext_part}")

        apply_oneeuro_to_file(input_csv_file_path, output_csv_file_path)

    print("\nAll CSV files processed and smoothed.")


if __name__ == "__main__":
    main()