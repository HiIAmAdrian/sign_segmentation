import pandas as pd
import numpy as np
import os
import glob
from OneEuroFilter import OneEuroFilter
from pathlib import Path

DFLT_FILTER_MIN_CUTOFF = 0.7
DFLT_FILTER_BETA = 0.007
DFLT_FILTER_DERIVATIVE_CUTOFF = 1.0
DFLT_MEDIAPIPE_LANDMARK_COUNT = 478
DFLT_EFFECTIVE_FPS = 60


def apply_oneeuro_to_single_file(
        input_csv_path_str: str,
        output_csv_path_str: str,
        landmark_count: int = DFLT_MEDIAPIPE_LANDMARK_COUNT,
        effective_fps: int = DFLT_EFFECTIVE_FPS,
        min_cutoff: float = DFLT_FILTER_MIN_CUTOFF,
        beta: float = DFLT_FILTER_BETA,
        d_cutoff: float = DFLT_FILTER_DERIVATIVE_CUTOFF
):
    """
    Aplică filtrul OneEuro pe datele de landmark-uri dintr-un singur fișier CSV.
    Scrie rezultatele netezite într-un nou fișier CSV.
    Returnează True dacă operațiunea a reușit, False altfel.
    """
    input_filename = Path(input_csv_path_str).name
    output_filename = Path(output_csv_path_str).name
    print(f"  FUNC_SMOOTH: Processing and smoothing: {input_filename} -> {output_filename}")

    try:
        df = pd.read_csv(input_csv_path_str)
    except pd.errors.EmptyDataError:
        print(f"    FUNC_SMOOTH: Warning - Input CSV {input_filename} is empty. Skipping.")
        return False
    except Exception as e:
        print(f"    FUNC_SMOOTH: Error reading CSV {input_filename}: {e}. Skipping.")
        return False

    if df.empty or not all(
            col in df.columns for col in ['sentence_id', 'frame_id', 'landmark_id', 'x_cam', 'y_cam', 'z_cam']):
        print(f"    FUNC_SMOOTH: Warning - CSV {input_filename} empty or missing required columns. Skipping.")
        return False

    smoothed_data_rows = []

    landmark_filters = {}

    unique_frame_ids = sorted(df['frame_id'].unique())
    if not unique_frame_ids:
        print(f"    FUNC_SMOOTH: No frame_ids found in {input_filename}. Skipping.")
        return False

    last_timestamp = -1

    for frame_id_val in unique_frame_ids:
        group = df[df['frame_id'] == frame_id_val].sort_values(by='landmark_id')

        current_timestamp = frame_id_val / effective_fps
        if current_timestamp < last_timestamp:
            print(
                f"    FUNC_SMOOTH: Warning - Timestamp jump detected for frame {frame_id_val}. Resetting filters for safety.")
            landmark_filters = {}
        last_timestamp = current_timestamp

        frame_landmarks_xyz_original = np.zeros((landmark_count, 3))
        for _, row in group.iterrows():
            lm_id = int(row['landmark_id'])
            if 0 <= lm_id < landmark_count:
                frame_landmarks_xyz_original[lm_id, 0] = row['x_cam']
                frame_landmarks_xyz_original[lm_id, 1] = row['y_cam']
                frame_landmarks_xyz_original[lm_id, 2] = row['z_cam']

        smoothed_frame_landmarks_xyz = np.copy(frame_landmarks_xyz_original)

        for lm_idx in range(landmark_count):
            original_x = frame_landmarks_xyz_original[lm_idx, 0]
            original_y = frame_landmarks_xyz_original[lm_idx, 1]
            original_z = frame_landmarks_xyz_original[lm_idx, 2]

            is_invalid_point = np.isclose(original_x, 0.0) and \
                               np.isclose(original_y, 0.0) and \
                               np.isclose(original_z, 0.0)

            if is_invalid_point:
                if lm_idx in landmark_filters:
                    del landmark_filters[lm_idx]
                smoothed_frame_landmarks_xyz[lm_idx, :] = 0.0
            else:
                if lm_idx not in landmark_filters:
                    config = {'freq': effective_fps, 'mincutoff': min_cutoff, 'beta': beta, 'dcutoff': d_cutoff}
                    landmark_filters[lm_idx] = [OneEuroFilter(**config), OneEuroFilter(**config),
                                                OneEuroFilter(**config)]

                filter_x, filter_y, filter_z = landmark_filters[lm_idx]
                sx = filter_x(original_x, current_timestamp)
                sy = filter_y(original_y, current_timestamp)
                sz = filter_z(original_z, current_timestamp)

                smoothed_frame_landmarks_xyz[lm_idx, 0] = sx if sx is not None else original_x
                smoothed_frame_landmarks_xyz[lm_idx, 1] = sy if sy is not None else original_y
                smoothed_frame_landmarks_xyz[lm_idx, 2] = sz if sz is not None else original_z

        sentence_id = group['sentence_id'].iloc[0] if not group.empty else (
            df['sentence_id'].iloc[0] if not df.empty else "unknown")
        for lm_idx_out in range(landmark_count):
            smoothed_data_rows.append({
                'sentence_id': sentence_id,
                'frame_id': frame_id_val,
                'landmark_id': lm_idx_out,
                'x_cam': smoothed_frame_landmarks_xyz[lm_idx_out, 0],
                'y_cam': smoothed_frame_landmarks_xyz[lm_idx_out, 1],
                'z_cam': smoothed_frame_landmarks_xyz[lm_idx_out, 2]
            })

    if smoothed_data_rows:
        df_smoothed = pd.DataFrame(smoothed_data_rows)
        df_smoothed.to_csv(output_csv_path_str, index=False, float_format='%.6f')
        print(f"    FUNC_SMOOTH: Successfully saved smoothed data to {output_filename}")
        return True
    else:
        print(f"    FUNC_SMOOTH: No data to smooth or write for {input_filename}")
        return False


if __name__ == "__main__":
    print("--- Running Landmark Smoothing (Standalone Mode) ---")

    participant_to_process_smooth = "p1"

    if participant_to_process_smooth == "p1":
        input_csv_dir_smooth = "./output_landmarks_csv_p1_FUNC"
        output_smoothed_dir_smooth = "./output_landmarks_csv_p1_FUNC_smoothed"
    elif participant_to_process_smooth == "p2":
        input_csv_dir_smooth = "./output_landmarks_csv_p2_FUNC"
        output_smoothed_dir_smooth = "./output_landmarks_csv_p2_FUNC_smoothed"
    else:
        print(f"Participant {participant_to_process_smooth} not configured for standalone smoothing.")
        exit()

    Path(output_smoothed_dir_smooth).mkdir(parents=True, exist_ok=True)

    input_csv_pattern_smooth = str(Path(input_csv_dir_smooth) / "sentence_*_mediapipe_landmarks_py.csv")
    csv_files_to_smooth = sorted(glob.glob(input_csv_pattern_smooth))

    if not csv_files_to_smooth:
        print(f"No input CSV files found matching pattern: {input_csv_pattern_smooth}")
        exit()

    print(f"Found {len(csv_files_to_smooth)} CSV files to process for smoothing.")

    for input_csv_path in csv_files_to_smooth:
        base_name = Path(input_csv_path).name
        name_part, ext_part = os.path.splitext(base_name)
        output_csv_path = Path(output_smoothed_dir_smooth) / f"{name_part}_oneeuro_smoothed{ext_part}"

        apply_oneeuro_to_single_file(input_csv_path, str(output_csv_path))

    print("\n--- Standalone Landmark Smoothing Finished ---")