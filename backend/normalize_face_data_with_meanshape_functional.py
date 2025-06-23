import pandas as pd
import numpy as np
import glob
from pathlib import Path

DFLT_MEDIAPIPE_LANDMARK_COUNT = 478
DFLT_MIN_VISIBLE_LANDMARKS_RATIO = 0.25
DFLT_MIN_LANDMARKS_FOR_RANSAC_ATTEMPT = 10
DFLT_ZERO_THRESHOLD = 1e-6
DFLT_RANSAC_ITERATIONS = 100
DFLT_RANSAC_SAMPLE_SIZE = 4
DFLT_RANSAC_INLIER_THRESHOLD = 0.035
DFLT_MIN_INLIERS_FOR_VALID_RANSAC_MODEL_RATIO = 0.15
DFLT_POST_RANSAC_DISTANCE_THRESHOLD = 0.015


def _load_landmarks_from_csv_grouped_internal(csv_path_str, landmark_count):
    try:
        df = pd.read_csv(csv_path_str)
        if df.empty: return {}, None
    except Exception:
        return {}, None
    frames_data = {}
    required_cols = ['frame_id', 'landmark_id', 'x_cam', 'y_cam', 'z_cam']
    if not all(col in df.columns for col in required_cols): return {}, None
    for frame_id, group in df.groupby('frame_id'):
        group = group.sort_values(by='landmark_id')
        landmarks_xyz = np.zeros((landmark_count, 3))
        present_lm_ids = group['landmark_id'].astype(int).values
        present_coords = group[['x_cam', 'y_cam', 'z_cam']].values
        for i, lm_id in enumerate(present_lm_ids):
            if 0 <= lm_id < landmark_count:
                landmarks_xyz[lm_id, :] = present_coords[i, :]
        frames_data[frame_id] = landmarks_xyz
    return frames_data, df


def _rigid_transform_3d_kabsch_internal(A, B):
    assert A.shape == B.shape
    if A.shape[0] < 3: return np.eye(3), np.zeros(3)
    centroid_A = np.mean(A, axis=0);
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A;
    BB = B - centroid_B
    H = AA.T @ BB;
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0: Vt[2, :] *= -1; R = Vt.T @ U.T
    t = centroid_B.T - R @ centroid_A.T
    return R, t


def _ransac_kabsch_alignment_internal(source_points_all, target_points_all,
                                      iterations, sample_size, inlier_threshold, min_inliers_for_valid_model):
    best_R, best_t, best_inlier_mask, max_inliers = None, None, None, -1
    num_points = source_points_all.shape[0]
    if num_points < sample_size: return None, None, None

    for _ in range(iterations):
        indices = np.random.choice(num_points, sample_size, replace=False)
        R_curr, t_curr = _rigid_transform_3d_kabsch_internal(source_points_all[indices], target_points_all[indices])
        if R_curr is None: continue
        transformed_source_all = (R_curr @ source_points_all.T).T + t_curr
        distances = np.linalg.norm(target_points_all - transformed_source_all, axis=1)
        current_inlier_mask = distances < inlier_threshold
        current_inliers_count = np.sum(current_inlier_mask)
        if current_inliers_count > max_inliers:
            max_inliers = current_inliers_count
            best_R, best_t, best_inlier_mask = R_curr, t_curr, current_inlier_mask

    if best_inlier_mask is None or np.sum(best_inlier_mask) < min_inliers_for_valid_model:
        return None, None, None

    num_best_inliers = np.sum(best_inlier_mask)
    if num_best_inliers >= sample_size:
        final_R, final_t = _rigid_transform_3d_kabsch_internal(source_points_all[best_inlier_mask],
                                                               target_points_all[best_inlier_mask])
        transformed_final = (final_R @ source_points_all.T).T + final_t
        final_distances = np.linalg.norm(target_points_all - transformed_final, axis=1)
        final_inlier_mask = final_distances < inlier_threshold
        if np.sum(final_inlier_mask) >= min_inliers_for_valid_model and np.sum(
                final_inlier_mask) >= num_best_inliers * 0.85:
            return final_R, final_t, final_inlier_mask
    return best_R, best_t, best_inlier_mask


def run_normalization_and_fill_for_file(
        input_smoothed_csv_path_str: str,
        output_normalized_csv_path_str: str,
        mean_shape_file_path_str: str,
        landmark_count: int = DFLT_MEDIAPIPE_LANDMARK_COUNT,
        min_visible_ratio: float = DFLT_MIN_VISIBLE_LANDMARKS_RATIO,
        min_for_ransac: int = DFLT_MIN_LANDMARKS_FOR_RANSAC_ATTEMPT,
        zero_thresh: float = DFLT_ZERO_THRESHOLD,
        ransac_iter: int = DFLT_RANSAC_ITERATIONS,
        ransac_sample: int = DFLT_RANSAC_SAMPLE_SIZE,
        ransac_thresh: float = DFLT_RANSAC_INLIER_THRESHOLD,
        min_inliers_ratio_ransac: float = DFLT_MIN_INLIERS_FOR_VALID_RANSAC_MODEL_RATIO,
        post_ransac_thresh: float = DFLT_POST_RANSAC_DISTANCE_THRESHOLD,
        debug_target_frame: int = -1
):
    """
    Procesează un singur fișier CSV cu landmark-uri netezite, aplică normalizarea
    Procrustes cu umplerea ocluziilor folosind RANSAC și o formă medie.
    Returnează True dacă operațiunea a reușit, False altfel.
    """
    input_filename = Path(input_smoothed_csv_path_str).name
    output_filename = Path(output_normalized_csv_path_str).name
    print(f"  FUNC_NORM: Processing normalization for: {input_filename} -> {output_filename}")

    try:
        mean_shape = np.load(mean_shape_file_path_str)
        if mean_shape.shape != (landmark_count, 3):
            print(f"    FUNC_NORM: Error - Mean shape {mean_shape_file_path_str} has wrong dimensions.")
            return False
    except Exception as e:
        print(f"    FUNC_NORM: Error loading mean shape {mean_shape_file_path_str}: {e}")
        return False

    frames_dict, original_df = _load_landmarks_from_csv_grouped_internal(input_smoothed_csv_path_str, landmark_count)
    if not frames_dict or original_df is None:
        print(f"    FUNC_NORM: Could not load data from {input_filename}. Skipping.")
        return False

    all_corrected_rows = []
    min_inliers_for_ransac_model_abs = int(landmark_count * min_inliers_ratio_ransac)

    for frame_id_key, current_frame_landmarks_xyz in frames_dict.items():
        original_frame_rows = original_df[original_df['frame_id'] == frame_id_key].copy()
        corrected_frame_output_xyz = np.copy(current_frame_landmarks_xyz)
        is_debug_this_frame = (debug_target_frame != -1 and int(frame_id_key) == debug_target_frame)

        original_zero_mask = np.all(np.abs(current_frame_landmarks_xyz) < zero_thresh, axis=1)
        non_zero_indices = np.where(~original_zero_mask)[0]
        num_non_zero = len(non_zero_indices)

        post_ransac_geo_outlier_mask = np.zeros(landmark_count, dtype=bool)

        if num_non_zero >= min_for_ransac and num_non_zero >= ransac_sample:
            mean_shape_subset = mean_shape[non_zero_indices]
            current_frame_subset = current_frame_landmarks_xyz[non_zero_indices]

            R_ransac, t_ransac, _ = _ransac_kabsch_alignment_internal(
                mean_shape_subset, current_frame_subset,
                ransac_iter, ransac_sample, ransac_thresh, min_inliers_for_ransac_model_abs
            )

            if R_ransac is not None:
                transformed_mean_robust = (R_ransac @ mean_shape.T).T + t_ransac
                distances_post = np.full(landmark_count, np.inf)
                distances_post[non_zero_indices] = np.linalg.norm(
                    current_frame_landmarks_xyz[non_zero_indices] - transformed_mean_robust[non_zero_indices], axis=1
                )
                pass2_inliers_full_mask = (distances_post < post_ransac_thresh)
                post_ransac_geo_outlier_mask = (~pass2_inliers_full_mask) & (~original_zero_mask)
                if is_debug_this_frame: print(
                    f"    FUNC_NORM Frame {frame_id_key}: RANSAC done. Geo outliers: {np.sum(post_ransac_geo_outlier_mask)}")
            elif is_debug_this_frame:
                print(f"    FUNC_NORM Frame {frame_id_key}: RANSAC failed.")
        elif is_debug_this_frame:
            print(f"    FUNC_NORM Frame {frame_id_key}: RANSAC skipped (num_non_zero={num_non_zero}).")

        final_unreliable_mask = original_zero_mask | post_ransac_geo_outlier_mask
        final_visible_mask = ~final_unreliable_mask
        num_final_visible = np.sum(final_visible_mask)

        if np.sum(
                final_unreliable_mask) > 0 and num_final_visible >= landmark_count * min_visible_ratio and num_final_visible >= 3:
            R_fill, t_fill = _rigid_transform_3d_kabsch_internal(mean_shape[final_visible_mask],
                                                                 current_frame_landmarks_xyz[final_visible_mask])
            transformed_mean_for_fill = (R_fill @ mean_shape.T).T + t_fill
            corrected_frame_output_xyz[final_unreliable_mask] = transformed_mean_for_fill[final_unreliable_mask]
            if is_debug_this_frame: print(
                f"    FUNC_NORM Frame {frame_id_key}: Filled {np.sum(final_unreliable_mask)} unreliable points.")
        elif is_debug_this_frame and np.sum(final_unreliable_mask) > 0:
            print(f"    FUNC_NORM Frame {frame_id_key}: Not enough visible points ({num_final_visible}) to fill.")

        if not original_frame_rows.empty:
            lm_id_to_coords = {i: corrected_frame_output_xyz[i] for i in range(landmark_count)}
            new_coords_list = [lm_id_to_coords.get(int(lm_id), [0, 0, 0]) for lm_id in
                               original_frame_rows['landmark_id']]
            original_frame_rows[['x_cam', 'y_cam', 'z_cam']] = np.array(new_coords_list)
            all_corrected_rows.append(original_frame_rows)

    if all_corrected_rows:
        output_df = pd.concat(all_corrected_rows, ignore_index=True)
        for col in ['frame_id', 'landmark_id']:
            if col in output_df.columns and col in original_df.columns:
                output_df[col] = output_df[col].astype(original_df[col].dtype)
        output_df.to_csv(output_normalized_csv_path_str, index=False, float_format='%.8f')
        print(f"    FUNC_NORM: Saved normalized & filled CSV to {output_filename}")
        return True
    else:
        print(f"    FUNC_NORM: No data to save for {input_filename}.")
        return False


if __name__ == "__main__":
    print("--- Running Landmark Normalization & Filling (Standalone Mode) ---")
    DFLT_MEAN_SHAPE_FILE_STANDALONE = Path("./p2_face_mean_shape/mean_face_shape_478_cleaned.npy")
    participant_to_process_norm = "p2"

    if participant_to_process_norm == "p1":
        input_csv_dir_norm = "./output_landmarks_csv_p1_FUNC_smoothed"
        output_csv_dir_norm = "./output_landmarks_csv_p1_FUNC_normalized"
    elif participant_to_process_norm == "p2":
        input_csv_dir_norm = "./output_landmarks_csv_p2_FUNC_smoothed"
        output_csv_dir_norm = "./output_landmarks_csv_p2_FUNC_normalized"
    else:
        print(f"Participant {participant_to_process_norm} not configured.")
        exit()

    Path(output_csv_dir_norm).mkdir(parents=True, exist_ok=True)

    input_csv_pattern_norm = str(Path(input_csv_dir_norm) / "*_oneeuro_smoothed.csv")
    csv_files_to_normalize = sorted(glob.glob(input_csv_pattern_norm))

    if not csv_files_to_normalize:
        print(f"No input CSV files found matching pattern: {input_csv_pattern_norm}")
        exit()
    if not DFLT_MEAN_SHAPE_FILE_STANDALONE.exists():
        print(f"Mean shape file not found: {DFLT_MEAN_SHAPE_FILE_STANDALONE}")
        exit()

    print(f"Found {len(csv_files_to_normalize)} CSV files to process for normalization.")

    for input_csv_path in csv_files_to_normalize:
        base_name = Path(input_csv_path).name
        output_csv_name = base_name.replace("_oneeuro_smoothed.csv", "_filled_ransac.csv")
        output_csv_path_full = Path(output_csv_dir_norm) / output_csv_name

        run_normalization_and_fill_for_file(
            input_csv_path,
            str(output_csv_path_full),
            str(DFLT_MEAN_SHAPE_FILE_STANDALONE)
        )
    print("\n--- Standalone Landmark Normalization & Filling Finished ---")