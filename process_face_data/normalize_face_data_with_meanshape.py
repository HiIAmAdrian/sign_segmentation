import pandas as pd
import numpy as np
import os
import glob

# import matplotlib.pyplot as plt # Uncomment for debug plotting
# from mpl_toolkits.mplot3d import Axes3D # Uncomment for debug plotting

# --- Configuration ---
MEAN_SHAPE_FILE = "./marinela_face_mean_shape/mean_face_shape_478_cleaned.npy"
INPUT_CSV_DIR = "output_landmarks_csv_marinela_smoothed"
INPUT_CSV_PATTERN = "*.csv"  # e.g., "sentence_008*_smoothed.csv" for targeted debugging
OUTPUT_CSV_DIR = "output_landmarks_csv_marinela_occlusion_normalised"  # Updated output dir name

MEDIAPIPE_LANDMARK_COUNT = 478
MIN_VISIBLE_LANDMARKS_RATIO = 0.25  # For final fill alignment
MIN_LANDMARKS_FOR_RANSAC_ATTEMPT = 10  # Min non-zero points to even try RANSAC
ZERO_THRESHOLD = 1e-6

# --- RANSAC Configuration ---
RANSAC_ITERATIONS = 100
RANSAC_SAMPLE_SIZE = 4  # Min 3 for 3D rigid, 4 can be more stable
RANSAC_INLIER_THRESHOLD = 0.035  # For RANSAC to find a coarse set of inliers (e.g., 3-5cm)
MIN_INLIERS_FOR_VALID_RANSAC_MODEL = int(MEDIAPIPE_LANDMARK_COUNT * 0.15)  # e.g., ~70 landmarks

# --- Post-RANSAC Outlier Rejection Configuration ---
POST_RANSAC_DISTANCE_THRESHOLD = 0.015  # Stricter threshold after robust pose (e.g., 1.5-3cm)


# --- Helper Functions ---
def load_landmarks_from_csv_grouped(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return {}, None
    except Exception as e:
        return {}, None

    frames_data = {}
    required_cols = ['frame_id', 'landmark_id', 'x_cam', 'y_cam', 'z_cam']
    if not all(col in df.columns for col in required_cols):
        return {}, None

    for frame_id, group in df.groupby('frame_id'):
        group = group.sort_values(by='landmark_id')
        landmarks_xyz = np.zeros((MEDIAPIPE_LANDMARK_COUNT, 3))
        present_lm_ids = group['landmark_id'].astype(int).values
        present_coords = group[['x_cam', 'y_cam', 'z_cam']].values
        for i, lm_id in enumerate(present_lm_ids):
            if 0 <= lm_id < MEDIAPIPE_LANDMARK_COUNT:
                landmarks_xyz[lm_id, :] = present_coords[i, :]
        frames_data[frame_id] = landmarks_xyz
    return frames_data, df


def rigid_transform_3d_kabsch(A, B):
    assert A.shape == B.shape
    if A.shape[0] < 3:  # Kabsch needs at least 3 points for a unique solution in 3D
        return np.eye(3), np.zeros(3)

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:  # Fix reflection
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B.T - R @ centroid_A.T
    return R, t


def ransac_kabsch_alignment(source_points_all, target_points_all,
                            iterations, sample_size, inlier_threshold, min_inliers_for_valid_model):
    best_R_from_sample = None
    best_t_from_sample = None
    best_inlier_mask_from_sample = None
    max_inliers_count = -1

    num_points = source_points_all.shape[0]
    if num_points < sample_size:
        return None, None, None

    for _ in range(iterations):
        indices = np.random.choice(num_points, sample_size, replace=False)
        source_sample = source_points_all[indices]
        target_sample = target_points_all[indices]

        R_curr, t_curr = rigid_transform_3d_kabsch(source_sample, target_sample)
        # Check if Kabsch returned identity due to insufficient points (already handled by A.shape[0] < 3)
        # but as a safeguard if sample became degenerate.
        if R_curr is None or (
                np.allclose(R_curr, np.eye(3)) and np.allclose(t_curr, np.zeros(3)) and source_sample.shape[0] < 3):
            continue

        transformed_source_all = (R_curr @ source_points_all.T).T + t_curr
        distances = np.linalg.norm(target_points_all - transformed_source_all, axis=1)
        current_inlier_mask = distances < inlier_threshold
        current_inliers_count = np.sum(current_inlier_mask)

        if current_inliers_count > max_inliers_count:
            max_inliers_count = current_inliers_count
            best_R_from_sample = R_curr
            best_t_from_sample = t_curr
            best_inlier_mask_from_sample = current_inlier_mask

    if best_inlier_mask_from_sample is None or np.sum(best_inlier_mask_from_sample) < min_inliers_for_valid_model:
        return None, None, None  # No good initial model found

    # Final refit using all inliers from the best model found by sampling
    num_best_inliers = np.sum(best_inlier_mask_from_sample)
    if num_best_inliers >= sample_size:  # Ensure enough points for Kabsch
        final_R, final_t = rigid_transform_3d_kabsch(source_points_all[best_inlier_mask_from_sample],
                                                     target_points_all[best_inlier_mask_from_sample])
        # Re-evaluate inliers with this globally refitted model
        transformed_source_all_final = (final_R @ source_points_all.T).T + final_t
        final_distances = np.linalg.norm(target_points_all - transformed_source_all_final, axis=1)
        final_inlier_mask = final_distances < inlier_threshold

        # Only accept refit if it doesn't significantly reduce inlier count or make it invalid
        if np.sum(final_inlier_mask) >= min_inliers_for_valid_model and \
                np.sum(final_inlier_mask) >= num_best_inliers * 0.9:  # Allow small drop
            return final_R, final_t, final_inlier_mask
        else:  # Refit made it worse, return model from sampling
            return best_R_from_sample, best_t_from_sample, best_inlier_mask_from_sample
    else:  # Not enough inliers from sample phase to do a global refit
        return best_R_from_sample, best_t_from_sample, best_inlier_mask_from_sample


# --- Main Processing Logic ---
def process_all_csvs_ransac_pass2():
    try:
        mean_shape = np.load(MEAN_SHAPE_FILE)
        if mean_shape.shape != (MEDIAPIPE_LANDMARK_COUNT, 3):
            print(f"Error: Mean shape file {MEAN_SHAPE_FILE} has incorrect dimensions.")
            return
        print(f"Successfully loaded mean shape from {MEAN_SHAPE_FILE}")
    except Exception as e:
        print(f"Error loading mean shape file {MEAN_SHAPE_FILE}: {e}")
        return

    if not os.path.exists(OUTPUT_CSV_DIR):
        os.makedirs(OUTPUT_CSV_DIR)
        print(f"Created output directory: {OUTPUT_CSV_DIR}")

    input_files = sorted(glob.glob(os.path.join(INPUT_CSV_DIR, INPUT_CSV_PATTERN)))
    if not input_files:
        print(f"No CSV files found in {INPUT_CSV_DIR} matching pattern {INPUT_CSV_PATTERN}")
        return
    print(f"Found {len(input_files)} CSV files to process.")

    for csv_file_path in input_files:
        filename = os.path.basename(csv_file_path)
        output_file_path = os.path.join(OUTPUT_CSV_DIR, filename.replace(".csv", "_filled_ransac.csv"))
        print(f"\nProcessing {filename}...")

        frames_dict, original_df = load_landmarks_from_csv_grouped(csv_file_path)
        if not frames_dict or original_df is None:
            continue

        all_corrected_rows = []
        frames_processed_count = 0

        for frame_id_key, current_frame_landmarks_xyz in frames_dict.items():
            frames_processed_count += 1
            if frames_processed_count % 200 == 0 and frames_processed_count > 0:
                print(f"  Processed {frames_processed_count} frames for {filename}...")

            original_frame_rows = original_df[original_df['frame_id'] == frame_id_key].copy()
            corrected_frame_output_xyz = np.copy(current_frame_landmarks_xyz)

            DEBUG_TARGET_FILENAME_CONTAINS = "sentence_008"
            DEBUG_TARGET_FRAME_ID = 154  # Set this to the frame you want to debug
            is_debug_frame = False
            if DEBUG_TARGET_FILENAME_CONTAINS in filename and int(frame_id_key) == DEBUG_TARGET_FRAME_ID:
                is_debug_frame = True
                print(f"\n--- DEBUGGING: {filename}, Frame {frame_id_key} ---")
                print(f"  RANSAC_INLIER_THRESHOLD = {RANSAC_INLIER_THRESHOLD}")
                print(f"  POST_RANSAC_DISTANCE_THRESHOLD = {POST_RANSAC_DISTANCE_THRESHOLD}")

            original_zero_mask = np.all(np.abs(current_frame_landmarks_xyz) < ZERO_THRESHOLD, axis=1)
            non_zero_mask_current_frame_indices = np.where(~original_zero_mask)[0]
            num_non_zero_initial = len(non_zero_mask_current_frame_indices)

            # This will be the final geometric outlier mask (from post-RANSAC check)
            post_ransac_geometric_outlier_mask = np.zeros(MEDIAPIPE_LANDMARK_COUNT, dtype=bool)
            R_robust_pose, t_robust_pose = np.eye(3), np.zeros(3)  # Initialize robust pose transform

            ransac_succeeded = False
            if num_non_zero_initial >= MIN_LANDMARKS_FOR_RANSAC_ATTEMPT and num_non_zero_initial >= RANSAC_SAMPLE_SIZE:
                mean_shape_subset_for_ransac = mean_shape[non_zero_mask_current_frame_indices]
                current_frame_subset_for_ransac = current_frame_landmarks_xyz[non_zero_mask_current_frame_indices]

                R_ransac_result, t_ransac_result, _ = ransac_kabsch_alignment(
                    # We don't need RANSAC's inlier mask directly here
                    mean_shape_subset_for_ransac,
                    current_frame_subset_for_ransac,
                    RANSAC_ITERATIONS,
                    RANSAC_SAMPLE_SIZE,
                    RANSAC_INLIER_THRESHOLD,
                    MIN_INLIERS_FOR_VALID_RANSAC_MODEL
                )

                if R_ransac_result is not None:
                    R_robust_pose, t_robust_pose = R_ransac_result, t_ransac_result
                    ransac_succeeded = True
                    if is_debug_frame:
                        print(f"  RANSAC successful for initial pose estimation.")

                    # --- Second Pass Outlier Detection using the RANSAC pose ---
                    transformed_mean_shape_robust_pose = (R_robust_pose @ mean_shape.T).T + t_robust_pose

                    distances_pass2 = np.full(MEDIAPIPE_LANDMARK_COUNT, np.inf)
                    # Calculate distances for all initially non-zero points in current frame
                    # to their counterparts in the robustly posed mean shape.
                    distances_pass2[non_zero_mask_current_frame_indices] = np.linalg.norm(
                        current_frame_landmarks_xyz[non_zero_mask_current_frame_indices] - \
                        transformed_mean_shape_robust_pose[non_zero_mask_current_frame_indices],
                        axis=1
                    )
                    # Inliers for the second pass are those within the POST_RANSAC_DISTANCE_THRESHOLD
                    pass2_inlier_mask_full = (distances_pass2 < POST_RANSAC_DISTANCE_THRESHOLD)

                    # Geometric outliers are non-zero points that are NOT inliers in this second pass
                    post_ransac_geometric_outlier_mask = (~pass2_inlier_mask_full) & (~original_zero_mask)

                    if is_debug_frame:
                        num_pass2_geo_outliers = np.sum(post_ransac_geometric_outlier_mask)
                        print(f"  Post-RANSAC check: Identified {num_pass2_geo_outliers} geometric outliers.")
                elif is_debug_frame:
                    print(f"  RANSAC failed to find a valid model for Frame {frame_id_key}.")
            elif is_debug_frame:
                print(
                    f"  Skipping RANSAC for Frame {frame_id_key} due to insufficient non-zero points ({num_non_zero_initial}).")

            # final_unreliable_mask combines original zeros and post-RANSAC geometric outliers
            final_unreliable_mask = original_zero_mask | post_ransac_geometric_outlier_mask
            final_visible_mask = ~final_unreliable_mask
            num_final_visible = np.sum(final_visible_mask)

            if is_debug_frame:
                print(f"  Original zero count: {np.sum(original_zero_mask)}")
                print(f"  Post-RANSAC geometric outlier count: {np.sum(post_ransac_geometric_outlier_mask)}")
                print(f"  Total unreliable count: {np.sum(final_unreliable_mask)}")
                print(f"  Final visible count for fill: {num_final_visible}")

            if np.sum(final_unreliable_mask) == 0:  # No unreliables, current_frame_xyz is fine
                pass
            elif num_final_visible < MEDIAPIPE_LANDMARK_COUNT * MIN_VISIBLE_LANDMARKS_RATIO:
                if is_debug_frame: print(
                    f"  Too few final visible points ({num_final_visible}) to fill. Original data (with potential outliers) kept.")
            else:  # We have unreliable points and enough visible points to attempt a fill
                source_points_for_fill = mean_shape[final_visible_mask]
                target_points_for_fill = current_frame_landmarks_xyz[final_visible_mask]

                if source_points_for_fill.shape[0] >= 3:
                    # Perform final alignment for filling using only the final_visible_mask
                    # This final_visible_mask has been cleaned by RANSAC + Post-RANSAC check
                    R_final_fill, t_final_fill = rigid_transform_3d_kabsch(source_points_for_fill,
                                                                           target_points_for_fill)
                    transformed_full_mean_shape_for_fill = (R_final_fill @ mean_shape.T).T + t_final_fill
                    corrected_frame_output_xyz[final_unreliable_mask] = transformed_full_mean_shape_for_fill[
                        final_unreliable_mask]
                    if is_debug_frame: print("  Successfully filled unreliable points using refined visible set.")
                elif is_debug_frame:
                    print(
                        f"  Not enough points ({source_points_for_fill.shape[0]}) for final Kabsch fill. Unreliable points remain.")

            # --- Optional Debug Plotting (Uncomment to use) ---
            # if is_debug_frame:
            #     import matplotlib.pyplot as plt
            #     from mpl_toolkits.mplot3d import Axes3D
            #     fig_debug = plt.figure(figsize=(12,10))
            #     ax_debug = fig_debug.add_subplot(111, projection='3d')

            #     current_nz_pts = current_frame_landmarks_xyz[non_zero_mask_current_frame_indices]
            #     ax_debug.scatter(current_nz_pts[:, 0], current_nz_pts[:, 1], current_nz_pts[:, 2],
            #                      c='blue', s=10, label=f'Current Frame (Non-Zero: {num_non_zero_initial})')

            #     if ransac_succeeded:
            #         transformed_mean_shape_ransac_aligned = (R_robust_pose @ mean_shape.T).T + t_robust_pose
            #         ax_debug.scatter(transformed_mean_shape_ransac_aligned[:, 0],
            #                          transformed_mean_shape_ransac_aligned[:, 1],
            #                          transformed_mean_shape_ransac_aligned[:, 2],
            #                          c='red', s=10, alpha=0.3, label='RANSAC Aligned Mean Shape')

            #     # Post-RANSAC geometric outliers (from current frame)
            #     post_ransac_outliers_current_frame = current_frame_landmarks_xyz[post_ransac_geometric_outlier_mask]
            #     if post_ransac_outliers_current_frame.shape[0] > 0:
            #         ax_debug.scatter(post_ransac_outliers_current_frame[:, 0],
            #                          post_ransac_outliers_current_frame[:, 1],
            #                          post_ransac_outliers_current_frame[:, 2],
            #                          c='orange', s=30, marker='x', label=f'Post-RANSAC Outliers ({post_ransac_outliers_current_frame.shape[0]})')

            #     # Final corrected output (non-zero points)
            #     final_plot_mask = ~np.all(np.isclose(corrected_frame_output_xyz, 0.0, atol=ZERO_THRESHOLD), axis=1)
            #     ax_debug.scatter(corrected_frame_output_xyz[final_plot_mask, 0],
            #                      corrected_frame_output_xyz[final_plot_mask, 1],
            #                      corrected_frame_output_xyz[final_plot_mask, 2],
            #                      c='green', s=5, alpha=0.6, label=f'Final Output ({np.sum(final_plot_mask)})')

            #     # Points that were originally zero
            #     original_zeros_pts = current_frame_landmarks_xyz[original_zero_mask]
            #     # Only plot if there are any, and to avoid clutter, maybe don't plot if all are (0,0,0)
            #     # ax_debug.scatter(original_zeros_pts[:, 0], original_zeros_pts[:, 1], original_zeros_pts[:, 2],
            #     #                  c='grey', s=5, marker='s', label='Original Zeros')

            #     ax_debug.set_xlabel("X"); ax_debug.set_ylabel("Y"); ax_debug.set_zlabel("Z (Depth)")
            #     ax_debug.set_xlim([-0.2, 0.2]); ax_debug.set_ylim([-0.2, 0.2]); ax_debug.set_zlim([0.3, 0.9])
            #     ax_debug.invert_zaxis()
            #     ax_debug.view_init(elev=20, azim=0)
            #     ax_debug.legend()
            #     ax_debug.set_title(f"Debug: {filename} - Fr {frame_id_key} - RANSAC Th: {RANSAC_INLIER_THRESHOLD}, Post Th: {POST_RANSAC_DISTANCE_THRESHOLD}")
            #     plt.show()
            # --- End Optional Debug Plotting ---

            if not original_frame_rows.empty:
                lm_id_to_coords = {i: corrected_frame_output_xyz[i] for i in range(MEDIAPIPE_LANDMARK_COUNT)}
                new_coords_list = []
                for lm_id in original_frame_rows['landmark_id'].astype(int):
                    if 0 <= lm_id < MEDIAPIPE_LANDMARK_COUNT:
                        new_coords_list.append(lm_id_to_coords[lm_id])
                    else:
                        new_coords_list.append([0, 0, 0])
                new_coords_array = np.array(new_coords_list)
                if new_coords_array.shape[0] == original_frame_rows.shape[0]:
                    original_frame_rows[['x_cam', 'y_cam', 'z_cam']] = new_coords_array
                else:
                    print(f"  Frame {frame_id_key}: Mismatch in row count for coord update. Skipping update.")
                all_corrected_rows.append(original_frame_rows)

        if all_corrected_rows:
            output_df = pd.concat(all_corrected_rows, ignore_index=True)
            for col in ['frame_id', 'landmark_id']:
                if col in output_df.columns and col in original_df.columns:
                    output_df[col] = output_df[col].astype(original_df[col].dtype)
            output_df.to_csv(output_file_path, index=False, float_format='%.8f')
            print(f"  Saved filled CSV to {output_file_path}")
        else:
            print(f"  No data to save for {filename}.")
    print("\nAll CSV files processed.")


if __name__ == "__main__":
    process_all_csvs_ransac_pass2()