import pandas as pd
import numpy as np
import os
import glob

# --- Configuration for Mean Shape Creation ---
# Select a few of your BEST quality CSV files (unoccluded, neutral expression, good depth)
# These should be CSVs that have ALREADY been processed by MediaPipe for depth,
# and ideally, ALREADY SMOOTHED by the OneEuroFilter if you have those.
CSV_FILES_FOR_MEAN_SHAPE_PATTERN = "output_landmarks_csv_catalin_smoothed_vX/sentence_00[1-5]_*_smoothed.csv"  # Example: first 5 smoothed sentences
OUTPUT_MEAN_SHAPE_FILE = "mean_face_shape_478.npy"  # Output file

MEDIAPIPE_LANDMARK_COUNT = 478  # Or 468, must match your data


def load_landmarks_from_csv(csv_path):
    """Loads all landmark data from a single CSV into a dictionary of {frame_id: landmarks_array}."""
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return {}
    except Exception as e:
        print(f"  Error reading {csv_path}: {e}")
        return {}

    frames_data = {}
    for frame_id, group in df.groupby('frame_id'):
        group = group.sort_values(by='landmark_id')
        landmarks_xyz = np.zeros((MEDIAPIPE_LANDMARK_COUNT, 3))
        for _, row in group.iterrows():
            lm_id = int(row['landmark_id'])
            if 0 <= lm_id < MEDIAPIPE_LANDMARK_COUNT:
                landmarks_xyz[lm_id, :] = [row['x_cam'], row['y_cam'], row['z_cam']]

        # Only consider frames where a substantial number of landmarks are valid (not 0,0,0)
        # This is a simple heuristic; you might need a more robust check.
        if np.sum(~np.all(np.isclose(landmarks_xyz, 0.0), axis=1)) > MEDIAPIPE_LANDMARK_COUNT * 0.8:  # e.g. >80% valid
            frames_data[frame_id] = landmarks_xyz
    return frames_data


def rigid_transform_3d_kabsch(A, B):
    """
    Finds the optimal rigid (rotation, translation) transformation from point set A to point set B
    using the Kabsch algorithm.
    A and B are NxD matrices (N points, D dimensions - here D=3).
    Assumes points in A and B correspond.
    Returns R (rotation matrix), t (translation vector).
    """
    assert A.shape == B.shape
    if A.shape[0] == 0:  # No points to align
        return np.eye(3), np.zeros(3)

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure a right-handed coordinate system (no reflection)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B.T - R @ centroid_A.T
    return R, t


def generalized_procrustes_analysis(shapes_list):
    """
    Aligns a list of K N-by-D shapes using Generalized Procrustes Analysis (GPA).
    shapes_list: A list of numpy arrays, each array being an (N, D) shape.
    Returns: List of aligned shapes and the mean shape.
    """
    if not shapes_list:
        return [], None

    # 1. Initialization: Select the first shape as the initial estimate of the mean shape.
    #    Or, more robustly, translate all shapes to origin.
    normalized_shapes = []
    for shape in shapes_list:
        centroid = np.mean(shape, axis=0)
        normalized_shapes.append(shape - centroid)  # Center each shape

    current_mean_shape = np.copy(normalized_shapes[0])  # Initial mean

    max_iters = 100
    tolerance = 1e-6  # Convergence tolerance

    for iteration in range(max_iters):
        # 2. Align all shapes to the current mean shape
        aligned_shapes_to_current_mean = []
        for shape in normalized_shapes:  # Use the already centered shapes
            # Only use common valid points for alignment if shapes have NaNs or zeros
            # For simplicity here, assuming all points in mean_shape and shape are for alignment
            # If you have (0,0,0)s, you need to select common non-zero points for A and B in rigid_transform_3d_kabsch

            # Create masks for valid points in both current_mean_shape and shape
            # This is important if some landmarks are consistently (0,0,0)
            valid_current_mean = ~np.all(np.isclose(current_mean_shape, 0.0), axis=1)
            valid_shape = ~np.all(np.isclose(shape, 0.0), axis=1)
            common_valid_mask = valid_current_mean & valid_shape

            if np.sum(common_valid_mask) < 10:  # Need enough points
                # print(f"Warning: Not enough common valid points to align a shape. Using unaligned (centered) shape.")
                aligned_shapes_to_current_mean.append(shape)  # Add centered shape
                continue

            R, t = rigid_transform_3d_kabsch(shape[common_valid_mask], current_mean_shape[common_valid_mask])
            aligned_shape = (R @ shape.T).T + t  # Apply transform to the full shape
            aligned_shapes_to_current_mean.append(aligned_shape)

        # 3. Update the mean shape
        new_mean_shape = np.mean(aligned_shapes_to_current_mean, axis=0)

        # Center the new mean shape
        new_mean_shape_centroid = np.mean(new_mean_shape, axis=0)
        new_mean_shape_centered = new_mean_shape - new_mean_shape_centroid

        # 4. Check for convergence
        diff = np.linalg.norm(new_mean_shape_centered - current_mean_shape)  # Compare centered means
        # print(f"GPA Iteration {iteration + 1}: Difference = {diff}")
        if diff < tolerance:
            break
        current_mean_shape = new_mean_shape_centered

    print(f"GPA converged in {iteration + 1} iterations.")
    # Final alignment of original shapes to the converged mean
    final_aligned_shapes = []
    for shape in shapes_list:  # Use original shapes for final alignment to the mean
        centroid = np.mean(shape, axis=0)  # Original centroid
        shape_centered = shape - centroid

        valid_current_mean = ~np.all(np.isclose(current_mean_shape, 0.0), axis=1)
        valid_shape_centered = ~np.all(np.isclose(shape_centered, 0.0), axis=1)
        common_valid_mask = valid_current_mean & valid_shape_centered

        if np.sum(common_valid_mask) < 10:
            print("Warning: Not enough common points for final alignment of a shape.")
            final_aligned_shapes.append(shape)  # Add original if alignment fails
            continue

        R, t_align_to_mean = rigid_transform_3d_kabsch(shape_centered[common_valid_mask],
                                                       current_mean_shape[common_valid_mask])
        # The 't' from Kabsch here aligns shape_centered to current_mean_shape (which is at origin).
        # We want the aligned shape to be at the origin too, so we only apply rotation.
        # Or, more correctly, align shape_centered to current_mean_shape, then use current_mean_shape.
        # The mean shape IS the target.
        # The aligned_shapes_to_current_mean from the last iteration are what we average.
        # The final_mean_shape is current_mean_shape.

    # The shapes aligned during the last iteration are already aligned to the current_mean_shape
    return aligned_shapes_to_current_mean, current_mean_shape


def main_create_mean_shape():
    csv_files = sorted(glob.glob(CSV_FILES_FOR_MEAN_SHAPE_PATTERN))
    if not csv_files:
        print(f"No CSV files found for mean shape creation matching: {CSV_FILES_FOR_MEAN_SHAPE_PATTERN}")
        return

    print(f"Found {len(csv_files)} CSV files for mean shape creation. Using up to first 100 good frames total.")

    all_good_frames_landmarks = []
    max_frames_to_average = 100  # Limit the number of frames to average

    for csv_path in csv_files:
        if len(all_good_frames_landmarks) >= max_frames_to_average:
            break
        print(f"  Loading frames from: {os.path.basename(csv_path)}")
        frames_dict = load_landmarks_from_csv(csv_path)
        for frame_id in sorted(frames_dict.keys()):  # Process frames in order
            if len(all_good_frames_landmarks) < max_frames_to_average:
                all_good_frames_landmarks.append(frames_dict[frame_id])
            else:
                break

    if len(all_good_frames_landmarks) < 2:  # Need at least 2 shapes for GPA
        print("Not enough valid frames collected to create a mean shape (need at least 2).")
        print("Please check CSV_FILES_FOR_MEAN_SHAPE_PATTERN and the quality of your CSVs.")
        return

    print(f"Collected {len(all_good_frames_landmarks)} frames for Generalized Procrustes Analysis.")

    # Perform Generalized Procrustes Analysis to align all shapes and find the mean
    _, mean_shape = generalized_procrustes_analysis(all_good_frames_landmarks)

    if mean_shape is not None:
        # Ensure the mean shape is centered at the origin (GPA should already do this for the returned mean)
        mean_shape_centroid = np.mean(mean_shape, axis=0)
        final_mean_shape = mean_shape - mean_shape_centroid

        np.save(OUTPUT_MEAN_SHAPE_FILE, final_mean_shape)
        print(f"Mean shape with {final_mean_shape.shape[0]} landmarks saved to: {OUTPUT_MEAN_SHAPE_FILE}")
        print("Mean shape centroid (should be near [0,0,0]):", np.mean(final_mean_shape, axis=0))
    else:
        print("Mean shape could not be computed.")


if __name__ == "__main__":
    main_create_mean_shape()