import pandas as pd
import numpy as np
import os
import glob

CSV_FILES_FOR_MEAN_SHAPE_PATTERN = "output_test/sentence_00[1-3]_mediapipe_landmarks_py_oneeuro_smoothed.csv"
OUTPUT_MEAN_SHAPE_FILE = "./p2_face_mean_shape/mean_face_shape_478.npy"

MEDIAPIPE_LANDMARK_COUNT = 478


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

        if np.sum(~np.all(np.isclose(landmarks_xyz, 0.0), axis=1)) > MEDIAPIPE_LANDMARK_COUNT * 0.8:
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
    if A.shape[0] == 0:
        return np.eye(3), np.zeros(3)

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


def generalized_procrustes_analysis(shapes_list):
    """
    Aligns a list of K N-by-D shapes using Generalized Procrustes Analysis (GPA).
    shapes_list: A list of numpy arrays, each array being an (N, D) shape.
    Returns: List of aligned shapes and the mean shape.
    """
    if not shapes_list:
        return [], None

    normalized_shapes = []
    for shape in shapes_list:
        centroid = np.mean(shape, axis=0)
        normalized_shapes.append(shape - centroid)

    current_mean_shape = np.copy(normalized_shapes[0])

    max_iters = 100
    tolerance = 1e-6

    for iteration in range(max_iters):
        aligned_shapes_to_current_mean = []
        for shape in normalized_shapes:
            valid_current_mean = ~np.all(np.isclose(current_mean_shape, 0.0), axis=1)
            valid_shape = ~np.all(np.isclose(shape, 0.0), axis=1)
            common_valid_mask = valid_current_mean & valid_shape

            if np.sum(common_valid_mask) < 10:
                aligned_shapes_to_current_mean.append(shape)
                continue

            R, t = rigid_transform_3d_kabsch(shape[common_valid_mask], current_mean_shape[common_valid_mask])
            aligned_shape = (R @ shape.T).T + t
            aligned_shapes_to_current_mean.append(aligned_shape)

        new_mean_shape = np.mean(aligned_shapes_to_current_mean, axis=0)

        new_mean_shape_centroid = np.mean(new_mean_shape, axis=0)
        new_mean_shape_centered = new_mean_shape - new_mean_shape_centroid

        diff = np.linalg.norm(new_mean_shape_centered - current_mean_shape)
        if diff < tolerance:
            break
        current_mean_shape = new_mean_shape_centered

    print(f"GPA converged in {iteration + 1} iterations.")
    final_aligned_shapes = []
    for shape in shapes_list:
        centroid = np.mean(shape, axis=0)
        shape_centered = shape - centroid

        valid_current_mean = ~np.all(np.isclose(current_mean_shape, 0.0), axis=1)
        valid_shape_centered = ~np.all(np.isclose(shape_centered, 0.0), axis=1)
        common_valid_mask = valid_current_mean & valid_shape_centered

        if np.sum(common_valid_mask) < 10:
            print("Warning: Not enough common points for final alignment of a shape.")
            final_aligned_shapes.append(shape)
            continue

        R, t_align_to_mean = rigid_transform_3d_kabsch(shape_centered[common_valid_mask],
                                                       current_mean_shape[common_valid_mask])

    return aligned_shapes_to_current_mean, current_mean_shape


def main_create_mean_shape():
    csv_files = sorted(glob.glob(CSV_FILES_FOR_MEAN_SHAPE_PATTERN))
    if not csv_files:
        print(f"No CSV files found for mean shape creation matching: {CSV_FILES_FOR_MEAN_SHAPE_PATTERN}")
        return

    print(f"Found {len(csv_files)} CSV files for mean shape creation. Using up to first 100 good frames total.")

    all_good_frames_landmarks = []
    max_frames_to_average = 100

    for csv_path in csv_files:
        if len(all_good_frames_landmarks) >= max_frames_to_average:
            break
        print(f"  Loading frames from: {os.path.basename(csv_path)}")
        frames_dict = load_landmarks_from_csv(csv_path)
        for frame_id in sorted(frames_dict.keys()):
            if len(all_good_frames_landmarks) < max_frames_to_average:
                all_good_frames_landmarks.append(frames_dict[frame_id])
            else:
                break

    if len(all_good_frames_landmarks) < 2:
        print("Not enough valid frames collected to create a mean shape (need at least 2).")
        print("Please check CSV_FILES_FOR_MEAN_SHAPE_PATTERN and the quality of your CSVs.")
        return

    print(f"Collected {len(all_good_frames_landmarks)} frames for Generalized Procrustes Analysis.")

    _, mean_shape = generalized_procrustes_analysis(all_good_frames_landmarks)

    if mean_shape is not None:
        mean_shape_centroid = np.mean(mean_shape, axis=0)
        final_mean_shape = mean_shape - mean_shape_centroid

        np.save(OUTPUT_MEAN_SHAPE_FILE, final_mean_shape)
        print(f"Mean shape with {final_mean_shape.shape[0]} landmarks saved to: {OUTPUT_MEAN_SHAPE_FILE}")
        print("Mean shape centroid (should be near [0,0,0]):", np.mean(final_mean_shape, axis=0))
    else:
        print("Mean shape could not be computed.")


if __name__ == "__main__":
    main_create_mean_shape()