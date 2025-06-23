import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

MEAN_SHAPE_FILE = "./p2_face_mean_shape/mean_face_shape_478.npy"
PLOT_TITLE_ORIGINAL = "Original Mean Face Shape (478 Landmarks)"
PLOT_TITLE_CLEANED = "Cleaned Mean Face Shape (Outliers Removed/Zeroed)"
PLOT_POINT_SIZE = 2

FIGURE_SIZE = (10, 7.5)
XLIMS = [-0.2, 0.2]
YLIMS = [-0.2, 0.2]
ZLIMS = [-0.3, 0.3]
ELEVATION = 20
AZIMUTH = 0

DISTANCE_PERCENTILE_THRESHOLD = 99.0

def filter_outliers_from_mean_shape(landmarks, percentile_threshold=99.5, fixed_threshold=None):
    """
    Filters outliers from the mean shape based on their distance from the origin (0,0,0).
    Outliers are set to (0,0,0).
    Returns a copy of the landmarks with outliers zeroed out, and the outlier mask.
    """
    if landmarks.shape[0] == 0:
        return np.copy(landmarks), np.array([], dtype=bool)

    distances = np.linalg.norm(landmarks, axis=1)

    if fixed_threshold is not None:
        threshold_value = fixed_threshold
        print(f"Using fixed distance threshold: {threshold_value:.4f}")
    else:
        if distances.size > 0:
            threshold_value = np.percentile(distances, percentile_threshold)
            print(f"Using {percentile_threshold}th percentile distance threshold: {threshold_value:.4f}")
        else:
            return np.copy(landmarks), np.array([], dtype=bool)


    outlier_mask = distances > threshold_value
    num_outliers = np.sum(outlier_mask)
    print(f"Identified {num_outliers} outlier landmarks.")

    if num_outliers > 0:
        print(f"Outlier indices: {np.where(outlier_mask)[0]}")


    cleaned_landmarks = np.copy(landmarks)
    cleaned_landmarks[outlier_mask] = [0, 0, 0]

    return cleaned_landmarks, outlier_mask

def plot_shape(ax, landmarks_data, title, is_cleaned_plot=False, outlier_mask_original=None):
    """Helper function to plot the shape."""
    ax.cla()

    valid_plot_mask = ~np.all(np.isclose(landmarks_data, 0.0), axis=1)
    plot_landmarks = landmarks_data[valid_plot_mask]

    if plot_landmarks.shape[0] == 0:
        print("No valid landmarks to plot after filtering (0,0,0).")
        ax.set_title(title + " (No data)")
        ax.set_xlim(XLIMS)
        ax.set_ylim(YLIMS)
        ax.set_zlim(ZLIMS)
        ax.invert_zaxis()
        ax.view_init(elev=ELEVATION, azim=AZIMUTH)
        return

    x_coords = plot_landmarks[:, 0]
    y_coords = plot_landmarks[:, 1]
    z_coords = plot_landmarks[:, 2]

    z_min_for_cmap = z_coords.min() if z_coords.size > 0 else ZLIMS[0]
    z_max_for_cmap = z_coords.max() if z_coords.size > 0 else ZLIMS[1]


    scatter = ax.scatter(x_coords, y_coords, z_coords,
                         s=PLOT_POINT_SIZE,
                         c=z_coords,
                         cmap='viridis_r',
                         vmin=z_min_for_cmap,
                         vmax=z_max_for_cmap,
                         depthshade=True)

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_zlabel("Z (meters - Relative Depth)")
    ax.set_title(title)

    ax.set_xlim(XLIMS)
    ax.set_ylim(YLIMS)
    ax.set_zlim(ZLIMS)

    x_span = XLIMS[1] - XLIMS[0]
    y_span = YLIMS[1] - YLIMS[0]
    z_span = ZLIMS[1] - ZLIMS[0]
    try:
        ax.set_box_aspect((x_span, y_span, z_span))
    except AttributeError:
        pass

    ax.invert_zaxis()
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)
    return scatter


def visualize_mean_shape_with_filtering(npy_file_path):
    try:
        original_landmarks = np.load(npy_file_path)
    except FileNotFoundError:
        print(f"Error: The file '{npy_file_path}' was not found.")
        return
    except Exception as e:
        print(f"Error loading .npy file: {e}")
        return

    if original_landmarks.ndim != 2 or original_landmarks.shape[1] != 3:
        print(f"Error: Expected data to be N_landmarks x 3 dimensions, but got {original_landmarks.shape}")
        return

    print(f"Loaded {original_landmarks.shape[0]} original landmarks.")
    print(f"Original mean shape centroid: {np.mean(original_landmarks, axis=0)}")

    cleaned_landmarks, outlier_mask = filter_outliers_from_mean_shape(
        original_landmarks,
        percentile_threshold=DISTANCE_PERCENTILE_THRESHOLD
    )
    cleaned_npy_filename = npy_file_path.replace(".npy", "_cleaned.npy")
    np.save(cleaned_npy_filename, cleaned_landmarks)
    print(f"Cleaned mean shape saved to: {cleaned_npy_filename}")


    fig = plt.figure(figsize=(FIGURE_SIZE[0]*2 + 1, FIGURE_SIZE[1]))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    print("\nPlotting original shape...")
    scatter1 = plot_shape(ax1, original_landmarks, PLOT_TITLE_ORIGINAL)

    print("\nPlotting cleaned shape...")
    scatter2 = plot_shape(ax2, cleaned_landmarks, PLOT_TITLE_CLEANED, is_cleaned_plot=True, outlier_mask_original=outlier_mask)

    if scatter1:
         cb1 = fig.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=10, label="Relative Depth (Z)")
    if scatter2:
         cb2 = fig.colorbar(scatter2, ax=ax2, shrink=0.5, aspect=10, label="Relative Depth (Z)")


    ax2.view_init(elev=20, azim=-60)
    ax2.set_title(PLOT_TITLE_CLEANED + "\n(View: elev=20, azim=-60)")


    plt.tight_layout(pad=2.0)
    plt.show()

if __name__ == "__main__":
    visualize_mean_shape_with_filtering(MEAN_SHAPE_FILE)