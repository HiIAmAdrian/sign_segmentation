import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting
import numpy as np
import cv2 # For creating video
import os
import glob # For finding files matching a pattern

# --- Configuration ---
# Directory containing your input CSV files
INPUT_CSV_DIR = "output_landmarks_csv_marinela_occlusion_normalised" # EXAMPLE: Replace with your input directory
# Pattern to match your CSV files within INPUT_CSV_DIR
INPUT_CSV_PATTERN = "sentence_008_mediapipe_landmarks_py_oneeuro_smoothed_filled_ransac.csv" # EXAMPLE: e.g., "*.csv" or "sentence_*_smoothed.csv"

OUTPUT_VIDEO_FILENAME = "face_reconstruction_output/face_reconstruction_all_sentences_video-marinela.mp4" # Output video name
FPS = 60 # Frames per second for the output video
PLOT_POINT_SIZE = 2
MEDIAPIPE_LANDMARK_COUNT = 478 # Adjust to 468 if needed

# LANDMARK_CONNECTIONS = [] # Keep this if you don't want connections
# Example if you have MediaPipe installed and want to draw connections:
try:
    from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION
    LANDMARK_CONNECTIONS = []
    print("Using MediaPipe FACEMESH_TESSELATION for connections.")
except ImportError:
    LANDMARK_CONNECTIONS = []
    print("MediaPipe not found or FACEMESH_TESSELATION not available. No landmark connections will be drawn.")


def plot_face_landmarks_3d(ax, landmarks_xyz, frame_id, sentence_id_str): # sentence_id is now a string
    """Plots 3D landmarks for a single frame."""
    ax.cla() # Clear previous frame

    # Filter out (0,0,0) landmarks, assuming they are invalid or not to be plotted
    valid_landmarks_mask = ~np.all(np.isclose(landmarks_xyz, 0, atol=1e-6), axis=1)
    landmarks_to_plot = landmarks_xyz[valid_landmarks_mask]

    if landmarks_to_plot.shape[0] == 0: # No valid landmarks to plot
        ax.set_title(f"Sentence {sentence_id_str}, Frame {frame_id} (No valid landmarks)")
        # Set default axis limits even if no data, as per your original script
        ax.set_xlim([-0.2, 0.2])
        ax.set_ylim([-0.2, 0.2])
        ax.set_zlim([0.3, 0.9]) # Original Z limits
        ax.invert_zaxis() # As per your original script
        ax.view_init(elev=20, azim=0) # As per your original script
        return

    x_coords = landmarks_to_plot[:, 0]
    y_coords = landmarks_to_plot[:, 1]
    z_coords = landmarks_to_plot[:, 2]

    # Scatter plot
    # Using fixed vmin/vmax for colormap as in your original script
    ax.scatter(x_coords, y_coords, z_coords, s=PLOT_POINT_SIZE, c=z_coords, cmap='viridis_r', vmin=0.3, vmax=1.0)

    # Plot connections if defined
    if LANDMARK_CONNECTIONS:
        for p1_idx, p2_idx in LANDMARK_CONNECTIONS:
            # Check if both landmarks exist and are valid (non-zero)
            if (0 <= p1_idx < MEDIAPIPE_LANDMARK_COUNT and
                0 <= p2_idx < MEDIAPIPE_LANDMARK_COUNT and
                valid_landmarks_mask[p1_idx] and
                valid_landmarks_mask[p2_idx]):
                point1 = landmarks_xyz[p1_idx]
                point2 = landmarks_xyz[p2_idx]
                ax.plot(
                    [point1[0], point2[0]],
                    [point1[1], point2[1]],
                    [point1[2], point2[2]],
                    'gray', alpha=0.4, linewidth=0.5
                )

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_zlabel("Z (meters - Depth)")
    ax.set_title(f"Sentence {sentence_id_str}, Frame {frame_id}")

    # Set axis limits and view as per your original script
    ax.set_xlim([-0.2, 0.2])
    ax.set_ylim([-0.2, 0.2])
    ax.set_zlim([0.3, 0.9]) # Original Z limits
    ax.invert_zaxis() # Match Z-axis inversion
    ax.view_init(elev=20, azim=0) # Match view


def main():
    output_video_dir = os.path.dirname(OUTPUT_VIDEO_FILENAME)
    if output_video_dir and not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir)
        print(f"Created output directory: {output_video_dir}")

    csv_file_search_path = os.path.join(INPUT_CSV_DIR, INPUT_CSV_PATTERN)
    csv_files_to_process = sorted(glob.glob(csv_file_search_path))

    if not csv_files_to_process:
        print(f"Error: No CSV files found matching pattern: {csv_file_search_path}")
        print(f"Please check INPUT_CSV_DIR ('{INPUT_CSV_DIR}') and INPUT_CSV_PATTERN ('{INPUT_CSV_PATTERN}')")
        return
    print(f"Found {len(csv_files_to_process)} CSV files to process.")

    fig_temp = plt.figure(figsize=(10, 7.5))
    ax_temp = fig_temp.add_subplot(111, projection='3d')
    dummy_landmarks = np.random.rand(MEDIAPIPE_LANDMARK_COUNT, 3) * 0.1
    dummy_landmarks[:, 2] += 0.5
    plot_face_landmarks_3d(ax_temp, dummy_landmarks, -1, "Initial")
    fig_temp.canvas.draw()
    buf = fig_temp.canvas.buffer_rgba()
    img_data_rgba = np.asarray(buf)
    img_data_rgb = img_data_rgba[:, :, :3]
    height, width, _ = img_data_rgb.shape
    plt.close(fig_temp)

    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_FILENAME,
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   FPS,
                                   (width, height))
    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for {OUTPUT_VIDEO_FILENAME}")
        return
    print(f"Video writer opened for {OUTPUT_VIDEO_FILENAME} ({width}x{height} @ {FPS}fps)")

    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(111, projection='3d')

    total_frames_processed_from_csvs = 0
    total_frames_written_to_video = 0

    # --- DEBUG: Option to process only a limited number of CSVs ---
    # MAX_CSVS_TO_PROCESS = 1
    # if MAX_CSVS_TO_PROCESS is not None:
    #     print(f"DEBUG: Limiting processing to first {MAX_CSVS_TO_PROCESS} CSV file(s).")
    #     csv_files_to_process = csv_files_to_process[:MAX_CSVS_TO_PROCESS]
    # --- END DEBUG ---

    for csv_idx, csv_file_path in enumerate(csv_files_to_process):
        print(f"\n[{csv_idx + 1}/{len(csv_files_to_process)}] Processing CSV: {os.path.basename(csv_file_path)}...")
        try:
            df = pd.read_csv(csv_file_path)
            print(f"  Successfully loaded {os.path.basename(csv_file_path)}. Shape: {df.shape}")
        except pd.errors.EmptyDataError:
            print(f"  Warning: CSV file {csv_file_path} is empty. Skipping.")
            continue
        except Exception as e:
            print(f"  Error reading CSV {csv_file_path}: {e}. Skipping.")
            continue

        if df.empty or not all(col in df.columns for col in ['frame_id', 'landmark_id', 'x_cam', 'y_cam', 'z_cam']):
            print(f"  Warning: CSV file {csv_file_path} is empty or has missing required columns. Skipping.")
            continue

        if 'sentence_id' in df.columns:
            sentence_id_val = df['sentence_id'].iloc[0] if not df.empty else "Unknown"
            sentence_id_str = str(sentence_id_val)
        else:
            sentence_id_str = os.path.splitext(os.path.basename(csv_file_path))[0]

        grouped_frames = df.groupby('frame_id')
        num_unique_frames = len(grouped_frames)
        print(f"  Found {num_unique_frames} unique frames in this CSV.")

        # --- DEBUG: Option to process only a limited number of frames per CSV ---
        # MAX_FRAMES_PER_CSV = 50 # Set to None to process all
        frames_processed_this_csv = 0
        # --- END DEBUG ---

        for frame_id, group in grouped_frames:
            # if MAX_FRAMES_PER_CSV is not None and frames_processed_this_csv >= MAX_FRAMES_PER_CSV:
            #     print(f"  DEBUG: Reached max frames ({MAX_FRAMES_PER_CSV}) for {os.path.basename(csv_file_path)}. Moving to next CSV.")
            #     break
            # frames_processed_this_csv += 1

            total_frames_processed_from_csvs += 1
            # More frequent progress update
            if total_frames_processed_from_csvs % (
                    FPS // 2) == 0 or total_frames_processed_from_csvs == 1:  # Approx every half second of video, or first frame
                print(
                    f"    Processing Frame ID {int(frame_id)} from {sentence_id_str} (Total CSV frames processed: {total_frames_processed_from_csvs})")

            group = group.sort_values(by='landmark_id')
            landmarks_xyz = np.zeros((MEDIAPIPE_LANDMARK_COUNT, 3))
            present_lm_ids = group['landmark_id'].astype(int).values
            present_coords = group[['x_cam', 'y_cam', 'z_cam']].values
            for i, lm_id_val in enumerate(present_lm_ids):
                if 0 <= lm_id_val < MEDIAPIPE_LANDMARK_COUNT:
                    landmarks_xyz[lm_id_val, :] = present_coords[i, :]

            plot_face_landmarks_3d(ax, landmarks_xyz, int(frame_id), sentence_id_str)
            fig.canvas.draw()  # This is often the slowest part per frame
            buf = fig.canvas.buffer_rgba()
            image_from_plot_rgba = np.asarray(buf)
            image_from_plot_rgb = image_from_plot_rgba[:, :, :3]
            image_bgr = cv2.cvtColor(image_from_plot_rgb, cv2.COLOR_RGB2BGR)
            video_writer.write(image_bgr)
            total_frames_written_to_video += 1

            # Original progress indicator
            if total_frames_written_to_video > 0 and total_frames_written_to_video % (FPS * 10) == 0:
                print(
                    f"  LOG UPDATE: Total frames written to video: {total_frames_written_to_video} (Current file: {os.path.basename(csv_file_path)}, Frame ID: {frame_id})")

        print(
            f"  Finished processing all {frames_processed_this_csv if 'MAX_FRAMES_PER_CSV' in locals() and MAX_FRAMES_PER_CSV is not None else num_unique_frames} frames for {os.path.basename(csv_file_path)}.")

    video_writer.release()
    plt.close(fig)
    print(f"\nVideo saved to {OUTPUT_VIDEO_FILENAME}.")
    print(f"Total frames processed from CSV(s): {total_frames_processed_from_csvs}")
    print(f"Total frames written to video: {total_frames_written_to_video}")

if __name__ == "__main__":
    main()