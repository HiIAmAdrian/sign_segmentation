import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting
import numpy as np
import cv2 # For creating video
import os
# import glob # No longer needed if processing a single file

# --- Configuration ---
# IMPORTANT: Set this to the exact path of the ONE CSV file you want to process
SINGLE_CSV_FILE_PATH = "output_landmarks_csv_catalin_smoothed/sentence_084_mediapipe_landmarks_py_oneeuro_smoothed.csv" # EXAMPLE
OUTPUT_VIDEO_FILENAME = "face_reconstruction_output/face_reconstruction_single_file_video84.mp4" # Can adjust output name
FPS = 60
PLOT_POINT_SIZE = 2
MEDIAPIPE_LANDMARK_COUNT = 478 # Adjust to 468 if needed

LANDMARK_CONNECTIONS = [] # No connections for now

def plot_face_landmarks_3d(ax, landmarks_xyz, frame_id, sentence_id):
    """Plots 3D landmarks for a single frame."""
    ax.cla()

    valid_landmarks_mask = ~np.all(np.isclose(landmarks_xyz, 0), axis=1)
    landmarks_to_plot = landmarks_xyz[valid_landmarks_mask]

    if landmarks_to_plot.shape[0] == 0:
        ax.set_title(f"Sentence {sentence_id}, Frame {frame_id} (No valid depth)")
        ax.set_xlim([-0.2, 0.2])
        ax.set_ylim([-0.2, 0.2])
        ax.set_zlim([0.3, 0.9])
        ax.invert_zaxis()
        ax.view_init(elev=20, azim=0)
        return

    x_coords = landmarks_to_plot[:, 0]
    y_coords = landmarks_to_plot[:, 1]
    z_coords = landmarks_to_plot[:, 2]

    ax.scatter(x_coords, y_coords, z_coords, s=PLOT_POINT_SIZE, c=z_coords, cmap='viridis_r', vmin=0.3, vmax=1.0)

    if LANDMARK_CONNECTIONS:
        for p1_idx, p2_idx in LANDMARK_CONNECTIONS:
            if p1_idx < len(landmarks_xyz) and p2_idx < len(landmarks_xyz):
                if valid_landmarks_mask[p1_idx] and valid_landmarks_mask[p2_idx]:
                    point1 = landmarks_xyz[p1_idx]
                    point2 = landmarks_xyz[p2_idx]
                    ax.plot(
                        [point1[0], point2[0]],
                        [point1[1], point2[1]],
                        [point1[2], point2[2]],
                        'gray', alpha=0.4
                    )

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_zlabel("Z (meters - Depth)")
    ax.set_title(f"Sentence {sentence_id}, Frame {frame_id}")

    ax.set_xlim([-0.2, 0.2])
    ax.set_ylim([-0.2, 0.2])
    ax.set_zlim([0.3, 0.9])
    ax.invert_zaxis()
    ax.view_init(elev=20, azim=0)


def main():
    output_video_dir = os.path.dirname(OUTPUT_VIDEO_FILENAME)
    if output_video_dir and not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir)
        print(f"Created output directory: {output_video_dir}")

    # --- Process a single specified CSV file ---
    if not os.path.exists(SINGLE_CSV_FILE_PATH):
        print(f"Error: Specified CSV file not found: {SINGLE_CSV_FILE_PATH}")
        print(f"Please check the path: {os.path.abspath(SINGLE_CSV_FILE_PATH)}")
        return
    print(f"Targeting single CSV file: {SINGLE_CSV_FILE_PATH}")
    csv_files_to_process = [SINGLE_CSV_FILE_PATH] # Create a list with one item
    # --- End of single file targeting ---


    fig_temp = plt.figure(figsize=(10, 7.5))
    ax_temp = fig_temp.add_subplot(111, projection='3d')
    dummy_landmarks = np.random.rand(MEDIAPIPE_LANDMARK_COUNT, 3) * 0.1
    dummy_landmarks[:, 2] += 0.5
    plot_face_landmarks_3d(ax_temp, dummy_landmarks, -1, -1)
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

    total_frames_processed = 0
    total_frames_written_to_video = 0

    # The loop will now iterate only once if csv_files_to_process has one item
    for csv_file_path in csv_files_to_process:
        print(f"Processing {os.path.basename(csv_file_path)}...")
        try:
            df = pd.read_csv(csv_file_path)
        except pd.errors.EmptyDataError:
            print(f"  Warning: CSV file {csv_file_path} is empty. Skipping.")
            continue
        except Exception as e:
            print(f"  Error reading CSV {csv_file_path}: {e}. Skipping.")
            continue

        if df.empty or not all(col in df.columns for col in ['sentence_id', 'frame_id', 'landmark_id', 'x_cam', 'y_cam', 'z_cam']):
            print(f"  Warning: CSV file {csv_file_path} is empty or has missing columns. Skipping.")
            continue

        # Get sentence_id from the first row (assuming it's consistent for the file)
        sentence_id = df['sentence_id'].iloc[0] if not df.empty else "UnknownSentence"


        for frame_id, group in df.groupby('frame_id'):
            total_frames_processed += 1
            group = group.sort_values(by='landmark_id')
            landmarks_xyz = group[['x_cam', 'y_cam', 'z_cam']].values

            if landmarks_xyz.shape[0] == 0:
                continue

            plot_face_landmarks_3d(ax, landmarks_xyz, int(frame_id), sentence_id)

            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            image_from_plot_rgba = np.asarray(buf)
            image_from_plot_rgb = image_from_plot_rgba[:, :, :3]
            image_bgr = cv2.cvtColor(image_from_plot_rgb, cv2.COLOR_RGB2BGR)

            video_writer.write(image_bgr)
            total_frames_written_to_video += 1

            if total_frames_written_to_video % (FPS * 5) == 0 and total_frames_written_to_video > 0 :
                 print(f"  ... frames written to video: {total_frames_written_to_video}")

    video_writer.release()
    plt.close(fig)
    print(f"Video saved to {OUTPUT_VIDEO_FILENAME}.")
    print(f"Total frames processed from CSV(s): {total_frames_processed}")
    print(f"Total frames written to video: {total_frames_written_to_video}")

if __name__ == "__main__":
    main()