import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting
import numpy as np
import cv2 # For creating video
import os
import glob

# --- Configuration ---
CSV_FILE_PATTERN = "output_data/sentence_*_dlib_landmarks_cpp.csv"  # Adjust if your CSV filenames differ
OUTPUT_VIDEO_FILENAME = "output_data/face_reconstruction_video.mp4"
FPS = 15 # Frame rate for the output video (adjust as needed)
PLOT_POINT_SIZE = 5
DLIB_LANDMARK_COUNT = 68 # Make sure this matches the number of landmarks in your CSV per frame

# Optional: Define connections between landmarks to draw lines
LANDMARK_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
    (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19),
    (19, 20), (20, 21), (22, 23), (23, 24), (24, 25), (25, 26), (27, 28), (28, 29),
    (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (30,35), (36, 37),
    (37, 38), (38, 39), (39, 40), (40, 41), (41, 36), (42, 43), (43, 44), (44, 45),
    (45, 46), (46, 47), (47, 42), (48, 49), (49, 50), (50, 51), (51, 52), (52, 53),
    (53, 54), (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48), (60, 61),
    (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 60)
]


def plot_face_landmarks_3d(ax, landmarks_xyz, frame_id, sentence_id):
    """Plots 3D landmarks for a single frame."""
    ax.cla()

    x_coords = landmarks_xyz[:, 0]
    y_coords = landmarks_xyz[:, 1]
    z_coords = landmarks_xyz[:, 2]

    ax.scatter(x_coords, y_coords, z_coords, s=PLOT_POINT_SIZE, c=z_coords, cmap='viridis_r', vmin=0.3, vmax=1.0)

    for p1_idx, p2_idx in LANDMARK_CONNECTIONS:
        if p1_idx < len(landmarks_xyz) and p2_idx < len(landmarks_xyz):
            if not (np.all(np.isclose(landmarks_xyz[p1_idx], 0)) or np.all(np.isclose(landmarks_xyz[p2_idx], 0))):
                ax.plot(
                    [landmarks_xyz[p1_idx, 0], landmarks_xyz[p2_idx, 0]],
                    [landmarks_xyz[p1_idx, 1], landmarks_xyz[p2_idx, 1]],
                    [landmarks_xyz[p1_idx, 2], landmarks_xyz[p2_idx, 2]],
                    'gray', alpha=0.6
                )

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_zlabel("Z (meters - Depth)")
    ax.set_title(f"Sentence {sentence_id}, Frame {frame_id}")

    ax.set_xlim([-0.2, 0.2])
    ax.set_ylim([-0.2, 0.2])
    ax.set_zlim([0.3, 0.9])
    ax.invert_zaxis()

    ax.view_init(elev=20, azim=0)  # <-- Adjusted for front-facing view



def main():
    # Ensure output directory exists
    output_video_dir = os.path.dirname(OUTPUT_VIDEO_FILENAME)
    if output_video_dir and not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir)
        print(f"Created output directory: {output_video_dir}")

    csv_files = sorted(glob.glob(CSV_FILE_PATTERN))
    if not csv_files:
        print(f"No CSV files found matching pattern: {CSV_FILE_PATTERN}")
        return

    print(f"Found CSV files: {csv_files}")

    # --- Setup for video writing using buffer_rgba ---
    # Create a temporary figure to get canvas dimensions
    fig_temp = plt.figure(figsize=(8, 6)) # Adjust figure size as needed
    ax_temp = fig_temp.add_subplot(111, projection='3d')
    # Plot dummy data to ensure canvas is rendered
    dummy_landmarks = np.zeros((DLIB_LANDMARK_COUNT, 3)) # Use zeros for dummy data
    plot_face_landmarks_3d(ax_temp, dummy_landmarks, -1, -1)
    fig_temp.canvas.draw() # Ensure canvas is drawn

    buf = fig_temp.canvas.buffer_rgba()
    img_data_rgba = np.asarray(buf)
    img_data_rgb = img_data_rgba[:, :, :3] # Slice to get RGB (drop alpha)

    height, width, _ = img_data_rgb.shape
    plt.close(fig_temp) # Close the temporary figure

    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_FILENAME,
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   FPS,
                                   (width, height))
    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for {OUTPUT_VIDEO_FILENAME}")
        return
    print(f"Video writer opened for {OUTPUT_VIDEO_FILENAME} ({width}x{height} @ {FPS}fps)")

    # Main figure for generating video frames
    fig = plt.figure(figsize=(8, 6)) # Use the same figsize
    ax = fig.add_subplot(111, projection='3d')

    total_frames_written = 0
    for csv_file_path in csv_files:
        print(f"Processing {csv_file_path}...")
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

        sentence_id = df['sentence_id'].iloc[0]

        for frame_id, group in df.groupby('frame_id'):
            group = group.sort_values(by='landmark_id')
            landmarks_xyz = group[['x_cam', 'y_cam', 'z_cam']].values

            if len(landmarks_xyz) != DLIB_LANDMARK_COUNT:
                # print(f"  Warning: Sentence {sentence_id}, Frame {frame_id} has {len(landmarks_xyz)} landmarks, expected {DLIB_LANDMARK_COUNT}. Skipping frame.")
                continue

            plot_face_landmarks_3d(ax, landmarks_xyz, int(frame_id), sentence_id)

            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            image_from_plot_rgba = np.asarray(buf)
            image_from_plot_rgb = image_from_plot_rgba[:, :, :3] # Get RGB from RGBA

            image_bgr = cv2.cvtColor(image_from_plot_rgb, cv2.COLOR_RGB2BGR)
            video_writer.write(image_bgr)
            total_frames_written +=1

            # Optional: Display plot (will be very slow for many frames)
            # plt.pause(0.001) # Reduced pause time

    video_writer.release()
    plt.close(fig)
    print(f"Video saved to {OUTPUT_VIDEO_FILENAME}. Total frames written: {total_frames_written}")


if __name__ == "__main__":
    main()