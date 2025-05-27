import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import os
import time
import datetime

# MediaPipe imports
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode  # Corrected access

# --- Configuration ---
MODEL_PATH = 'model/face_landmarker_v2_with_blendshapes.task'  # IMPORTANT: Use a model that outputs blendshapes!


# The regular face_landmarker.task might not.
# Download 'face_landmarker_v2_with_blendshapes.task'
# from MediaPipe's model card.

def create_blendshape_landmarker_options(model_path=MODEL_PATH):
    """Creates FaceLandmarkerOptions specifically for blendshape extraction."""
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,  # Process frame by frame from video
        num_faces=1,  # Assuming you want blendshapes for the primary detected face
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,  # CRITICAL: Enable blendshapes
        output_facial_transformation_matrixes=False  # Keep false unless you need them
    )
    return options


def process_bag_for_blendshapes(bag_filename, output_csv_dir, sentence_num, landmarker_options, trim_duration_sec=0.0):
    print(
        f"Blendshapes: Processing sentence {sentence_num} from {bag_filename} (skipping first {trim_duration_sec:.2f}s)...")

    csv_filename_base = f"sentence_{sentence_num:03d}_mediapipe_blendshapes.csv"  # Distinct filename
    csv_filepath = os.path.join(output_csv_dir, csv_filename_base)

    pipeline = None
    frames_with_blendshapes_written = 0

    try:
        pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, bag_filename, repeat_playback=False)

        profile = pipeline.start(config)
        playback_device = profile.get_device()
        if not playback_device.is_playback():
            print(f"  Blendshapes Error: Device from BAG {bag_filename} is not playback.")
            return
        playback = playback_device.as_playback()
        playback.set_real_time(False)

        try:
            total_duration_timedelta = playback.get_duration()
            total_duration_ns = int(total_duration_timedelta.total_seconds() * 1_000_000_000)
        except Exception:
            total_duration_ns = -1  # Duration unknown

        if trim_duration_sec > 0.001:
            seek_delta = datetime.timedelta(seconds=trim_duration_sec)
            print(f"  Blendshapes: Seeking to {trim_duration_sec:.2f}s...")
            playback.seek(seek_delta)
            time.sleep(0.2)

        # We only need the color stream for MediaPipe input
        # Alignment and depth are not strictly needed if only extracting blendshapes,
        # but keeping align might be simpler if future features need it.
        # For pure blendshapes, could skip align and depth processing.
        align_to = rs.stream.color
        align = rs.align(align_to)

        with FaceLandmarker.create_from_options(landmarker_options) as landmarker, \
                open(csv_filepath, 'w') as csv_file:

            csv_file.write("sentence_id,frame_id,blendshape_name,score\n")

            last_timestamp_ms = 0
            consecutive_no_frame_count = 0
            MAX_CONSECUTIVE_NO_FRAMES = 300
            current_frame_id_for_csv = 0  # Keep a running frame ID for the CSV output

            while True:
                frames_tuple_or_set = None
                try:
                    frames_tuple_or_set = pipeline.try_wait_for_frames(100)
                except RuntimeError:
                    pass  # Let logic below handle

                actual_frameset = None
                no_frame_received = False

                if isinstance(frames_tuple_or_set, tuple):
                    success, received_frameset = frames_tuple_or_set
                    if success and received_frameset:
                        actual_frameset = received_frameset
                    else:
                        no_frame_received = True
                elif frames_tuple_or_set:
                    actual_frameset = frames_tuple_or_set
                else:
                    no_frame_received = True

                if no_frame_received:
                    consecutive_no_frame_count += 1
                    current_position_ns = playback.get_position()
                    if total_duration_ns > 0 and current_position_ns >= total_duration_ns - 100_000_000:
                        print(
                            f"  Blendshapes: Position ({current_position_ns / 1e9:.2f}s) near/at duration. End of {bag_filename}.")
                        break
                    if consecutive_no_frame_count > MAX_CONSECUTIVE_NO_FRAMES:
                        print(f"  Blendshapes: Max no-frame polls. End of {bag_filename}.")
                        break
                    time.sleep(0.01)
                    continue

                consecutive_no_frame_count = 0

                # Align frames to get color (depth not strictly needed for blendshapes only)
                aligned_frames = align.process(actual_frameset)
                color_frame_rs = aligned_frames.get_color_frame()

                if not color_frame_rs:
                    current_frame_id_for_csv += 1  # Still increment frame ID even if no color
                    continue

                color_image_np = np.asanyarray(color_frame_rs.get_data())
                color_image_format = color_frame_rs.get_profile().format()
                rgb_image_np = None

                if color_image_format == rs.format.rgb8:
                    rgb_image_np = color_image_np
                elif color_image_format == rs.format.bgr8:
                    rgb_image_np = cv2.cvtColor(color_image_np, cv2.COLOR_BGR2RGB)
                else:
                    if color_image_np.ndim == 3 and color_image_np.shape[2] == 3:
                        rgb_image_np = cv2.cvtColor(color_image_np, cv2.COLOR_BGR2RGB)
                    elif color_image_np.ndim == 2:  # Grayscale
                        rgb_image_np = cv2.cvtColor(color_image_np, cv2.COLOR_GRAY2RGB)
                    else:
                        print(
                            f"  Blendshapes Error: Cannot handle color format {color_image_format} for {bag_filename}")
                        current_frame_id_for_csv += 1
                        continue

                if rgb_image_np is None:  # Should not happen if logic above is correct
                    current_frame_id_for_csv += 1
                    continue

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image_np)
                timestamp_ms = int(color_frame_rs.get_timestamp())
                if timestamp_ms <= last_timestamp_ms and last_timestamp_ms > 0:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms

                face_landmarker_result = None
                try:
                    if landmarker_options.running_mode == VisionRunningMode.VIDEO:
                        face_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    # Add IMAGE mode if you plan to use it, but VIDEO is typical for bag files
                except Exception as e_mp:
                    print(f"  Blendshapes Error: MediaPipe detection failed for {bag_filename}: {e_mp}")
                    current_frame_id_for_csv += 1
                    continue

                wrote_blendshapes_this_frame = False
                if face_landmarker_result and face_landmarker_result.face_blendshapes:
                    # Assuming one face detected, get its blendshapes
                    blendshapes_for_face = face_landmarker_result.face_blendshapes[0]
                    for blendshape_category in blendshapes_for_face:
                        blendshape_name = blendshape_category.category_name
                        score = blendshape_category.score
                        csv_file.write(f"{sentence_num},{current_frame_id_for_csv},{blendshape_name},{score:.6f}\n")
                    wrote_blendshapes_this_frame = True

                if wrote_blendshapes_this_frame:
                    frames_with_blendshapes_written += 1  # Count frames where blendshapes were actually outputted
                    if frames_with_blendshapes_written % 100 == 0 and frames_with_blendshapes_written > 0:  # Print progress less frequently
                        print(
                            f"  Blendshapes: Processed {frames_with_blendshapes_written} frames with blendshapes for {bag_filename}...")

                current_frame_id_for_csv += 1  # Increment for every frame iteration from BAG

                current_position_ns_after_proc = playback.get_position()
                if total_duration_ns > 0 and current_position_ns_after_proc >= total_duration_ns - 10_000_000:
                    print(f"  Blendshapes: Position near/at duration. End of {bag_filename}.")
                    break

    except RuntimeError as e_rt:
        print(f"  Blendshapes RealSense/Runtime Error for {bag_filename}: {e_rt}")
    except Exception as e_gen:
        print(f"  Blendshapes General Error for {bag_filename}: {e_gen}")
    finally:
        if pipeline:
            try:
                pipeline.stop()
            except Exception as e_stop:
                print(f"  Blendshapes: Error stopping pipeline for {bag_filename}: {e_stop}")
        print(
            f"  Blendshapes: Saved data to {csv_filepath} ({frames_with_blendshapes_written} frames had blendshapes written).")
        print(f"  Blendshapes: Finished processing {bag_filename}")


if __name__ == "__main__":
    # --- Ensure Model File Exists ---
    model_dir = os.path.dirname(MODEL_PATH)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: MediaPipe model file '{MODEL_PATH}' not found.")
        print("Please download 'face_landmarker_v2_with_blendshapes.task' from MediaPipe model card")
        print(f"and place it at: {os.path.abspath(MODEL_PATH)}")
        exit()

    # --- Configuration for your files ---
    bag_files_directory = r"D:\SegmentationThesis\output_realsense60fps+tesla Catalin"  # YOUR BAG FILE DIRECTORY
    output_blendshapes_csv_directory = "./output_blendshapes_csv_catalin"  # Separate directory for blendshape CSVs

    start_sentence_id = 1
    max_sentences_to_check = 100  # Or however many sentences/BAGs you have
    trim_start_seconds = 1.0  # Same trim as your other script, if desired

    # --- Setup Directories ---
    if not os.path.exists(bag_files_directory):
        print(f"Error: BAG files directory not found: {os.path.abspath(bag_files_directory)}")
        exit()
    if not os.path.exists(output_blendshapes_csv_directory):
        os.makedirs(output_blendshapes_csv_directory, exist_ok=True)
        print(f"Created output blendshapes CSV directory: {os.path.abspath(output_blendshapes_csv_directory)}")

    # --- Create Landmarker Options ---
    landmarker_opts_for_blendshapes = create_blendshape_landmarker_options()

    # --- Find BAG Files ---
    bag_files_to_process_list = []
    print(f"Looking for BAG files in: {os.path.abspath(bag_files_directory)}")
    try:
        all_files_in_bag_dir = sorted(os.listdir(bag_files_directory))
        for i in range(max_sentences_to_check):
            current_sentence_id_val = start_sentence_id + i
            expected_bag_name_val = f"sentence_{current_sentence_id_val:03d}_realsense.bag"
            full_bag_path = os.path.join(bag_files_directory, expected_bag_name_val)
            if os.path.exists(full_bag_path):
                bag_files_to_process_list.append((full_bag_path, current_sentence_id_val))
                print(f"  Found: {full_bag_path}")
    except FileNotFoundError:
        print(f"Error: BAG files directory not found during listing: {os.path.abspath(bag_files_directory)}")
        exit()

    if not bag_files_to_process_list:
        print(f"No BAG files found matching the pattern in {os.path.abspath(bag_files_directory)}")

    # --- Process Files ---
    for bag_path_val, s_num_val in bag_files_to_process_list:
        print(f"\nProcessing for Blendshapes: {bag_path_val} for sentence {s_num_val}")
        process_bag_for_blendshapes(bag_path_val, output_blendshapes_csv_directory,
                                    s_num_val, landmarker_opts_for_blendshapes,
                                    trim_duration_sec=trim_start_seconds)

    print("\nAll specified BAG files processed for blendshapes.")