import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import os
import time
import datetime

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = 'model/face_landmarker_v2_with_blendshapes.task'


def create_landmarker_options(running_mode_enum_val, model_path=MODEL_PATH):
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=running_mode_enum_val,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )
    return options


def process_bag_file(bag_filename, output_csv_dir, sentence_num, landmarker_options, trim_duration_sec=0.0):
    print(
        f"Python: Processing sentence {sentence_num} from {bag_filename} (skipping first {trim_duration_sec:.2f}s) [MediaPipe Python]...")

    csv_filename_base = f"sentence_{sentence_num:03d}_mediapipe_landmarks_py.csv"
    csv_filepath = os.path.join(output_csv_dir, csv_filename_base)

    pipeline = None
    frames_processed_for_csv = 0

    try:
        pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, bag_filename, repeat_playback=False)

        profile = pipeline.start(config)
        playback_device = profile.get_device()  # Get the device
        if not playback_device.is_playback():
            print(f"  Python Error: Device from BAG file {bag_filename} is not a playback device.")
            return
        playback = playback_device.as_playback()  # Cast to playback
        playback.set_real_time(False)

        # Get total duration of the BAG file
        try:
            # get_duration() returns a timedelta object. We need its total seconds or nanoseconds.
            total_duration_timedelta = playback.get_duration()
            total_duration_ns = int(total_duration_timedelta.total_seconds() * 1_000_000_000)
            print(f"  Python: BAG file total duration: {total_duration_timedelta} ({total_duration_ns / 1e9:.2f} s)")
        except Exception as e_dur:
            print(
                f"  Python Warning: Could not get BAG file duration for {bag_filename}: {e_dur}. End detection might be less reliable.")
            total_duration_ns = -1  # Indicate duration unknown

        if trim_duration_sec > 0.001:
            seek_delta = datetime.timedelta(seconds=trim_duration_sec)
            print(f"  Python: Seeking playback to {trim_duration_sec:.2f} s (using timedelta: {seek_delta})...")
            playback.seek(seek_delta)
            print("  Python: Playback seek complete.")
            time.sleep(0.2)  # Give it a moment to settle after seek

        align_to = rs.stream.color
        align = rs.align(align_to)

        depth_sensor = profile.get_device().first_depth_sensor()
        if not depth_sensor:
            print(f"  Python Error: No depth sensor found in BAG: {bag_filename}")
            return
        depth_scale = depth_sensor.get_depth_scale()
        print(f"  Python: Depth Scale: {depth_scale}")

        color_stream = None
        for stream in profile.get_streams():
            if stream.stream_type() == rs.stream.color and stream.is_video_stream_profile():
                color_stream = stream.as_video_stream_profile()
                break

        if not color_stream:
            print(f"  Python Error: No color stream profile found in BAG: {bag_filename}")
            if pipeline: pipeline.stop()
            return

        depth_intrinsics_rs = color_stream.get_intrinsics()
        print(f"  Python: Aligned Depth Intrinsics (from color stream context): {depth_intrinsics_rs}")

        with FaceLandmarker.create_from_options(landmarker_options) as landmarker, \
                open(csv_filepath, 'w') as csv_file:

            csv_file.write("sentence_id,frame_id,landmark_id,x_cam,y_cam,z_cam\n")

            last_timestamp_ms = 0
            consecutive_no_frame_count = 0
            MAX_CONSECUTIVE_NO_FRAMES = 300  # Number of timeouts before assuming end

            while True:
                frames_tuple_or_set = None
                try:
                    frames_tuple_or_set = pipeline.try_wait_for_frames(100)
                except RuntimeError as e:
                    # This catch might be too broad if try_wait_for_frames doesn't raise on simple timeout
                    print(f"  Python: RealSense runtime error during try_wait_for_frames for {bag_filename}: {e}")
                    break

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
                    current_position_ns = playback.get_position()  # Get current position in nanoseconds

                    # Check if near or at the end based on duration
                    if total_duration_ns > 0 and current_position_ns >= total_duration_ns - 100_000_000:  # Within 0.1s of end
                        print(
                            f"  Python: Playback position ({current_position_ns / 1e9:.2f}s) reached or exceeded duration ({total_duration_ns / 1e9:.2f}s). Exiting loop for {bag_filename}.")
                        break

                    if consecutive_no_frame_count > MAX_CONSECUTIVE_NO_FRAMES:
                        print(
                            f"  Python: Exceeded max consecutive no-frame polls ({MAX_CONSECUTIVE_NO_FRAMES}). Current position {current_position_ns / 1e9:.2f}s. Assuming end of BAG for {bag_filename}.")
                        break
                    time.sleep(0.01)
                    continue

                consecutive_no_frame_count = 0

                aligned_frames = align.process(actual_frameset)
                color_frame_rs = aligned_frames.get_color_frame()
                depth_frame_rs = aligned_frames.get_depth_frame()

                if not color_frame_rs or not depth_frame_rs:
                    continue

                color_image_np = np.asanyarray(color_frame_rs.get_data())
                color_image_format = color_frame_rs.get_profile().format()

                if color_image_format == rs.format.rgb8:
                    rgb_image_np = color_image_np
                elif color_image_format == rs.format.bgr8:
                    rgb_image_np = cv2.cvtColor(color_image_np, cv2.COLOR_BGR2RGB)
                else:
                    if color_image_np.ndim == 3 and color_image_np.shape[2] == 3:
                        rgb_image_np = cv2.cvtColor(color_image_np, cv2.COLOR_BGR2RGB)
                    elif color_image_np.ndim == 2:
                        rgb_image_np = cv2.cvtColor(color_image_np, cv2.COLOR_GRAY2RGB)
                    else:
                        print(
                            f"  Python Error: Cannot handle color format {color_image_format} with shape {color_image_np.shape} for {bag_filename}")
                        continue

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image_np)
                timestamp_ms = int(color_frame_rs.get_timestamp())

                if timestamp_ms <= last_timestamp_ms and last_timestamp_ms > 0:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms

                try:
                    if landmarker_options.running_mode == VisionRunningMode.VIDEO:
                        face_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    elif landmarker_options.running_mode == VisionRunningMode.IMAGE:
                        face_landmarker_result = landmarker.detect(mp_image)
                    else:
                        print(f"  Python Error: Unsupported running mode for this script for {bag_filename}.")
                        break
                except Exception as e_mp:
                    print(f"  Python Error: MediaPipe detection failed for {bag_filename}: {e_mp}")
                    continue

                if face_landmarker_result and face_landmarker_result.face_landmarks:
                    landmarks_list_normalized = face_landmarker_result.face_landmarks[0]
                    depth_image_np = np.asanyarray(depth_frame_rs.get_data())
                    color_w, color_h = color_frame_rs.get_width(), color_frame_rs.get_height()
                    depth_w, depth_h = depth_frame_rs.get_width(), depth_frame_rs.get_height()

                    num_landmarks_written_this_frame = 0
                    for landmark_idx, landmark in enumerate(landmarks_list_normalized):
                        norm_x, norm_y = landmark.x, landmark.y
                        pixel_x = min(max(int(norm_x * color_w), 0), color_w - 1)
                        pixel_y = min(max(int(norm_y * color_h), 0), color_h - 1)
                        depth_units = 0
                        if 0 <= pixel_x < depth_w and 0 <= pixel_y < depth_h:
                            depth_units = depth_image_np[pixel_y, pixel_x]
                        depth_m = depth_units * depth_scale
                        if depth_m > 0.001:
                            point3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics_rs,
                                                                      [float(pixel_x), float(pixel_y)], depth_m)
                            csv_file.write(
                                f"{sentence_num},{frames_processed_for_csv},{landmark_idx},{point3d[0]:.6f},{point3d[1]:.6f},{point3d[2]:.6f}\n")
                            num_landmarks_written_this_frame += 1
                        else:
                            csv_file.write(f"{sentence_num},{frames_processed_for_csv},{landmark_idx},0.0,0.0,0.0\n")
                    if num_landmarks_written_this_frame > 0:
                        frames_processed_for_csv += 1
                        if frames_processed_for_csv > 0 and frames_processed_for_csv % 30 == 0:
                            print(f"  Python Processed {frames_processed_for_csv} frames for CSV for {bag_filename}...")

                # Check end of file after processing a frame too
                current_position_ns_after_proc = playback.get_position()
                if total_duration_ns > 0 and current_position_ns_after_proc >= total_duration_ns - 10_000_000:  # very close to end (10ms)
                    print(
                        f"  Python: Playback position ({current_position_ns_after_proc / 1e9:.2f}s) near/at duration ({total_duration_ns / 1e9:.2f}s) after processing. Exiting loop for {bag_filename}.")
                    break


    except RuntimeError as e_rt:
        print(f"  Python RealSense/Runtime Error during processing of {bag_filename}: {e_rt}")
    except Exception as e_gen:
        print(f"  Python General Error during processing of {bag_filename}: {e_gen}")
    finally:
        if pipeline:
            try:
                pipeline.stop()
                print(f"  Python: Pipeline stopped for {bag_filename}")
            except Exception as e_stop:
                print(f"  Python: Error stopping pipeline for {bag_filename}: {e_stop}")
        print(f"  Python Saved MediaPipe landmark data to {csv_filepath} ({frames_processed_for_csv} frames written).")
        print(f"  Python Finished processing {bag_filename}")


if __name__ == "__main__":
    model_dir = os.path.dirname(MODEL_PATH)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: MediaPipe model file not found at {MODEL_PATH}")
        print(f"Please download 'face_landmarker.task' and place it at: {os.path.abspath(MODEL_PATH)}")
        exit()

    bag_files_directory = r"D:\SegmentationThesis\output_realsense60fps+tesla Marinela"
    output_csv_directory = "./output_landmarks_csv_marinela"
    start_sentence_id = 1
    max_sentences_to_check = 100  # Check for up to 100 sentences
    trim_start_seconds = 0.3

    if not os.path.exists(bag_files_directory):
        print(f"Error: BAG files directory not found: {os.path.abspath(bag_files_directory)}")
        exit()
    if not os.path.exists(output_csv_directory):
        os.makedirs(output_csv_directory, exist_ok=True)
        print(f"Created output CSV directory: {os.path.abspath(output_csv_directory)}")

    landmarker_opts = create_landmarker_options(running_mode_enum_val=VisionRunningMode.VIDEO)

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
        print(
            f"No BAG files found matching the pattern 'sentence_XXX_realsense.bag' in {os.path.abspath(bag_files_directory)}")

    for bag_path_val, s_num_val in bag_files_to_process_list:
        print(f"\nProcessing: {bag_path_val} for sentence {s_num_val}")
        process_bag_file(bag_path_val, output_csv_directory, s_num_val, landmarker_opts,
                         trim_duration_sec=trim_start_seconds)
    print("\nAll specified BAG files processed.")