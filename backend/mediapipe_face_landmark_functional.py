import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import time
import datetime
from pathlib import Path
import traceback

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def create_landmarker_options_for_landmarks_functional(running_mode_enum_val, model_asset_path_str):
    """Creează FaceLandmarkerOptions pentru extracția de landmark-uri."""
    print(
        f"    CREATE_OPTS_BS: Received model_asset_path_str: '{model_asset_path_str}', type: {type(model_asset_path_str)}")
    try:
        resolved_model_path = str(
            Path(model_asset_path_str).resolve(strict=True))
        print(f"    CREATE_OPTS_BS: Resolved model_asset_path: '{resolved_model_path}'")
    except FileNotFoundError:
        print(f"    CREATE_OPTS_BS: ERROR - Model file NOT FOUND at path: {model_asset_path_str}")
        raise
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_asset_path_str)),
        running_mode=running_mode_enum_val,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )
    return options


def run_landmark_extraction_for_bag(
        bag_file_path_str: str,
        output_csv_dir_str: str,
        sentence_num_int: int,
        landmarker_model_file_path_str: str,
        trim_duration_sec: float = 0.0,
        target_fps_for_mediapipe: int = 60
):
    """
    Procesează un singur fișier .bag pentru a extrage landmark-uri faciale 3D.
    Scrie rezultatele într-un fișier CSV în directorul specificat.
    Returnează calea către fișierul CSV creat sau None dacă eșuează.
    """
    bag_filename = Path(bag_file_path_str).name
    print(
        f"  FUNC_LM: Processing landmarks for sentence {sentence_num_int} from {bag_filename} (trim: {trim_duration_sec:.2f}s)...")

    if not Path(landmarker_model_file_path_str).exists():
        print(f"    FUNC_LM: ERROR - MediaPipe landmark model not found at {landmarker_model_file_path_str}")
        return None

    Path(output_csv_dir_str).mkdir(parents=True, exist_ok=True)

    csv_filename_base = f"sentence_{sentence_num_int:03d}_mediapipe_landmarks_py.csv"
    csv_filepath = Path(output_csv_dir_str) / csv_filename_base

    pipeline = None
    frames_processed_for_csv = 0

    landmarker_opts = create_landmarker_options_for_landmarks_functional(
        VisionRunningMode.VIDEO,
        landmarker_model_file_path_str
    )

    try:
        pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, bag_file_path_str, repeat_playback=False)

        profile = pipeline.start(config)
        playback_device = profile.get_device()
        if not playback_device.is_playback():
            print(f"    FUNC_LM: Error - Device from BAG {bag_filename} is not playback.")
            return None
        playback = playback_device.as_playback()
        playback.set_real_time(False)

        total_duration_ns = -1
        try:
            total_duration_timedelta = playback.get_duration()
            total_duration_ns = int(total_duration_timedelta.total_seconds() * 1_000_000_000)
        except Exception:
            pass

        if trim_duration_sec > 0.001:
            playback.seek(datetime.timedelta(seconds=trim_duration_sec))
            time.sleep(0.1)

        align = rs.align(rs.stream.color)

        depth_sensor = profile.get_device().first_depth_sensor()
        if not depth_sensor: print(
            f"    FUNC_LM: Warning - No depth sensor found in BAG {bag_filename}. 3D landmarks might be inaccurate."); return None  # Sau continuă cu 2D
        depth_scale = depth_sensor.get_depth_scale()

        color_stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_intrinsics_rs = color_stream_profile.get_intrinsics()

        with FaceLandmarker.create_from_options(landmarker_opts) as landmarker, \
                open(csv_filepath, 'w') as csv_file:
            csv_file.write("sentence_id,frame_id,landmark_id,x_cam,y_cam,z_cam\n")

            last_mp_timestamp_ms = 0
            consecutive_no_frames = 0
            MAX_CONSECUTIVE_TIMEOUTS = 50

            while True:
                try:
                    frameset = pipeline.wait_for_frames(1000)
                    if not frameset:
                        consecutive_no_frames += 1
                        if consecutive_no_frames > MAX_CONSECUTIVE_TIMEOUTS:
                            print(
                                f"    FUNC_LM: Exceeded max timeouts waiting for frames from {bag_filename}. Assuming end.")
                            break
                        continue
                except RuntimeError as e_wait:
                    print(
                        f"    FUNC_LM: Timeout waiting for frames from {bag_filename} (RuntimeError: {e_wait}). Checking position.")
                    consecutive_no_frames += 1
                    try:
                        current_pos_ns = playback.get_position()
                        if total_duration_ns > 0 and current_pos_ns >= total_duration_ns - 10_000_000:
                            print(f"    FUNC_LM: Playback position indicates end of BAG file for {bag_filename}.")
                            break
                        if consecutive_no_frames > MAX_CONSECUTIVE_TIMEOUTS:
                            print(f"    FUNC_LM: Exceeded max timeouts, assuming end of BAG for {bag_filename}.")
                            break
                    except Exception:
                        break
                    continue

                consecutive_no_frames = 0
                aligned_frames = align.process(frameset)
                color_frame_rs = aligned_frames.get_color_frame()
                depth_frame_rs = aligned_frames.get_depth_frame()

                if not color_frame_rs or not depth_frame_rs: continue

                color_image_np = np.asanyarray(color_frame_rs.get_data())
                if color_image_np.shape[2] == 4:
                    color_image_np = cv2.cvtColor(color_image_np, cv2.COLOR_RGBA2RGB)
                elif color_image_np.shape[2] == 3 and color_frame_rs.get_profile().format() == rs.format.bgr8:
                    color_image_np = cv2.cvtColor(color_image_np, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_image_np)

                current_rs_timestamp_ms = int(color_frame_rs.get_timestamp())
                if current_rs_timestamp_ms <= last_mp_timestamp_ms and last_mp_timestamp_ms > 0:
                    current_mp_timestamp_ms = last_mp_timestamp_ms + 1
                else:
                    current_mp_timestamp_ms = current_rs_timestamp_ms
                last_mp_timestamp_ms = current_mp_timestamp_ms

                face_landmarker_result = landmarker.detect_for_video(mp_image, current_mp_timestamp_ms)

                if face_landmarker_result and face_landmarker_result.face_landmarks:
                    landmarks_list_normalized = face_landmarker_result.face_landmarks[0]
                    depth_image_np = np.asanyarray(depth_frame_rs.get_data())
                    color_w, color_h = color_frame_rs.get_width(), color_frame_rs.get_height()
                    depth_w, depth_h = depth_frame_rs.get_width(), depth_frame_rs.get_height()

                    num_lm_written_this_frame = 0
                    for landmark_idx, landmark_norm in enumerate(landmarks_list_normalized):
                        px = min(max(int(landmark_norm.x * color_w), 0), color_w - 1)
                        py = min(max(int(landmark_norm.y * color_h), 0), color_h - 1)

                        depth_units = 0
                        if 0 <= px < depth_w and 0 <= py < depth_h:
                            depth_units = depth_image_np[py, px]

                        depth_m = depth_units * depth_scale
                        if depth_m > 0.001:
                            point3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics_rs, [float(px), float(py)],
                                                                      depth_m)
                            csv_file.write(
                                f"{sentence_num_int},{frames_processed_for_csv},{landmark_idx},{point3d[0]:.6f},{point3d[1]:.6f},{point3d[2]:.6f}\n")
                            num_lm_written_this_frame += 1
                        else:
                            csv_file.write(
                                f"{sentence_num_int},{frames_processed_for_csv},{landmark_idx},0.0,0.0,0.0\n")

                    if num_lm_written_this_frame > 0:
                        frames_processed_for_csv += 1


    except RuntimeError as e_rt:
        print(f"    FUNC_LM: RealSense Runtime Error for {bag_filename}: {e_rt}")
        # traceback.print_exc()
        return None
    except Exception as e_gen:
        print(f"    FUNC_LM: General Error for {bag_filename}: {e_gen}")
        traceback.print_exc()
        return None
    finally:
        if pipeline:
            try:
                pipeline.stop()
            except Exception:
                pass
        print(f"  FUNC_LM: Saved landmark data to {csv_filepath} ({frames_processed_for_csv} frames written).")

    return csv_filepath if frames_processed_for_csv > 0 else None


if __name__ == "__main__":
    print("--- Running MediaPipe Face Landmark Extraction (Standalone Mode) ---")

    DFLT_LANDMARK_MODEL_PATH = Path('model/face_landmarker_v2_with_blendshapes.task')

    participant_to_process = "p1"

    if participant_to_process == "p1":
        standalone_bag_dir = Path("D:/SegmentationThesis/output_realsense60fps+tesla p1")
        standalone_output_csv_dir = Path("./output_landmarks_csv_p1_FUNC")
        standalone_trim_sec = 1.0
    elif participant_to_process == "p2":
        standalone_bag_dir = Path("D:/SegmentationThesis/output_realsense60fps+tesla p2")
        standalone_output_csv_dir = Path("./output_landmarks_csv_p2_FUNC")
        standalone_trim_sec = 0.3
    else:
        print(f"Participant {participant_to_process} not configured for standalone run.")
        exit()

    standalone_output_csv_dir.mkdir(parents=True, exist_ok=True)

    start_sentence_id_standalone = 1
    max_sentences_standalone = 100

    if not DFLT_LANDMARK_MODEL_PATH.exists():
        print(f"ERROR: MediaPipe model file not found at {DFLT_LANDMARK_MODEL_PATH.resolve()}")
        exit()
    if not standalone_bag_dir.exists():
        print(f"Error: BAG files directory not found: {standalone_bag_dir.resolve()}")
        exit()

    bag_files_found_standalone = []
    for i in range(max_sentences_standalone):
        s_id = start_sentence_id_standalone + i
        bag_f_name = f"sentence_{s_id:03d}_realsense.bag"
        full_path = standalone_bag_dir / bag_f_name
        if full_path.exists():
            bag_files_found_standalone.append((str(full_path), s_id))

    print(f"Found {len(bag_files_found_standalone)} BAG files to process in {standalone_bag_dir}")

    for bag_p, s_num in bag_files_found_standalone:
        output_csv = run_landmark_extraction_for_bag(
            bag_p,
            str(standalone_output_csv_dir),
            s_num,
            str(DFLT_LANDMARK_MODEL_PATH),
            trim_duration_sec=standalone_trim_sec
        )
        if output_csv:
            print(f"  Standalone: Successfully processed {Path(bag_p).name}, output: {output_csv}")
        else:
            print(f"  Standalone: Failed to process {Path(bag_p).name}")

    print("\n--- Standalone MediaPipe Face Landmark Extraction Finished ---")