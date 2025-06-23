import traceback

import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import time
import datetime
from pathlib import Path

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def create_landmarker_options_for_blendshapes_functional(running_mode_enum_val, model_asset_path_str):
    """Creează FaceLandmarkerOptions specific pentru extracția de blendshape-uri."""
    print(
        f"    CREATE_OPTS_BS: Received model_asset_path_str: '{model_asset_path_str}', type: {type(model_asset_path_str)}")
    try:
        resolved_model_path = str(
            Path(model_asset_path_str).resolve(strict=True))
        print(f"    CREATE_OPTS_BS: Resolved model_asset_path: '{resolved_model_path}'")
    except FileNotFoundError:
        print(f"    CREATE_OPTS_BS: ERROR - Model file NOT FOUND at path: {model_asset_path_str}")
        raise
    hardcoded_model_path = "C:/Users/adrian.stan/Desktop/School/segmentation/backend/model/face_landmarker_v2_with_blendshapes.task"
    print(f"    CREATE_OPTS_BS: Using HARDCODED model path: {hardcoded_model_path}")
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(hardcoded_model_path)),
        running_mode=running_mode_enum_val,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    return options


def run_blendshape_extraction_for_bag(
        bag_file_path_str: str,
        output_csv_dir_str: str,
        sentence_num_int: int,
        landmarker_model_file_path_str: str,
        trim_duration_sec: float = 0.0
):
    """
    Procesează un singur fișier .bag pentru a extrage scorurile blendshape-urilor faciale.
    Scrie rezultatele într-un fișier CSV în directorul specificat.
    Returnează calea către fișierul CSV creat sau None dacă eșuează.
    """
    bag_filename = Path(bag_file_path_str).name
    print(
        f"  FUNC_BS: Processing blendshapes for sentence {sentence_num_int} from {bag_filename} (trim: {trim_duration_sec:.2f}s)...")

    if not Path(landmarker_model_file_path_str).exists():
        print(f"    FUNC_BS: ERROR - MediaPipe blendshape model not found at {landmarker_model_file_path_str}")
        return None

    Path(output_csv_dir_str).mkdir(parents=True, exist_ok=True)

    csv_filename_base = f"sentence_{sentence_num_int:03d}_mediapipe_blendshapes.csv"
    csv_filepath = Path(output_csv_dir_str) / csv_filename_base

    pipeline = None
    frames_with_blendshapes_written = 0

    landmarker_opts = create_landmarker_options_for_blendshapes_functional(
        VisionRunningMode.VIDEO,
        landmarker_model_file_path_str
    )

    try:
        pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, bag_file_path_str, repeat_playback=False)
        config.enable_stream(rs.stream.color)
        profile = pipeline.start(config)
        playback_device = profile.get_device()
        if not playback_device.is_playback():
            print(f"    FUNC_BS: Error - Device from BAG {bag_filename} is not playback.")
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

        print(f"    FUNC_BS DEBUG: Type of landmarker_opts: {type(landmarker_opts)}")
        print(f"    FUNC_BS DEBUG: Content of landmarker_opts: {landmarker_opts}")
        with FaceLandmarker.create_from_options(landmarker_opts) as landmarker, \
                open(csv_filepath, 'w') as csv_file:
            csv_file.write("sentence_id,frame_id,blendshape_name,score\n")

            last_mp_timestamp_ms = 0
            consecutive_no_frames = 0
            MAX_CONSECUTIVE_TIMEOUTS = 100
            current_sequential_frame_id = 0

            while True:
                try:
                    frameset = pipeline.wait_for_frames(1000)
                    if not frameset:
                        consecutive_no_frames += 1
                        if consecutive_no_frames > MAX_CONSECUTIVE_TIMEOUTS: break
                        continue
                except RuntimeError:
                    consecutive_no_frames += 1
                    try:
                        current_pos_ns = playback.get_position()
                        if total_duration_ns > 0 and current_pos_ns >= total_duration_ns - 10_000_000: break
                        if consecutive_no_frames > MAX_CONSECUTIVE_TIMEOUTS: break
                    except Exception:
                        break
                    continue

                consecutive_no_frames = 0
                color_frame_rs = frameset.get_color_frame()

                if not color_frame_rs:
                    current_sequential_frame_id += 1
                    continue

                color_image_np = np.asanyarray(color_frame_rs.get_data())
                rgb_image_np = None
                if color_image_np.shape[2] == 4:
                    rgb_image_np = cv2.cvtColor(color_image_np, cv2.COLOR_RGBA2RGB)
                elif color_image_np.shape[2] == 3 and color_frame_rs.get_profile().format() == rs.format.bgr8:
                    rgb_image_np = cv2.cvtColor(color_image_np, cv2.COLOR_BGR2RGB)
                elif color_image_np.shape[2] == 3 and color_frame_rs.get_profile().format() == rs.format.rgb8:
                    rgb_image_np = color_image_np
                else:
                    current_sequential_frame_id += 1
                    continue

                if rgb_image_np is None:
                    current_sequential_frame_id += 1
                    continue

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image_np)

                current_rs_timestamp_ms = int(color_frame_rs.get_timestamp())
                if current_rs_timestamp_ms <= last_mp_timestamp_ms and last_mp_timestamp_ms > 0:
                    current_mp_timestamp_ms = last_mp_timestamp_ms + 1
                else:
                    current_mp_timestamp_ms = current_rs_timestamp_ms
                last_mp_timestamp_ms = current_mp_timestamp_ms

                face_landmarker_result = landmarker.detect_for_video(mp_image, current_mp_timestamp_ms)

                wrote_blendshapes_this_frame = False
                if face_landmarker_result and face_landmarker_result.face_blendshapes:
                    blendshapes_for_face = face_landmarker_result.face_blendshapes[0]
                    for blendshape_category in blendshapes_for_face:
                        blendshape_name = blendshape_category.category_name
                        score = blendshape_category.score
                        csv_file.write(
                            f"{sentence_num_int},{current_sequential_frame_id},{blendshape_name},{score:.6f}\n")
                    wrote_blendshapes_this_frame = True

                if wrote_blendshapes_this_frame:
                    frames_with_blendshapes_written += 1

                current_sequential_frame_id += 1

    except RuntimeError as e_rt:
        print(f"    FUNC_BS: RealSense Runtime Error for {bag_filename}: {e_rt}")
        return None
    except Exception as e_gen:
        print(f"    FUNC_BS: General Error for {bag_filename}: {e_gen}")
        traceback.print_exc()
        return None
    finally:
        if pipeline:
            try:
                pipeline.stop()
            except Exception:
                pass
        print(
            f"  FUNC_BS: Saved blendshape data to {csv_filepath} ({frames_with_blendshapes_written} frames had blendshapes).")

    return csv_filepath if frames_with_blendshapes_written > 0 else None


if __name__ == "__main__":
    print("--- Running MediaPipe Blendshape Extraction (Standalone Mode) ---")

    DFLT_BLENDSHAPE_MODEL_PATH_STANDALONE = Path('model/face_landmarker_v2_with_blendshapes.task')

    participant_to_process_standalone = "p1"

    if participant_to_process_standalone == "p1":
        standalone_bag_dir_bs = Path("D:/SegmentationThesis/output_realsense60fps+tesla p1")
        standalone_output_csv_dir_bs = Path("./output_blendshapes_csv_p1_FUNC")
        standalone_trim_sec_bs = 1.0
    elif participant_to_process_standalone == "p2":
        standalone_bag_dir_bs = Path("D:/SegmentationThesis/output_realsense60fps+tesla p2")
        standalone_output_csv_dir_bs = Path("./output_blendshapes_csv_p2_FUNC")
        standalone_trim_sec_bs = 0.3
    else:
        print(f"Participant {participant_to_process_standalone} not configured for standalone blendshape run.")
        exit()

    standalone_output_csv_dir_bs.mkdir(parents=True, exist_ok=True)

    start_sentence_id_bs = 1
    max_sentences_bs = 100

    if not DFLT_BLENDSHAPE_MODEL_PATH_STANDALONE.exists():
        print(f"ERROR: MediaPipe model file not found at {DFLT_BLENDSHAPE_MODEL_PATH_STANDALONE.resolve()}")
        exit()
    if not standalone_bag_dir_bs.exists():
        print(f"Error: BAG files directory not found: {standalone_bag_dir_bs.resolve()}")
        exit()

    bag_files_found_bs = []
    for i in range(max_sentences_bs):
        s_id = start_sentence_id_bs + i
        bag_f_name = f"sentence_{s_id:03d}_realsense.bag"
        full_path = standalone_bag_dir_bs / bag_f_name
        if full_path.exists():
            bag_files_found_bs.append((str(full_path), s_id))

    print(f"Found {len(bag_files_found_bs)} BAG files to process in {standalone_bag_dir_bs} for blendshapes.")

    for bag_p, s_num in bag_files_found_bs:
        output_csv = run_blendshape_extraction_for_bag(
            bag_p,
            str(standalone_output_csv_dir_bs),
            s_num,
            str(DFLT_BLENDSHAPE_MODEL_PATH_STANDALONE),
            trim_duration_sec=standalone_trim_sec_bs
        )
        if output_csv:
            print(f"  Standalone_BS: Successfully processed {Path(bag_p).name}, output: {output_csv}")
        else:
            print(f"  Standalone_BS: Failed to process {Path(bag_p).name}")

    print("\n--- Standalone MediaPipe Blendshape Extraction Finished ---")