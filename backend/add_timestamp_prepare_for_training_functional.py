import re
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import pyrealsense2 as rs
import datetime
import time
import pickle

def pivot_landmarks(df_long):
    if df_long.empty: return pd.DataFrame()
    if not all(c in df_long.columns for c in ['frame_id', 'landmark_id', 'x_cam', 'y_cam', 'z_cam']):
        print("    FUNC_ATS: Pivot Landmarks - Missing required columns.")
        return pd.DataFrame()
    try:
        df_wide = df_long.pivot_table(index='frame_id', columns='landmark_id', values=['x_cam', 'y_cam', 'z_cam'])
        df_wide.columns = [f'landmark_{int(col[1])}_{col[0]}' for col in df_wide.columns]
        return df_wide.sort_index()
    except Exception as e:
        print(f"    FUNC_ATS: Error during landmark pivot: {e}")
        return pd.DataFrame()


def pivot_blendshapes(df_long):
    if df_long.empty: return pd.DataFrame()
    if not all(c in df_long.columns for c in ['frame_id', 'blendshape_name', 'score']):
        print("    FUNC_ATS: Pivot Blendshapes - Missing required columns.")
        return pd.DataFrame()
    try:
        df_wide = df_long.pivot_table(index='frame_id', columns='blendshape_name', values='score')
        return df_wide.sort_index()
    except Exception as e:
        print(f"    FUNC_ATS: Error during blendshape pivot: {e}")
        return pd.DataFrame()


def extract_timestamps_from_bag(bag_filepath_str, trim_sec=0.0):
    print(f"    FUNC_ATS: Extracting timestamps from: {Path(bag_filepath_str).name} (trim: {trim_sec}s)")
    timestamps_ms_list = []
    pipeline = None
    frames_extracted = 0
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, str(bag_filepath_str), repeat_playback=False)

        profile = pipeline.start(config)
        playback_device = profile.get_device().as_playback()
        playback_device.set_real_time(False)

        if trim_sec > 0.001:
            playback_device.seek(datetime.timedelta(seconds=trim_sec))
            time.sleep(0.1)

        last_ts = -1
        consecutive_no_frame = 0
        MAX_CONSECUTIVE_NO_FRAMES_ATS = 150

        while True:
            success, frameset = pipeline.try_wait_for_frames(200)
            if not success or not frameset:
                consecutive_no_frame += 1
                if consecutive_no_frame > MAX_CONSECUTIVE_NO_FRAMES_ATS:
                    break
                try:
                    current_pos_ns = playback_device.get_position()
                    total_dur_ns = int(playback_device.get_duration().total_seconds() * 1e9)
                    if total_dur_ns > 0 and current_pos_ns >= total_dur_ns - 10_000_000: break
                except Exception:
                    pass
                time.sleep(0.005)
                continue
            consecutive_no_frame = 0
            color_frame = frameset.get_color_frame()
            if color_frame:
                ts_ms = int(color_frame.get_timestamp())
                if ts_ms <= last_ts and last_ts > 0: ts_ms = last_ts + 1
                last_ts = ts_ms
                timestamps_ms_list.append(ts_ms)
                frames_extracted += 1
    except RuntimeError as e_rt:
        print(
            f"    FUNC_ATS: Runtime error (e.g. end of file) during BAG processing for {Path(bag_filepath_str).name}: {e_rt}")
    except Exception as e_gen:
        print(f"    FUNC_ATS: General Error processing BAG {Path(bag_filepath_str).name}: {e_gen}")
    finally:
        if pipeline:
            try:
                pipeline.stop()
            except Exception:
                pass
    return np.array(timestamps_ms_list, dtype=np.float64) if timestamps_ms_list else np.array([], dtype=np.float64)


def run_add_timestamps_and_combine_facial(
        landmarks_normalized_csv_path_str: str,
        blendshapes_csv_path_str: str,
        original_bag_file_path_str: str,
        output_facial_pkl_path_str: str,
        trim_start_seconds_from_bag: float = 0.0
):
    """
    Combină datele de landmark-uri (normalizate) și blendshape-uri, le asociază timestamp-uri
    precise din fișierul .bag original, normalizează indexul temporal și salvează
    DataFrame-ul rezultat într-un fișier PKL.
    Returnează True dacă operațiunea a reușit, False altfel.
    """
    lm_csv_path = Path(landmarks_normalized_csv_path_str)
    bs_csv_path = Path(blendshapes_csv_path_str)
    bag_filepath = Path(original_bag_file_path_str)
    output_path = Path(output_facial_pkl_path_str)

    print(
        f"  FUNC_ATS: Preparing final facial data for {lm_csv_path.name.split('_')[1] if '_' in lm_csv_path.name else lm_csv_path.name}")

    if not lm_csv_path.exists():
        print(f"    FUNC_ATS: Error - Normalized landmark file not found: {lm_csv_path}")
        return False
    if not bag_filepath.exists():
        print(f"    FUNC_ATS: Error - Original BAG file not found: {bag_filepath}")
        return False
    if bs_csv_path and not bs_csv_path.exists():
        print(
            f"    FUNC_ATS: Warning - Blendshape file specified but not found: {bs_csv_path}. Proceeding without blendshapes.")

    realsense_timestamps_ms = _extract_timestamps_from_bag_internal(str(bag_filepath),
                                                                    trim_sec=trim_start_seconds_from_bag)
    if realsense_timestamps_ms is None or len(realsense_timestamps_ms) == 0:
        print(f"    FUNC_ATS: Error - No timestamps extracted from {bag_filepath.name}. Cannot proceed.")
        return False

    try:
        df_landmarks_long = pd.read_csv(lm_csv_path)
        if df_landmarks_long.empty or 'frame_id' not in df_landmarks_long.columns:
            print(f"    FUNC_ATS: Landmark file {lm_csv_path.name} is empty or missing 'frame_id'.")
            return False
        df_landmarks_long['frame_id'] = df_landmarks_long['frame_id'].astype(int)
        max_lm_frame_id = df_landmarks_long['frame_id'].max()

        if max_lm_frame_id >= len(realsense_timestamps_ms):
            print(
                f"    FUNC_ATS: Warning - Max landmark frame_id {max_lm_frame_id} >= num timestamps {len(realsense_timestamps_ms)}. Truncating landmarks.")
            df_landmarks_long = df_landmarks_long[df_landmarks_long['frame_id'] < len(realsense_timestamps_ms)]
        if df_landmarks_long.empty: print(f"    FUNC_ATS: Landmarks empty after timestamp check."); return False

        df_landmarks_wide = _pivot_landmarks_internal(df_landmarks_long)
        if df_landmarks_wide.empty: print(f"    FUNC_ATS: Pivoted landmarks empty."); return False

        valid_lm_indices = df_landmarks_wide.index[df_landmarks_wide.index < len(realsense_timestamps_ms)]
        df_landmarks_wide = df_landmarks_wide.loc[valid_lm_indices]
        if df_landmarks_wide.empty: print(
            f"    FUNC_ATS: Landmarks empty after timestamp index validation."); return False

        timestamps_for_lm_frames = realsense_timestamps_ms[df_landmarks_wide.index.astype(int)]
        df_landmarks_wide['realsense_timestamp_ms'] = timestamps_for_lm_frames
        df_landmarks_wide.set_index('realsense_timestamp_ms', inplace=True)
    except Exception as e:
        print(f"    FUNC_ATS: Error processing landmark data from {lm_csv_path.name}: {e}");
        return False

    df_blendshapes_wide = pd.DataFrame()
    if bs_csv_path and bs_csv_path.exists():
        try:
            df_blendshapes_long = pd.read_csv(bs_csv_path)
            if not df_blendshapes_long.empty and 'frame_id' in df_blendshapes_long.columns:
                df_blendshapes_long['frame_id'] = df_blendshapes_long['frame_id'].astype(int)
                max_bs_frame_id = df_blendshapes_long['frame_id'].max()
                if max_bs_frame_id >= len(realsense_timestamps_ms):
                    df_blendshapes_long = df_blendshapes_long[
                        df_blendshapes_long['frame_id'] < len(realsense_timestamps_ms)]

                if not df_blendshapes_long.empty:
                    df_blendshapes_wide_temp = _pivot_blendshapes_internal(df_blendshapes_long)
                    if not df_blendshapes_wide_temp.empty:
                        valid_bs_indices = df_blendshapes_wide_temp.index[
                            df_blendshapes_wide_temp.index < len(realsense_timestamps_ms)]
                        df_blendshapes_wide_temp = df_blendshapes_wide_temp.loc[valid_bs_indices]
                        if not df_blendshapes_wide_temp.empty:
                            timestamps_for_bs_frames = realsense_timestamps_ms[
                                df_blendshapes_wide_temp.index.astype(int)]
                            df_blendshapes_wide_temp['realsense_timestamp_ms'] = timestamps_for_bs_frames
                            df_blendshapes_wide = df_blendshapes_wide_temp.set_index('realsense_timestamp_ms')
        except Exception as e:
            print(f"    FUNC_ATS: Error processing blendshape file {bs_csv_path.name}: {e}")

    if not df_blendshapes_wide.empty:
        df_facial_combined = pd.merge(df_landmarks_wide, df_blendshapes_wide, left_index=True, right_index=True,
                                      how='outer', suffixes=('_lm', '_bs'))
        df_facial_combined.index.name = 'realsense_timestamp_ms'
        df_facial_combined = df_facial_combined.ffill().bfill().fillna(0)
    else:
        print(f"    FUNC_ATS: Proceeding with landmarks data only for {lm_csv_path.name} (no valid blendshapes).")
        df_facial_combined = df_landmarks_wide.copy()
        if df_facial_combined.index.name != 'realsense_timestamp_ms':
            df_facial_combined.index.name = 'realsense_timestamp_ms'

    if df_facial_combined.empty:
        print(f"    FUNC_ATS: Error - Combined facial data is empty. Skipping save for {lm_csv_path.name}.")
        return False

    df_facial_combined = df_facial_combined.reset_index()
    df_facial_combined['realsense_timestamp_ms'] = pd.to_timedelta(df_facial_combined['realsense_timestamp_ms'],
                                                                   unit='ms', errors='coerce')
    df_facial_combined.dropna(subset=['realsense_timestamp_ms'], inplace=True)
    if df_facial_combined.empty: print(
        f"    FUNC_ATS: Facial data empty after final timestamp conversion."); return False

    df_facial_combined = df_facial_combined.set_index('realsense_timestamp_ms').sort_index()
    if not df_facial_combined.empty:
        first_ts = df_facial_combined.index.min()
        df_facial_combined.index = df_facial_combined.index - first_ts
    df_facial_combined = df_facial_combined.sort_index()
    df_facial_combined.index.name = 'normalized_timestamp_us'

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(df_facial_combined, f)
        print(f"    FUNC_ATS: Successfully saved processed facial data to: {output_path}")
        return True
    except Exception as e:
        print(f"    FUNC_ATS: Error saving PKL {output_path}: {e}");
        return False


if __name__ == "__main__":
    print("--- Running Add Timestamps & Combine Facial (Standalone Mode) ---")
    participant_to_process_ats = "p2"

    if participant_to_process_ats == "p1":
        standalone_landmarks_norm_dir = Path("./output_landmarks_csv_p1_FUNC_normalized")
        standalone_blendshapes_dir = Path("./output_blendshapes_csv_p1_FUNC")
        standalone_bag_dir_ats = Path("D:/SegmentationThesis/output_realsense60fps+tesla p1")
        standalone_output_pkl_dir_ats = Path("./final_facial_data_processed_p1_FUNC")
        standalone_trim_bag = 1.0
    elif participant_to_process_ats == "p2":
        standalone_landmarks_norm_dir = Path("./output_landmarks_csv_p2_FUNC_normalized")
        standalone_blendshapes_dir = Path("./output_blendshapes_csv_p2_FUNC")
        standalone_bag_dir_ats = Path("D:/SegmentationThesis/output_realsense60fps+tesla p2")
        standalone_output_pkl_dir_ats = Path("./final_facial_data_processed_p2_FUNC")
        standalone_trim_bag = 0.3
    else:
        print(f"Participant {participant_to_process_ats} not configured.")
        exit()

    standalone_output_pkl_dir_ats.mkdir(parents=True, exist_ok=True)

    landmark_files_standalone = sorted(glob.glob(str(standalone_landmarks_norm_dir / "sentence_*_filled_ransac.csv")))
    if not landmark_files_standalone:
        print(f"No normalized landmark files found in {standalone_landmarks_norm_dir}")
        exit()

    for lm_norm_csv_path_str in landmark_files_standalone:
        lm_path = Path(lm_norm_csv_path_str)
        s_id_match = re.search(r"sentence_(\d+)", lm_path.name)
        if not s_id_match:
            print(f"Could not get sentence_id from {lm_path.name}. Skipping.");
            continue
        sentence_id_val = int(s_id_match.group(1))

        bs_csv_name = f"sentence_{sentence_id_val:03d}_mediapipe_blendshapes.csv"
        bs_path_str = str(standalone_blendshapes_dir / bs_csv_name)

        bag_name = f"sentence_{sentence_id_val:03d}_realsense.bag"
        bag_path_str = str(standalone_bag_dir_ats / bag_name)

        output_pkl_name = f"sentence_{sentence_id_val:03d}_facial_processed_bagts.pkl"
        output_pkl_path_str = str(standalone_output_pkl_dir_ats / output_pkl_name)

        print(f"\nStandalone ATS: Processing sentence {sentence_id_val}")
        print(f"  LM_NORM: {lm_path.name}")
        print(f"  BS_CSV (optional): {Path(bs_path_str).name if Path(bs_path_str).exists() else 'Not found/used'}")
        print(f"  BAG: {Path(bag_path_str).name}")

        run_add_timestamps_and_combine_facial(
            lm_norm_csv_path_str,
            bs_path_str,
            bag_path_str,
            output_pkl_path_str,
            trim_start_seconds_from_bag=standalone_trim_bag
        )
    print("\n--- Standalone Add Timestamps & Combine Facial Finished ---")