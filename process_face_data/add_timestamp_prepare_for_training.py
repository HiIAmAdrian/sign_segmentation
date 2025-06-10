import pandas as pd
import numpy as np
from pathlib import Path
import glob
import traceback
import pyrealsense2 as rs
import datetime
import time
import pickle

# --- Configuration ---
LANDMARKS_CSV_DIR = Path("./output_landmarks_csv_catalin_occlusion_normalised")  # Modifică la directorul tău
BLENDSHAPES_CSV_DIR = Path("./output_blendshapes_csv_catalin")  # Modifică la directorul tău
ORIGINAL_BAG_FILES_DIR = Path("D:/SegmentationThesis/output_realsense60fps+tesla Catalin")  # Modifică la directorul tău

OUTPUT_FACIAL_PKL_BASE_DIR = Path("./final_facial_data_processed_trimmed")  # Directorul părinte pentru output-uri
OUTPUT_FACIAL_PKL_BASE_DIR.mkdir(parents=True, exist_ok=True)

TRIM_DURATIONS_PER_PARTICIPANT_KEY = {
    "catalin": 1.0,
    "marinela": 0.3,
}


def get_sentence_id_from_filename(filename_path):
    name_parts = filename_path.stem.split('_')
    for part in name_parts:
        if part.isdigit():
            return int(part)
    return None


def get_bag_filename_from_sentence_id_and_participant(sentence_id, participant_key_name):
    # Ajustează această funcție pentru a se potrivi cu modul în care sunt numite fișierele .bag
    # dacă numele participantului este în numele fișierului .bag sau dacă ai directoare separate
    # Exemplu simplu, presupunând că numele participantului NU e în numele fișierului .bag:
    return f"sentence_{int(sentence_id):03d}_realsense.bag"


def pivot_landmarks(df_long):
    if df_long.empty: return pd.DataFrame()
    try:  # Adaugă try-except pentru a prinde erori de pivotare dacă frame_id lipsește
        df_wide = df_long.pivot_table(index='frame_id', columns='landmark_id', values=['x_cam', 'y_cam', 'z_cam'])
        df_wide.columns = [f'landmark_{int(col[1])}_{col[0]}' for col in df_wide.columns]
        return df_wide.sort_index()
    except KeyError as e:
        print(f"    Error pivoting landmarks (missing expected columns like 'frame_id' or 'landmark_id'): {e}")
        return pd.DataFrame()


def pivot_blendshapes(df_long):
    if df_long.empty: return pd.DataFrame()
    try:  # Adaugă try-except
        df_wide = df_long.pivot_table(index='frame_id', columns='blendshape_name', values='score')
        return df_wide.sort_index()
    except KeyError as e:
        print(f"    Error pivoting blendshapes (missing expected columns like 'frame_id' or 'blendshape_name'): {e}")
        return pd.DataFrame()


def extract_timestamps_from_bag(bag_filepath, trim_sec=0.0):
    print(f"    Extracting timestamps from: {bag_filepath.name} (trim: {trim_sec}s)")
    timestamps_ms_list = []
    pipeline = None
    frames_extracted = 0
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, str(bag_filepath), repeat_playback=False)
        config.enable_stream(rs.stream.color)
        profile = pipeline.start(config)
        playback_device = profile.get_device().as_playback()
        playback_device.set_real_time(False)
        if trim_sec > 0.001:
            seek_delta = datetime.timedelta(seconds=trim_sec)
            playback_device.seek(seek_delta)
            time.sleep(0.2)
        last_ts = -1
        consecutive_no_frame = 0
        MAX_CONSECUTIVE_NO_FRAMES = 500  # Mărit
        while True:
            success, frameset = pipeline.try_wait_for_frames(200)
            if not success or not frameset:
                consecutive_no_frame += 1
                if consecutive_no_frame > MAX_CONSECUTIVE_NO_FRAMES: break
                try:
                    current_pos_ns = playback_device.get_position()
                    total_dur_ns = int(playback_device.get_duration().total_seconds() * 1e9)
                    if total_dur_ns > 0 and current_pos_ns >= total_dur_ns - 20_000_000: break  # 20ms de la final
                except Exception:
                    pass
                time.sleep(0.005);
                continue
            consecutive_no_frame = 0
            color_frame = frameset.get_color_frame()
            if color_frame:
                ts_ms = int(color_frame.get_timestamp())
                if ts_ms <= last_ts and last_ts > 0: ts_ms = last_ts + 1
                last_ts = ts_ms
                timestamps_ms_list.append(ts_ms)
                frames_extracted += 1
                # if frames_extracted % 500 == 0: print(f"      Extracted {frames_extracted} timestamps...") # Comentat pentru a reduce output-ul
    except RuntimeError as e_rt:
        print(f"    Runtime error: {e_rt}")
    except Exception as e_gen:
        print(f"    General Error: {e_gen}"); traceback.print_exc()
    finally:
        if pipeline:
            try:
                pipeline.stop()
            except Exception:
                pass
    print(f"    Finished extracting. Total timestamps: {len(timestamps_ms_list)}")
    return np.array(timestamps_ms_list, dtype=np.float64)


print("--- Starting Facial Data Preparation (Pivot and Timestamp Extraction from BAG with Trim) ---")

# Construiește maparea pentru blendshapes o singură dată
blendshape_files_map_all_participants = {
    (participant_key, get_sentence_id_from_filename(Path(f))): Path(f)
    for participant_key, participant_blendshape_dir_str_part in TRIM_DURATIONS_PER_PARTICIPANT_KEY.items()
    # Folosește cheile definite
    for f in glob.glob(str(BLENDSHAPES_CSV_DIR / f"*{participant_key}*csv"))  # Filtrează fișierele blendshape
    if get_sentence_id_from_filename(Path(f)) is not None
}

overall_processed_count = 0
overall_error_count = 0

for participant_key_name, trim_duration_for_this_participant in TRIM_DURATIONS_PER_PARTICIPANT_KEY.items():
    print(
        f"\n===== Processing Facial Data for Participant: {participant_key_name} (Trim: {trim_duration_for_this_participant}s) =====")

    current_output_facial_dir = OUTPUT_FACIAL_PKL_BASE_DIR / f"final_facial_data_processed_{participant_key_name}_trimmed"
    current_output_facial_dir.mkdir(parents=True, exist_ok=True)

    # Caută fișierele de landmark-uri care conțin numele participantului
    participant_landmark_files = sorted([
        p for p in LANDMARKS_CSV_DIR.glob("*.csv") if participant_key_name.lower() in p.name.lower()
    ])

    if not participant_landmark_files:
        print(f"  No landmark CSV files found for participant {participant_key_name}. Skipping.")
        continue

    processed_count_participant = 0
    error_count_participant = 0

    for lm_csv_path in participant_landmark_files:
        sentence_id = get_sentence_id_from_filename(lm_csv_path)
        if sentence_id is None:
            print(f"  Warning: Could not get sentence_id from {lm_csv_path.name}. Skipping.")
            error_count_participant += 1
            continue

        print(f"\n  Processing Facial Sentence ID: {sentence_id:03d} (from {lm_csv_path.name})")

        bag_filename = get_bag_filename_from_sentence_id_and_participant(sentence_id, participant_key_name)
        # Construiește calea către fișierul .bag pe baza numelui participantului
        # Presupunem că ORIGINAL_BAG_FILES_DIR este un director părinte, iar sub el sunt directoare per participant
        # sau că numele participantului este în numele directorului ORIGINAL_BAG_FILES_DIR
        # Pentru acest exemplu, vom presupune că ORIGINAL_BAG_FILES_DIR este specific participantului
        # dacă cheile din TRIM_DURATIONS_PER_PARTICIPANT_KEY sunt folosite pentru a construi căi.

        # Găsește directorul BAG specific participantului
        participant_bag_dir = None
        for p_key, p_dir_str_part_for_bag in TRIM_DURATIONS_PER_PARTICIPANT_KEY.items():
            # Acest if este un exemplu, trebuie să-l adaptezi la structura ta
            if p_key == participant_key_name:  # Găsește calea BAG pe baza cheii participantului
                # Presupunem că ORIGINAL_BAG_FILES_DIR este o cale generală și adăugăm numele participantului
                # sau că ai o mapare directă.
                # Pentru exemplul din config:
                participant_bag_dir_candidate = Path(
                    str(ORIGINAL_BAG_FILES_DIR).replace("Marinela", participant_key_name).replace("Catalin",
                                                                                                  participant_key_name))
                if participant_bag_dir_candidate.exists():  # Verifică dacă directorul există
                    participant_bag_dir = participant_bag_dir_candidate
                else:  # Fallback la ORIGINAL_BAG_FILES_DIR dacă nu găsește un director specific
                    participant_bag_dir = ORIGINAL_BAG_FILES_DIR

                break  # Ieși din buclă după ce ai găsit directorul

        if participant_bag_dir is None:
            print(
                f"    Error: Could not determine BAG directory for participant {participant_key_name}. Skipping sentence.")
            error_count_participant += 1
            continue

        bag_filepath = participant_bag_dir / bag_filename
        if not bag_filepath.exists():
            print(f"    Error: Original .bag file not found: {bag_filepath}. Skipping sentence.")
            error_count_participant += 1;
            continue

        realsense_timestamps_ms = extract_timestamps_from_bag(bag_filepath, trim_sec=trim_duration_for_this_participant)
        if realsense_timestamps_ms is None or len(realsense_timestamps_ms) == 0:
            error_count_participant += 1;
            continue
        try:
            df_landmarks_long = pd.read_csv(lm_csv_path)
            if df_landmarks_long.empty or 'frame_id' not in df_landmarks_long.columns: error_count_participant += 1; continue
            df_landmarks_long['frame_id'] = df_landmarks_long['frame_id'].astype(int)
            max_lm_frame_id = df_landmarks_long['frame_id'].max()
            if len(realsense_timestamps_ms) <= max_lm_frame_id:
                df_landmarks_long = df_landmarks_long[df_landmarks_long['frame_id'] < len(realsense_timestamps_ms)]
            if df_landmarks_long.empty: error_count_participant += 1; continue
        except Exception as e:
            error_count_participant += 1; continue
        df_landmarks_wide = pivot_landmarks(df_landmarks_long)
        if df_landmarks_wide.empty: error_count_participant += 1; continue

        # Verifică dacă indexul (frame_id) există în realsense_timestamps_ms
        valid_indices_lm = df_landmarks_wide.index[df_landmarks_wide.index < len(realsense_timestamps_ms)]
        df_landmarks_wide = df_landmarks_wide.loc[valid_indices_lm]
        if df_landmarks_wide.empty: print(
            f"    Landmarks empty after index validation for {lm_csv_path.name}"); error_count_participant += 1; continue

        timestamps_for_lm_frames = realsense_timestamps_ms[df_landmarks_wide.index.astype(int)]
        df_landmarks_wide['realsense_timestamp_ms'] = timestamps_for_lm_frames
        df_landmarks_wide.set_index('realsense_timestamp_ms', inplace=True)

        df_blendshapes_wide = pd.DataFrame()
        bs_file_key = (participant_key_name, sentence_id)  # Cheie pentru map-ul de blendshapes
        if bs_file_key in blendshape_files_map_all_participants:
            bs_csv_path = blendshape_files_map_all_participants[bs_file_key]
            try:
                df_blendshapes_long = pd.read_csv(bs_csv_path)
                if not df_blendshapes_long.empty and 'frame_id' in df_blendshapes_long.columns:
                    df_blendshapes_long['frame_id'] = df_blendshapes_long['frame_id'].astype(int)
                    max_bs_frame_id = df_blendshapes_long['frame_id'].max()
                    if len(realsense_timestamps_ms) <= max_bs_frame_id:
                        df_blendshapes_long = df_blendshapes_long[
                            df_blendshapes_long['frame_id'] < len(realsense_timestamps_ms)]
                    if not df_blendshapes_long.empty:
                        df_bs_wide_tmp = pivot_blendshapes(df_blendshapes_long)
                        if not df_bs_wide_tmp.empty:
                            valid_indices_bs = df_bs_wide_tmp.index[df_bs_wide_tmp.index < len(realsense_timestamps_ms)]
                            df_bs_wide_tmp = df_bs_wide_tmp.loc[valid_indices_bs]
                            if not df_bs_wide_tmp.empty:
                                ts_bs_frames = realsense_timestamps_ms[df_bs_wide_tmp.index.astype(int)]
                                df_bs_wide_tmp['realsense_timestamp_ms'] = ts_bs_frames
                                df_blendshapes_wide = df_bs_wide_tmp.set_index('realsense_timestamp_ms')
            except Exception:
                pass

        if not df_blendshapes_wide.empty:
            df_facial_combined = pd.merge(df_landmarks_wide, df_blendshapes_wide, left_index=True, right_index=True,
                                          how='outer')
            df_facial_combined.ffill(inplace=True);
            df_facial_combined.bfill(inplace=True);
            df_facial_combined.fillna(0, inplace=True)
        else:
            df_facial_combined = df_landmarks_wide.copy()
        if df_facial_combined.empty: error_count_participant += 1; continue

        df_facial_combined.index = pd.to_timedelta(df_facial_combined.index, unit='ms', errors='coerce')
        df_facial_combined.dropna(subset=[df_facial_combined.index.name], inplace=True)
        if df_facial_combined.empty: error_count_participant += 1; continue

        first_ts = df_facial_combined.index.min()
        df_facial_combined.index = df_facial_combined.index - first_ts
        df_facial_combined = df_facial_combined.sort_index()

        output_filename = f"sentence_{sentence_id:03d}_facial_processed_bagts.pkl"
        output_path = current_output_facial_dir / output_filename
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(df_facial_combined, f)
            processed_count_participant += 1
        except Exception as e:
            error_count_participant += 1; traceback.print_exc()
    print(
        f"  Participant {participant_key_name}: Processed {processed_count_participant} facial files, Errors/Skipped: {error_count_participant}.")
    overall_processed_count += processed_count_participant
    overall_error_count += error_count_participant

print(f"\n--- Facial Data Preparation Summary ---")
print(f"Overall successfully processed and saved {overall_processed_count} facial data files.")
if overall_error_count > 0: print(f"Overall encountered errors for {overall_error_count} files.")
print(f"Processed facial data saved into subdirectories under: {OUTPUT_FACIAL_PKL_BASE_DIR}")