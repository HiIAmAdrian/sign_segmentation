import pandas as pd
import numpy as np
from pathlib import Path
import glob
import traceback
import pyrealsense2 as rs # Pentru a citi timestamp-urile din .bag
import datetime # Pentru a gestiona trim_duration_sec
import time     # Pentru sleep după seek
import pickle  # Pentru a salva datele procesate

# --- Configuration ---
# Input directories for "long" format CSVs
LANDMARKS_CSV_DIR = Path("./output_landmarks_csv_catalin_occlusion_normalised")
BLENDSHAPES_CSV_DIR = Path("./output_blendshapes_csv_catalin")
# !!! Directorul cu fișierele .bag originale !!!
ORIGINAL_BAG_FILES_DIR = Path("D:/SegmentationThesis/output_realsense60fps+tesla Catalin")

# Output directory for "wide" format processed facial data
OUTPUT_FACIAL_PKL_DIR = Path("./final_facial_data_processed_catalin")
OUTPUT_FACIAL_PKL_DIR.mkdir(parents=True, exist_ok=True)

# Parametru de trim, trebuie să fie IDENTIC cu cel folosit la generarea CSV-urilor
TRIM_START_SECONDS_BAG = 1

# Naming convention assumptions (ADJUST IF NECESSARY)
def get_sentence_id_from_filename(filename_path):
    name_parts = filename_path.stem.split('_')
    for part in name_parts:
        if part.isdigit():
            return int(part)
    return None

def get_bag_filename_from_sentence_id(sentence_id):
    # !!! AJUSTEAZĂ ACEASTĂ FUNCȚIE PENTRU A SE POTRIVI CU NUMELE FIȘIERELOR TALE .BAG !!!
    return f"sentence_{int(sentence_id):03d}_realsense.bag"

# --- Helper Functions ---
def pivot_landmarks(df_long):
    if df_long.empty: return pd.DataFrame()
    df_wide = df_long.pivot_table(index='frame_id', columns='landmark_id', values=['x_cam', 'y_cam', 'z_cam'])
    df_wide.columns = [f'landmark_{int(col[1])}_{col[0]}' for col in df_wide.columns]
    return df_wide.sort_index()

def pivot_blendshapes(df_long):
    if df_long.empty: return pd.DataFrame()
    df_wide = df_long.pivot_table(index='frame_id', columns='blendshape_name', values='score')
    return df_wide.sort_index()

def extract_timestamps_from_bag(bag_filepath, trim_sec=0.0):
    """Extracts timestamps of color frames from a .bag file."""
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
        MAX_CONSECUTIVE_NO_FRAMES = 300 # Mai mare pentru citirea din fișier

        while True:
            # try_wait_for_frames returnează (success_flag, frameset_object)
            success, frameset = pipeline.try_wait_for_frames(200) # Mărește ușor timeout-ul pentru fișiere

            if not success or not frameset: # Verifică AMBELE
                consecutive_no_frame += 1
                if consecutive_no_frame > MAX_CONSECUTIVE_NO_FRAMES:
                    print(f"    Max no-frame polls ({MAX_CONSECUTIVE_NO_FRAMES}) reached in {bag_filepath.name}. Extracted {frames_extracted} timestamps.")
                    break
                # Verifică dacă am ajuns la sfârșit, chiar dacă nu primim cadre
                try:
                    current_pos_ns = playback_device.get_position()
                    total_dur_ns = int(playback_device.get_duration().total_seconds() * 1e9)
                    if total_dur_ns > 0 and current_pos_ns >= total_dur_ns - 10_000_000: # Aproape de final
                        print(f"    Playback position indicates end of BAG file {bag_filepath.name}. Extracted {frames_extracted} timestamps.")
                        break
                except Exception: # Ignoră erorile de la get_position dacă stream-ul s-a oprit
                    pass
                time.sleep(0.005) # Așteaptă puțin dacă nu s-au primit cadre
                continue

            consecutive_no_frame = 0 # Reset counter if frames were received

            color_frame = frameset.get_color_frame() # Acum 'frameset' este obiectul corect
            if color_frame:
                ts_ms = int(color_frame.get_timestamp())
                if ts_ms <= last_ts and last_ts > 0:
                    ts_ms = last_ts + 1 # Sau o mică incrementare, e.g., +10ms
                last_ts = ts_ms
                timestamps_ms_list.append(ts_ms)
                frames_extracted += 1
                if frames_extracted % 500 == 0 and frames_extracted > 0:
                    print(f"      Extracted {frames_extracted} timestamps...")
            else:
                print(f"    Warning: No color frame in frameset for {bag_filepath.name}, frame count {frames_extracted}")
                # Nu ar trebui să se întâmple des dacă stream-ul e activat și fișierul e OK


    except RuntimeError as e_rt: # "Frame didn't arrive within X" sau altele
        print(f"    Runtime error during BAG processing for {bag_filepath.name}: {e_rt}")
        # Această eroare poate indica și sfârșitul fișierului dacă timeout-ul e atins repetat
    except Exception as e_gen:
        print(f"    General Error processing BAG file {bag_filepath.name} for timestamps: {e_gen}")
        traceback.print_exc()
    finally:
        if pipeline:
            try:
                pipeline.stop()
            except Exception: pass
    print(f"    Finished extracting timestamps from {bag_filepath.name}. Total: {len(timestamps_ms_list)}")
    return np.array(timestamps_ms_list, dtype=np.float64)

# --- Main Processing Logic ---
print("--- Starting Facial Data Preparation (Pivot and Timestamp Extraction from BAG) ---")

landmark_files = sorted(glob.glob(str(LANDMARKS_CSV_DIR / "*.csv")))
blendshape_files_map = {get_sentence_id_from_filename(Path(f)): Path(f)
                        for f in glob.glob(str(BLENDSHAPES_CSV_DIR / "*.csv"))
                        if get_sentence_id_from_filename(Path(f)) is not None}

if not landmark_files:
    print(f"Error: No landmark CSV files found in {LANDMARKS_CSV_DIR}")
    exit()

processed_count = 0
error_count = 0

for lm_csv_path_str in landmark_files:
    lm_csv_path = Path(lm_csv_path_str)
    sentence_id = get_sentence_id_from_filename(lm_csv_path)

    if sentence_id is None:
        print(f"Warning: Could not determine sentence ID for landmark file: {lm_csv_path.name}. Skipping.")
        error_count +=1
        continue

    print(f"\nProcessing Sentence ID: {sentence_id:03d} (from {lm_csv_path.name})")

    # --- Identificare și procesare fișier .bag ---
    bag_filename = get_bag_filename_from_sentence_id(sentence_id)
    bag_filepath = ORIGINAL_BAG_FILES_DIR / bag_filename
    if not bag_filepath.exists():
        print(f"  Error: Original .bag file not found: {bag_filepath}. Skipping sentence.")
        error_count += 1
        continue

    realsense_timestamps_ms = extract_timestamps_from_bag(bag_filepath, trim_sec=TRIM_START_SECONDS_BAG)
    if realsense_timestamps_ms is None or len(realsense_timestamps_ms) == 0:
        print(f"  Error: No timestamps extracted from {bag_filepath.name}. Skipping sentence.")
        error_count += 1
        continue

    # --- Load Landmark Data ---
    try:
        df_landmarks_long = pd.read_csv(lm_csv_path)
        if df_landmarks_long.empty:
            print(f"  Warning: Landmark file {lm_csv_path.name} is empty. Skipping.")
            error_count +=1; continue
        if 'frame_id' not in df_landmarks_long.columns:
            print(f"  Error: 'frame_id' column missing in {lm_csv_path.name}. Skipping."); error_count += 1; continue
        df_landmarks_long['frame_id'] = df_landmarks_long['frame_id'].astype(int)
        max_lm_frame_id = df_landmarks_long['frame_id'].max()
    except Exception as e:
        print(f"  Error reading landmark file {lm_csv_path.name}: {e}"); error_count +=1; continue

    # Verifică dacă avem suficiente timestamp-uri pentru frame_id-urile din CSV
    if len(realsense_timestamps_ms) <= max_lm_frame_id:
        print(f"  Warning: Not enough timestamps ({len(realsense_timestamps_ms)}) extracted from BAG for max frame_id ({max_lm_frame_id}) in {lm_csv_path.name}. Some facial frames might be lost or misaligned.")
        # Decide cum gestionezi: trunchiază df_landmarks_long sau continui cu riscul de erori de indexare
        # Pentru siguranță, trunchiem:
        df_landmarks_long = df_landmarks_long[df_landmarks_long['frame_id'] < len(realsense_timestamps_ms)]
        if df_landmarks_long.empty:
            print("    Landmark data became empty after timestamp count check. Skipping.")
            error_count +=1; continue


    # --- Pivot Landmarks & Adaugă Timestamp-uri Reale ---
    df_landmarks_wide = pivot_landmarks(df_landmarks_long)
    if df_landmarks_wide.empty: print(f"  Warning: Pivoted landmarks empty. Skipping."); error_count +=1; continue

    # Asociază timestamp-ul real pe baza indexului (frame_id)
    # Indexul lui df_landmarks_wide este frame_id
    timestamps_for_lm_frames = realsense_timestamps_ms[df_landmarks_wide.index.astype(int)]
    df_landmarks_wide['realsense_timestamp_ms'] = timestamps_for_lm_frames
    df_landmarks_wide.set_index('realsense_timestamp_ms', inplace=True)

    # --- Load, Pivot Blendshapes & Adaugă Timestamp-uri Reale ---
    df_blendshapes_wide = pd.DataFrame()
    if sentence_id in blendshape_files_map:
        bs_csv_path = blendshape_files_map[sentence_id]
        print(f"  Processing blendshapes from: {bs_csv_path.name}")
        try:
            df_blendshapes_long = pd.read_csv(bs_csv_path)
            if not df_blendshapes_long.empty:
                if 'frame_id' not in df_blendshapes_long.columns:
                    print(f"    Warning: 'frame_id' missing in {bs_csv_path.name}."); df_blendshapes_long['frame_id'] = -1 # Sau altă gestionare
                df_blendshapes_long['frame_id'] = df_blendshapes_long['frame_id'].astype(int)
                max_bs_frame_id = df_blendshapes_long['frame_id'].max()

                if len(realsense_timestamps_ms) <= max_bs_frame_id:
                    print(f"    Warning: Not enough timestamps ({len(realsense_timestamps_ms)}) for max blendshape frame_id ({max_bs_frame_id}). Truncating.")
                    df_blendshapes_long = df_blendshapes_long[df_blendshapes_long['frame_id'] < len(realsense_timestamps_ms)]

                if not df_blendshapes_long.empty:
                    df_blendshapes_wide_temp = pivot_blendshapes(df_blendshapes_long)
                    if not df_blendshapes_wide_temp.empty:
                        timestamps_for_bs_frames = realsense_timestamps_ms[df_blendshapes_wide_temp.index.astype(int)]
                        df_blendshapes_wide_temp['realsense_timestamp_ms'] = timestamps_for_bs_frames
                        df_blendshapes_wide = df_blendshapes_wide_temp.set_index('realsense_timestamp_ms')
                    else: print(f"    Warning: Pivoted blendshapes empty for {bs_csv_path.name}.")
            else: print(f"    Warning: Blendshape file {bs_csv_path.name} is empty.")
        except Exception as e: print(f"    Error processing blendshape file {bs_csv_path.name}: {e}")
    else: print(f"  Warning: No corresponding blendshape file for sentence ID {sentence_id:03d}.")

    # --- Combinare, Normalizare Timestamp-uri, Salvare ---
    if not df_blendshapes_wide.empty:
        print("  Merging landmarks and blendshapes using precise timestamps...")
        df_facial_combined = pd.merge(
            df_landmarks_wide, df_blendshapes_wide,
            left_index=True,  # Folosește indexul 'realsense_timestamp_ms'
            right_index=True,  # Folosește indexul 'realsense_timestamp_ms'
            how='outer',  # Păstrează toate timestamp-urile, vor apărea NaN-uri
            suffixes=('_lm', '_bs')
            # Adaugă sufixe dacă există coloane cu același nume (nu ar trebui pentru feature-uri)
        )
        # După merge, indexul ar trebui să fie timestamp-urile, dar s-ar putea să-și piardă numele.
        # Să ne asigurăm că numele indexului este setat corect înainte de operațiunile următoare.
        df_facial_combined.index.name = 'realsense_timestamp_ms'  # SETEAZĂ NUMELE INDEXULUI

        df_facial_combined.ffill(inplace=True)
        df_facial_combined.bfill(inplace=True)
        df_facial_combined.fillna(0, inplace=True)
    else:
        print("  Proceeding with landmarks data only (no valid blendshapes).")
        df_facial_combined = df_landmarks_wide.copy()
        # Și aici, asigură-te că indexul are numele corect dacă df_landmarks_wide este folosit direct
        if df_facial_combined.index.name != 'realsense_timestamp_ms':
            df_facial_combined.index.name = 'realsense_timestamp_ms'

    if df_facial_combined.empty:
        print(f"  Error: Combined facial data is empty for sentence ID {sentence_id:03d}. Skipping.")
        error_count += 1
        continue

        # Acum df_facial_combined.index.name ar trebui să fie 'realsense_timestamp_ms'
        # Conversia la TimedeltaIndex:
        # Mai întâi, resetează indexul pentru a-l avea ca o coloană, apoi convertește și setează-l la loc
    df_facial_combined = df_facial_combined.reset_index()
    df_facial_combined['realsense_timestamp_ms'] = pd.to_timedelta(df_facial_combined['realsense_timestamp_ms'],
                                                                   unit='ms', errors='coerce')
    df_facial_combined.dropna(subset=['realsense_timestamp_ms'], inplace=True)  # Elimină dacă conversia a eșuat
    if df_facial_combined.empty:
        print(f"  Error: Facial data empty after timestamp conversion for sentence ID {sentence_id:03d}. Skipping.")
        error_count += 1
        continue
    df_facial_combined = df_facial_combined.set_index('realsense_timestamp_ms').sort_index()

    # Normalizare Timestamp-uri (pentru a începe de la zero)
    # Acest pas se face pe indexul Timedelta
    first_ts = df_facial_combined.index.min()
    df_facial_combined.index = df_facial_combined.index - first_ts
    df_facial_combined = df_facial_combined.sort_index()  # Asigură sortarea după normalizare

    # --- Salvare ---
    output_filename = f"sentence_{sentence_id:03d}_facial_processed_bagts.pkl"
    output_path = OUTPUT_FACIAL_PKL_DIR / output_filename
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(df_facial_combined, f)
        print(f"  Successfully saved processed facial data to: {output_path}")
        processed_count += 1
    except Exception as e:
        print(f"  Error saving processed facial data for sentence {sentence_id:03d} to {output_path}: {e}")
        error_count += 1
        traceback.print_exc()

# --- Sumar ---
print(f"\n--- Facial Data Preparation Summary (from BAG Timestamps) ---")
print(f"Successfully processed and saved {processed_count} facial data files.")
if error_count > 0: print(f"Encountered errors for {error_count} files.")
print(f"Processed facial data saved to: {OUTPUT_FACIAL_PKL_DIR}")