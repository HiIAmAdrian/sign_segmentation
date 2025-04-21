import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import traceback

# --- Configuration ---

# Input Directories/Files
PROCESSED_DATA_DIR = Path("./processed_combined_data_both_gloves")
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "combined_interpolated_processed_sequences.pkl"
EAF_ANNOTATION_DIR = Path("./elan_annotations")  # Directory containing ALL your .eaf files
# !!! Need original suit CSVs for accurate timestamps !!!
ORIGINAL_SUIT_CSV_DIR = Path("./suit")  # From previous script

# Output File
OUTPUT_ANNOTATION_PKL = Path("./processed_combined_data_both_gloves/annotations.pkl")  # New name for I/O only

# EAF Parsing Settings
TARGET_TIER_ID = "default"  # Tier name in ELAN containing sign annotations
TARGET_ANNOTATION_VALUE = "i"  # The label used in ELAN for sign segments ('I')

# Synchronization Offset (if video didn't start at same ms as MoCap)
SYNC_OFFSET_MS = 0  # Adjust if necessary based on your sync analysis

# Label Mapping (Simplified: I=1, O=0)
LABEL_O = 0
LABEL_I = 1


# --- Helper Functions ---
# (parse_eaf, get_mocap_timestamps, map_time_to_frame remain the same as previous version)
def parse_eaf(eaf_path, target_tier_id="default", target_annotation_value="i"):
    """Parses an ELAN EAF file to extract time segments for a specific tier and value."""
    segments = []
    try:
        tree = ET.parse(eaf_path)
        root = tree.getroot()
        time_slots = {slot.attrib['TIME_SLOT_ID']: int(slot.attrib['TIME_VALUE'])
                      for slot in root.findall(".//TIME_SLOT")}
        tier = None
        for t in root.findall(f".//TIER[@TIER_ID='{target_tier_id}']"):
            tier = t;
            break
        if tier is None:
            print(f"  --> Warning: Tier '{target_tier_id}' not found in {eaf_path.name}")
            return []
        for annotation in tier.findall(".//ALIGNABLE_ANNOTATION"):
            value_element = annotation.find("ANNOTATION_VALUE")
            # Handle annotations with potentially empty values robustly
            value = value_element.text if value_element is not None and value_element.text is not None else ""
            if value == target_annotation_value:
                t1_id = annotation.attrib['TIME_SLOT_REF1']
                t2_id = annotation.attrib['TIME_SLOT_REF2']
                if t1_id in time_slots and t2_id in time_slots:
                    start_ms = time_slots[t1_id]
                    end_ms = time_slots[t2_id]
                    segments.append((start_ms, end_ms))
                else:
                    print(
                        f"  --> Warning: Time slot refs {t1_id}/{t2_id} not found for annotation ID {annotation.attrib.get('ANNOTATION_ID', 'N/A')} in {eaf_path.name}")
        segments.sort()
        return segments
    except ET.ParseError as e:
        print(f"  --> Error: Failed to parse XML in {eaf_path.name}: {e}")
        return None
    except Exception as e:
        print(f"  --> Error: Unexpected error parsing {eaf_path.name}: {e}")
        traceback.print_exc();
        return None


def get_mocap_timestamps(mocap_csv_path):
    """Loads the frame_timestamp column from the original MoCap CSV."""
    try:
        df = pd.read_csv(mocap_csv_path, usecols=['frame_timestamp'], converters={'frame_timestamp': float})
        timestamps_ms = df['frame_timestamp'].sort_values().to_numpy()
        if np.isnan(timestamps_ms).any():
            print(f"  --> Warning: NaNs found in timestamps of {mocap_csv_path.name}. Attempting to fill.")
            timestamps_ms = pd.Series(timestamps_ms).ffill().bfill().fillna(0).to_numpy()
        return timestamps_ms
    except FileNotFoundError:
        print(f"  --> Error: Original MoCap CSV not found: {mocap_csv_path.name}");
        return None
    except ValueError:
        print(f"  --> Error: 'frame_timestamp' column not found or invalid in {mocap_csv_path.name}");
        return None
    except Exception as e:
        print(f"  --> Error loading timestamps from {mocap_csv_path.name}: {e}");
        return None


def map_time_to_frame(target_ms, mocap_timestamps_ms):
    """Finds the index of the MoCap frame closest to the target time."""
    if mocap_timestamps_ms is None or len(mocap_timestamps_ms) == 0: return -1
    time_diff = np.abs(mocap_timestamps_ms - target_ms)
    closest_frame_index = np.argmin(time_diff)
    return int(closest_frame_index)


# --- Main Logic ---

# 1. Load processed data info (lengths, IDs, and splits)
print(f"Loading processed data info from: {PROCESSED_DATA_FILE}")
sequence_info = {}  # Store {mocap_filename: {'length': frame_count, 'split': 'train'/'val'/'test'}}
all_mocap_filenames_in_pkl = set()
try:
    with open(PROCESSED_DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    for split in ['train', 'val', 'test']:
        ids_key = f'{split}_ids'
        x_key = f'X_{split}'
        if ids_key in data and x_key in data:
            for i, seq_info_dict in enumerate(data[ids_key]):
                # Use 'filename' from the identifier dictionary
                mocap_filename = seq_info_dict.get('filename')
                if mocap_filename:
                    all_mocap_filenames_in_pkl.add(mocap_filename)
                    seq_data = data[x_key][i]
                    length = seq_data.shape[0] if seq_data is not None and hasattr(seq_data, 'shape') else 0
                    if length > 0:
                        sequence_info[mocap_filename] = {'length': length, 'split': split}
                    else:
                        print(
                            f"Warning: Sequence {mocap_filename} from split '{split}' has invalid length ({length}). It will be skipped.")
                else:
                    print(
                        f"Warning: Missing 'filename' key in identifier dictionary for sequence {i} in split '{split}'.")

        else:
            print(f"Warning: Keys '{ids_key}' or '{x_key}' not found in processed data file for split '{split}'.")

    if not sequence_info:
        print("FATAL Error: No valid sequence information loaded from the .pkl file. Cannot proceed.")
        exit()
    print(f"Loaded info for {len(sequence_info)} sequences from .pkl file.")

except Exception as e:
    print(f"FATAL Error loading processed data file: {e}")
    traceback.print_exc()
    exit()

# 2. Initialize results dictionary
y_splits = {'train': [], 'val': [], 'test': []}
temp_results = {}  # Store {mocap_filename: labels_array} temporarily

# 3. Iterate through EAF files and process annotations
print(f"\n--- Processing EAF annotations from: {EAF_ANNOTATION_DIR} ---")
eaf_files = list(EAF_ANNOTATION_DIR.glob("*.eaf"))
processed_eaf_count = 0
error_eaf_count = 0

if not eaf_files:
    print(f"Error: No .eaf files found in {EAF_ANNOTATION_DIR}. Cannot generate labels.")
    exit()

for eaf_path in eaf_files:
    print(f"\nProcessing EAF file: {eaf_path.name}")

    # --- Determine corresponding MoCap filename ---
    # This assumes a simple naming convention (e.g., removing '.eaf' and adding '_suit.csv')
    # !!! ADJUST THIS LOGIC if your naming differs !!!
    try:
        base_name = eaf_path.stem  # Filename without extension
        # Try removing common annotation suffixes if present
        if base_name.endswith("_annotations"): base_name = base_name[:-12]
        if base_name.endswith("_elan"): base_name = base_name[:-5]
        # Assume it matches the base name of the suit file
        corresponding_mocap_filename = f"{base_name}_suit.csv"
    except Exception:
        print(f"  --> Error: Could not determine MoCap filename from EAF name: {eaf_path.name}. Skipping.")
        error_eaf_count += 1
        continue

    # --- Check if MoCap file info exists ---
    if corresponding_mocap_filename not in sequence_info:
        print(
            f"  --> Warning: MoCap sequence info for '{corresponding_mocap_filename}' (derived from {eaf_path.name}) not found in the processed data .pkl file. Skipping EAF.")
        error_eaf_count += 1
        continue

    mocap_info = sequence_info[corresponding_mocap_filename]
    total_mocap_frames = mocap_info['length']

    # --- Load MoCap Timestamps ---
    original_mocap_path = ORIGINAL_SUIT_CSV_DIR / corresponding_mocap_filename
    mocap_timestamps_ms = get_mocap_timestamps(original_mocap_path)
    if mocap_timestamps_ms is None:
        print(f"  --> Error: Could not load timestamps for {corresponding_mocap_filename}. Skipping EAF.")
        error_eaf_count += 1
        continue

    # --- Parse EAF ---
    video_segments_ms = parse_eaf(eaf_path, TARGET_TIER_ID, TARGET_ANNOTATION_VALUE)
    if video_segments_ms is None:
        print(f"  --> Error: Failed to parse EAF file {eaf_path.name}. Skipping.")
        error_eaf_count += 1
        continue
    if not video_segments_ms:
        print(
            f"  --> Info: No annotations found with value '{TARGET_ANNOTATION_VALUE}' on tier '{TARGET_TIER_ID}'. Creating all 'O' labels.")

    # --- Create B-I-O (actually I/O) Label Array ---
    labels = np.full(total_mocap_frames, LABEL_O, dtype=int)  # Initialize all to 'O' (0)

    for video_start_ms, video_end_ms in video_segments_ms:
        mocap_target_start_ms = video_start_ms + SYNC_OFFSET_MS
        mocap_target_end_ms = video_end_ms + SYNC_OFFSET_MS

        start_frame = map_time_to_frame(mocap_target_start_ms, mocap_timestamps_ms)
        end_frame = map_time_to_frame(mocap_target_end_ms, mocap_timestamps_ms)

        if start_frame == -1 or end_frame == -1:
            print(f"  --> Error mapping time for segment ({video_start_ms}-{video_end_ms})ms. Skipping.")
            continue

        start_frame = max(0, start_frame)
        end_frame = min(total_mocap_frames - 1, end_frame)

        if start_frame > end_frame:
            print(
                f"  --> Warning: Start frame ({start_frame}) > end frame ({end_frame}) for segment ({video_start_ms}-{video_end_ms})ms. Marking only start frame as 'I'.")
            labels[start_frame] = LABEL_I  # Mark single frame as Inside
        else:
            # Mark frames from start_frame up to and including end_frame as 'I' (1)
            labels[start_frame: end_frame + 1] = LABEL_I

    # Store the generated labels temporarily
    temp_results[corresponding_mocap_filename] = labels
    processed_eaf_count += 1

# 4. Assign results to correct splits
print("\n--- Assigning generated labels to splits ---")
processed_filenames_from_eaf = set(temp_results.keys())
for mocap_filename, info in sequence_info.items():
    split = info['split']
    if mocap_filename in temp_results:
        y_splits[split].append(temp_results[mocap_filename])
    else:
        # Handle sequences from PKL that didn't have a matching EAF processed
        print(
            f"Warning: No annotation generated for sequence '{mocap_filename}' (expected in '{split}' split). Check EAF files/naming.")
        # Option: Append None, or an array of zeros, or skip this sequence from X/Y lists later
        # Appending array of zeros (all 'O') for simplicity, but this might be incorrect.
        print(f"  --> Creating all 'O' labels for {mocap_filename}.")
        labels = np.full(info['length'], LABEL_O, dtype=int)
        y_splits[split].append(labels)

# 5. Final Verification and Saving
print(f"\n--- Summary ---")
print(f"Successfully processed {processed_eaf_count} EAF files.")
if error_eaf_count > 0:
    print(f"Skipped {error_eaf_count} EAF files due to errors.")

final_label_count = sum(len(v) for v in y_splits.values())
print(f"Generated label arrays for {final_label_count} sequences.")

# Final length check
print("Verifying label array lengths match feature array lengths...")
lengths_match = True
error_sequences = []
for split in ['train', 'val', 'test']:
    if f'X_{split}' in data:
        x_list = data[f'X_{split}']
        y_list = y_splits.get(split, [])  # Use .get for safety
        if len(x_list) != len(y_list):
            print(
                f"FATAL Error: Mismatch in number of sequences for split '{split}' after processing (X: {len(x_list)}, Y: {len(y_list)}).")
            print("  This likely means some EAF files were missing or had errors.")
            lengths_match = False
            continue  # Check other splits maybe
        for i in range(len(x_list)):
            seq_id = data[f'{split}_ids'][i].get('filename', f"Index_{i}")
            if x_list[i] is not None and y_list[i] is not None:
                if x_list[i].shape[0] != y_list[i].shape[0]:
                    print(
                        f"FATAL Error: Length mismatch for sequence '{seq_id}' in split '{split}' (X: {x_list[i].shape[0]}, Y: {y_list[i].shape[0]})")
                    lengths_match = False
                    error_sequences.append(seq_id)
            elif x_list[i] is None or y_list[i] is None:
                print(f"Warning: Found None sequence for '{seq_id}' in split '{split}' during final check.")

if lengths_match and final_label_count > 0:
    print(f"\nSaving final I/O annotation lists to: {OUTPUT_ANNOTATION_PKL}")
    try:
        with open(OUTPUT_ANNOTATION_PKL, 'wb') as f:
            pickle.dump(y_splits, f)
        print("Annotation lists saved successfully.")
    except Exception as e:
        print(f"Error saving annotation pickle file: {e}")
elif final_label_count == 0:
    print("\nNo label sequences were generated. Nothing saved.")
else:
    print("\nAnnotations not saved due to length mismatch errors for sequences:")
    for seq_id in error_sequences: print(f" - {seq_id}")

print("\n--- Annotation Processing Finished ---")
