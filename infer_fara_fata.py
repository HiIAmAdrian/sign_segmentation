import tensorflow as tf
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from pympi.Elan import Eaf
import traceback
import os

MODEL_NAME_SUFFIX_FROM_TRAIN = "BiGRU_2Cls_OI_TSG"
MODEL_PATH = Path(f"./trained_models_2_classes_OI_ts_gloves/best_{MODEL_NAME_SUFFIX_FROM_TRAIN}.keras")

SCALER_PATH = Path("./final_data_ts_gloves_only/ts_gloves_only_scaler.pkl")
PKL_DATA_DIR = Path("./final_data_ts_gloves_only")
PKL_FEATURES_FILE = PKL_DATA_DIR / "all_data_features_ts_gloves_only.pkl"

SEQUENCE_TO_INFER_INDEX = 0
CHOOSE_SET = "test"

TRIM_SECONDS_OFFSET_DEFAULT = 0.3
MIN_SEGMENT_DURATION_MS = 250
MIN_I_FRAMES_FOR_POSTPROCESS = 6

OUTPUT_EAF_DIR_BASE = Path("./inference_output_eaf_2_classes_OI_ts_gloves")
OUTPUT_EAF_DIR = OUTPUT_EAF_DIR_BASE / f"{MODEL_NAME_SUFFIX_FROM_TRAIN}_Set_{CHOOSE_SET}"
OUTPUT_EAF_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES_MODEL = 2
LABEL_O_2CLASS, LABEL_I_2CLASS = 0, 1


def load_model_and_scaler(model_path, scaler_path=None):
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    scaler = None
    if scaler_path and scaler_path.exists():
        print(f"Loading scaler from: {scaler_path}")
        with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
    return model, scaler


def get_sequence_for_inference_from_set(pkl_file_path, set_name, seq_index):
    print(f"Loading sequence from: {pkl_file_path}, set: {set_name}, index: {seq_index}")
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)
    df_key, ids_key = f'X_{set_name}_df_indexed', f'{set_name}_ids'
    if df_key not in data or not data[df_key] or seq_index >= len(data[df_key]):
        raise ValueError(f"Data/Index error for {df_key} in {pkl_file_path}")
    df = data[df_key][seq_index]
    id_info = data[ids_key][seq_index] if ids_key in data and len(data[ids_key]) > seq_index else {}
    stem = Path(id_info.get('filename', f"unknown_{set_name}_{seq_index}")).stem.split('.')[0]
    if df is None or df.empty: raise ValueError("Sequence is empty or None.")
    trim_sec = id_info.get('trim_seconds_applied', TRIM_SECONDS_OFFSET_DEFAULT)
    if trim_sec is None: trim_sec = TRIM_SECONDS_OFFSET_DEFAULT
    return df, df.index, stem, trim_sec


def scale_data(data_array, scaler):
    if scaler is None: print("Warning: Scaler is None, returning data as is."); return data_array
    if data_array.ndim == 1: data_array = data_array.reshape(1, -1)
    if data_array.shape[1] != scaler.n_features_in_:
        raise ValueError(f"Scaler feature mismatch. Data: {data_array.shape[1]}, Scaler: {scaler.n_features_in_}")
    return scaler.transform(data_array)


def predict_sequence_simple(model, feature_sequence_scaled, model_max_len_from_train, num_classes_from_model_output):
    seq_len_actual = feature_sequence_scaled.shape[0]
    num_features_actual = feature_sequence_scaled.shape[1]
    num_features_model_expected = model.input_shape[-1]
    if num_features_actual != num_features_model_expected:
        raise ValueError(
            f"Feature count mismatch for prediction. Data: {num_features_actual}, Model: {num_features_model_expected}")

    input_data_for_model = np.zeros((1, model_max_len_from_train, num_features_model_expected), dtype=np.float32)
    len_to_copy = min(seq_len_actual, model_max_len_from_train)
    input_data_for_model[0, :len_to_copy, :] = feature_sequence_scaled[:len_to_copy, :]

    pred_probs_all_steps = model.predict(input_data_for_model, verbose=0)
    pred_probs_for_sequence = pred_probs_all_steps[0, :len_to_copy, :]
    predicted_labels = np.argmax(pred_probs_for_sequence, axis=-1)

    if seq_len_actual > model_max_len_from_train:
        print(
            f"Warning: Actual sequence length ({seq_len_actual}) > model's trained max length ({model_max_len_from_train}).")
        full_labels = np.full(seq_len_actual, LABEL_O_2CLASS, dtype=int)
        full_labels[:len_to_copy] = predicted_labels
        full_probs = np.zeros((seq_len_actual, num_classes_from_model_output), dtype=np.float32)
        full_probs[:len_to_copy, :] = pred_probs_for_sequence
        one_hot_o = tf.keras.utils.to_categorical(LABEL_O_2CLASS, num_classes=num_classes_from_model_output)
        full_probs[len_to_copy:, :] = one_hot_o
        return full_labels, full_probs
    return predicted_labels, pred_probs_for_sequence


def post_process_oi_sequence(pred_labels_oi, pred_probs_oi, min_i_duration_frames=3):
    print(f"Post-process (O/I): Starting for {len(pred_labels_oi)} frames, min_i_frames={min_i_duration_frames}.")
    if len(pred_labels_oi) == 0: return pred_labels_oi

    corrected_seq = list(pred_labels_oi)
    n = len(corrected_seq)
    idx = 0
    while idx < n:
        if corrected_seq[idx] == LABEL_I_2CLASS:
            start_i_idx = idx
            while idx < n and corrected_seq[idx] == LABEL_I_2CLASS: idx += 1
            duration_i = idx - start_i_idx
            if duration_i < min_i_duration_frames:
                for k_fill_o in range(start_i_idx, idx): corrected_seq[k_fill_o] = LABEL_O_2CLASS
        else:
            idx += 1
    print(f"Post-process (O/I): Finished.")
    return np.array(corrected_seq, dtype=int)


def oi_to_segments(oi_labels, timedelta_index_us_values):
    segments = []
    in_segment = False
    start_time_ms = 0
    if len(oi_labels) != len(timedelta_index_us_values):
        m = min(len(oi_labels), len(timedelta_index_us_values))
        oi_labels, timedelta_index_us_values = oi_labels[:m], timedelta_index_us_values[:m]
        if m == 0: return segments
    for i, label in enumerate(oi_labels):
        current_time_ms = int(timedelta_index_us_values[i] / 1000)
        if label == LABEL_I_2CLASS:
            if not in_segment: start_time_ms = current_time_ms; in_segment = True
        elif label == LABEL_O_2CLASS:
            if in_segment:
                end_ms = int(timedelta_index_us_values[i - 1] / 1000) if i > 0 else start_time_ms
                if end_ms >= start_time_ms: segments.append((start_time_ms, end_ms, "SIGN"))
                in_segment = False
    if in_segment:
        end_ms = int(timedelta_index_us_values[-1] / 1000) if len(timedelta_index_us_values) > 0 else start_time_ms
        if end_ms >= start_time_ms: segments.append((start_time_ms, end_ms, "SIGN"))
    return segments


def create_eaf_file(output_eaf_path, segments, absolute_media_uri=None, relative_media_path_for_storage=None):
    eaf = Eaf(author="InferScript_TSG")
    tier_id = "PredictedSigns_TSG"
    eaf.add_tier(tier_id)
    if "default-lt" not in eaf.linguistic_types: eaf.add_linguistic_type("default-lt", timealignable=True)
    if tier_id in eaf.tiers and (eaf.tiers[tier_id][2] is None or eaf.tiers[tier_id][2] == ''):
        ea, p, dl = eaf.tiers[tier_id]
        eaf.tiers[tier_id] = (ea, p, "default-lt", dl)
    if absolute_media_uri and relative_media_path_for_storage:
        mt = "video/mp4"
        mt = "audio/x-wav" if ".wav" in relative_media_path_for_storage.lower() else mt
        print(f"Linking media: URI='{absolute_media_uri}', Relative Path='{relative_media_path_for_storage}'")
        eaf.add_linked_file(file_path=absolute_media_uri, relpath=relative_media_path_for_storage, mimetype=mt)
    elif relative_media_path_for_storage:
        mt = "video/mp4"
        mt = "audio/x-wav" if ".wav" in relative_media_path_for_storage.lower() else mt
        print(f"Warning: Linking media with relative path only: {relative_media_path_for_storage}")
        eaf.add_linked_file(file_path=relative_media_path_for_storage, relpath=relative_media_path_for_storage,
                            mimetype=mt)
    else:
        print("No media path provided for EAF.")
    for s, e, v in segments:
        if e <= s: print(f"Skipping segment with duration<=0ms: Start={s}, End={e}"); continue
        try:
            eaf.add_annotation(tier_id, int(s), int(e), value=v)
        except Exception as err_ann:
            print(f"Error adding annotation ({s}-{e}, {v}): {err_ann}"); traceback.print_exc()
    try:
        Path(output_eaf_path).parent.mkdir(parents=True, exist_ok=True)
        eaf.to_file(str(output_eaf_path))
        print(f"EAF saved: {output_eaf_path}")
    except Exception as err_save:
        print(f"Error saving EAF {output_eaf_path}: {err_save}"); traceback.print_exc()


if __name__ == "__main__":
    model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
    if model is None: print("Exiting: model loading failure."); exit()
    if scaler is None: print(
        "Warning: Scaler not loaded. Ensure data in PKL is already scaled if this is intended.")

    print(f"--- Inferring on Set: {CHOOSE_SET}, Index: {SEQUENCE_TO_INFER_INDEX} (TS+Gloves Data, 2 Classes) ---")
    model.summary(line_length=120)
    max_len_model, feat_model_expected, n_cls_model = model.input_shape[1], model.input_shape[2], model.output_shape[-1]
    if n_cls_model != NUM_CLASSES_MODEL: print(
        f"FATAL: Model output classes ({n_cls_model}) != expected ({NUM_CLASSES_MODEL})"); exit()
    print(f"Model expects: max_len={max_len_model}, features={feat_model_expected}, classes={n_cls_model}")

    try:
        df_unscaled, t_index, stem, trim_sec_for_current_seq = get_sequence_for_inference_from_set(
            PKL_FEATURES_FILE, CHOOSE_SET, SEQUENCE_TO_INFER_INDEX)

        print(
            f"Loaded '{stem}': {df_unscaled.shape[0]} frames, {df_unscaled.shape[1]} raw features. Using trim_offset: {trim_sec_for_current_seq:.2f}s")

        if df_unscaled.shape[1] != feat_model_expected:
            print(
                f"Warning: Raw feature count from PKL ({df_unscaled.shape[1]}) != model expected features ({feat_model_expected}). This might be an issue if scaler expects {feat_model_expected}.")

        vals_scaled = scale_data(df_unscaled.values, scaler)
        print(f"Scaled sequence shape: {vals_scaled.shape}")
        if vals_scaled.shape[1] != feat_model_expected:
            print(
                f"FATAL: Feature count after scaling ({vals_scaled.shape[1]}) != model expected ({feat_model_expected}).")
            exit()

        raw_labels, raw_probs = predict_sequence_simple(model, vals_scaled, max_len_model, n_cls_model)
        print("Raw 2-class labels (first 50):", raw_labels[:50])
        unique_r, c_r = np.unique(raw_labels, return_counts=True)
        print("Counts raw:", dict(zip(unique_r, c_r)))

        final_labels = post_process_oi_sequence(raw_labels, raw_probs,
                                                min_i_duration_frames=MIN_I_FRAMES_FOR_POSTPROCESS)
        print("Final 2-class labels (post-proc, first 50):", final_labels[:50])
        unique_f, c_f = np.unique(final_labels, return_counts=True)
        print("Counts final:", dict(zip(unique_f, c_f)))

        ts_us_vals = t_index.to_series().dt.total_seconds() * 1_000_000
        segs_unfilt = oi_to_segments(final_labels, ts_us_vals)
        print(f"Generated {len(segs_unfilt)} unfiltered segments (O/I).")

        segs_filt = [s for s in segs_unfilt if (s[1] - s[0]) >= MIN_SEGMENT_DURATION_MS]
        print(f"Retained {len(segs_filt)} segs after duration filter (min_dur={MIN_SEGMENT_DURATION_MS}ms).")
        if segs_filt:
            [print(f"  Filt Seg {i} (rel): S={s[0]}ms, E={s[1]}ms, Dur={s[1] - s[0]}ms") for i, s in
             enumerate(segs_filt[:5])]
        else:
            print("  No segs after duration filter.")

        adj_segs_eaf = []
        trim_ms = int(trim_sec_for_current_seq * 1000)
        target_segs = segs_filt
        if trim_ms != 0:
            print(f"Applying trim offset of {trim_ms}ms to {len(target_segs)} segments.")
            for s_start, s_end, s_val in target_segs: adj_segs_eaf.append((s_start + trim_ms, s_end + trim_ms, s_val))
            if adj_segs_eaf: [print(f"  Adj Seg {i}: S={s[0]}ms, E={s[1]}ms") for i, s in enumerate(adj_segs_eaf[:2])]
        else:
            print("No trim offset to apply (offset is 0). Using filtered segments as is.")
            adj_segs_eaf = target_segs

        media_filename = f"{stem}_realsense.mp4"
        eaf_filename = f"{stem}_pred_2clsTSG_set{CHOOSE_SET}{SEQUENCE_TO_INFER_INDEX}_trim{trim_sec_for_current_seq:.1f}s_minDur{MIN_SEGMENT_DURATION_MS}ms_minI{MIN_I_FRAMES_FOR_POSTPROCESS}f.eaf"
        proj_r = Path.cwd()
        abs_vid_p = (proj_r / "videos" / media_filename).resolve()
        out_eaf_p = (proj_r / OUTPUT_EAF_DIR / eaf_filename).resolve()
        abs_vid_u, rel_vid_p = None, None
        if abs_vid_p.exists():
            abs_vid_u = abs_vid_p.as_uri()
            try:
                rel_vid_p = Path(os.path.relpath(abs_vid_p, out_eaf_p.parent)).as_posix()
            except ValueError:
                rel_vid_p = (Path("..") / "videos" / media_filename).as_posix()
        else:
            rel_vid_p = (Path("..") / "videos" / media_filename).as_posix()
            print(f"Warning: Video for EAF not found: {abs_vid_p}")
        create_eaf_file(out_eaf_p, adj_segs_eaf, abs_vid_u, rel_vid_p)

    except Exception as e_main:
        print(f"An error occurred in main inference block: {e_main}"); traceback.print_exc()