import tensorflow as tf
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from pympi.Elan import Eaf
import traceback
import os

# --- Configurații ---
# !!! ACTUALIZEAZĂ ACEASTĂ CALE la modelul antrenat PENTRU 2 CLASE !!!
# (ex: cel cu sau fără strat Conv, dar fără caracteristici istorice)
MODEL_PATH = Path("./5-trained_models_2_classes_OI_refined/bigru_best_2class_OI.keras")
SCALER_PATH = Path("./final_combined_data_for_training_ALL_SIGNERS/final_features_ts_facial_scaler.pkl")
PKL_DATA_FOR_INFERENCE_DIR = Path("./final_combined_data_for_training_ALL_SIGNERS")
PKL_DATA_FOR_INFERENCE_FILE = PKL_DATA_FOR_INFERENCE_DIR / "all_data_final_features_ts_facial.pkl"

SEQUENCE_TO_INFER_INDEX_IN_TEST_SET = 0

TRIM_SECONDS_OFFSET = 0.3
MIN_SEGMENT_DURATION_MS = 200
MIN_I_FRAMES_FOR_POSTPROCESS = 5  # Numărul minim de cadre 'I' consecutive pentru a nu fi eliminate de post_process_oi_sequence

OUTPUT_EAF_DIR = Path("./inference_output_eaf_2_classes_OI_refined")
OUTPUT_EAF_DIR.mkdir(parents=True, exist_ok=True)

# --- Configurație Clase ---
NUM_CLASSES_MODEL = 2  # Modelul a fost antrenat cu 2 clase
LABEL_O_2CLASS, LABEL_I_2CLASS = 0, 1


# --------------------------

# --- Funcții Ajutătoare ---
def load_model_and_scaler(model_path, scaler_path=None):
    print(f"Loading 2-class model from: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    scaler = None
    if scaler_path and scaler_path.exists():
        print(f"Loading scaler from: {scaler_path}")
        with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
    return model, scaler


def get_sequence_for_inference(pkl_file_path, seq_index_in_test):
    print(f"Loading sequence for inference from: {pkl_file_path}, test index: {seq_index_in_test}")
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)
    if 'X_test_df_indexed' not in data or not data['X_test_df_indexed']: raise ValueError("X_test_df_indexed empty.")
    if seq_index_in_test >= len(data['X_test_df_indexed']): raise ValueError(f"Seq index out of bounds.")
    df = data['X_test_df_indexed'][seq_index_in_test]
    idx = data['test_ids'][seq_index_in_test] if 'test_ids' in data and len(
        data['test_ids']) > seq_index_in_test else {}
    stem = Path(idx.get('filename', f"unk_{seq_index_in_test}")).stem.split('.')[0]
    if df is None or df.empty: raise ValueError(f"Selected sequence is None or empty.")
    return df, df.index, stem


def scale_data(data_array, scaler):
    if scaler is None: return data_array
    if data_array.ndim == 1: data_array = data_array.reshape(1, -1)
    if data_array.shape[1] != scaler.n_features_in_: raise ValueError(f"Scaler feature mismatch.")
    return scaler.transform(data_array)


def predict_sequence_simple(model, feature_sequence_scaled, model_max_len_from_train, num_classes_from_model_output):
    seq_len_actual = feature_sequence_scaled.shape[0]
    num_features_actual = feature_sequence_scaled.shape[1]
    num_features_model_expected = model.input_shape[-1]
    if num_features_actual != num_features_model_expected: raise ValueError(f"Feature count mismatch for prediction.")

    input_data_for_model = np.zeros((1, model_max_len_from_train, num_features_model_expected), dtype=np.float32)
    len_to_copy = min(seq_len_actual, model_max_len_from_train)
    input_data_for_model[0, :len_to_copy, :] = feature_sequence_scaled[:len_to_copy, :]

    # print(f"Predicting with input shape: {input_data_for_model.shape}")
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
    """O post-procesare simplă pentru secvențe O-I: elimină segmentele 'I' prea scurte."""
    # pred_probs_oi nu este folosit momentan în această versiune simplă, dar e păstrat pentru compatibilitate/extensii
    print(
        f"Post-process (O/I): Starting for sequence of length {len(pred_labels_oi)} with min_i_duration_frames={min_i_duration_frames}.")
    if len(pred_labels_oi) == 0: return pred_labels_oi

    corrected_seq = list(pred_labels_oi)
    n = len(corrected_seq)

    idx = 0
    while idx < n:
        if corrected_seq[idx] == LABEL_I_2CLASS:
            start_i_idx = idx
            while idx < n and corrected_seq[idx] == LABEL_I_2CLASS:
                idx += 1
            end_i_idx = idx  # idx este acum primul cadru DUPĂ secvența de I
            duration_i = end_i_idx - start_i_idx
            if duration_i < min_i_duration_frames:
                # print(f"  Post-process (O/I): Converting short I segment (frames {start_i_idx}-{end_i_idx-1}, duration {duration_i}) to O.")
                for k_fill_o in range(start_i_idx, end_i_idx):
                    corrected_seq[k_fill_o] = LABEL_O_2CLASS
        else:  # corrected_seq[idx] == LABEL_O_2CLASS
            idx += 1

    print(f"Post-process (O/I): Finished.")
    return np.array(corrected_seq, dtype=int)


def oi_to_segments(oi_labels, timedelta_index_us_values):
    segments = []
    in_segment = False
    start_time_ms = 0

    if len(oi_labels) != len(timedelta_index_us_values):
        min_len = min(len(oi_labels), len(timedelta_index_us_values))
        oi_labels = oi_labels[:min_len];
        timedelta_index_us_values = timedelta_index_us_values[:min_len]
        if min_len == 0: return segments

    for i, label in enumerate(oi_labels):
        current_time_ms = int(timedelta_index_us_values[i] / 1000)
        if label == LABEL_I_2CLASS:
            if not in_segment:
                start_time_ms = current_time_ms
                in_segment = True
        elif label == LABEL_O_2CLASS:
            if in_segment:
                end_time_current_segment_ms = int(timedelta_index_us_values[i - 1] / 1000) if i > 0 else start_time_ms
                if end_time_current_segment_ms >= start_time_ms:  # Permitem segmente de 0ms dacă e cazul (ex: 1 cadru I)
                    # Filtrul de durată se va ocupa de ele mai târziu
                    segments.append((start_time_ms, end_time_current_segment_ms, "SIGN"))
                in_segment = False

    if in_segment:  # Verifică dacă ultimul segment 'I' e deschis
        end_time_final_segment_ms = int(timedelta_index_us_values[-1] / 1000) if len(
            timedelta_index_us_values) > 0 else start_time_ms
        if end_time_final_segment_ms >= start_time_ms:
            segments.append((start_time_ms, end_time_final_segment_ms, "SIGN"))
    return segments


def create_eaf_file(output_eaf_path, segments, absolute_media_uri=None, relative_media_path_for_storage=None):
    eafob = Eaf(author="InferenceScript");
    tier_id = "PredictedSigns";
    eafob.add_tier(tier_id)
    if "default-lt" not in eafob.linguistic_types: eafob.add_linguistic_type("default-lt", timealignable=True)
    if tier_id in eafob.tiers and (eafob.tiers[tier_id][2] is None or eafob.tiers[tier_id][2] == ''):
        e_a, p, d_l = eafob.tiers[tier_id][0], eafob.tiers[tier_id][1], eafob.tiers[tier_id][3];
        eafob.tiers[tier_id] = (e_a, p, "default-lt", d_l)
    if absolute_media_uri and relative_media_path_for_storage:
        m = "video/mp4";
        m = "audio/x-wav" if ".wav" in relative_media_path_for_storage.lower() else m
        print(f"Linking media: URI='{absolute_media_uri}', Relative Path='{relative_media_path_for_storage}'")
        eafob.add_linked_file(file_path=absolute_media_uri, relpath=relative_media_path_for_storage, mimetype=m)
    elif relative_media_path_for_storage:
        m = "video/mp4";
        m = "audio/x-wav" if ".wav" in relative_media_path_for_storage.lower() else m
        print(f"Warning: Linking with relative path: {relative_media_path_for_storage}")
        eafob.add_linked_file(file_path=relative_media_path_for_storage, relpath=relative_media_path_for_storage,
                              mimetype=m)
    else:
        print("No media path.")
    for s, e, v in segments:
        if e <= s: print(
            f"Skipping invalid segment (duration<=0ms): {s}-{e}"); continue  # Modificat pentru a prinde și 0ms
        try:
            eafob.add_annotation(tier_id, int(s), int(e), value=v)
        except Exception as err:
            print(f"Error adding annotation ({s}-{e}, {v}): {err}"); traceback.print_exc()
    try:
        Path(output_eaf_path).parent.mkdir(parents=True, exist_ok=True);
        eafob.to_file(str(output_eaf_path));
        print(f"EAF saved: {output_eaf_path}")
    except Exception as err:
        print(f"Error saving EAF {output_eaf_path}: {err}"); traceback.print_exc()


if __name__ == "__main__":
    trained_model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
    if trained_model is None or scaler is None: print("Exiting: model/scaler loading failure."); exit()

    print("--- Trained 2-Class Model Summary for Inference ---")
    trained_model.summary(line_length=120)
    model_max_len_from_train = trained_model.input_shape[1]
    num_features_model_expected = trained_model.input_shape[2]
    num_classes_from_model_output = trained_model.output_shape[-1]
    if num_classes_from_model_output != NUM_CLASSES_MODEL:
        print(f"FATAL: Model output classes ({num_classes_from_model_output}) != expected ({NUM_CLASSES_MODEL})");
        exit()
    print(
        f"Model expects: max_len={model_max_len_from_train}, num_features={num_features_model_expected}, num_classes={NUM_CLASSES_MODEL}")

    try:
        df_sequence_unscaled, timedelta_index, original_filename_stem = get_sequence_for_inference(
            PKL_DATA_FOR_INFERENCE_FILE, SEQUENCE_TO_INFER_INDEX_IN_TEST_SET)
        print(
            f"Loaded '{original_filename_stem}': {df_sequence_unscaled.shape[0]} frames, {df_sequence_unscaled.shape[1]} raw features.")
        sequence_values_scaled = scale_data(df_sequence_unscaled.values, scaler)
        if sequence_values_scaled.shape[1] != num_features_model_expected:
            print(
                f"FATAL: Feature count mismatch. Scaled: {sequence_values_scaled.shape[1]}, Model: {num_features_model_expected}");
            exit()

        raw_predicted_labels, raw_predicted_probs = predict_sequence_simple(
            trained_model, sequence_values_scaled, model_max_len_from_train, NUM_CLASSES_MODEL)

        print("Raw 2-class predicted labels (first 50):", raw_predicted_labels[:50])
        unique_r, counts_r = np.unique(raw_predicted_labels, return_counts=True);
        print("Counts raw 2-class:", dict(zip(unique_r, counts_r)))

        print("Applying O/I post-processing...")
        final_predicted_labels = post_process_oi_sequence(
            raw_predicted_labels,
            raw_predicted_probs,  # Trecem și probs, chiar dacă nu sunt folosite activ acum
            min_i_duration_frames=MIN_I_FRAMES_FOR_POSTPROCESS
        )
        print("Final 2-class labels post-processed (first 50):", final_predicted_labels[:50])
        unique_f, counts_f = np.unique(final_predicted_labels, return_counts=True);
        print("Counts final 2-class:", dict(zip(unique_f, counts_f)))

        timedelta_index_us_values = timedelta_index.to_series().dt.total_seconds() * 1_000_000

        segments_unfiltered = oi_to_segments(final_predicted_labels, timedelta_index_us_values.values)
        print(f"Generated {len(segments_unfiltered)} unfiltered segments (O/I based).")

        segments_filtered_by_duration = []
        if segments_unfiltered:
            for start_ms, end_ms, value in segments_unfiltered:
                if (end_ms - start_ms) >= MIN_SEGMENT_DURATION_MS:
                    segments_filtered_by_duration.append((start_ms, end_ms, value))
        print(
            f"Retained {len(segments_filtered_by_duration)} segments after duration filter (min_duration={MIN_SEGMENT_DURATION_MS}ms).")

        if segments_filtered_by_duration:
            for idx, seg in enumerate(segments_filtered_by_duration[:5]): print(
                f"  Filt Seg {idx} (rel): S={seg[0]}ms, E={seg[1]}ms")
        else:
            print("  No segments after duration filter.")

        adjusted_segments_for_eaf = []
        trim_ms = int(TRIM_SECONDS_OFFSET * 1000)
        target_segments = segments_filtered_by_duration
        if trim_ms != 0:
            print(f"Applying trim offset of {trim_ms}ms to {len(target_segments)} segments.")
            for s, e, v in target_segments: adjusted_segments_for_eaf.append((s + trim_ms, e + trim_ms, v))
            if adjusted_segments_for_eaf:
                for idx, seg in enumerate(adjusted_segments_for_eaf[:2]): print(
                    f"  Adj Seg {idx}: S={seg[0]}ms, E={seg[1]}ms")
        else:
            print("No trim offset to apply. Using filtered segments as is.")
            adjusted_segments_for_eaf = target_segments

        media_fn = f"{original_filename_stem}_realsense.mp4"
        eaf_fn = f"{original_filename_stem}_pred_2class_trim{TRIM_SECONDS_OFFSET}s_min{MIN_SEGMENT_DURATION_MS}ms_minI{MIN_I_FRAMES_FOR_POSTPROCESS}f.eaf"  # Nume actualizat
        proj_root = Path.cwd();
        abs_vid_path = (proj_root / "videos" / media_fn).resolve();
        out_eaf_path = (proj_root / OUTPUT_EAF_DIR / eaf_fn).resolve()
        abs_vid_uri, rel_vid_path = None, None
        if abs_vid_path.exists():
            abs_vid_uri = abs_vid_path.as_uri()
            try:
                rel_vid_path = Path(os.path.relpath(abs_vid_path, out_eaf_path.parent)).as_posix()
            except ValueError:
                rel_vid_path = (Path("..") / "videos" / media_fn).as_posix()
        else:
            print(f"Warning: Video for EAF not found: {abs_vid_path}");
            rel_vid_path = (Path("..") / "videos" / media_fn).as_posix()

        create_eaf_file(out_eaf_path, adjusted_segments_for_eaf, abs_vid_uri, rel_vid_path)

    except Exception as e:
        print(f"An error occurred: {e}");
        traceback.print_exc()