import tensorflow as tf
import pickle
from pathlib import Path
import numpy as np
import pandas as pd


DFLT_LABEL_O_2CLASS, DFLT_LABEL_I_2CLASS = 0, 1

DFLT_MIN_I_FRAMES_FOR_POSTPROCESS = 5
DFLT_MIN_SEGMENT_DURATION_MS = 100
DFLT_TRIM_SECONDS_OFFSET_FOR_EAF_OUTPUT = 0.0

def scale_data_for_inference(data_array: np.ndarray, scaler):
    """Scalează datele de intrare folosind scaler-ul pre-antrenat."""
    if scaler is None:
        print("  INFER_LOGIC: Scaler is None, returning data as is.")
        return data_array
    if data_array.ndim == 1:
        data_array = data_array.reshape(1, -1)

    if data_array.shape[1] != scaler.n_features_in_:
        raise ValueError(
            f"Feature mismatch for scaling. Data has {data_array.shape[1]} features, "
            f"scaler was fit on {scaler.n_features_in_} features."
        )
    return scaler.transform(data_array)


def predict_sequence_for_inference(
        model,
        feature_sequence_scaled: np.ndarray,
        model_expected_max_len: int,
        padding_type: str = 'post',
        padding_value: float = 0.0
):
    """
    Rulează predicția pe o singură secvență de caracteristici scalate.
    Gestionează padding-ul dacă lungimea secvenței este diferită de cea așteptată de model.
    """
    actual_seq_len = feature_sequence_scaled.shape[0]
    num_features_actual = feature_sequence_scaled.shape[1]

    model_input_features = model.input_shape[-1]
    if num_features_actual != model_input_features:
        raise ValueError(
            f"Feature count mismatch for prediction. Input: {num_features_actual}, Model expects: {model_input_features}")

    padded_input_sequence = pad_sequences(
        [feature_sequence_scaled],
        maxlen=model_expected_max_len,
        padding=padding_type,
        truncating=padding_type,
        dtype='float32',
        value=padding_value
    )

    pred_probs_all_steps = model.predict(padded_input_sequence, verbose=0)
    len_for_output = min(actual_seq_len, model_expected_max_len)

    pred_probs_for_original_length = pred_probs_all_steps[0, :len_for_output, :]
    predicted_labels_for_original_length = np.argmax(pred_probs_for_original_length, axis=-1)

    if actual_seq_len > model_expected_max_len:
        print(
            f"  INFER_LOGIC: Warning - Actual sequence length ({actual_seq_len}) was truncated to model's max length ({model_expected_max_len}).")
        num_classes_output = pred_probs_for_original_length.shape[-1]

        final_labels = np.full(actual_seq_len, DFLT_LABEL_O_2CLASS, dtype=int)
        final_labels[:len_for_output] = predicted_labels_for_original_length

        final_probs = np.zeros((actual_seq_len, num_classes_output), dtype=np.float32)
        final_probs[:len_for_output, :] = pred_probs_for_original_length
        one_hot_o = tf.keras.utils.to_categorical([DFLT_LABEL_O_2CLASS], num_classes=num_classes_output)[0]
        final_probs[len_for_output:, :] = one_hot_o

        return final_labels, final_probs

    return predicted_labels_for_original_length, pred_probs_for_original_length


def post_process_oi_labels_functional(
        pred_labels_oi: np.ndarray,
        min_i_duration_frames: int = DFLT_MIN_I_FRAMES_FOR_POSTPROCESS,
        label_o_val: int = DFLT_LABEL_O_2CLASS,
        label_i_val: int = DFLT_LABEL_I_2CLASS
) -> np.ndarray:
    """Post-procesează o secvență de etichete O/I pentru a elimina segmentele 'I' prea scurte."""
    if len(pred_labels_oi) == 0: return pred_labels_oi

    corrected_seq = list(pred_labels_oi)
    n = len(corrected_seq)
    idx = 0
    while idx < n:
        if corrected_seq[idx] == label_i_val:
            start_i_idx = idx
            while idx < n and corrected_seq[idx] == label_i_val:
                idx += 1
            duration_i = idx - start_i_idx
            if duration_i < min_i_duration_frames:
                for k_fill_o in range(start_i_idx, idx):
                    corrected_seq[k_fill_o] = label_o_val
        else:
            idx += 1
    return np.array(corrected_seq, dtype=int)


def extract_segments_from_oi_labels(
        oi_labels: np.ndarray,
        timedelta_index_ms_values: np.ndarray,
        label_o_val: int = DFLT_LABEL_O_2CLASS,
        label_i_val: int = DFLT_LABEL_I_2CLASS,
        min_segment_duration_ms: int = DFLT_MIN_SEGMENT_DURATION_MS,
        trim_offset_ms: int = int(DFLT_TRIM_SECONDS_OFFSET_FOR_EAF_OUTPUT * 1000)
) -> list[dict]:
    """
    Convertește o secvență de etichete O/I în segmente temporale.
    Returnează o listă de dicționare: [{'start_ms': val, 'end_ms': val}, ...].
    """
    segments = []
    in_segment = False
    start_time_ms_current_segment = 0

    if len(oi_labels) != len(timedelta_index_ms_values):
        print(
            f"  INFER_LOGIC: ERROR - Length mismatch between labels ({len(oi_labels)}) and timestamps ({len(timedelta_index_ms_values)}). Cannot extract segments.")
        return []
    if len(oi_labels) == 0: return []

    for i, label in enumerate(oi_labels):
        current_time_ms_of_frame = int(timedelta_index_ms_values[i])

        if label == label_i_val:
            if not in_segment:
                start_time_ms_current_segment = current_time_ms_of_frame
                in_segment = True
        elif label == label_o_val:
            if in_segment:
                end_time_ms_current_segment = int(
                    timedelta_index_ms_values[i - 1]) if i > 0 else start_time_ms_current_segment

                if (end_time_ms_current_segment - start_time_ms_current_segment) >= min_segment_duration_ms:
                    segments.append({
                        "start_ms": start_time_ms_current_segment + trim_offset_ms,
                        "end_ms": end_time_ms_current_segment + trim_offset_ms
                    })
                in_segment = False

    if in_segment:
        end_time_ms_current_segment = int(timedelta_index_ms_values[-1])
        if (end_time_ms_current_segment - start_time_ms_current_segment) >= min_segment_duration_ms:
            segments.append({
                "start_ms": start_time_ms_current_segment + trim_offset_ms,
                "end_ms": end_time_ms_current_segment + trim_offset_ms
            })

    return segments


def run_inference_for_sequence_data(
        model_to_use,
        feature_sequence_scaled_np: np.ndarray,
        original_timedelta_index: pd.TimedeltaIndex,
        model_max_len: int,
        num_model_output_classes: int,
        min_i_frames_postprocess: int = DFLT_MIN_I_FRAMES_FOR_POSTPROCESS,
        min_segment_duration_ms_filter: int = DFLT_MIN_SEGMENT_DURATION_MS,
        label_o: int = DFLT_LABEL_O_2CLASS,
        label_i: int = DFLT_LABEL_I_2CLASS
) -> list[dict]:
    """
    Rulează întregul proces de inferență și extragere de segmente pentru o singură secvență.
    """
    if model_to_use is None:
        raise ValueError("Model not provided for inference.")
    if feature_sequence_scaled_np is None or feature_sequence_scaled_np.size == 0:
        raise ValueError("Feature sequence is empty or None.")
    if original_timedelta_index is None or original_timedelta_index.empty:
        raise ValueError("TimedeltaIndex is empty or None.")

    raw_predicted_labels, raw_predicted_probs = predict_sequence_for_inference(
        model_to_use,
        feature_sequence_scaled_np,
        model_expected_max_len=model_max_len,
    )

    final_predicted_labels = post_process_oi_labels_functional(
        raw_predicted_labels,
        min_i_duration_frames=min_i_frames_postprocess,
        label_o_val=label_o,
        label_i_val=label_i
    )

    num_labels_to_map = len(final_predicted_labels)
    if num_labels_to_map > len(original_timedelta_index):
        print(
            f"  INFER_LOGIC: Warning - More labels ({num_labels_to_map}) than timestamps ({len(original_timedelta_index)}). Truncating labels.")
        final_predicted_labels = final_predicted_labels[:len(original_timedelta_index)]

    relevant_timestamps_ms = (original_timedelta_index[:num_labels_to_map].total_seconds() * 1000).to_numpy()

    segments = extract_segments_from_oi_labels(
        final_predicted_labels,
        relevant_timestamps_ms,
        label_o_val=label_o,
        label_i_val=label_i,
        min_segment_duration_ms=min_segment_duration_ms_filter,
        trim_offset_ms=0
    )
    return segments


if __name__ == "__main__":
    print("--- Running Inference Logic (Standalone Test Mode) ---")

    test_model_path = Path("../trained_models_final/bilstm_best_final.keras")
    test_scaler_path = Path("../final_combined_data_for_training_ALL_SIGNERS/final_features_ts_facial_scaler.pkl")
    test_data_pkl_path = Path("../final_combined_data_for_training_ALL_SIGNERS/all_data_final_features_ts_facial.pkl")
    test_seq_index = 0

    if not all([test_model_path.exists(), test_scaler_path.exists(), test_data_pkl_path.exists()]):
        print("Error: One or more files for standalone test are missing. Adjust paths.")
        print(f"Model: {test_model_path.resolve()} (Exists: {test_model_path.exists()})")
        print(f"Scaler: {test_scaler_path.resolve()} (Exists: {test_scaler_path.exists()})")
        print(f"Data PKL: {test_data_pkl_path.resolve()} (Exists: {test_data_pkl_path.exists()})")
        exit()

    print(f"Standalone: Loading model from {test_model_path}")
    test_model = tf.keras.models.load_model(test_model_path, compile=False)
    print(f"Standalone: Loading scaler from {test_scaler_path}")
    with open(test_scaler_path, 'rb') as f:
        test_scaler = pickle.load(f)

    print(f"Standalone: Loading data from {test_data_pkl_path}")
    with open(test_data_pkl_path, 'rb') as f:
        test_data_dict = pickle.load(f)

    if 'X_test_df_indexed' not in test_data_dict or not test_data_dict['X_test_df_indexed'] or \
            test_seq_index >= len(test_data_dict['X_test_df_indexed']):
        print("Error: Cannot get test sequence from PKL for standalone test.")
        exit()

    df_seq_unscaled_test = test_data_dict['X_test_df_indexed'][test_seq_index]
    original_td_index_test = df_seq_unscaled_test.index
    feature_names_test = test_data_dict.get('feature_names', list(df_seq_unscaled_test.columns))

    if df_seq_unscaled_test.empty: print("Error: Test sequence is empty."); exit()

    print(f"Standalone: Using test sequence {test_seq_index} with shape {df_seq_unscaled_test.shape}")

    df_seq_reordered_test = df_seq_unscaled_test.reindex(columns=feature_names_test, fill_value=0.0)

    data_to_scale_test = df_seq_reordered_test.values
    scaled_sequence_test = scale_data_for_inference(data_to_scale_test, test_scaler)

    model_input_len_test = test_model.input_shape[1]
    model_output_classes_test = test_model.output_shape[-1]

    if model_output_classes_test != 2 and model_output_classes_test != 3:
        print(
            f"Warning: Model output classes ({model_output_classes_test}) might not match expected O/I (2) or O/B/I (3).")

    print("\n--- Running Standalone Inference ---")
    final_segments_test = run_inference_for_sequence_data(
        model_to_use=test_model,
        feature_sequence_scaled_np=scaled_sequence_test,
        original_timedelta_index=original_td_index_test,
        model_max_len=model_input_len_test,
        num_model_output_classes=model_output_classes_test,
        min_i_frames_postprocess=DFLT_MIN_I_FRAMES_FOR_POSTPROCESS,
        min_segment_duration_ms_filter=DFLT_MIN_SEGMENT_DURATION_MS,
        label_o=DFLT_LABEL_O_2CLASS,
        label_i=DFLT_LABEL_I_2CLASS
    )

    print(f"\n--- Standalone Inference Results (first 5 segments) ---")
    if final_segments_test:
        for idx, seg in enumerate(final_segments_test[:5]):
            print(f"  Segment {idx}: Start={seg['start_ms']}ms, End={seg['end_ms']}ms")
    else:
        print("  No segments found.")

    print("\n--- Standalone Inference Logic Test Finished ---")