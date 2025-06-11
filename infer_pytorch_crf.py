# --- START OF FILE infer_pytorch_crf.py ---
import torch
import torch.nn as nn  # Necesar pentru a defini arhitectura modelului la încărcare
from torchcrf import CRF  # Necesar pentru a defini arhitectura modelului la încărcare
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence  # Necesar pentru model

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from pympi.Elan import Eaf
import traceback
import os

# --- Configurații ---
# !!! ACTUALIZEAZĂ ACEASTĂ CALE la modelul .pt antrenat cu CRF !!!
MODEL_PT_PATH = Path("./trained_models_final_pytorch_optimized_crf/bigru_optcrf_best.pt")  # Sau bilstm_optcrf_best.pt
MODEL_TYPE_FOR_INFERENCE = 'gru'  # Setează 'lstm' sau 'gru' corespunzător modelului încărcat
RNN_UNITS_FOR_INFERENCE = 128  # Setează LSTM_UNITS sau GRU_UNITS din scriptul de antrenament (ex: 128)

SCALER_PATH = Path("./final_combined_data_for_training_ALL_SIGNERS/final_features_ts_facial_scaler.pkl")
PKL_DATA_FOR_INFERENCE_DIR = Path("./final_combined_data_for_training_ALL_SIGNERS")
PKL_DATA_FOR_INFERENCE_FILE = PKL_DATA_FOR_INFERENCE_DIR / "all_data_final_features_ts_facial.pkl"

SEQUENCE_TO_INFER_INDEX_IN_TEST_SET = 0

# !!! CONSTANTA PENTRU TRUNCHIERE !!!
TRIM_SECONDS_OFFSET = 0.3  # Setează valoarea corectă (din output-ul anterior era 0.3s)

OUTPUT_EAF_DIR = Path("./inference_output_pytorch_crf")
OUTPUT_EAF_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 3  # O, B, I
LABEL_O, LABEL_B, LABEL_I = 0, 1, 2
NUM_FEATURES_EXPECTED = 253  # Din output-ul anterior (fără eticheta anterioară)

# Necesar pentru a reconstrui arhitectura modelului înainte de a încărca state_dict
# Asigură-te că acești parametri sunt aceiași cu cei folosiţi la antrenament
# pentru modelul specific pe care îl încarci (LSTM_UNITS sau GRU_UNITS, DROPOUT_RATE)
LSTM_UNITS = 128  # Valoare din scriptul tău de antrenament
GRU_UNITS = 128  # Valoare din scriptul tău de antrenament
DROPOUT_RATE_MODEL = 0.4  # Valoare din scriptul tău de antrenament

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- Definirea Arhitecturii Modelului (trebuie să fie identică cu cea din train_pytorch_optimized_crf.py) ---
class RecurrentModelWithCRF(nn.Module):
    def __init__(self, input_dim, hidden_dim_rnn_output, num_actual_classes, recurrent_type='lstm', dropout_rate=0.3):
        super(RecurrentModelWithCRF, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_rnn_output = hidden_dim_rnn_output
        self.num_actual_classes = num_actual_classes

        RecurrentLayer = nn.LSTM if recurrent_type == 'lstm' else nn.GRU
        self.rnn = RecurrentLayer(input_dim, hidden_dim_rnn_output // 2,
                                  num_layers=1, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(dropout_rate)
        self.hidden2tag = nn.Linear(hidden_dim_rnn_output, self.num_actual_classes)
        self.crf = CRF(num_tags=self.num_actual_classes, batch_first=True)

    def _get_rnn_features(self, sentences, lengths):
        # lengths tensor needs to be on CPU for pack_padded_sequence
        # If lengths is already a tensor on CPU, .cpu() is a no-op. If on GPU, it's moved.
        packed_input = pack_padded_sequence(sentences, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_input)
        rnn_out, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=sentences.size(1))
        rnn_out = self.dropout(rnn_out)  # Aplică dropout și la inferență DUPĂ RNN, înainte de CRF emissions
        # Deși nn.Dropout e inactiv în .eval(), e bine să fie consistent cu antrenamentul.
        # Dacă vrei să fie activ, nu apela model.eval() înainte de predict.
        # Dar standard e să fie inactiv.
        emission_scores = self.hidden2tag(rnn_out)
        return emission_scores

    def predict(self, sentences, lengths, mask):  # mask-ul poate fi derivat din lengths dacă nu e dat
        self.eval()  # Setează modelul în modul de evaluare (dezactivează dropout, batchnorm updates etc.)
        with torch.no_grad():
            emission_scores = self._get_rnn_features(sentences, lengths)
            if mask is None:  # Creează masca dacă nu e furnizată
                mask = torch.arange(sentences.size(1), device=sentences.device)[None, :] < lengths[:, None]
            predicted_paths = self.crf.decode(emissions=emission_scores, mask=mask)
        return predicted_paths

    # compute_loss nu este necesar pentru inferență


# --- Funcții Ajutătoare ---
def load_pytorch_model_and_scaler(model_path, model_type, rnn_units_param, num_features, num_classes_param,
                                  dropout_rate_param, scaler_path_param, device_param):
    print(f"Loading PyTorch CRF model state_dict from: {model_path}")

    # Determină hidden_dim_rnn_output bazat pe rnn_units (deoarece e bidirecțional)
    hidden_dim_rnn_output = rnn_units_param * 2

    # Recreează arhitectura modelului
    model = RecurrentModelWithCRF(
        input_dim=num_features,
        hidden_dim_rnn_output=hidden_dim_rnn_output,
        num_actual_classes=num_classes_param,
        recurrent_type=model_type,
        dropout_rate=dropout_rate_param
    )
    model.load_state_dict(torch.load(model_path, map_location=device_param))
    model.to(device_param)
    model.eval()  # Important: setează modelul în modul de evaluare

    scaler = None
    if scaler_path_param and scaler_path_param.exists():
        print(f"Loading scaler from: {scaler_path_param}")
        with open(scaler_path_param, 'rb') as f:
            scaler = pickle.load(f)
    return model, scaler


def get_sequence_for_inference(pkl_file_path, seq_index_in_test):
    # ... (identic cu versiunea anterioară)
    print(f"Loading sequence for inference from: {pkl_file_path}, test index: {seq_index_in_test}")
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)
    if 'X_test_df_indexed' not in data or not data['X_test_df_indexed']:
        raise ValueError("X_test_df_indexed not found or empty in PKL.")
    if seq_index_in_test >= len(data['X_test_df_indexed']):
        raise ValueError(
            f"Sequence index {seq_index_in_test} out of bounds for X_test_df_indexed (len: {len(data['X_test_df_indexed'])}).")
    df_sequence_unscaled = data['X_test_df_indexed'][seq_index_in_test]
    full_id_dict = data['test_ids'][seq_index_in_test] if 'test_ids' in data and len(
        data['test_ids']) > seq_index_in_test else {}
    original_filename_stem_from_id = full_id_dict.get('filename', f"unknown_sequence_{seq_index_in_test}")
    original_filename_stem = Path(original_filename_stem_from_id).stem.split('.')[0]
    if df_sequence_unscaled is None or df_sequence_unscaled.empty:
        raise ValueError(f"Selected sequence (test index {seq_index_in_test}) is None or empty.")
    return df_sequence_unscaled, df_sequence_unscaled.index, original_filename_stem


def scale_data(data_array, scaler):
    # ... (identic)
    if scaler is None: return data_array
    if data_array.ndim == 1: data_array = data_array.reshape(1, -1)
    if data_array.shape[1] != scaler.n_features_in_:
        raise ValueError(
            f"Number of features in data_array ({data_array.shape[1]}) does not match scaler's expected features ({scaler.n_features_in_})")
    return scaler.transform(data_array)


def pad_sequence_for_pytorch_inference(sequence_np, max_len, padding_value=0.0):
    # sequence_np este (seq_actual_len, num_features)
    actual_len = sequence_np.shape[0]
    num_features = sequence_np.shape[1]

    padded_sequence = np.full((max_len, num_features), padding_value, dtype='float32')
    len_to_copy = min(actual_len, max_len)
    padded_sequence[:len_to_copy, :] = sequence_np[:len_to_copy, :]

    return padded_sequence, actual_len


def bio_to_segments(bio_labels, timedelta_index_us_values, min_segment_duration_ms=100):  # Adăugat filtru
    # ... (identic, dar cu filtrul de durată minimă adăugat sau aplicat după)
    segments = []
    in_segment = False
    start_time_ms = 0
    if len(bio_labels) != len(timedelta_index_us_values):
        print(
            f"Warning: bio_labels length ({len(bio_labels)}) vs timedelta_index_us_values length ({len(timedelta_index_us_values)}) mismatch. Truncating.")
        min_len = min(len(bio_labels), len(timedelta_index_us_values))
        bio_labels, timedelta_index_us_values = bio_labels[:min_len], timedelta_index_us_values[:min_len]
        if min_len == 0: return segments

    raw_segments = []  # Segmente înainte de filtrare
    for i, label in enumerate(bio_labels):
        current_time_ms = int(timedelta_index_us_values[i] / 1000)
        if label == LABEL_B:
            if in_segment:
                end_time_prev_segment_ms = int(timedelta_index_us_values[i - 1] / 1000) if i > 0 else start_time_ms
                if end_time_prev_segment_ms > start_time_ms:
                    raw_segments.append((start_time_ms, end_time_prev_segment_ms, "SIGN"))
            start_time_ms = current_time_ms
            in_segment = True
        elif label == LABEL_I:
            if not in_segment: start_time_ms = current_time_ms; in_segment = True
        elif label == LABEL_O:
            if in_segment:
                end_time_current_segment_ms = int(timedelta_index_us_values[i - 1] / 1000) if i > 0 else start_time_ms
                if end_time_current_segment_ms > start_time_ms:
                    raw_segments.append((start_time_ms, end_time_current_segment_ms, "SIGN"))
                in_segment = False
    if in_segment:
        end_time_final_segment_ms = int(timedelta_index_us_values[-1] / 1000) if len(
            timedelta_index_us_values) > 0 else start_time_ms
        if end_time_final_segment_ms > start_time_ms:
            raw_segments.append((start_time_ms, end_time_final_segment_ms, "SIGN"))

    # Filtrare segmente scurte
    for start_ms, end_ms, value in raw_segments:
        if (end_ms - start_ms) >= min_segment_duration_ms:
            segments.append((start_ms, end_ms, value))
        # else:
        # print(f"Filtered out short segment: {start_ms}-{end_ms} (duration {end_ms-start_ms}ms)")
    return segments


def create_eaf_file(output_eaf_path, segments, absolute_media_uri=None, relative_media_path_for_storage=None):
    # ... (identic)
    eafob = Eaf(author="InferenceScriptPyTorchCRF")
    tier_id = "PredictedSignsCRF"
    eafob.add_tier(tier_id)
    if "default-lt" not in eafob.linguistic_types: eafob.add_linguistic_type("default-lt", timealignable=True)
    if tier_id in eafob.tiers and (eafob.tiers[tier_id][2] is None or eafob.tiers[tier_id][2] == ''):
        eafob.tiers[tier_id] = (eafob.tiers[tier_id][0], eafob.tiers[tier_id][1], "default-lt", eafob.tiers[tier_id][3])
    if absolute_media_uri and relative_media_path_for_storage:
        mime_type = "video/mp4"
        if ".wav" in relative_media_path_for_storage.lower(): mime_type = "audio/x-wav"
        eafob.add_linked_file(file_path=absolute_media_uri, relpath=relative_media_path_for_storage, mimetype=mime_type)
    elif relative_media_path_for_storage:
        mime_type = "video/mp4"
        if ".wav" in relative_media_path_for_storage.lower(): mime_type = "audio/x-wav"
        eafob.add_linked_file(file_path=relative_media_path_for_storage, relpath=relative_media_path_for_storage,
                              mimetype=mime_type)
    else:
        print("No media path provided.")
    for start_ms, end_ms, annotation_value in segments:
        if end_ms <= start_ms: continue
        try:
            eafob.add_annotation(tier_id, int(start_ms), int(end_ms), value=annotation_value)
        except Exception as e:
            print(f"Error adding annotation ({start_ms}-{end_ms}, {annotation_value}): {e}")
    try:
        Path(output_eaf_path).parent.mkdir(parents=True, exist_ok=True); eafob.to_file(str(output_eaf_path))
    except Exception as e:
        print(f"Error saving EAF file {output_eaf_path}: {e}")
    print(f"EAF file saved to: {output_eaf_path}")


# --- Main Execution ---
if __name__ == "__main__":
    # Determină rnn_units pe baza constantei MODEL_TYPE_FOR_INFERENCE
    rnn_units = GRU_UNITS if MODEL_TYPE_FOR_INFERENCE == 'gru' else LSTM_UNITS

    model, scaler = load_pytorch_model_and_scaler(
        MODEL_PT_PATH,
        MODEL_TYPE_FOR_INFERENCE,
        rnn_units,  # Acesta este units per direcție pentru RNN intern
        NUM_FEATURES_EXPECTED,  # Numărul de caracteristici de input
        NUM_CLASSES,
        DROPOUT_RATE_MODEL,  # Rata de dropout folosită la antrenament
        SCALER_PATH,
        device
    )

    if model is None: print("Exiting due to model loading failure."); exit()
    if scaler is None: print("Error: Scaler not loaded. Exiting."); exit()

    print(f"--- PyTorch CRF Model for Inference ({MODEL_TYPE_FOR_INFERENCE.upper()}) ---")
    # print(model) # Poți decomenta pentru a vedea structura

    try:
        df_sequence_unscaled, timedelta_index, original_filename_stem = get_sequence_for_inference(
            PKL_DATA_FOR_INFERENCE_FILE,
            SEQUENCE_TO_INFER_INDEX_IN_TEST_SET
        )
        print(f"Loaded sequence for '{original_filename_stem}' with {df_sequence_unscaled.shape[0]} frames.")

        sequence_values_scaled = scale_data(df_sequence_unscaled.values, scaler)
        if sequence_values_scaled.shape[1] != NUM_FEATURES_EXPECTED:
            print(
                f"FATAL: Feature count mismatch. Scaled: {sequence_values_scaled.shape[1]}, Expected: {NUM_FEATURES_EXPECTED}")
            exit()

        # Pentru inferență cu PyTorch, trebuie să avem un max_len.
        # Ideal, ar fi MAX_LEN_CALCULATED din scriptul de antrenament.
        # Dacă nu îl avem, putem folosi lungimea secvenței curente sau o valoare mare.
        # Dar modelul a fost antrenat cu un max_len specific pentru padding.
        # Să presupunem că MAX_LEN_CALCULATED este cunoscut (ex: din config sau log-ul de antrenament)
        # Pentru acest exemplu, îl vom hardcoda sau îl vom lua ca lungimea secvenței curente dacă e mai mic.
        # Cel mai bine e să-l salvezi/încarci împreună cu modelul.
        # Din outputul tău anterior: MAX_LEN_CALCULATED = 2997
        max_len_for_padding = 2997  # Folosește valoarea cu care s-a antrenat modelul CRF

        sequence_padded_np, actual_length = pad_sequence_for_pytorch_inference(
            sequence_values_scaled,
            max_len=max_len_for_padding
        )

        # Convert to PyTorch tensors
        features_tensor = torch.tensor(sequence_padded_np, dtype=torch.float32).unsqueeze(0).to(
            device)  # (1, max_len, num_features)
        lengths_tensor = torch.tensor([actual_length], dtype=torch.long).to(device)  # (1,)
        # Masca este creată în interiorul model.predict dacă este None

        print(f"Predicting with input shape: {features_tensor.shape}, actual length: {actual_length}")

        # Predicție - model.predict returnează o listă de liste
        predicted_paths_batch = model.predict(features_tensor, lengths_tensor, mask=None)
        predicted_labels_crf = np.array(predicted_paths_batch[0])[
                               :actual_length]  # Luăm prima (și singura) secvență și o tăiem la lungimea reală

        print("CRF Predicted labels (first 50):", predicted_labels_crf[:50])
        unique_crf, counts_crf = np.unique(predicted_labels_crf, return_counts=True)
        print("Counts of CRF predicted labels:", dict(zip(unique_crf, counts_crf)))

        # Post-procesarea euristică poate fi încă utilă, deși CRF ar trebui să ajute mult.
        # Pentru CRF, s-ar putea să nu mai ai nevoie de `post_process_bio_sequence_with_probs`
        # ci de o funcție mai simplă de corectare a unor pattern-uri B-I-O dacă CRF-ul nu le-a rezolvat perfect.
        # Momentan, o vom omite pentru a vedea output-ul pur al CRF.
        # Dacă vrei să o aplici, ar trebui adaptată pentru a nu mai folosi probabilități softmax,
        # ci doar etichetele decodate de CRF.

        final_predicted_labels = predicted_labels_crf  # Folosim direct output-ul CRF
        print("Using direct CRF output for segmentation.")

        timedelta_index_us_values = timedelta_index.to_series().dt.total_seconds() * 1_000_000
        # Aplicăm filtrul de durată minimă aici
        MIN_SEGMENT_DURATION_MS_INFER = 200  # Ajustează după nevoie
        segments = bio_to_segments(final_predicted_labels, timedelta_index_us_values.values,
                                   min_segment_duration_ms=MIN_SEGMENT_DURATION_MS_INFER)

        print(f"Generated {len(segments)} segments (after CRF and min duration filter).")
        if segments:
            for seg_idx, seg in enumerate(segments[:5]):
                print(f"  Segment {seg_idx}: Start={seg[0]}ms, End={seg[1]}ms, Label='{seg[2]}'")
        else:
            print("  No segments generated.")

        adjusted_segments_for_eaf = []
        trim_ms_offset_global = int(TRIM_SECONDS_OFFSET * 1000)
        if trim_ms_offset_global > 0:
            print(f"Applying global trim offset of {trim_ms_offset_global}ms.")
            for start_ms, end_ms, value in segments:
                adjusted_segments_for_eaf.append(
                    (start_ms + trim_ms_offset_global, end_ms + trim_ms_offset_global, value))
        else:
            adjusted_segments_for_eaf = segments

        media_file_name_guess = f"{original_filename_stem}_realsense.mp4"
        eaf_file_name = f"{original_filename_stem}_predicted_pytorch_crf_offset_{TRIM_SECONDS_OFFSET}s.eaf"
        project_root = Path.cwd()
        absolute_video_path = (project_root / "videos" / media_file_name_guess).resolve()
        output_eaf_full_path = (project_root / OUTPUT_EAF_DIR / eaf_file_name).resolve()
        # ... (restul logicii pentru căi media și salvare EAF)
        absolute_video_uri_str = None
        relative_path_str_for_storage = None
        if absolute_video_path.exists():
            absolute_video_uri_str = absolute_video_path.as_uri()
            try:
                relative_path_os = os.path.relpath(absolute_video_path, output_eaf_full_path.parent)
                relative_path_str_for_storage = Path(relative_path_os).as_posix()
            except ValueError:
                relative_path_str_for_storage = (Path("..") / "videos" / media_file_name_guess).as_posix()
        else:
            print(f"Warning: Video path for EAF does not exist: {absolute_video_path}")
            relative_path_str_for_storage = (Path("..") / "videos" / media_file_name_guess).as_posix()

        create_eaf_file(output_eaf_full_path, adjusted_segments_for_eaf,
                        absolute_media_uri=absolute_video_uri_str,
                        relative_media_path_for_storage=relative_path_str_for_storage)

    except Exception as e:
        print(f"An error occurred during Pytorch CRF inference: {e}")
        traceback.print_exc()

# --- END OF FILE infer_pytorch_crf.py ---