import tensorflow as tf
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import traceback
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.utils import class_weight

# --- Configuration ---
FINAL_DATA_DIR = Path("./final_combined_data_for_training_ALL_SIGNERS")
FINAL_FEATURES_DATA_FILE = FINAL_DATA_DIR / "all_data_final_features_ts_facial.pkl"
FINAL_ANNOTATION_FILE = FINAL_DATA_DIR / "annotations_bio_final_combined.pkl"
MODEL_SAVE_DIR = Path("./trained_models_final_prev_label")  # Director nou
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

LSTM_UNITS = 128
GRU_UNITS = 128
DROPOUT_RATE = 0.4
L2_REG_FACTOR = 0.0
NUM_CLASSES = 3
LABEL_O, LABEL_B, LABEL_I = 0, 1, 2
BATCH_SIZE = 8
EPOCHS = 50
PATIENCE = 15
PADDING_TYPE = 'post'
MAX_LEN_CALCULATED = None
PROB_THRESHOLD_B_WHEN_OI = 0.3
PROB_THRESHOLD_I_WHEN_BO = 0.4

print(f"TensorFlow Version: {tf.__version__}")
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"Num GPUs Available: {len(gpu_devices)}")
    for gpu in gpu_devices: print(f"GPU: {gpu}")
    try:
        for gpu in gpu_devices: tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth set to True.")
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")
else:
    print("No GPU found by TensorFlow. Training will use CPU.")


# --- Funcție pentru augmentarea datelor cu eticheta anterioară ---
def augment_with_previous_label(X_sequences_raw, y_sequences_raw, num_classes):
    X_augmented = []
    if not X_sequences_raw or not y_sequences_raw:
        return X_augmented

    num_original_features = X_sequences_raw[0].shape[1]

    for seq_idx in range(len(X_sequences_raw)):
        x_seq = X_sequences_raw[seq_idx]
        y_seq_labels = y_sequences_raw[seq_idx]  # Etichete 0, 1, 2

        if x_seq.shape[0] == 0:  # Secvență goală de caracteristici
            X_augmented.append(np.empty((0, num_original_features + num_classes), dtype=x_seq.dtype))
            continue

        new_x_frames_for_seq = np.zeros((x_seq.shape[0], num_original_features + num_classes), dtype=x_seq.dtype)

        for t in range(x_seq.shape[0]):
            current_features = x_seq[t, :]
            if t == 0:
                prev_label_one_hot = np.zeros(num_classes,
                                              dtype=x_seq.dtype)  # Placeholder pentru primul cadru (ex: clasa O)
                # Sau un placeholder distinct, de ex. un vector cu toți 1/num_classes
                # prev_label_one_hot[LABEL_O] = 1.0 # Presupunem că începe cu O dacă nu știm
            else:
                prev_label_one_hot = tf.keras.utils.to_categorical(y_seq_labels[t - 1], num_classes=num_classes)

            new_x_frames_for_seq[t, :num_original_features] = current_features
            new_x_frames_for_seq[t, num_original_features:] = prev_label_one_hot

        X_augmented.append(new_x_frames_for_seq)
    return X_augmented


# --- Load Data ---
print(f"Loading final processed features from: {FINAL_FEATURES_DATA_FILE}")
try:
    with open(FINAL_FEATURES_DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    X_train_raw_orig, X_val_raw_orig, X_test_raw_orig = data['X_train'], data['X_val'], data['X_test']
    feature_names = data.get('feature_names', [])
    if not X_train_raw_orig: print(f"Error: X_train is empty in {FINAL_FEATURES_DATA_FILE}"); exit()

    NUM_ORIG_FEATURES = X_train_raw_orig[0].shape[1] if X_train_raw_orig and X_train_raw_orig[0] is not None and len(
        X_train_raw_orig[0].shape) > 1 else 0
    if NUM_ORIG_FEATURES == 0 and X_train_raw_orig:
        for seq in X_train_raw_orig:
            if seq is not None and len(seq.shape) > 1 and seq.shape[1] > 0: NUM_ORIG_FEATURES = seq.shape[1]; break
    if NUM_ORIG_FEATURES == 0: print("Error: Could not determine number of original features."); exit()

    print(f"Found {len(X_train_raw_orig)} train, {len(X_val_raw_orig)} val, {len(X_test_raw_orig)} test sequences.")
    print(f"Number of original features per frame: {NUM_ORIG_FEATURES}")
except Exception as e:
    print(f"Error loading feature data: {e}"); traceback.print_exc(); exit()

print(f"Loading final annotations from: {FINAL_ANNOTATION_FILE}")
try:
    with open(FINAL_ANNOTATION_FILE, 'rb') as f:
        annotations = pickle.load(f)
    y_train_raw, y_val_raw, y_test_raw = annotations['train'], annotations['val'], annotations['test']
    # Validările de lungime se fac DUPĂ augmentare acum, pentru X_train_raw etc.
    print("Annotations loaded.")
except Exception as e:
    print(f"Error loading annotation data: {e}"); traceback.print_exc(); exit()

# --- Augment Data with Previous Label ---
print("Augmenting data with previous label feature...")
X_train_raw = augment_with_previous_label(X_train_raw_orig, y_train_raw, NUM_CLASSES)
X_val_raw = augment_with_previous_label(X_val_raw_orig, y_val_raw, NUM_CLASSES)
# Pentru X_test, folosim y_test_raw pentru augmentare (teacher forcing simulat pentru evaluare)
# ÎNTR-UN SCENARIU REAL DE INFERENȚĂ, AICI AR TREBUI SĂ FIE PREDICȚIA MODELULUI PENTRU CADRUL ANTERIOR
X_test_raw = augment_with_previous_label(X_test_raw_orig, y_test_raw, NUM_CLASSES)

if not X_train_raw: print("Error: X_train_raw became empty after augmentation."); exit()
NUM_FEATURES = X_train_raw[0].shape[1]  # Noul număr de caracteristici
print(f"Number of features after augmentation: {NUM_FEATURES}")

# Validare lungimi după augmentare (X_raw e lista de array-uri augmentate)
try:
    for name, X, y in [("Train", X_train_raw, y_train_raw), ("Val", X_val_raw, y_val_raw),
                       ("Test", X_test_raw, y_test_raw)]:
        if len(X) != len(y): raise AssertionError(
            f"{name} X length ({len(X)}) != Y length ({len(y)}) post-augmentation")
        for i in range(len(X)):
            if X[i].shape[0] != y[i].shape[0]: raise AssertionError(
                f"{name} seq {i}: X len {X[i].shape[0]} != Y len {y[i].shape[0]} post-augmentation")
    print("Data lengths validated post-augmentation.")
except Exception as e:
    print(f"Error during post-augmentation validation: {e}"); traceback.print_exc(); exit()

all_sequence_lengths = [seq.shape[0] for dataset in [X_train_raw, X_val_raw, X_test_raw] for seq in dataset if
                        seq is not None and seq.shape[0] > 0]
if not all_sequence_lengths: print("Error: No sequences (post-augmentation) to determine max length."); exit()
MAX_LEN_CALCULATED = np.max(all_sequence_lengths)
print(f"Determined MAX_LEN from augmented data: {MAX_LEN_CALCULATED}")

print("Padding sequences...")
X_train_padded = tf.keras.preprocessing.sequence.pad_sequences(X_train_raw, padding=PADDING_TYPE, dtype='float32',
                                                               maxlen=MAX_LEN_CALCULATED, value=0.0)
X_val_padded = tf.keras.preprocessing.sequence.pad_sequences(X_val_raw, padding=PADDING_TYPE, dtype='float32',
                                                             maxlen=MAX_LEN_CALCULATED, value=0.0)
X_test_padded = tf.keras.preprocessing.sequence.pad_sequences(X_test_raw, padding=PADDING_TYPE, dtype='float32',
                                                              maxlen=MAX_LEN_CALCULATED, value=0.0)
y_train_padded = tf.keras.preprocessing.sequence.pad_sequences(y_train_raw, padding=PADDING_TYPE, value=LABEL_O,
                                                               maxlen=MAX_LEN_CALCULATED)
y_val_padded = tf.keras.preprocessing.sequence.pad_sequences(y_val_raw, padding=PADDING_TYPE, value=LABEL_O,
                                                             maxlen=MAX_LEN_CALCULATED)
y_test_padded = tf.keras.preprocessing.sequence.pad_sequences(y_test_raw, padding=PADDING_TYPE, value=LABEL_O,
                                                              maxlen=MAX_LEN_CALCULATED)
effective_max_len = MAX_LEN_CALCULATED
print(f"Sequences padded to length: {effective_max_len}")

print("One-hot encoding labels...")
y_train_one_hot = tf.keras.utils.to_categorical(y_train_padded.astype(int), num_classes=NUM_CLASSES)
y_val_one_hot = tf.keras.utils.to_categorical(y_val_padded.astype(int), num_classes=NUM_CLASSES)
y_test_one_hot = tf.keras.utils.to_categorical(y_test_padded.astype(int), num_classes=NUM_CLASSES)
print(f"Shape X_train_padded: {X_train_padded.shape}, y_train_one_hot: {y_train_one_hot.shape}")

y_train_labels_flat_for_weights = []
for i in range(len(y_train_raw)):
    true_seq_len = len(y_train_raw[i])
    y_train_labels_flat_for_weights.extend(y_train_padded[i, :true_seq_len])

class_weights_dict = None
sample_weights_train = None
if y_train_labels_flat_for_weights:
    unique_classes_in_train = np.unique(y_train_labels_flat_for_weights)
    print(f"Unique classes found in training labels for weight calculation: {unique_classes_in_train}")
    expected_classes = np.arange(NUM_CLASSES)
    if not np.array_equal(np.sort(unique_classes_in_train), expected_classes) and len(
            unique_classes_in_train) < NUM_CLASSES:
        print(
            f"Warning: Not all {NUM_CLASSES} classes are present in training labels. Found {unique_classes_in_train}.")

    class_weights_values = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=unique_classes_in_train,
        y=y_train_labels_flat_for_weights
    )
    temp_class_weights_dict = {unique_classes_in_train[i]: class_weights_values[i] for i in
                               range(len(unique_classes_in_train))}
    class_weights_dict = {cls: temp_class_weights_dict.get(cls, 1.0) for cls in range(NUM_CLASSES)}

    target_b_weight = 30.0
    if LABEL_B in class_weights_dict:
        print(f"Original B-weight from 'balanced': {class_weights_dict[LABEL_B]}")
        class_weights_dict[LABEL_B] = target_b_weight
    print(f"ADJUSTED Class Weights: {class_weights_dict}")

    sample_weights_train = np.ones_like(y_train_padded, dtype=float)
    for i in range(y_train_padded.shape[0]):
        original_length = len(y_train_raw[i])
        for j in range(original_length):
            label = y_train_padded[i, j]
            sample_weights_train[i, j] = class_weights_dict.get(label, 1.0)
        if original_length < MAX_LEN_CALCULATED:
            sample_weights_train[i, original_length:] = 0.0
    print(f"Shape of sample_weights_train: {sample_weights_train.shape}")
else:
    print("Warning: Could not calculate class weights.")


def build_recurrent_model(model_type='lstm', units=128, input_shape=None, num_classes=3, dropout_rate=0.3, l2_reg=0.0):
    # ... (Funcția build_recurrent_model rămâne la fel) ...
    if input_shape is None: raise ValueError("input_shape must be provided")
    input_layer = tf.keras.layers.Input(shape=input_shape, name="Input_Layer")
    masked_input = tf.keras.layers.Masking(mask_value=0.0, name="Masking_Layer")(
        input_layer)  # Presupunând că padding-ul pentru caracteristici e 0.0
    regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None

    if model_type == 'lstm':
        recurrent_output = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units, return_sequences=True,
                                 kernel_regularizer=regularizer,
                                 recurrent_regularizer=regularizer),
            name="BiLSTM_Layer"
        )(masked_input)
    elif model_type == 'gru':
        recurrent_output = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(units, return_sequences=True,
                                kernel_regularizer=regularizer,
                                recurrent_regularizer=regularizer),
            name="BiGRU_Layer"
        )(masked_input)
    else:
        raise ValueError("model_type must be 'lstm' or 'gru'")
    dropout_output = tf.keras.layers.Dropout(dropout_rate, name="Dropout_Layer")(recurrent_output)
    output_layer = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=regularizer),
        name="Output_Layer"
    )(dropout_output)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model


def check_bio_sequence_validity(sequence):
    # ... (Funcția check_bio_sequence_validity rămâne la fel) ...
    if not isinstance(sequence, (list, np.ndarray)) or len(sequence) == 0: return True
    seq = [int(s) for s in sequence];
    n = len(seq)
    first_non_o_idx = -1
    for i in range(n):
        if seq[i] != LABEL_O: first_non_o_idx = i; break
    if first_non_o_idx != -1 and seq[first_non_o_idx] == LABEL_I: return False
    for i in range(n):
        curr, prev = seq[i], (seq[i - 1] if i > 0 else None)
        nxt = seq[i + 1] if i < n - 1 else None
        if curr == LABEL_I and (prev is None or prev == LABEL_O): return False
        if curr == LABEL_B and prev == LABEL_B: return False
        if curr == LABEL_B and nxt == LABEL_O: return False
        if curr == LABEL_B and nxt is not None and nxt != LABEL_I: return False
    return True


def post_process_bio_sequence_with_probs(pred_labels_argmax, pred_probs_seq):
    # ... (Funcția post_process_bio_sequence_with_probs rămâne la fel) ...
    # Această funcție ar trebui revizuită pentru a fi mai robustă dacă este folosită extensiv.
    # Momentan, va funcționa pe baza regulilor simple definite anterior.
    if not isinstance(pred_labels_argmax, (list, np.ndarray)) or len(pred_labels_argmax) == 0:
        return pred_labels_argmax
    corrected_seq = list(pred_labels_argmax)
    n = len(corrected_seq)
    first_non_o_idx = -1
    for i in range(n):
        if corrected_seq[i] != LABEL_O: first_non_o_idx = i; break
    if first_non_o_idx != -1 and corrected_seq[first_non_o_idx] == LABEL_I:
        if pred_probs_seq[first_non_o_idx, LABEL_B] > PROB_THRESHOLD_B_WHEN_OI:
            corrected_seq[first_non_o_idx] = LABEL_B
    i = 0
    while i < n - 1:
        current_label = corrected_seq[i]
        next_label_pred = corrected_seq[i + 1]
        if current_label == LABEL_O and next_label_pred == LABEL_I:
            if pred_probs_seq[i, LABEL_B] > PROB_THRESHOLD_B_WHEN_OI:
                corrected_seq[i] = LABEL_B
        elif current_label == LABEL_B and next_label_pred == LABEL_B:
            corrected_seq[i + 1] = LABEL_I
        elif current_label == LABEL_B and next_label_pred == LABEL_O:
            if pred_probs_seq[i + 1, LABEL_I] > PROB_THRESHOLD_I_WHEN_BO and pred_probs_seq[i + 1, LABEL_I] > \
                    pred_probs_seq[i + 1, LABEL_O]:
                corrected_seq[i + 1] = LABEL_I
            elif i + 1 < n:
                corrected_seq[i + 1] = LABEL_I
        i += 1
    return np.array(corrected_seq, dtype=int)


def analyze_predictions(model, X_test_data, y_test_original_unpadded, y_test_padded_indices,
                        model_name_str, class_names=['O', 'B', 'I'], b_tolerance_frames=30,
                        apply_post_processing=False):
    # ... (Funcția analyze_predictions rămâne la fel ca în versiunea anterioară) ...
    title_suffix = " (Post-Processed)" if apply_post_processing else " (Raw Predictions)"
    print(f"\n--- Detailed Analysis for {model_name_str}{title_suffix} on Test Set ---")
    if not apply_post_processing:
        print(f"--- (B-Class Tolerance for Raw: +/- {b_tolerance_frames} frames) ---")

    y_pred_probs_all_seqs = model.predict(X_test_data, batch_size=BATCH_SIZE)
    y_pred_classes_raw = np.argmax(y_pred_probs_all_seqs, axis=-1)
    y_pred_classes_to_evaluate = np.copy(y_pred_classes_raw)

    if apply_post_processing:
        print("Applying post-processing to predicted sequences...")
        for i in range(y_pred_classes_raw.shape[0]):
            original_length = len(y_test_original_unpadded[i])
            raw_pred_seq_segment_argmax = y_pred_classes_raw[i, :original_length]
            raw_pred_seq_segment_probs = y_pred_probs_all_seqs[i, :original_length, :]
            corrected_seq_segment = post_process_bio_sequence_with_probs(raw_pred_seq_segment_argmax,
                                                                         raw_pred_seq_segment_probs)
            y_pred_classes_to_evaluate[i, :original_length] = corrected_seq_segment
            if original_length < MAX_LEN_CALCULATED:
                y_pred_classes_to_evaluate[i, original_length:] = LABEL_O

    valid_predicted_sequences_count = 0
    total_sequences_evaluated = len(y_test_original_unpadded)
    tp_b_tolerant = 0;
    fn_b_tolerant = 0;
    total_real_b = 0
    all_predicted_b_for_fp_calc = []

    for i in range(total_sequences_evaluated):
        true_seq_len = len(y_test_original_unpadded[i])
        current_true_labels = y_test_padded_indices[i, :true_seq_len]
        current_pred_labels_for_eval = y_pred_classes_to_evaluate[i, :true_seq_len]
        if check_bio_sequence_validity(current_pred_labels_for_eval):
            valid_predicted_sequences_count += 1
        real_b_indices_in_seq = np.where(current_true_labels == LABEL_B)[0]
        pred_b_indices_in_seq = np.where(current_pred_labels_for_eval == LABEL_B)[0]
        total_real_b += len(real_b_indices_in_seq)
        for pred_b_idx in pred_b_indices_in_seq:
            all_predicted_b_for_fp_calc.append({'seq_idx': i, 'frame_idx': pred_b_idx, 'matched_to_real_b': False})

    for i in range(total_sequences_evaluated):
        true_seq_len = len(y_test_original_unpadded[i])  # Necesar din nou pentru a accesa y_test_padded_indices
        current_true_labels = y_test_padded_indices[i, :true_seq_len]
        real_b_indices_in_seq = np.where(current_true_labels == LABEL_B)[0]
        for real_b_idx in real_b_indices_in_seq:
            found_match_for_this_real_b = False
            best_candidate_pred_location = None;
            min_dist = float('inf')
            for pred_b_loc in all_predicted_b_for_fp_calc:
                if pred_b_loc['seq_idx'] == i and not pred_b_loc['matched_to_real_b']:
                    dist = abs(pred_b_loc['frame_idx'] - real_b_idx)
                    if dist <= b_tolerance_frames and dist < min_dist:
                        min_dist = dist;
                        best_candidate_pred_location = pred_b_loc
            if best_candidate_pred_location is not None:
                tp_b_tolerant += 1
                best_candidate_pred_location['matched_to_real_b'] = True
                found_match_for_this_real_b = True
            if not found_match_for_this_real_b: fn_b_tolerant += 1

    fp_b_tolerant = sum(1 for pred_b_loc in all_predicted_b_for_fp_calc if not pred_b_loc['matched_to_real_b'])
    precision_b_tolerant = tp_b_tolerant / (tp_b_tolerant + fp_b_tolerant) if (tp_b_tolerant + fp_b_tolerant) > 0 else 0
    recall_b_tolerant = tp_b_tolerant / total_real_b if total_real_b > 0 else 0
    f1_b_tolerant = 2 * (precision_b_tolerant * recall_b_tolerant) / (precision_b_tolerant + recall_b_tolerant) if (
                                                                                                                               precision_b_tolerant + recall_b_tolerant) > 0 else 0

    print("\nMetrics for B-Class (with tolerance +/- {} frames, on sequences being evaluated):".format(
        b_tolerance_frames))
    print(f"  Total Real B-frames in Test: {total_real_b}")
    print(f"  Tolerant True Positives (TP_B): {tp_b_tolerant}")
    print(f"  Tolerant False Positives (FP_B): {fp_b_tolerant}")
    print(f"  Tolerant False Negatives (FN_B): {fn_b_tolerant}")
    print(f"  Tolerant Precision (B): {precision_b_tolerant:.4f}")
    print(f"  Tolerant Recall (B):    {recall_b_tolerant:.4f}")
    print(f"  Tolerant F1-Score (B):  {f1_b_tolerant:.4f}")
    print(f"\nSequence Validity Check ({model_name_str}{' - Postprocessed' if apply_post_processing else ' - Raw'}):")
    print(f"  Total sequences evaluated: {total_sequences_evaluated}")
    print(f"  Valid B-I-O predicted sequences: {valid_predicted_sequences_count}")
    if total_sequences_evaluated > 0: print(
        f"  Percentage of valid sequences: {(valid_predicted_sequences_count / total_sequences_evaluated) * 100:.2f}%")

    print("\nStandard Classification Report (exact frame-by-frame, non-padded):")
    report_labels = list(range(NUM_CLASSES))
    target_names_report = [class_names[i] if i < len(class_names) else f"Class_{i}" for i in report_labels]
    true_labels_flat_all_for_report, pred_labels_flat_all_for_report = [], []
    for i in range(len(y_test_original_unpadded)):
        true_seq_len = len(y_test_original_unpadded[i])
        true_labels_flat_all_for_report.extend(y_test_padded_indices[i, :true_seq_len])
        pred_labels_flat_all_for_report.extend(y_pred_classes_to_evaluate[i, :true_seq_len])
    if true_labels_flat_all_for_report:
        unique_true_plus_pred = np.unique(
            np.concatenate((true_labels_flat_all_for_report, pred_labels_flat_all_for_report)))
        current_report_labels = sorted([l for l in report_labels if l in unique_true_plus_pred])
        current_target_names = [class_names[i] if i < len(class_names) else f"Class_{i}" for i in current_report_labels]
        if not current_report_labels:
            print("No common labels for classification report. Skipping report.")
        else:
            report = classification_report(true_labels_flat_all_for_report, pred_labels_flat_all_for_report,
                                           target_names=current_target_names, labels=current_report_labels,
                                           zero_division=0)
            print(report)
        print("\nStandard Confusion Matrix (exact frame-by-frame, non-padded):")
        cm = confusion_matrix(true_labels_flat_all_for_report, pred_labels_flat_all_for_report, labels=report_labels)
        plt.figure(figsize=(8, 6))
        cm_tick_labels = [class_names[i] if i < len(class_names) else f"Class_{i}" for i in report_labels]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_tick_labels, yticklabels=cm_tick_labels)
        plt.title(f'Standard Confusion Matrix - {model_name_str}{" - Postprocessed" if apply_post_processing else ""}');
        plt.ylabel('Actual');
        plt.xlabel('Predicted')
        cm_filename = f"{model_name_str}{'_postprocessed' if apply_post_processing else ''}_standard_confusion_matrix.png"
        cm_path = MODEL_SAVE_DIR / cm_filename
        plt.savefig(cm_path);
        plt.close()
        print(f"Standard confusion matrix saved to {cm_path}")
    else:
        print("No non-padded labels for standard report/matrix.")


def plot_training_history(history, model_name_suffix):
    model_name_prefix = "Final"
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name_prefix}_{model_name_suffix} Loss');
    plt.xlabel('Epochs');
    plt.ylabel('Loss');
    plt.legend();
    plt.grid(True)
    plt.subplot(1, 2, 2)
    accuracy_key = 'accuracy';
    val_accuracy_key = 'val_accuracy'
    if 'weighted_accuracy' in history.history and history.history[
        'weighted_accuracy']: accuracy_key = 'weighted_accuracy'
    if 'val_weighted_accuracy' in history.history and history.history[
        'val_weighted_accuracy']: val_accuracy_key = 'val_weighted_accuracy'
    plt.plot(history.history[accuracy_key], label=f'Train {accuracy_key.replace("_", " ").title()}')
    plt.plot(history.history[val_accuracy_key], label=f'Val {val_accuracy_key.replace("_", " ").title()}')
    plt.title(f'{model_name_prefix}_{model_name_suffix} Accuracy');
    plt.xlabel('Epochs');
    plt.ylabel('Accuracy');
    plt.legend();
    plt.grid(True)
    plt.tight_layout()
    hist_path = MODEL_SAVE_DIR / f"{model_name_prefix}_{model_name_suffix}_training_history.png"
    plt.savefig(hist_path);
    plt.close()
    print(f"Training history plot saved to {hist_path}")


# --- Train and Evaluate BiLSTM ---
print("\n--- Training BiLSTM Model ---")
input_shape_model = (effective_max_len, NUM_FEATURES)  # NUM_FEATURES este acum augmentat
bilstm_model = build_recurrent_model(model_type='lstm', units=LSTM_UNITS, input_shape=input_shape_model,
                                     num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE, l2_reg=L2_REG_FACTOR)
bilstm_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy',
                     metrics=['accuracy'], weighted_metrics=['accuracy'])
bilstm_model.summary()
bilstm_checkpoint_path = MODEL_SAVE_DIR / "bilstm_best_prev_label.keras"  # Nume nou
bilstm_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(filepath=str(bilstm_checkpoint_path), monitor='val_loss', save_best_only=True,
                                       verbose=1)
]
print(f"Starting BiLSTM training with sample weights and augmented features.")
history_bilstm = bilstm_model.fit(X_train_padded, y_train_one_hot, batch_size=BATCH_SIZE, epochs=EPOCHS,
                                  validation_data=(X_val_padded, y_val_one_hot), callbacks=bilstm_callbacks,
                                  sample_weight=sample_weights_train, verbose=1)
print("\nEvaluating BiLSTM model on Test Set...")
eval_results_bilstm = bilstm_model.evaluate(X_test_padded, y_test_one_hot, verbose=0, batch_size=BATCH_SIZE)
bilstm_loss, bilstm_accuracy = eval_results_bilstm[0], eval_results_bilstm[1]
bilstm_weighted_accuracy = eval_results_bilstm[2] if len(eval_results_bilstm) > 2 else bilstm_accuracy
print(
    f"BiLSTM Test Loss: {bilstm_loss:.4f}, Accuracy: {bilstm_accuracy:.4f}, Weighted Accuracy: {bilstm_weighted_accuracy:.4f}")
plot_training_history(history_bilstm, "BiLSTM_PrevLabel")
analyze_predictions(bilstm_model, X_test_padded, y_test_raw, y_test_padded, "BiLSTM_PrevLabel", b_tolerance_frames=30,
                    apply_post_processing=False)
analyze_predictions(bilstm_model, X_test_padded, y_test_raw, y_test_padded, "BiLSTM_PrevLabel", b_tolerance_frames=30,
                    apply_post_processing=True)

# --- Train and Evaluate BiGRU ---
print("\n--- Training BiGRU Model ---")
bigru_model = build_recurrent_model(model_type='gru', units=GRU_UNITS, input_shape=input_shape_model,
                                    num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE, l2_reg=L2_REG_FACTOR)
bigru_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy',
                    metrics=['accuracy'], weighted_metrics=['accuracy'])
bigru_model.summary()
bigru_checkpoint_path = MODEL_SAVE_DIR / "bigru_best_prev_label.keras"  # Nume nou
bigru_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(filepath=str(bigru_checkpoint_path), monitor='val_loss', save_best_only=True,
                                       verbose=1)
]
print(f"Starting BiGRU training with sample weights and augmented features.")
history_bigru = bigru_model.fit(X_train_padded, y_train_one_hot, batch_size=BATCH_SIZE, epochs=EPOCHS,
                                validation_data=(X_val_padded, y_val_one_hot), callbacks=bigru_callbacks,
                                sample_weight=sample_weights_train, verbose=1)
print("\nEvaluating BiGRU model on Test Set...")
eval_results_bigru = bigru_model.evaluate(X_test_padded, y_test_one_hot, verbose=0, batch_size=BATCH_SIZE)
bigru_loss, bigru_accuracy = eval_results_bigru[0], eval_results_bigru[1]
bigru_weighted_accuracy = eval_results_bigru[2] if len(eval_results_bigru) > 2 else bigru_accuracy
print(
    f"BiGRU Test Loss: {bigru_loss:.4f}, Accuracy: {bigru_accuracy:.4f}, Weighted Accuracy: {bigru_weighted_accuracy:.4f}")
plot_training_history(history_bigru, "BiGRU_PrevLabel")
analyze_predictions(bigru_model, X_test_padded, y_test_raw, y_test_padded, "BiGRU_PrevLabel", b_tolerance_frames=30,
                    apply_post_processing=False)
analyze_predictions(bigru_model, X_test_padded, y_test_raw, y_test_padded, "BiGRU_PrevLabel", b_tolerance_frames=30,
                    apply_post_processing=True)

print("\n--- Training Finished ---")