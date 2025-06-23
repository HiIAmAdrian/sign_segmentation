import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import traceback
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.utils import class_weight

FINAL_DATA_DIR = Path("./final_combined_data_for_training_ALL_SIGNERS")
FINAL_FEATURES_DATA_FILE = FINAL_DATA_DIR / "all_data_final_features_ts_facial.pkl"
FINAL_ANNOTATION_FILE = FINAL_DATA_DIR / "annotations_bio_final_combined.pkl"
MODEL_SAVE_DIR = Path("./trained_models_final")
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

LSTM_UNITS = 128
GRU_UNITS = 128
DROPOUT_RATE = 0.4
L2_REG_FACTOR = 0.0
NUM_CLASSES = 3
BATCH_SIZE = 8
EPOCHS = 50
PATIENCE = 15
PADDING_TYPE = 'post'
MAX_LEN_CALCULATED = None

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

print(f"Loading final processed features from: {FINAL_FEATURES_DATA_FILE}")
try:
    with open(FINAL_FEATURES_DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    X_train_raw, X_val_raw, X_test_raw = data['X_train'], data['X_val'], data['X_test']
    feature_names = data.get('feature_names', [])
    if not X_train_raw: print(f"Error: X_train is empty in {FINAL_FEATURES_DATA_FILE}"); exit()
    NUM_FEATURES = X_train_raw[0].shape[1] if X_train_raw and X_train_raw[0] is not None and len(
        X_train_raw[0].shape) > 1 else 0
    if NUM_FEATURES == 0 and X_train_raw:
        for seq in X_train_raw:
            if seq is not None and len(seq.shape) > 1 and seq.shape[1] > 0: NUM_FEATURES = seq.shape[1]; break
    if NUM_FEATURES == 0: print("Error: Could not determine number of features."); exit()
    print(
        f"Found {len(X_train_raw)} train, {len(X_val_raw)} val, {len(X_test_raw)} test sequences. Features: {NUM_FEATURES}")
except Exception as e:
    print(f"Error loading feature data: {e}"); traceback.print_exc(); exit()

print(f"Loading final annotations from: {FINAL_ANNOTATION_FILE}")
try:
    with open(FINAL_ANNOTATION_FILE, 'rb') as f:
        annotations = pickle.load(f)
    y_train_raw, y_val_raw, y_test_raw = annotations['train'], annotations['val'], annotations['test']
    for name, X, y in [("Train", X_train_raw, y_train_raw), ("Val", X_val_raw, y_val_raw),
                       ("Test", X_test_raw, y_test_raw)]:
        if len(X) != len(y): raise AssertionError(f"{name} X length ({len(X)}) != Y length ({len(y)})")
        for i in range(len(X)):
            if X[i].shape[0] != y[i].shape[0]: raise AssertionError(
                f"{name} seq {i}: X len {X[i].shape[0]} != Y len {y[i].shape[0]}")
    print("Annotations loaded and validated.")
except Exception as e:
    print(f"Error loading annotation data: {e}"); traceback.print_exc(); exit()

all_sequence_lengths = [seq.shape[0] for dataset in [X_train_raw, X_val_raw, X_test_raw] for seq in dataset if
                        seq is not None]
if not all_sequence_lengths: print("Error: No sequences to determine max length."); exit()
MAX_LEN_CALCULATED = np.max(all_sequence_lengths)
print(f"Determined MAX_LEN from data: {MAX_LEN_CALCULATED}")

print("Padding sequences...")
X_train_padded = tf.keras.preprocessing.sequence.pad_sequences(X_train_raw, padding=PADDING_TYPE, dtype='float32',
                                                               maxlen=MAX_LEN_CALCULATED, value=0.0)
X_val_padded = tf.keras.preprocessing.sequence.pad_sequences(X_val_raw, padding=PADDING_TYPE, dtype='float32',
                                                             maxlen=MAX_LEN_CALCULATED, value=0.0)
X_test_padded = tf.keras.preprocessing.sequence.pad_sequences(X_test_raw, padding=PADDING_TYPE, dtype='float32',
                                                              maxlen=MAX_LEN_CALCULATED, value=0.0)
y_train_padded = tf.keras.preprocessing.sequence.pad_sequences(y_train_raw, padding=PADDING_TYPE, value=0,
                                                               maxlen=MAX_LEN_CALCULATED)
y_val_padded = tf.keras.preprocessing.sequence.pad_sequences(y_val_raw, padding=PADDING_TYPE, value=0,
                                                             maxlen=MAX_LEN_CALCULATED)
y_test_padded = tf.keras.preprocessing.sequence.pad_sequences(y_test_raw, padding=PADDING_TYPE, value=0,
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
            f"Warning: Not all {NUM_CLASSES} classes (0 to {NUM_CLASSES - 1}) are present in the non-padded training labels. Found only {unique_classes_in_train}. Class/Sample weighting might be suboptimal or skipped.")

    class_weights_values = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=unique_classes_in_train,
        y=y_train_labels_flat_for_weights
    )
    temp_class_weights_dict = {unique_classes_in_train[i]: class_weights_values[i] for i in
                               range(len(unique_classes_in_train))}
    class_weights_dict = {cls: temp_class_weights_dict.get(cls, 1.0) for cls in range(NUM_CLASSES)}

    target_b_weight = 10.0
    if 1 in class_weights_dict:
        print(f"Original B-weight from 'balanced': {class_weights_dict[1]}")
        class_weights_dict[1] = target_b_weight
    print(f"ADJUSTED Class Weights (for all {NUM_CLASSES} classes): {class_weights_dict}")

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
    print("Warning: Could not calculate class weights, y_train_labels_flat_for_weights is empty.")


def build_recurrent_model(model_type='lstm', units=128, input_shape=None, num_classes=3, dropout_rate=0.3, l2_reg=0.0):
    if input_shape is None: raise ValueError("input_shape must be provided")
    input_layer = tf.keras.layers.Input(shape=input_shape, name="Input_Layer")
    masked_input = tf.keras.layers.Masking(mask_value=0.0, name="Masking_Layer")(input_layer)
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


def analyze_predictions(model, X_test_data, y_test_original_unpadded, y_test_padded_indices,
                        model_name_str, class_names=['O', 'B', 'I'], b_tolerance_frames=30):
    print(
        f"\n--- Detailed Analysis for {model_name_str} on Test Set (B-Class Tolerance: +/- {b_tolerance_frames} frames) ---")

    y_pred_probs = model.predict(X_test_data, batch_size=BATCH_SIZE)
    y_pred_classes = np.argmax(y_pred_probs, axis=-1)

    tp_b_tolerant = 0
    fn_b_tolerant = 0
    total_real_b = 0
    all_predicted_b_locations = []

    for i in range(len(y_test_original_unpadded)):
        true_seq_len = len(y_test_original_unpadded[i])
        current_true_labels = y_test_padded_indices[i, :true_seq_len]
        current_pred_labels = y_pred_classes[i, :true_seq_len]

        real_b_indices_in_seq = np.where(current_true_labels == 1)[0]
        pred_b_indices_in_seq = np.where(current_pred_labels == 1)[0]

        total_real_b += len(real_b_indices_in_seq)

        for pred_b_idx in pred_b_indices_in_seq:
            all_predicted_b_locations.append({'seq_idx': i, 'frame_idx': pred_b_idx, 'matched_to_real_b': False})

        for real_b_idx in real_b_indices_in_seq:
            found_match_for_this_real_b = False
            best_candidate_pred_location = None
            min_dist = float('inf')

            for pred_b_loc in all_predicted_b_locations:
                if pred_b_loc['seq_idx'] == i and not pred_b_loc['matched_to_real_b']:
                    dist = abs(pred_b_loc['frame_idx'] - real_b_idx)
                    if dist <= b_tolerance_frames:
                        if dist < min_dist:
                            min_dist = dist
                            best_candidate_pred_location = pred_b_loc

            if best_candidate_pred_location is not None:
                tp_b_tolerant += 1
                best_candidate_pred_location['matched_to_real_b'] = True
                found_match_for_this_real_b = True

            if not found_match_for_this_real_b:
                fn_b_tolerant += 1

    fp_b_tolerant = sum(1 for pred_b_loc in all_predicted_b_locations if not pred_b_loc['matched_to_real_b'])

    precision_b_tolerant = tp_b_tolerant / (tp_b_tolerant + fp_b_tolerant) if (tp_b_tolerant + fp_b_tolerant) > 0 else 0
    recall_b_tolerant = tp_b_tolerant / total_real_b if total_real_b > 0 else 0
    f1_b_tolerant = 2 * (precision_b_tolerant * recall_b_tolerant) / (precision_b_tolerant + recall_b_tolerant) if (
                                                                                                                               precision_b_tolerant + recall_b_tolerant) > 0 else 0

    print("\nMetrics for B-Class (with tolerance +/- {} frames):".format(b_tolerance_frames))
    print(f"  Total Real B-frames in Test: {total_real_b}")
    print(f"  Tolerant True Positives (TP_B): {tp_b_tolerant}")
    print(f"  Tolerant False Positives (FP_B): {fp_b_tolerant}")
    print(f"  Tolerant False Negatives (FN_B): {fn_b_tolerant}")
    print(f"  Tolerant Precision (B): {precision_b_tolerant:.4f}")
    print(f"  Tolerant Recall (B):    {recall_b_tolerant:.4f}")
    print(f"  Tolerant F1-Score (B):  {f1_b_tolerant:.4f}")

    print("\nStandard Classification Report (exact frame-by-frame, non-padded):")
    report_labels = list(range(NUM_CLASSES))
    target_names_report = [class_names[i] for i in report_labels]
    true_labels_flat_all_for_report, pred_labels_flat_all_for_report = [], []
    for i in range(len(y_test_original_unpadded)):
        true_seq_len = len(y_test_original_unpadded[i])
        true_labels_flat_all_for_report.extend(y_test_padded_indices[i, :true_seq_len])
        pred_labels_flat_all_for_report.extend(y_pred_classes[i, :true_seq_len])
    if true_labels_flat_all_for_report:
        report = classification_report(true_labels_flat_all_for_report, pred_labels_flat_all_for_report,
                                       target_names=target_names_report, labels=report_labels, zero_division=0)
        print(report)
        print("\nStandard Confusion Matrix (exact frame-by-frame, non-padded):")
        cm = confusion_matrix(true_labels_flat_all_for_report, pred_labels_flat_all_for_report, labels=report_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names_report,
                    yticklabels=target_names_report)
        plt.title(f'Standard Confusion Matrix - {model_name_str}');
        plt.ylabel('Actual');
        plt.xlabel('Predicted')
        cm_path = MODEL_SAVE_DIR / f"{model_name_str}_standard_confusion_matrix.png"
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
    plt.title(f'{model_name_prefix}_{model_name_suffix} Loss')
    plt.xlabel('Epochs');
    plt.ylabel('Loss');
    plt.legend();
    plt.grid(True)
    plt.subplot(1, 2, 2)
    accuracy_key = 'accuracy';
    val_accuracy_key = 'val_accuracy'
    if 'weighted_accuracy' in history.history and history.history['weighted_accuracy']:
        accuracy_key = 'weighted_accuracy'
    if 'val_weighted_accuracy' in history.history and history.history['val_weighted_accuracy']:
        val_accuracy_key = 'val_weighted_accuracy'
    plt.plot(history.history[accuracy_key], label=f'Train {accuracy_key.replace("_", " ").title()}')
    plt.plot(history.history[val_accuracy_key], label=f'Val {val_accuracy_key.replace("_", " ").title()}')
    plt.title(f'{model_name_prefix}_{model_name_suffix} Accuracy')
    plt.xlabel('Epochs');
    plt.ylabel('Accuracy');
    plt.legend();
    plt.grid(True)
    plt.tight_layout()
    hist_path = MODEL_SAVE_DIR / f"{model_name_prefix}_{model_name_suffix}_training_history.png"
    plt.savefig(hist_path);
    plt.close()
    print(f"Training history plot saved to {hist_path}")


print("\n--- Training BiLSTM Model ---")
input_shape_model = (effective_max_len, NUM_FEATURES)
bilstm_model = build_recurrent_model(model_type='lstm', units=LSTM_UNITS, input_shape=input_shape_model,
                                     num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE, l2_reg=L2_REG_FACTOR)
bilstm_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy',
                     metrics=['accuracy'], weighted_metrics=['accuracy'])
bilstm_model.summary()
bilstm_checkpoint_path = MODEL_SAVE_DIR / "bilstm_best_final.keras"
bilstm_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(filepath=str(bilstm_checkpoint_path), monitor='val_loss', save_best_only=True,
                                       verbose=1)
]
print(f"Starting BiLSTM training with sample weights (derived from class_weights: {class_weights_dict})")
history_bilstm = bilstm_model.fit(X_train_padded, y_train_one_hot, batch_size=BATCH_SIZE, epochs=EPOCHS,
                                  validation_data=(X_val_padded, y_val_one_hot), callbacks=bilstm_callbacks,
                                  sample_weight=sample_weights_train, verbose=1)
print("\nEvaluating BiLSTM model on Test Set...")
eval_results_bilstm = bilstm_model.evaluate(X_test_padded, y_test_one_hot, verbose=0, batch_size=BATCH_SIZE)
bilstm_loss = eval_results_bilstm[0]
bilstm_accuracy = eval_results_bilstm[1]
bilstm_weighted_accuracy = eval_results_bilstm[2] if len(eval_results_bilstm) > 2 else bilstm_accuracy
print(
    f"BiLSTM Test Loss: {bilstm_loss:.4f}, BiLSTM Test Accuracy: {bilstm_accuracy:.4f}, BiLSTM Test Weighted Accuracy: {bilstm_weighted_accuracy:.4f}")
plot_training_history(history_bilstm, "BiLSTM")
analyze_predictions(bilstm_model, X_test_padded, y_test_raw, y_test_padded, "BiLSTM_Final", b_tolerance_frames=30)

print("\n--- Training BiGRU Model ---")
bigru_model = build_recurrent_model(model_type='gru', units=GRU_UNITS, input_shape=input_shape_model,
                                    num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE, l2_reg=L2_REG_FACTOR)
bigru_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy',
                    metrics=['accuracy'], weighted_metrics=['accuracy'])
bigru_model.summary()
bigru_checkpoint_path = MODEL_SAVE_DIR / "bigru_best_final.keras"
bigru_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(filepath=str(bigru_checkpoint_path), monitor='val_loss', save_best_only=True,
                                       verbose=1)
]
print(f"Starting BiGRU training with sample weights (derived from class_weights: {class_weights_dict})")
history_bigru = bigru_model.fit(X_train_padded, y_train_one_hot, batch_size=BATCH_SIZE, epochs=EPOCHS,
                                validation_data=(X_val_padded, y_val_one_hot), callbacks=bigru_callbacks,
                                sample_weight=sample_weights_train, verbose=1)
print("\nEvaluating BiGRU model on Test Set...")
eval_results_bigru = bigru_model.evaluate(X_test_padded, y_test_one_hot, verbose=0, batch_size=BATCH_SIZE)
bigru_loss = eval_results_bigru[0]
bigru_accuracy = eval_results_bigru[1]
bigru_weighted_accuracy = eval_results_bigru[2] if len(eval_results_bigru) > 2 else bigru_accuracy
print(
    f"BiGRU Test Loss: {bigru_loss:.4f}, BiGRU Test Accuracy: {bigru_accuracy:.4f}, BiGRU Test Weighted Accuracy: {bigru_weighted_accuracy:.4f}")
plot_training_history(history_bigru, "BiGRU")
analyze_predictions(bigru_model, X_test_padded, y_test_raw, y_test_padded, "BiGRU_Final", b_tolerance_frames=30)

print("\n--- Training Finished ---")
print(f"Best BiLSTM model saved to: {bilstm_checkpoint_path}")
print(f"Best BiGRU model saved to: {bigru_checkpoint_path}")
print(f"Training plots and confusion matrices saved to: {MODEL_SAVE_DIR}")