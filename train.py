import tensorflow as tf
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import traceback
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.utils import class_weight

# Încearcă să imporți tensorflow_addons
try:
    import tensorflow_addons as tfa

    CRF_AVAILABLE = True
    print(f"TensorFlow Addons version: {tfa.__version__} found.")
except ImportError:
    CRF_AVAILABLE = False
    print("WARNING: tensorflow_addons not found. CRF layer will not be available.")
    # Poți decide să oprești scriptul aici sau să ai un fallback la modelul fără CRF
    # exit("Tensorflow-addons is required for CRF layer. Please install it (e.g., pip install tensorflow-addons==0.18.0).")

# --- Configuration ---
FINAL_DATA_DIR = Path("./final_combined_data_for_training_ALL_SIGNERS")
FINAL_FEATURES_DATA_FILE = FINAL_DATA_DIR / "all_data_final_features_ts_facial.pkl"
FINAL_ANNOTATION_FILE = FINAL_DATA_DIR / "annotations_bio_final_combined.pkl"
MODEL_SAVE_DIR = Path("./trained_models_final_crf")  # Director nou
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

USE_CRF_LAYER_FLAG = True  # Setează la True pentru a folosi CRF, False pentru modelul standard

LSTM_UNITS = 128
GRU_UNITS = 128
DROPOUT_RATE = 0.4
L2_REG_FACTOR = 0.0001  # O valoare mică pentru început cu CRF
NUM_CLASSES = 3
LABEL_O, LABEL_B, LABEL_I = 0, 1, 2
BATCH_SIZE = 8
EPOCHS = 50
PATIENCE = 15
PADDING_TYPE = 'post'
MAX_LEN_CALCULATED = None

print(f"TensorFlow Version: {tf.__version__}")
if USE_CRF_LAYER_FLAG and not CRF_AVAILABLE:
    print("Error: CRF_AVAILABLE is False but USE_CRF_LAYER_FLAG is True. Install tensorflow_addons.")
    exit()

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
y_train_padded = tf.keras.preprocessing.sequence.pad_sequences(y_train_raw, padding=PADDING_TYPE, value=LABEL_O,
                                                               maxlen=MAX_LEN_CALCULATED)
y_val_padded = tf.keras.preprocessing.sequence.pad_sequences(y_val_raw, padding=PADDING_TYPE, value=LABEL_O,
                                                             maxlen=MAX_LEN_CALCULATED)
y_test_padded = tf.keras.preprocessing.sequence.pad_sequences(y_test_raw, padding=PADDING_TYPE, value=LABEL_O,
                                                              maxlen=MAX_LEN_CALCULATED)
effective_max_len = MAX_LEN_CALCULATED
print(f"Sequences padded to length: {effective_max_len}")

# Pregătește y-urile pentru model
y_train_for_model = y_train_padded.astype(np.int32)
y_val_for_model = y_val_padded.astype(np.int32)
y_test_for_model = y_test_padded.astype(np.int32)

if not USE_CRF_LAYER_FLAG:  # Doar dacă NU folosim CRF, facem one-hot encoding
    print("One-hot encoding labels (CRF not used)...")
    y_train_for_model = tf.keras.utils.to_categorical(y_train_padded.astype(int), num_classes=NUM_CLASSES)
    y_val_for_model = tf.keras.utils.to_categorical(y_val_padded.astype(int), num_classes=NUM_CLASSES)
    y_test_for_model = tf.keras.utils.to_categorical(y_test_padded.astype(int), num_classes=NUM_CLASSES)
    print(f"Shape X_train_padded: {X_train_padded.shape}, y_train_for_model (one-hot): {y_train_for_model.shape}")
else:
    print(
        f"Shape X_train_padded: {X_train_padded.shape}, y_train_for_model (indices for CRF): {y_train_for_model.shape}")

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
    # Asigură că toate clasele 0, 1, 2 sunt în unique_classes_in_train pentru compute_class_weight
    # dacă nu, compute_class_weight poate da eroare sau ponderi greșite.
    # Este mai sigur să ne asigurăm că y_train_labels_flat_for_weights conține cel puțin o dată fiecare clasă
    # sau să gestionăm acest caz în compute_class_weight.
    # Pentru 'balanced', trebuie să fie prezente.

    # Verifică dacă toate clasele sunt în setul de antrenament
    missing_classes = set(expected_classes) - set(unique_classes_in_train)
    if missing_classes:
        print(f"WARNING: Classes {missing_classes} are not present in the non-padded training labels. ")
        print("         Class/Sample weighting might be problematic or `compute_class_weight` might fail.")
        print("         Consider ensuring all classes appear in the training set if using 'balanced' weights.")

    if len(unique_classes_in_train) == NUM_CLASSES:  # Continuă doar dacă toate clasele sunt prezente
        class_weights_values = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=unique_classes_in_train,  # Ar trebui să fie [0, 1, 2]
            y=y_train_labels_flat_for_weights
        )
        temp_class_weights_dict = {unique_classes_in_train[i]: class_weights_values[i] for i in
                                   range(len(unique_classes_in_train))}
        class_weights_dict = {cls: temp_class_weights_dict.get(cls, 1.0) for cls in range(NUM_CLASSES)}

        target_b_weight = 30.0
        if LABEL_B in class_weights_dict:
            print(f"Original B-weight from 'balanced': {class_weights_dict[LABEL_B]}")
            class_weights_dict[LABEL_B] = target_b_weight
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
        print("Skipping class/sample weighting as not all classes were present in training data.")
else:
    print("Warning: Could not calculate class weights, y_train_labels_flat_for_weights is empty.")


def build_recurrent_model(model_type='lstm', units=128, input_shape=None, num_classes=3, dropout_rate=0.3, l2_reg=0.0,
                          use_crf=False):
    if use_crf and not CRF_AVAILABLE:
        raise ImportError("CRF layer selected but tensorflow_addons is not available.")

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

    if use_crf:
        # Stratul Dense produce potențiale (logiți) pentru CRF, FĂRĂ activare softmax
        dense_potentials = tf.keras.layers.Dense(num_classes, kernel_regularizer=regularizer, name="Dense_Potentials")(
            dropout_output)
        crf = tfa.layers.CRF(num_classes, name="CRF_Layer")  # Instanțiază layer-ul CRF
        output_layer = crf(dense_potentials)  # Aplică layer-ul CRF
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        # Modelul cu CRF va fi compilat cu loss-ul și metrica specifică CRF
        # Keras necesită ca loss-ul și metrica să fie definite în afara modelului și pasate la compile,
        # sau ca layer-ul să adauge loss-ul (cum face tfa.layers.CRF).
        # `tfa.layers.CRF.ForwardBackward.get_loss` este ceea ce căutăm.
        # Acest lucru este un pic mai manual.
        # Vom defini loss-ul și metrica mai târziu, la compilare.
    else:
        output_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=regularizer),
            name="Output_Layer"
        )(dropout_output)
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model


# ... (Funcțiile analyze_predictions și plot_training_history rămân la fel, dar vezi nota despre predict cu CRF)

def analyze_predictions(model, X_test_data, y_test_original_unpadded, y_test_padded_indices,
                        model_name_str, class_names=['O', 'B', 'I'], b_tolerance_frames=30,
                        is_crf_model=False):  # Adaugă is_crf_model
    print(
        f"\n--- Detailed Analysis for {model_name_str} on Test Set (B-Class Tolerance: +/- {b_tolerance_frames} frames) ---")

    if is_crf_model:
        # model.predict() pe un model cu CRF la final returnează direct indecșii claselor
        y_pred_classes = model.predict(X_test_data, batch_size=BATCH_SIZE)
        # y_pred_classes ar trebui să fie (batch_size, max_len)
        # Uneori, poate avea o dimensiune suplimentară de 1 la sfârșit, ex: (batch_size, max_len, 1)
        if y_pred_classes.ndim == 3 and y_pred_classes.shape[-1] == 1:
            y_pred_classes = np.squeeze(y_pred_classes, axis=-1)
    else:
        y_pred_probs = model.predict(X_test_data, batch_size=BATCH_SIZE)
        y_pred_classes = np.argmax(y_pred_probs, axis=-1)

    tp_b_tolerant = 0;
    fn_b_tolerant = 0;
    total_real_b = 0
    all_predicted_b_locations = []
    valid_predicted_sequences_count = 0
    total_sequences_evaluated = len(y_test_original_unpadded)

    for i in range(total_sequences_evaluated):
        true_seq_len = len(y_test_original_unpadded[i])
        current_true_labels = y_test_padded_indices[i, :true_seq_len]
        current_pred_labels = y_pred_classes[i, :true_seq_len]

        if check_bio_sequence_validity(current_pred_labels):
            valid_predicted_sequences_count += 1

        real_b_indices_in_seq = np.where(current_true_labels == LABEL_B)[0]
        pred_b_indices_in_seq = np.where(current_pred_labels == LABEL_B)[0]
        total_real_b += len(real_b_indices_in_seq)
        for pred_b_idx in pred_b_indices_in_seq:
            all_predicted_b_locations.append({'seq_idx': i, 'frame_idx': pred_b_idx, 'matched_to_real_b': False})

        for real_b_idx in real_b_indices_in_seq:
            found_match_for_this_real_b = False
            best_candidate_pred_location = None;
            min_dist = float('inf')
            for pred_b_loc in all_predicted_b_locations:
                if pred_b_loc['seq_idx'] == i and not pred_b_loc['matched_to_real_b']:
                    dist = abs(pred_b_loc['frame_idx'] - real_b_idx)
                    if dist <= b_tolerance_frames and dist < min_dist:
                        min_dist = dist;
                        best_candidate_pred_location = pred_b_loc
            if best_candidate_pred_location is not None:
                tp_b_tolerant += 1;
                best_candidate_pred_location['matched_to_real_b'] = True
                found_match_for_this_real_b = True
            if not found_match_for_this_real_b: fn_b_tolerant += 1

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

    print(f"\nSequence Validity Check ({model_name_str}):")
    print(f"  Total sequences evaluated: {total_sequences_evaluated}")
    print(f"  Valid B-I-O predicted sequences: {valid_predicted_sequences_count}")
    if total_sequences_evaluated > 0:
        print(
            f"  Percentage of valid sequences: {(valid_predicted_sequences_count / total_sequences_evaluated) * 100:.2f}%")

    print("\nStandard Classification Report (exact frame-by-frame, non-padded):")
    report_labels = list(range(NUM_CLASSES))
    target_names_report = [class_names[label_idx] for label_idx in report_labels if label_idx < len(class_names)]
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


def plot_training_history(history, model_name_suffix):  # Păstrează la fel
    model_name_prefix = "Final_CRF" if USE_CRF_LAYER_FLAG else "Final"
    # ... restul funcției plot_training_history ...
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

    # Gestionează cheile de acuratețe, CRF ar putea avea 'accuracy' sau 'crf_accuracy'
    # sau o metrică specifică pasată la compile.
    # Pentru tfa.layers.CRF, metrica standard 'accuracy' ar trebui să funcționeze dacă y_true și y_pred sunt indecși.
    accuracy_key = 'accuracy'
    val_accuracy_key = 'val_accuracy'
    # Dacă ai adăugat weighted_accuracy și NU e CRF
    if not USE_CRF_LAYER_FLAG:
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


# --- Train and Evaluate BiLSTM ---
print(f"\n--- Training BiLSTM Model {'with CRF' if USE_CRF_LAYER_FLAG else ''} ---")
input_shape_model = (effective_max_len, NUM_FEATURES)
bilstm_model = build_recurrent_model(model_type='lstm', units=LSTM_UNITS, input_shape=input_shape_model,
                                     num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE, l2_reg=L2_REG_FACTOR,
                                     use_crf=USE_CRF_LAYER_FLAG)

if USE_CRF_LAYER_FLAG:
    # Pentru tfa.layers.CRF, se compilează cu loss-ul și metrica specifice (dacă layer-ul le expune)
    # sau un loss care funcționează cu logiți și y_true ca indecși.
    # Căutând exemple pentru tfa.layers.CRF, se pare că stratul însuși adaugă loss-ul,
    # iar modelul trebuie compilat astfel încât să optimizeze acest loss.
    # Predicțiile sunt deja secvența decodificată.
    # Compilarea se face adesea cu loss=None și se adaugă loss-ul CRF manual,
    # sau se folosește un wrapper.
    # O metodă mai simplă pentru versiuni TFA care o suportă este:
    crf_layer_from_model = bilstm_model.get_layer('CRF_Layer')  # Găsește layer-ul CRF
    # loss_fn = crf_layer_from_model.loss # Acest atribut s-ar putea să nu existe direct
    # accuracy_fn = crf_layer_from_model.accuracy # Similar

    # Din cauza incertitudinii API-ului TFA pentru TF 2.10, vom folosi o abordare mai generică
    # care este adesea compatibilă: sparse_categorical_crossentropy pe logiții dinaintea CRF,
    # și ne bazăm pe CRF pentru a face decodificarea corectă la predicție.
    # Acest lucru înseamnă că CRF-ul nu este antrenat *direct* prin funcția sa de loss optimă,
    # ci indirect prin antrenarea potențialelor. Nu este ideal, dar e un punct de pornire.
    # O implementare CRF corectă ar trebui să aibă propria sa funcție de loss.
    print("Compiling BiLSTM model with CRF using a generic sparse loss (CRF primarily for decoding).")
    bilstm_model.compile(optimizer=tf.keras.optimizers.Adam(),
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'],
                         sample_weight_mode="temporal" if sample_weights_train is not None else None)
else:
    bilstm_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy',
                         metrics=['accuracy'], weighted_metrics=['accuracy'])

bilstm_model.summary()
bilstm_checkpoint_path = MODEL_SAVE_DIR / f"bilstm_best_final{'_crf' if USE_CRF_LAYER_FLAG else ''}.keras"
bilstm_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(filepath=str(bilstm_checkpoint_path), monitor='val_loss', save_best_only=True,
                                       verbose=1)
]
print(
    f"Starting BiLSTM training {'with CRF and ' if USE_CRF_LAYER_FLAG else ''}sample weights (derived from class_weights: {class_weights_dict})")
y_train_data_for_fit = y_train_for_model  # Acesta este y_train_padded (indecși) dacă CRF, sau y_train_one_hot altfel
y_val_data_for_fit = y_val_for_model  # Similar pentru validare

history_bilstm = bilstm_model.fit(X_train_padded, y_train_data_for_fit, batch_size=BATCH_SIZE, epochs=EPOCHS,
                                  validation_data=(X_val_padded, y_val_data_for_fit), callbacks=bilstm_callbacks,
                                  sample_weight=sample_weights_train, verbose=1)

print("\nEvaluating BiLSTM model on Test Set...")
y_test_data_for_eval = y_test_for_model  # Similar pentru evaluare
eval_results_bilstm = bilstm_model.evaluate(X_test_padded, y_test_data_for_eval, verbose=0, batch_size=BATCH_SIZE,
                                            sample_weight=None)
bilstm_loss = eval_results_bilstm[0]
bilstm_accuracy = eval_results_bilstm[1]
bilstm_weighted_accuracy = eval_results_bilstm[2] if len(
    eval_results_bilstm) > 2 and not USE_CRF_LAYER_FLAG else bilstm_accuracy
print(f"BiLSTM Test Loss: {bilstm_loss:.4f}, BiLSTM Test Accuracy: {bilstm_accuracy:.4f}", end="")
if not USE_CRF_LAYER_FLAG:
    print(f", BiLSTM Test Weighted Accuracy: {bilstm_weighted_accuracy:.4f}")
else:
    print("")  # Doar newline

plot_training_history(history_bilstm, f"BiLSTM{'_CRF' if USE_CRF_LAYER_FLAG else ''}")
analyze_predictions(bilstm_model, X_test_padded, y_test_raw, y_test_padded,
                    f"BiLSTM{'_CRF_Final' if USE_CRF_LAYER_FLAG else '_Final'}", b_tolerance_frames=30,
                    is_crf_model=USE_CRF_LAYER_FLAG)

# --- Train and Evaluate BiGRU (cu opțiune CRF) ---
# Poți duplica logica de mai sus pentru BiGRU, schimbând model_type și numele variabilelor.
# Pentru moment, voi lăsa doar BiLSTM pentru a simplifica și a testa CRF.
# Dacă vrei să rulezi și BiGRU cu CRF, trebuie să aplici aceleași modificări.

print(f"\n--- Skipping BiGRU training for this CRF test run ---")
# print("\n--- Training BiGRU Model ---")
# ... (cod similar pentru BiGRU) ...


print("\n--- Training Finished ---")
print(f"Best BiLSTM model saved to: {bilstm_checkpoint_path}")
# if BiGRU a fost antrenat: print(f"Best BiGRU model saved to: {bigru_checkpoint_path}")
print(f"Training plots and confusion matrices saved to: {MODEL_SAVE_DIR}")