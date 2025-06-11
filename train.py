import tensorflow as tf
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import traceback
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
from sklearn.utils import class_weight

# --- Configuration ---
FINAL_DATA_DIR = Path("./final_combined_data_for_training_ALL_SIGNERS")
FINAL_FEATURES_DATA_FILE = FINAL_DATA_DIR / "all_data_final_features_ts_facial.pkl"
FINAL_ANNOTATION_FILE = FINAL_DATA_DIR / "annotations_bio_final_combined.pkl"  # Folosim adnotările B-I-O originale
MODEL_SAVE_DIR = Path("./5-dropout02-trained_models_2_classes_OI_refined")  # Director nou
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# --- Hiperparametri Model ---
LSTM_UNITS = 128  # Poți experimenta: 64, 192, 256
GRU_UNITS = 128  # Poți experimenta: 64, 192, 256
DROPOUT_RATE = 0.4  # Poți experimenta: 0.3, 0.5
L2_REG_FACTOR = 0.0  # Poți experimenta: 0.0001, 0.001
LEARNING_RATE = 0.005  # Poți experimenta: 0.0005, 0.0001

# --- Configurație Clase ---
NUM_CLASSES_ORIG = 3
LABEL_O_3CLASS, LABEL_B_3CLASS, LABEL_I_3CLASS = 0, 1, 2  # Etichetele B-I-O originale

NUM_CLASSES_TARGET = 2  # Trecem la O și I
LABEL_O_2CLASS, LABEL_I_2CLASS = 0, 1  # Noile etichete

# --- Configurație Antrenament ---
BATCH_SIZE = 16  # Ai putea încerca să mărești dacă memoria permite
EPOCHS = 70  # Poți crește numărul de epoci dacă nu apare overfitting rapid
PATIENCE = 20  # Răbdare mai mare pentru EarlyStopping, mai ales dacă LR scade
PADDING_TYPE = 'post'
MAX_LEN_CALCULATED = None  # Se va calcula

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


# --- Funcție pentru maparea etichetelor B-I-O la O-I ---
def map_labels_to_2_classes(y_sequences_3_classes):
    y_sequences_2_classes = []
    for seq_3_class in y_sequences_3_classes:
        if seq_3_class is None:
            y_sequences_2_classes.append(None)
            continue
        # 0 (O original) -> 0 (O_2CLASS)
        # 1 (B original) -> 0 (O_2CLASS) << MODIFICARE CHEIE
        # 2 (I original) -> 1 (I_2CLASS) << MODIFICARE CHEIE
        seq_2_class = np.where(seq_3_class == LABEL_I_3CLASS, LABEL_I_2CLASS, LABEL_O_2CLASS)
        y_sequences_2_classes.append(seq_2_class.astype(np.int32))
    return y_sequences_2_classes


print(f"Loading final processed features from: {FINAL_FEATURES_DATA_FILE}")
try:
    with open(FINAL_FEATURES_DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    # X_train_raw, etc., sunt deja array-uri NumPy scalate din PKL
    X_train_raw, X_val_raw, X_test_raw = data['X_train'], data['X_val'], data['X_test']
    feature_names = data.get('feature_names', [])
    if not X_train_raw or X_train_raw[0] is None: print(f"Error: X_train is empty or first seq is None"); exit()

    NUM_FEATURES = X_train_raw[0].shape[1]  # Numărul de caracteristici din datele încărcate
    print(
        f"Found {len(X_train_raw)} train, {len(X_val_raw)} val, {len(X_test_raw)} test sequences. Features: {NUM_FEATURES}")
except Exception as e:
    print(f"Error loading feature data: {e}");
    traceback.print_exc();
    exit()

print(f"Loading original 3-class annotations from: {FINAL_ANNOTATION_FILE}")
try:
    with open(FINAL_ANNOTATION_FILE, 'rb') as f:
        annotations_3_class = pickle.load(f)
    y_train_raw_3_class, y_val_raw_3_class, y_test_raw_3_class = annotations_3_class['train'], annotations_3_class[
        'val'], annotations_3_class['test']

    print("Mapping 3-class labels (B,I,O) to 2-class labels (O,I)...")
    y_train_raw_2_class = map_labels_to_2_classes(y_train_raw_3_class)
    y_val_raw_2_class = map_labels_to_2_classes(y_val_raw_3_class)
    y_test_raw_2_class = map_labels_to_2_classes(y_test_raw_3_class)
    print("Labels mapped to 2 classes.")

    # Validare lungimi după mapare
    for name, X, y in [("Train", X_train_raw, y_train_raw_2_class),
                       ("Val", X_val_raw, y_val_raw_2_class),
                       ("Test", X_test_raw, y_test_raw_2_class)]:
        if len(X) != len(y): raise AssertionError(f"{name} X length ({len(X)}) != Y length ({len(y)})")
        for i in range(len(X)):
            if X[i] is None or y[i] is None: raise AssertionError(f"{name} sequence {i} is None for X or Y.")
            if X[i].shape[0] != y[i].shape[0]: raise AssertionError(
                f"{name} seq {i}: X len {X[i].shape[0]} != Y len {y[i].shape[0]}")
    print("Annotations re-validated after 2-class mapping.")
except Exception as e:
    print(f"Error loading or mapping annotation data: {e}");
    traceback.print_exc();
    exit()

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

# Padăm cu noua etichetă O_2CLASS (care e 0)
y_train_padded_2_class = tf.keras.preprocessing.sequence.pad_sequences(y_train_raw_2_class, padding=PADDING_TYPE,
                                                                       value=LABEL_O_2CLASS, maxlen=MAX_LEN_CALCULATED)
y_val_padded_2_class = tf.keras.preprocessing.sequence.pad_sequences(y_val_raw_2_class, padding=PADDING_TYPE,
                                                                     value=LABEL_O_2CLASS, maxlen=MAX_LEN_CALCULATED)
y_test_padded_2_class = tf.keras.preprocessing.sequence.pad_sequences(y_test_raw_2_class, padding=PADDING_TYPE,
                                                                      value=LABEL_O_2CLASS, maxlen=MAX_LEN_CALCULATED)
effective_max_len = MAX_LEN_CALCULATED
print(f"Sequences padded to length: {effective_max_len}")

print("One-hot encoding 2-class labels...")
y_train_one_hot = tf.keras.utils.to_categorical(y_train_padded_2_class.astype(int), num_classes=NUM_CLASSES_TARGET)
y_val_one_hot = tf.keras.utils.to_categorical(y_val_padded_2_class.astype(int), num_classes=NUM_CLASSES_TARGET)
y_test_one_hot = tf.keras.utils.to_categorical(y_test_padded_2_class.astype(int), num_classes=NUM_CLASSES_TARGET)
print(f"Shape X_train_padded: {X_train_padded.shape}, y_train_one_hot: {y_train_one_hot.shape}")

# --- Calcul Ponderi Clase și Sample Weights (pentru 2 clase) ---
y_train_labels_flat_for_weights = []
for i in range(len(y_train_raw_2_class)):  # Folosim y_train_raw_2_class
    true_seq_len = len(y_train_raw_2_class[i])
    y_train_labels_flat_for_weights.extend(y_train_padded_2_class[i, :true_seq_len])

class_weights_dict = None;
sample_weights_train = None
if y_train_labels_flat_for_weights:
    unique_classes_in_train = np.unique(y_train_labels_flat_for_weights)
    print(f"Unique 2-classes in training labels for weights: {unique_classes_in_train}")

    all_possible_2_classes = np.arange(NUM_CLASSES_TARGET)  # [0, 1]

    class_weights_values = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=all_possible_2_classes,
        y=y_train_labels_flat_for_weights
    )
    class_weights_dict = {all_possible_2_classes[i]: class_weights_values[i] for i in
                          range(len(all_possible_2_classes))}
    print(f"Calculated Class Weights for 2 classes (O,I): {class_weights_dict}")

    sample_weights_train = np.ones_like(y_train_padded_2_class, dtype=float)
    for i in range(y_train_padded_2_class.shape[0]):
        original_length = len(y_train_raw_2_class[i])
        for j in range(original_length):
            label = y_train_padded_2_class[i, j]
            sample_weights_train[i, j] = class_weights_dict.get(label, 1.0)
        if original_length < MAX_LEN_CALCULATED:
            sample_weights_train[i, original_length:] = 0.0
    print(f"Shape of sample_weights_train: {sample_weights_train.shape}")
else:
    print("Warning: Could not calculate class weights.")


def build_recurrent_model(model_type='lstm', units=128, input_shape=None, num_classes_out=2, dropout_rate=0.3,
                          l2_reg=0.0, add_conv_layer=False):  # Parametru nou
    if input_shape is None: raise ValueError("input_shape must be provided")
    input_layer = tf.keras.layers.Input(shape=input_shape, name="Input_Layer")
    current_layer = tf.keras.layers.Masking(mask_value=0.0, name="Masking_Layer")(input_layer)
    regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None

    # --- SUGESTIE DE RAFINARE: Strat Conv1D opțional ---
    if add_conv_layer:
        # Poți experimenta cu numărul de filtre, kernel_size, și dacă adaugi BatchNormalization/MaxPooling
        current_layer = tf.keras.layers.Conv1D(filters=units // 2, kernel_size=5, padding="same", activation="relu",
                                               name="Conv1D_FeatureExtractor")(current_layer)
        current_layer = tf.keras.layers.BatchNormalization(name="BatchNorm_AfterConv")(current_layer)
        # current_layer = tf.keras.layers.MaxPooling1D(pool_size=2, name="MaxPool_AfterConv")(current_layer) # Atenție la reducerea lungimii secvenței
    # -------------------------------------------------

    RecurrentLayer = tf.keras.layers.LSTM if model_type == 'lstm' else tf.keras.layers.GRU
    # --- SUGESTIE DE RAFINARE: Stivuirea straturilor RNN ---
    # Primul strat BiRNN
    x = tf.keras.layers.Bidirectional(
        RecurrentLayer(units, return_sequences=True, kernel_regularizer=regularizer, recurrent_regularizer=regularizer),
        name=f"Bi{model_type.upper()}_Layer_1"
    )(current_layer)
    x = tf.keras.layers.Dropout(dropout_rate, name="Dropout_Layer_1")(x)

    # # Al doilea strat BiRNN (opțional, decomentează pentru a încerca)
    # x = tf.keras.layers.Bidirectional(
    #     RecurrentLayer(units // 2, return_sequences=True, kernel_regularizer=regularizer, recurrent_regularizer=regularizer), # Poate unități mai puține
    #     name=f"Bi{model_type.upper()}_Layer_2"
    # )(x)
    # x = tf.keras.layers.Dropout(dropout_rate, name="Dropout_Layer_2")(x)
    # -----------------------------------------------------

    recurrent_output = x  # Output-ul de la ultimul strat RNN/Dropout

    output_layer = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(num_classes_out, activation='softmax', kernel_regularizer=regularizer),
        name="Output_Layer"
    )(recurrent_output)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model


# --- Funcția analyze_predictions (adaptată pentru 2 clase) ---
def analyze_predictions_2_classes(model, X_test_data, y_test_original_unpadded_2_class, y_test_padded_indices_2_class,
                                  model_name_str, class_names=['O', 'I']):
    print(f"\n--- Detailed Analysis for {model_name_str} (2 Classes O,I) on Test Set ---")

    y_pred_probs_all_seqs = model.predict(X_test_data, batch_size=BATCH_SIZE)
    y_pred_classes_raw = np.argmax(y_pred_probs_all_seqs, axis=-1)
    y_pred_classes_to_evaluate = np.copy(y_pred_classes_raw)

    print("\nStandard Classification Report (2 Classes, exact frame-by-frame, non-padded):")
    report_labels = list(range(NUM_CLASSES_TARGET))
    target_names_report = class_names

    true_labels_flat, pred_labels_flat = [], []
    for i in range(len(y_test_original_unpadded_2_class)):
        true_seq_len = len(y_test_original_unpadded_2_class[i])
        if true_seq_len == 0: continue
        true_labels_flat.extend(y_test_padded_indices_2_class[i, :true_seq_len])
        pred_labels_flat.extend(y_pred_classes_to_evaluate[i, :true_seq_len])

    if true_labels_flat:
        # --- SUGESTIE DE RAFINARE: Calcul F1-score specific pentru clasa 'I' ---
        f1_i_class = f1_score(true_labels_flat, pred_labels_flat, labels=[LABEL_I_2CLASS], average='macro',
                              zero_division=0)
        print(f"Frame-level F1-Score for Class 'I' (Signs): {f1_i_class:.4f}")
        # -----------------------------------------------------------------------

        unique_true_plus_pred = np.unique(np.concatenate((true_labels_flat, pred_labels_flat)))
        current_report_labels = sorted([l for l in report_labels if l in unique_true_plus_pred])
        current_target_names = [class_names[i] for i in current_report_labels if i < len(class_names)]

        if not current_report_labels:
            print("No common labels for report.")
        else:
            report = classification_report(true_labels_flat, pred_labels_flat,
                                           target_names=current_target_names, labels=current_report_labels,
                                           zero_division=0)
            print(report)

        print("\nStandard Confusion Matrix (2 Classes, exact frame-by-frame, non-padded):")
        cm = confusion_matrix(true_labels_flat, pred_labels_flat, labels=report_labels)
        plt.figure(figsize=(6, 4));
        cm_tick_labels = class_names
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_tick_labels, yticklabels=cm_tick_labels)
        plt.title(f'Confusion Matrix - {model_name_str} (2 Classes)');
        plt.ylabel('Actual');
        plt.xlabel('Predicted')
        cm_filename = f"{model_name_str}_2class_confusion_matrix.png"
        cm_path = MODEL_SAVE_DIR / cm_filename
        plt.savefig(cm_path);
        plt.close();
        print(f"2-Class confusion matrix saved to {cm_path}")
    else:
        print("No non-padded labels for 2-class report/matrix.")


# --- plot_training_history (rămâne la fel) ---
def plot_training_history(history, model_name_suffix):
    model_name_prefix = "2ClassOI"
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1);
    plt.plot(history.history['loss'], label='Train Loss');
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name_prefix}_{model_name_suffix} Loss');
    plt.xlabel('Epochs');
    plt.ylabel('Loss');
    plt.legend();
    plt.grid(True)
    plt.subplot(1, 2, 2);
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
    plt.tight_layout();
    hist_path = MODEL_SAVE_DIR / f"{model_name_prefix}_{model_name_suffix}_training_history.png"
    plt.savefig(hist_path);
    plt.close();
    print(f"Training history plot saved to {hist_path}")


# --- Antrenament și Evaluare ---
input_shape_model = (effective_max_len, NUM_FEATURES)

# --- SUGESTIE DE RAFINARE: Optimizator cu rată de învățare customizată ---
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
# ----------------------------------------------------------------------

# --- SUGESTIE DE RAFINARE: Callback pentru reducerea ratei de învățare ---
reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=PATIENCE // 2, min_lr=0.00001, verbose=1
)
# ----------------------------------------------------------------------

# --- Train and Evaluate BiGRU (2 Classes) ---
# Ne concentrăm pe BiGRU deoarece a performat mai bine anterior cu 2 clase. Poți adăuga și BiLSTM dacă dorești.
print("\n--- Training BiGRU Model (2 Classes O,I - Refined) ---")
# --- SUGESTIE DE RAFINARE: Trecem add_conv_layer=True/False ---
USE_CONV_LAYER = False  # Setează la True pentru a experimenta cu Conv1D
# --------------------------------------------------------------
bigru_model_2class = build_recurrent_model(model_type='gru', units=GRU_UNITS, input_shape=input_shape_model,
                                           num_classes_out=NUM_CLASSES_TARGET, dropout_rate=DROPOUT_RATE,
                                           l2_reg=L2_REG_FACTOR,
                                           add_conv_layer=USE_CONV_LAYER)
bigru_model_2class.compile(optimizer=optimizer, loss='categorical_crossentropy',  # Folosim optimizatorul customizat
                           metrics=['accuracy'], weighted_metrics=['accuracy'])
bigru_model_2class.summary(line_length=120)
bigru_checkpoint_path = MODEL_SAVE_DIR / f"bigru_best_2class_OI{'_conv' if USE_CONV_LAYER else ''}.keras"
bigru_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(filepath=str(bigru_checkpoint_path), monitor='val_loss', save_best_only=True,
                                       verbose=1),
    reduce_lr_callback  # Adăugăm callback-ul ReduceLROnPlateau
]
history_bigru_2class = bigru_model_2class.fit(X_train_padded, y_train_one_hot, batch_size=BATCH_SIZE, epochs=EPOCHS,
                                              validation_data=(X_val_padded, y_val_one_hot), callbacks=bigru_callbacks,
                                              sample_weight=sample_weights_train, verbose=1)

print("\nEvaluating BiGRU 2-class model on Test Set...")
eval_results_bigru = bigru_model_2class.evaluate(X_test_padded, y_test_one_hot, verbose=0, batch_size=BATCH_SIZE)
print(
    f"BiGRU 2-Class Test Loss: {eval_results_bigru[0]:.4f}, Accuracy: {eval_results_bigru[1]:.4f}, Weighted Acc: {eval_results_bigru[2] if len(eval_results_bigru) > 2 else eval_results_bigru[1]:.4f}")
plot_training_history(history_bigru_2class, f"BiGRU_2Class_OI{'_conv' if USE_CONV_LAYER else ''}")
analyze_predictions_2_classes(bigru_model_2class, X_test_padded, y_test_raw_2_class, y_test_padded_2_class,
                              f"BiGRU_2Class_OI{'_conv' if USE_CONV_LAYER else ''}")

print("\n--- 2-Class Training Finished ---")
print(f"Best BiGRU 2-class model saved to: {bigru_checkpoint_path}")