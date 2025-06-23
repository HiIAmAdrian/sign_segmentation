import tensorflow as tf
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import traceback
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
from sklearn.utils import class_weight

DATA_BASE_DIR = Path("./final_data_ts_gloves_only")
FINAL_FEATURES_DATA_FILE = DATA_BASE_DIR / "all_data_features_ts_gloves_only.pkl"
FINAL_ANNOTATION_FILE = DATA_BASE_DIR / "annotations_bio_ts_gloves_only.pkl"
MODEL_SAVE_DIR = Path("./trained_models_2_classes_OI_ts_gloves")  # Director nou
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

LSTM_UNITS = 128
GRU_UNITS = 128
DROPOUT_RATE = 0.4
L2_REG_FACTOR = 0.0
LEARNING_RATE = 0.001

NUM_CLASSES_ORIG_BIO = 3
LABEL_O_3CLASS, LABEL_B_3CLASS, LABEL_I_3CLASS = 0, 1, 2

NUM_CLASSES_TARGET = 2
LABEL_O_2CLASS, LABEL_I_2CLASS = 0, 1

BATCH_SIZE = 16
EPOCHS = 70
PATIENCE = 20
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
    print("No GPU found. Training on CPU.")


def map_labels_to_2_classes(y_sequences_3_classes):
    y_sequences_2_classes = []
    for seq_3_class in y_sequences_3_classes:
        if seq_3_class is None: y_sequences_2_classes.append(None); continue
        seq_2_class = np.where(seq_3_class == LABEL_I_3CLASS, LABEL_I_2CLASS, LABEL_O_2CLASS)
        y_sequences_2_classes.append(seq_2_class.astype(np.int32))
    return y_sequences_2_classes


print(f"Loading TS+Gloves features from: {FINAL_FEATURES_DATA_FILE}")
try:
    with open(FINAL_FEATURES_DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    X_train_raw, X_val_raw, X_test_raw = data['X_train'], data['X_val'], data['X_test']
    if not X_train_raw or X_train_raw[0] is None: print(f"Error: X_train empty/None."); exit()
    NUM_FEATURES = X_train_raw[0].shape[1]
    print(
        f"Found {len(X_train_raw)} train, {len(X_val_raw)} val, {len(X_test_raw)} test sequences. Features: {NUM_FEATURES}")
except Exception as e:
    print(f"Error loading feature data: {e}"); traceback.print_exc(); exit()

print(f"Loading 3-class annotations (to be mapped) from: {FINAL_ANNOTATION_FILE}")
try:
    with open(FINAL_ANNOTATION_FILE, 'rb') as f:
        annotations_3_class = pickle.load(f)
    y_train_raw_3_class, y_val_raw_3_class, y_test_raw_3_class = annotations_3_class['train'], annotations_3_class[
        'val'], annotations_3_class['test']

    print("Mapping 3-class labels to 2-class labels (O,I)...")
    y_train_raw_2_class = map_labels_to_2_classes(y_train_raw_3_class)
    y_val_raw_2_class = map_labels_to_2_classes(y_val_raw_3_class)
    y_test_raw_2_class = map_labels_to_2_classes(y_test_raw_3_class)

    for name, X, y in [("Train", X_train_raw, y_train_raw_2_class), ("Val", X_val_raw, y_val_raw_2_class),
                       ("Test", X_test_raw, y_test_raw_2_class)]:
        if len(X) != len(y): raise AssertionError(f"{name} X/Y length mismatch")
        for i in range(len(X)):
            if X[i] is None or y[i] is None: raise AssertionError(f"{name} seq {i} None")
            if X[i].shape[0] != y[i].shape[0]: raise AssertionError(f"{name} seq {i} X/Y frame mismatch")
    print("Annotations mapped and validated for 2 classes.")
except Exception as e:
    print(f"Error loading/mapping annotation data: {e}"); traceback.print_exc(); exit()

all_lengths = [s.shape[0] for d in [X_train_raw, X_val_raw, X_test_raw] for s in d if s is not None]
if not all_lengths: print("Error: No sequences for MAX_LEN."); exit()
MAX_LEN_CALCULATED = np.max(all_lengths);
effective_max_len = MAX_LEN_CALCULATED
print(f"MAX_LEN: {MAX_LEN_CALCULATED}")

X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(X_train_raw, padding=PADDING_TYPE, dtype='float32',
                                                            maxlen=MAX_LEN_CALCULATED, value=0.0)
X_val_pad = tf.keras.preprocessing.sequence.pad_sequences(X_val_raw, padding=PADDING_TYPE, dtype='float32',
                                                          maxlen=MAX_LEN_CALCULATED, value=0.0)
X_test_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_raw, padding=PADDING_TYPE, dtype='float32',
                                                           maxlen=MAX_LEN_CALCULATED, value=0.0)
y_train_pad_2cls = tf.keras.preprocessing.sequence.pad_sequences(y_train_raw_2_class, padding=PADDING_TYPE,
                                                                 value=LABEL_O_2CLASS, maxlen=MAX_LEN_CALCULATED)
y_val_pad_2cls = tf.keras.preprocessing.sequence.pad_sequences(y_val_raw_2_class, padding=PADDING_TYPE,
                                                               value=LABEL_O_2CLASS, maxlen=MAX_LEN_CALCULATED)
y_test_pad_2cls = tf.keras.preprocessing.sequence.pad_sequences(y_test_raw_2_class, padding=PADDING_TYPE,
                                                                value=LABEL_O_2CLASS, maxlen=MAX_LEN_CALCULATED)
print(f"Padding done. Effective max_len: {MAX_LEN_CALCULATED}")

y_train_oh = tf.keras.utils.to_categorical(y_train_pad_2cls.astype(int), num_classes=NUM_CLASSES_TARGET)
y_val_oh = tf.keras.utils.to_categorical(y_val_pad_2cls.astype(int), num_classes=NUM_CLASSES_TARGET)
y_test_oh = tf.keras.utils.to_categorical(y_test_pad_2cls.astype(int), num_classes=NUM_CLASSES_TARGET)
print(f"One-hot encoding done. X_train_pad: {X_train_pad.shape}, y_train_oh: {y_train_oh.shape}")

y_flat_weights = []
for i in range(len(y_train_raw_2_class)):
    if y_train_raw_2_class[i] is not None:
        y_flat_weights.extend(y_train_pad_2cls[i, :len(y_train_raw_2_class[i])])

class_w_dict, sample_w_train = None, None
if y_flat_weights:
    classes_unique = np.unique(y_flat_weights)
    all_possible_cls = np.arange(NUM_CLASSES_TARGET)
    class_w_vals = class_weight.compute_class_weight(class_weight='balanced', classes=all_possible_cls,
                                                     y=y_flat_weights)
    class_w_dict = {all_possible_cls[i]: class_w_vals[i] for i in range(len(all_possible_cls))}
    print(f"Class Weights (2 classes): {class_w_dict}")
    sample_w_train = np.ones_like(y_train_pad_2cls, dtype=float)
    for i in range(y_train_pad_2cls.shape[0]):
        if y_train_raw_2_class[i] is not None:
            l_orig = len(y_train_raw_2_class[i])
            for j in range(l_orig): sample_w_train[i, j] = class_w_dict.get(y_train_pad_2cls[i, j], 1.0)
            if l_orig < MAX_LEN_CALCULATED: sample_w_train[i, l_orig:] = 0.0
        else:
            sample_w_train[i, :] = 0.0
else:
    print("Warning: No class weights calculated because y_flat_weights is empty.")


def build_recurrent_model(model_type='gru', units=128, input_shape=None, num_classes_out=2, dropout_rate=0.3,
                          l2_reg=0.0, add_conv_layer=False):
    if input_shape is None: raise ValueError("input_shape required")
    inp = tf.keras.layers.Input(shape=input_shape, name="Input_Layer")
    curr = tf.keras.layers.Masking(mask_value=0.0, name="Masking_Layer")(inp)
    reg = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    if add_conv_layer:
        curr = tf.keras.layers.Conv1D(filters=units // 2, kernel_size=5, padding="same", activation="relu",
                                      name="Conv1D_FeatExt")(curr)
        curr = tf.keras.layers.BatchNormalization(name="BatchNorm_Conv")(curr)
    RecLayer = tf.keras.layers.LSTM if model_type == 'lstm' else tf.keras.layers.GRU
    x = tf.keras.layers.Bidirectional(
        RecLayer(units, return_sequences=True, kernel_regularizer=reg, recurrent_regularizer=reg),
        name=f"Bi{model_type.upper()}_L1")(curr)
    x = tf.keras.layers.Dropout(dropout_rate, name="Drop_1")(x)
    out_dense = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(num_classes_out, activation='softmax', kernel_regularizer=reg), name="Output_Layer")(x)
    return tf.keras.Model(inputs=inp, outputs=out_dense)


def analyze_predictions_2_classes(model, X_data, y_orig_2cls_list, y_pad_2cls_array, model_name, class_names=['O', 'I']):
    print(f"\n--- Analysis for {model_name} (2 Classes O,I) on provided data ---")
    y_probs = model.predict(X_data, batch_size=BATCH_SIZE)
    y_preds_pad = np.argmax(y_probs, axis=-1)
    true_flat, pred_flat = [], []

    for i in range(len(y_orig_2cls_list)):
        if y_orig_2cls_list[i] is None: continue
        original_length = len(y_orig_2cls_list[i])
        if original_length == 0: continue

        true_flat.extend(y_pad_2cls_array[i, :original_length])
        pred_flat.extend(y_preds_pad[i, :original_length])

    if true_flat and pred_flat:
        f1_i = f1_score(true_flat, pred_flat, labels=[LABEL_I_2CLASS], average='binary', zero_division=0)
        print(f"Frame-level F1-Score for Class 'I' (Signs): {f1_i:.4f}")

        print(classification_report(true_flat, pred_flat, target_names=class_names, labels=list(range(len(class_names))), zero_division=0))

        cm = confusion_matrix(true_flat, pred_flat, labels=list(range(len(class_names))))
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'CM - {model_name} (2 Classes)')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(MODEL_SAVE_DIR / f"{model_name}_2cls_CM.png")
        plt.close()
    else:
        print("No valid labels found for report/matrix.")


def plot_training_history(history, model_name_suffix):
    prefix = "2Cls_TSGloves"
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{prefix}_{model_name_suffix} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    acc_k, val_acc_k = 'accuracy', 'val_accuracy'
    if 'weighted_accuracy' in history.history and 'val_weighted_accuracy' in history.history:
        acc_k = 'weighted_accuracy'
        val_acc_k = 'val_weighted_accuracy'
    elif 'accuracy' not in history.history:
        acc_k = next(iter([k for k in history.history.keys() if 'acc' in k and 'val' not in k]), 'accuracy')
        val_acc_k = next(iter([k for k in history.history.keys() if 'acc' in k and 'val' in k]), 'val_accuracy')

    plt.plot(history.history[acc_k], label=f'Train {acc_k.replace("_", " ").title()}')
    plt.plot(history.history[val_acc_k], label=f'Val {val_acc_k.replace("_", " ").title()}')
    plt.title(f'{prefix}_{model_name_suffix} Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(MODEL_SAVE_DIR / f"{prefix}_{model_name_suffix}_hist.png")
    plt.close()


input_shape_model = (effective_max_len, NUM_FEATURES)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=PATIENCE // 2, min_lr=1e-6,
                                                    verbose=1)
USE_CONV = False

print(f"\n--- Training BiGRU (2 Classes, TS+Gloves {'+Conv' if USE_CONV else ''}) ---")
model_name_suffix_bigru = f"BiGRU_2Cls_OI_TSG{'_Conv' if USE_CONV else ''}"
model_bigru_tsg_2cls = build_recurrent_model(model_type='gru', units=GRU_UNITS, input_shape=input_shape_model,
                                       num_classes_out=NUM_CLASSES_TARGET, dropout_rate=DROPOUT_RATE,
                                       l2_reg=L2_REG_FACTOR, add_conv_layer=USE_CONV)
model_bigru_tsg_2cls.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'],
                       weighted_metrics=['accuracy'])
model_bigru_tsg_2cls.summary(line_length=120)
checkpoint_path_bigru = MODEL_SAVE_DIR / f"best_{model_name_suffix_bigru}.keras"
callbacks_list_bigru = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_path_bigru), monitor='val_loss', save_best_only=True,
                                       verbose=1),
    reduce_lr_cb
]
history_bigru = model_bigru_tsg_2cls.fit(X_train_pad, y_train_oh, batch_size=BATCH_SIZE, epochs=EPOCHS,
                             validation_data=(X_val_pad, y_val_oh), callbacks=callbacks_list_bigru,
                             sample_weight=sample_w_train, verbose=1)

print(f"\nEvaluating {model_name_suffix_bigru} on Test Set...")
eval_res_bigru = model_bigru_tsg_2cls.evaluate(X_test_pad, y_test_oh, verbose=0, batch_size=BATCH_SIZE)
print(
    f"Test Loss: {eval_res_bigru[0]:.4f}, Acc: {eval_res_bigru[1]:.4f}, Weighted Acc: {eval_res_bigru[2] if len(eval_res_bigru) > 2 else eval_res_bigru[1]:.4f}")
plot_training_history(history_bigru, model_name_suffix_bigru)
analyze_predictions_2_classes(model_bigru_tsg_2cls, X_test_pad, y_test_raw_2_class, y_test_pad_2cls, model_name_suffix_bigru)

print(f"\n--- Training Finished ({model_name_suffix_bigru}) ---")
print(f"Best BiGRU model saved to: {checkpoint_path_bigru}")


print(f"\n--- Training BiLSTM (2 Classes, TS+Gloves {'+Conv' if USE_CONV else ''}) ---")
model_name_suffix_bilstm = f"BiLSTM_2Cls_OI_TSG{'_Conv' if USE_CONV else ''}"
optimizer_bilstm = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model_bilstm_tsg_2cls = build_recurrent_model(model_type='lstm', units=LSTM_UNITS, input_shape=input_shape_model,
                                       num_classes_out=NUM_CLASSES_TARGET, dropout_rate=DROPOUT_RATE,
                                       l2_reg=L2_REG_FACTOR, add_conv_layer=USE_CONV)
model_bilstm_tsg_2cls.compile(optimizer=optimizer_bilstm, loss='categorical_crossentropy', metrics=['accuracy'],
                       weighted_metrics=['accuracy'])
model_bilstm_tsg_2cls.summary(line_length=120)
checkpoint_path_bilstm = MODEL_SAVE_DIR / f"best_{model_name_suffix_bilstm}.keras"
reduce_lr_cb_bilstm = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=PATIENCE // 2, min_lr=1e-6, verbose=1)

callbacks_list_bilstm = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_path_bilstm), monitor='val_loss', save_best_only=True,
                                       verbose=1),
    reduce_lr_cb_bilstm
]
history_bilstm = model_bilstm_tsg_2cls.fit(X_train_pad, y_train_oh, batch_size=BATCH_SIZE, epochs=EPOCHS,
                             validation_data=(X_val_pad, y_val_oh), callbacks=callbacks_list_bilstm,
                             sample_weight=sample_w_train, verbose=1)

print(f"\nEvaluating {model_name_suffix_bilstm} on Test Set...")
eval_res_bilstm = model_bilstm_tsg_2cls.evaluate(X_test_pad, y_test_oh, verbose=0, batch_size=BATCH_SIZE)
print(
    f"Test Loss: {eval_res_bilstm[0]:.4f}, Acc: {eval_res_bilstm[1]:.4f}, Weighted Acc: {eval_res_bilstm[2] if len(eval_res_bilstm) > 2 else eval_res_bilstm[1]:.4f}")
plot_training_history(history_bilstm, model_name_suffix_bilstm)
analyze_predictions_2_classes(model_bilstm_tsg_2cls, X_test_pad, y_test_raw_2_class, y_test_pad_2cls, model_name_suffix_bilstm)

print(f"\n--- Training Finished ({model_name_suffix_bilstm}) ---")
print(f"Best BiLSTM model saved to: {checkpoint_path_bilstm}")

print("\n\nAll training complete.")