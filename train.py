from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Bidirectional, LSTM, GRU, Dense, TimeDistributed, Masking, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

# --- Configuration ---

PROCESSED_DATA_DIR = Path("./processed_combined_data_both_gloves") # Dir containing the .pkl files
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "combined_interpolated_processed_sequences.pkl"
SCALER_FILE = PROCESSED_DATA_DIR / "combined_interpolated_scaler.pkl"

# !!! IMPORTANT: Path to your annotation file !!!
# You need to create this file. It should contain a dictionary like:
# {'y_train': [...], 'y_val': [...], 'y_test': [...]}
# where each value is a list of NumPy arrays (int labels 0, 1, 2)
ANNOTATION_FILE = Path("./processed_combined_data_both_gloves/annotations.pkl")

MODEL_SAVE_DIR = Path("./trained_models")
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Model Hyperparameters (tune these)
LSTM_UNITS = 128
GRU_UNITS = 128
DROPOUT_RATE = 0.4 # Increased dropout for potentially small dataset
NUM_CLASSES = 2 # B, I, O (0, 1, 2) - Adjust if using BIOES etc.

# Training Parameters
BATCH_SIZE = 32 # Adjust based on memory
EPOCHS = 100 # Max epochs, EarlyStopping will prevent overfitting
PATIENCE = 10 # For EarlyStopping

# Padding - Choose 'pre' or 'post'. 'post' is often slightly preferred.
PADDING_TYPE = 'post'
# Max sequence length - None pads to the longest sequence in the batch/dataset
# Can set a specific number (e.g., 3000) if memory is an issue and sequences are very long
MAX_LEN = None

# --- Load Data ---

print(f"Loading processed data from: {PROCESSED_DATA_FILE}")
try:
    with open(PROCESSED_DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    X_train_raw = data['X_train']
    X_val_raw = data['X_val']
    X_test_raw = data['X_test']
    # train_ids = data['train_ids'] # Not used directly in training, but useful for matching
    # val_ids = data['val_ids']
    # test_ids = data['test_ids']
    feature_names = data['feature_names']
    NUM_FEATURES = len(feature_names)
    print(f"Found {len(X_train_raw)} train, {len(X_val_raw)} val, {len(X_test_raw)} test sequences.")
    print(f"Number of features per frame: {NUM_FEATURES}")
except FileNotFoundError:
    print(f"Error: Processed data file not found at {PROCESSED_DATA_FILE}")
    exit()
except KeyError as e:
    print(f"Error: Missing key {e} in processed data file. Ensure the file is correct.")
    exit()

print(f"Loading annotations from: {ANNOTATION_FILE}")
try:
    with open(ANNOTATION_FILE, 'rb') as f:
        annotations = pickle.load(f)
    y_train_raw = annotations['train']
    y_val_raw = annotations['val']
    y_test_raw = annotations['test']
    # Basic validation
    assert len(X_train_raw) == len(y_train_raw)
    assert len(X_val_raw) == len(y_val_raw)
    assert len(X_test_raw) == len(y_test_raw)
    print("Annotations loaded successfully.")
except FileNotFoundError:
    print(f"Error: Annotation file not found at {ANNOTATION_FILE}")
    print("Please create this file containing lists of integer label sequences (0=O, 1=B, 2=I) named 'y_train', 'y_val', 'y_test'.")
    exit()
except (KeyError, AssertionError) as e:
    print(f"Error: Annotation file format incorrect or lengths don't match feature sequences: {e}")
    exit()

# --- Prepare Data for Model ---

# 1. Padding Sequences
# We pad both features (X) and labels (y)
print("Padding sequences...")
X_train_padded = pad_sequences(X_train_raw, padding=PADDING_TYPE, dtype='float32', maxlen=MAX_LEN)
X_val_padded = pad_sequences(X_val_raw, padding=PADDING_TYPE, dtype='float32', maxlen=MAX_LEN)
X_test_padded = pad_sequences(X_test_raw, padding=PADDING_TYPE, dtype='float32', maxlen=MAX_LEN)

# Pad labels - Use a value unlikely to be a real class label (e.g., -1 if classes are 0,1,2)
# Or pad with 0 if using sparse categorical crossentropy (but we use one-hot below)
# Let's pad with 0 and handle it via one-hot encoding dimensions later.
y_train_padded = pad_sequences(y_train_raw, padding=PADDING_TYPE, value=0, maxlen=MAX_LEN) # Pad with 'O' class index
y_val_padded = pad_sequences(y_val_raw, padding=PADDING_TYPE, value=0, maxlen=MAX_LEN)
y_test_padded = pad_sequences(y_test_raw, padding=PADDING_TYPE, value=0, maxlen=MAX_LEN)

# Determine max length after padding (useful for input shape)
# If MAX_LEN was None, it's the length of the longest sequence
effective_max_len = X_train_padded.shape[1]
print(f"Sequences padded to length: {effective_max_len}")

# 2. One-Hot Encode Labels
print("One-hot encoding labels...")
# Ensure labels are integers before one-hot encoding
y_train_padded = y_train_padded.astype(int)
y_val_padded = y_val_padded.astype(int)
y_test_padded = y_test_padded.astype(int)

y_train_one_hot = to_categorical(y_train_padded, num_classes=NUM_CLASSES)
y_val_one_hot = to_categorical(y_val_padded, num_classes=NUM_CLASSES)
y_test_one_hot = to_categorical(y_test_padded, num_classes=NUM_CLASSES)

print(f"Shape of X_train_padded: {X_train_padded.shape}")
print(f"Shape of y_train_one_hot: {y_train_one_hot.shape}")

# --- Build Model ---

def build_recurrent_model(model_type='lstm', units=128, input_shape=None, num_classes=3, dropout_rate=0.3):
    """Builds a Bidirectional LSTM or GRU model for sequence tagging."""
    if input_shape is None:
        raise ValueError("input_shape must be provided (e.g., (None, num_features))")

    input_layer = Input(shape=input_shape, name="Input_Layer")

    # Masking layer to ignore padding (assuming padding value 0.0 for features)
    # Note: Ensure your padding value in pad_sequences matches mask_value
    masked_input = Masking(mask_value=0.0, name="Masking_Layer")(input_layer)

    # Bidirectional Recurrent Layer
    if model_type == 'lstm':
        recurrent_layer = Bidirectional(
            LSTM(units, return_sequences=True), name="BiLSTM_Layer"
        )(masked_input)
    elif model_type == 'gru':
        recurrent_layer = Bidirectional(
            GRU(units, return_sequences=True), name="BiGRU_Layer"
        )(masked_input)
    else:
        raise ValueError("model_type must be 'lstm' or 'gru'")

    # Dropout for regularization
    dropout_layer = Dropout(dropout_rate, name="Dropout_Layer")(recurrent_layer)

    # TimeDistributed Dense Output Layer
    # Applies the same dense layer to each time step
    output_layer = TimeDistributed(
        Dense(num_classes, activation='softmax'), name="Output_Layer"
    )(dropout_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# --- Train and Evaluate BiLSTM ---

print("\n--- Training BiLSTM Model ---")
input_shape_bilstm = (effective_max_len, NUM_FEATURES) # Or (None, NUM_FEATURES) if not padding globally

bilstm_model = build_recurrent_model(
    model_type='lstm',
    units=LSTM_UNITS,
    input_shape=input_shape_bilstm,
    num_classes=NUM_CLASSES,
    dropout_rate=DROPOUT_RATE
)

bilstm_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss='categorical_crossentropy', # Use this for one-hot encoded labels
    metrics=['accuracy'] # Frame-level accuracy
    # Add weighted metrics later if class imbalance is an issue
)

bilstm_model.summary()

# Callbacks
bilstm_checkpoint_path = MODEL_SAVE_DIR / "bilstm_best.keras" # Use .keras format
bilstm_callbacks = [
    EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath=bilstm_checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
]

# Train the model
history_bilstm = bilstm_model.fit(
    X_train_padded, y_train_one_hot,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val_padded, y_val_one_hot),
    callbacks=bilstm_callbacks,
    verbose=1
)

# Load best weights just to be sure (if EarlyStopping restored them)
# bilstm_model.load_weights(bilstm_checkpoint_path) # Usually handled by restore_best_weights

# Evaluate on Test Set
print("\nEvaluating BiLSTM model on Test Set...")
bilstm_loss, bilstm_accuracy = bilstm_model.evaluate(X_test_padded, y_test_one_hot, verbose=0)
print(f"BiLSTM Test Loss: {bilstm_loss:.4f}")
print(f"BiLSTM Test Accuracy: {bilstm_accuracy:.4f}")

# Save the final model (optional, checkpoint saves the best one)
# bilstm_model.save(MODEL_SAVE_DIR / "bilstm_final.keras")

# Plot training history (optional)
def plot_history(history, model_name):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(MODEL_SAVE_DIR / f"{model_name}_training_history.png")
    # plt.show() # Uncomment to display plots immediately

plot_history(history_bilstm, "BiLSTM")


# --- Train and Evaluate BiGRU ---

print("\n--- Training BiGRU Model ---")
input_shape_bigru = (effective_max_len, NUM_FEATURES)

bigru_model = build_recurrent_model(
    model_type='gru',
    units=GRU_UNITS,
    input_shape=input_shape_bigru,
    num_classes=NUM_CLASSES,
    dropout_rate=DROPOUT_RATE
)

bigru_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

bigru_model.summary()

# Callbacks
bigru_checkpoint_path = MODEL_SAVE_DIR / "bigru_best.keras"
bigru_callbacks = [
    EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath=bigru_checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
]

# Train the model
history_bigru = bigru_model.fit(
    X_train_padded, y_train_one_hot,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val_padded, y_val_one_hot),
    callbacks=bigru_callbacks,
    verbose=1
)

# Evaluate on Test Set
print("\nEvaluating BiGRU model on Test Set...")
bigru_loss, bigru_accuracy = bigru_model.evaluate(X_test_padded, y_test_one_hot, verbose=0)
print(f"BiGRU Test Loss: {bigru_loss:.4f}")
print(f"BiGRU Test Accuracy: {bigru_accuracy:.4f}")

# Plot training history
plot_history(history_bigru, "BiGRU")

print("\n--- Training Finished ---")
print(f"Best BiLSTM model saved to: {bilstm_checkpoint_path}")
print(f"Best BiGRU model saved to: {bigru_checkpoint_path}")
print(f"Training plots saved to: {MODEL_SAVE_DIR}")