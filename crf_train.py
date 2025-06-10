# --- START OF FILE train_pytorch_optimized_crf.py ---

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF  # Import the optimized CRF layer

import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import traceback
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.utils import class_weight  # Still used for analysis/potential future use
import copy

# --- Configuration ---
FINAL_DATA_DIR = Path("./final_combined_data_for_training_ALL_SIGNERS")
FINAL_FEATURES_DATA_FILE = FINAL_DATA_DIR / "all_data_final_features_ts_facial.pkl"
FINAL_ANNOTATION_FILE = FINAL_DATA_DIR / "annotations_bio_final_combined.pkl"
MODEL_SAVE_DIR = Path("./trained_models_final_pytorch_optimized_crf")
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

LSTM_UNITS = 128
GRU_UNITS = 128
DROPOUT_RATE = 0.4
L2_REG_FACTOR = 0.0
NUM_CLASSES = 3  # O, B, I (actual classes for CRF emissions)
LABEL_O, LABEL_B, LABEL_I = 0, 1, 2
BATCH_SIZE = 8  # Can potentially increase this now
EPOCHS = 50
PATIENCE = 15
MAX_LEN_CALCULATED = None

# --- PyTorch Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Num GPUs Available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU found by PyTorch. Training will use CPU.")

# --- Load Data ---
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
    print(f"Error loading feature data: {e}");
    traceback.print_exc();
    exit()

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
    print(f"Error loading annotation data: {e}");
    traceback.print_exc();
    exit()

all_sequence_lengths = [seq.shape[0] for dataset in [X_train_raw, X_val_raw, X_test_raw] for seq in dataset if
                        seq is not None]
if not all_sequence_lengths: print("Error: No sequences to determine max length."); exit()
MAX_LEN_CALCULATED = np.max(all_sequence_lengths)
print(f"Determined MAX_LEN from data: {MAX_LEN_CALCULATED}")
effective_max_len = MAX_LEN_CALCULATED


# --- Preprocessing and PyTorch Dataset/DataLoader ---
def pad_sequences_numpy(sequences, maxlen, padding_value=0.0, dtype='float32'):
    padded_sequences = np.full((len(sequences), maxlen) + sequences[0].shape[1:], padding_value, dtype=dtype)
    for i, seq in enumerate(sequences):
        length = seq.shape[0]
        padded_sequences[i, :length] = seq
    return padded_sequences


def pad_labels_numpy(sequences, maxlen, padding_value=LABEL_O, dtype='int64'):
    padded_sequences = np.full((len(sequences), maxlen), padding_value, dtype=dtype)
    for i, seq in enumerate(sequences):
        length = len(seq)
        padded_sequences[i, :length] = seq
    return padded_sequences


print("Padding sequences (NumPy)...")
X_train_padded_np = pad_sequences_numpy(X_train_raw, maxlen=effective_max_len, dtype='float32', padding_value=0.0)
X_val_padded_np = pad_sequences_numpy(X_val_raw, maxlen=effective_max_len, dtype='float32', padding_value=0.0)
X_test_padded_np = pad_sequences_numpy(X_test_raw, maxlen=effective_max_len, dtype='float32', padding_value=0.0)

y_train_padded_np = pad_labels_numpy(y_train_raw, maxlen=effective_max_len, padding_value=LABEL_O)
y_val_padded_np = pad_labels_numpy(y_val_raw, maxlen=effective_max_len, padding_value=LABEL_O)
y_test_padded_np = pad_labels_numpy(y_test_raw, maxlen=effective_max_len, padding_value=LABEL_O)

print(f"Sequences padded to length: {effective_max_len}")
print(f"Shape X_train_padded_np: {X_train_padded_np.shape}, y_train_padded_np: {y_train_padded_np.shape}")

train_lengths = [len(seq) for seq in X_train_raw]
val_lengths = [len(seq) for seq in X_val_raw]
test_lengths = [len(seq) for seq in X_test_raw]

# --- Class/Sample Weights (Original calculation for reference/potential future use) ---
# This part is kept for context, but sample_weights_train_np won't be directly used in CRF loss
# as pytorch-crf uses a mask for padding.
y_train_labels_flat_for_weights = []
for i in range(len(y_train_raw)):
    true_seq_len = train_lengths[i]
    y_train_labels_flat_for_weights.extend(y_train_padded_np[i, :true_seq_len])

class_weights_dict_calc = None
if y_train_labels_flat_for_weights:
    unique_classes_in_train = np.unique(y_train_labels_flat_for_weights)
    class_weights_values = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=unique_classes_in_train,
        y=y_train_labels_flat_for_weights
    )
    temp_class_weights_dict = {unique_classes_in_train[i]: class_weights_values[i] for i in
                               range(len(unique_classes_in_train))}
    class_weights_dict_calc = {cls: temp_class_weights_dict.get(cls, 1.0) for cls in range(NUM_CLASSES)}
    target_b_weight = 30.0
    if LABEL_B in class_weights_dict_calc: class_weights_dict_calc[LABEL_B] = target_b_weight
    print(f"Reference Class Weights (not directly used in pytorch-crf loss): {class_weights_dict_calc}")
else:
    print("Warning: Could not calculate class weights for reference.")


# --- PyTorch Datasets and DataLoaders ---
class SequenceDataset(Dataset):
    def __init__(self, features, labels, lengths):  # Removed sample_weights from here
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.lengths = torch.tensor(lengths, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Create mask here based on length
        mask = torch.zeros(effective_max_len, dtype=torch.bool)
        mask[:self.lengths[idx]] = True
        return self.features[idx], self.labels[idx], self.lengths[idx], mask


train_dataset = SequenceDataset(X_train_padded_np, y_train_padded_np, train_lengths)
val_dataset = SequenceDataset(X_val_padded_np, y_val_padded_np, val_lengths)
test_dataset = SequenceDataset(X_test_padded_np, y_test_padded_np, test_lengths)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# --- PyTorch Model with Optimized CRF ---
class RecurrentModelWithCRF(nn.Module):
    def __init__(self, input_dim, hidden_dim_rnn_output, num_actual_classes, recurrent_type='lstm', dropout_rate=0.3):
        super(RecurrentModelWithCRF, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_rnn_output = hidden_dim_rnn_output  # This is output of BiRNN (e.g., LSTM_UNITS*2)
        self.num_actual_classes = num_actual_classes  # O, B, I

        if recurrent_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim_rnn_output // 2,
                               num_layers=1, bidirectional=True, batch_first=True)
        elif recurrent_type == 'gru':
            self.rnn = nn.GRU(input_dim, hidden_dim_rnn_output // 2,
                              num_layers=1, bidirectional=True, batch_first=True)
        else:
            raise ValueError("recurrent_type must be 'lstm' or 'gru'")

        self.dropout = nn.Dropout(dropout_rate)
        # Maps the output of the RNN to emission scores for actual classes
        self.hidden2tag = nn.Linear(hidden_dim_rnn_output, self.num_actual_classes)

        # CRF layer from pytorch-crf library
        self.crf = CRF(num_tags=self.num_actual_classes, batch_first=True)

    def _get_rnn_features(self, sentences, lengths):
        # sentences: (batch, seq_len, input_dim)
        # lengths: (batch,)
        # The lengths tensor needs to be on CPU for pack_padded_sequence
        packed_input = pack_padded_sequence(sentences, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_input)
        rnn_out, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=sentences.size(1))
        rnn_out = self.dropout(rnn_out)
        emission_scores = self.hidden2tag(rnn_out)  # (batch, seq_len, self.num_actual_classes)
        return emission_scores

    def compute_loss(self, sentences, tags, lengths, mask):
        # sentences: (batch, seq_len, input_dim)
        # tags: (batch, seq_len) - original labels 0,1,2
        # lengths: (batch,)
        # mask: (batch, seq_len) - boolean mask, True for valid tokens

        emission_scores = self._get_rnn_features(sentences, lengths)  # (batch, seq_len, self.num_actual_classes)

        # The CRF layer's forward method returns the log likelihood.
        # Loss is the negative log likelihood.
        # Mask is crucial here for handling padding.
        log_likelihood = self.crf(emissions=emission_scores, tags=tags, mask=mask, reduction='mean')
        return -log_likelihood  # NLL loss

    def predict(self, sentences, lengths, mask):
        # sentences: (batch, seq_len, input_dim)
        # lengths: (batch,)
        # mask: (batch, seq_len)
        self.eval()
        with torch.no_grad():
            emission_scores = self._get_rnn_features(sentences, lengths)  # (batch, seq_len, self.num_actual_classes)
            # The decode method returns a list of lists, where each inner list is the best path for a sequence.
            predicted_paths = self.crf.decode(emissions=emission_scores, mask=mask)
        return predicted_paths


# --- BIO Sequence Validity and Post-processing (Unchanged from previous PyTorch version) ---
def check_bio_sequence_validity(sequence):
    if not isinstance(sequence, (list, np.ndarray)) or len(sequence) == 0: return True
    seq = [int(s) for s in sequence]
    n = len(seq)
    first_non_o_idx = -1
    for i in range(n):
        if seq[i] != LABEL_O: first_non_o_idx = i; break
    if first_non_o_idx != -1 and seq[first_non_o_idx] == LABEL_I: return False
    for i in range(n):
        current_label, prev_label = seq[i], (seq[i - 1] if i > 0 else None)
        # next_label = seq[i+1] if i < n-1 else None # Not used in this simplified check
        if current_label == LABEL_I:
            if prev_label is None or prev_label == LABEL_O: return False
    return True


def post_process_bio_sequence(sequence_pred):
    if not isinstance(sequence_pred, (list, np.ndarray)) or len(sequence_pred) == 0: return sequence_pred
    corrected_seq = list(sequence_pred)
    n = len(corrected_seq)
    first_non_o_idx = -1
    for i in range(n):
        if corrected_seq[i] != LABEL_O: first_non_o_idx = i; break
    if first_non_o_idx != -1 and corrected_seq[first_non_o_idx] == LABEL_I:
        corrected_seq[first_non_o_idx] = LABEL_B
    i = 0
    while i < n:
        current_label = corrected_seq[i]
        if i + 1 < n:
            next_label = corrected_seq[i + 1]
            if current_label == LABEL_O and next_label == LABEL_I:
                corrected_seq[i] = LABEL_B
            elif current_label == LABEL_B and next_label == LABEL_B:
                corrected_seq[i + 1] = LABEL_I
        i += 1
    return np.array(corrected_seq, dtype=int)


# --- Analysis and Plotting (Largely unchanged, uses model.predict) ---
def analyze_predictions(model_pytorch, dataloader_torch, y_original_unpadded_list, y_padded_indices_np,
                        model_name_str, class_names=['O', 'B', 'I'], b_tolerance_frames=30,
                        apply_post_processing=False):
    print(f"\n--- Detailed Analysis for {model_name_str} on Test Set ---")
    if apply_post_processing:
        print("--- (Results AFTER Post-Processing) ---")
    else:
        print(f"--- (B-Class Tolerance: +/- {b_tolerance_frames} frames) ---")

    all_pred_paths_unpadded = []
    model_pytorch.eval()
    with torch.no_grad():
        for features_batch, _, lengths_batch, mask_batch in dataloader_torch:
            features_batch = features_batch.to(device)
            # lengths_batch = lengths_batch.to(device) # Not strictly needed for predict if mask is used
            mask_batch = mask_batch.to(device)
            predicted_paths_batch = model_pytorch.predict(features_batch, lengths_batch,
                                                          mask_batch)  # lengths_batch for _get_rnn_features
            all_pred_paths_unpadded.extend(predicted_paths_batch)

    y_pred_classes_padded_np = np.full_like(y_padded_indices_np, LABEL_O, dtype=int)
    for i, path in enumerate(all_pred_paths_unpadded):
        path_len = len(path)
        if path_len > 0:
            y_pred_classes_padded_np[i, :path_len] = path

    y_pred_classes_to_evaluate = np.copy(y_pred_classes_padded_np)

    if apply_post_processing:
        print("Applying post-processing to predicted sequences...")
        for i in range(y_pred_classes_padded_np.shape[0]):
            original_length = len(y_original_unpadded_list[i])
            raw_pred_seq_segment = y_pred_classes_padded_np[i, :original_length]
            corrected_seq_segment = post_process_bio_sequence(raw_pred_seq_segment)
            y_pred_classes_to_evaluate[i, :original_length] = corrected_seq_segment
            if original_length < y_pred_classes_to_evaluate.shape[1]:
                y_pred_classes_to_evaluate[i, original_length:] = LABEL_O

    tp_b_tolerant, fn_b_tolerant, total_real_b = 0, 0, 0
    all_predicted_b_locations = []
    valid_predicted_sequences_count = 0
    total_sequences_evaluated = len(y_original_unpadded_list)

    for i in range(total_sequences_evaluated):
        true_seq_len = len(y_original_unpadded_list[i])
        current_true_labels = y_padded_indices_np[i, :true_seq_len]
        current_pred_labels = y_pred_classes_to_evaluate[i, :true_seq_len]

        if check_bio_sequence_validity(current_pred_labels): valid_predicted_sequences_count += 1
        real_b_indices_in_seq = np.where(current_true_labels == LABEL_B)[0]
        pred_b_indices_in_seq = np.where(current_pred_labels == LABEL_B)[0]
        total_real_b += len(real_b_indices_in_seq)
        for pred_b_idx in pred_b_indices_in_seq:
            all_predicted_b_locations.append({'seq_idx': i, 'frame_idx': pred_b_idx, 'matched_to_real_b': False})
        for real_b_idx in real_b_indices_in_seq:
            found_match_for_this_real_b, best_candidate_pred_location, min_dist = False, None, float('inf')
            for pred_b_loc in all_predicted_b_locations:
                if pred_b_loc['seq_idx'] == i and not pred_b_loc['matched_to_real_b']:
                    dist = abs(pred_b_loc['frame_idx'] - real_b_idx)
                    if dist <= b_tolerance_frames and dist < min_dist: min_dist, best_candidate_pred_location = dist, pred_b_loc
            if best_candidate_pred_location: tp_b_tolerant += 1;best_candidate_pred_location[
                'matched_to_real_b'] = True;found_match_for_this_real_b = True
            if not found_match_for_this_real_b: fn_b_tolerant += 1

    fp_b_tolerant = sum(1 for pred_b_loc in all_predicted_b_locations if not pred_b_loc['matched_to_real_b'])
    precision_b_tolerant = tp_b_tolerant / (tp_b_tolerant + fp_b_tolerant) if (tp_b_tolerant + fp_b_tolerant) > 0 else 0
    recall_b_tolerant = tp_b_tolerant / total_real_b if total_real_b > 0 else 0
    f1_b_tolerant = 2 * (precision_b_tolerant * recall_b_tolerant) / (precision_b_tolerant + recall_b_tolerant) if (
                                                                                                                               precision_b_tolerant + recall_b_tolerant) > 0 else 0

    if not apply_post_processing:
        print(f"\nMetrics for B-Class (tolerance +/- {b_tolerance_frames} frames):")
        print(f"  Total Real B: {total_real_b}, TP_B: {tp_b_tolerant}, FP_B: {fp_b_tolerant}, FN_B: {fn_b_tolerant}")
        print(
            f"  Precision (B): {precision_b_tolerant:.4f}, Recall (B): {recall_b_tolerant:.4f}, F1-Score (B): {f1_b_tolerant:.4f}")

    print(
        f"\nSequence Validity ({model_name_str}{' - PostP' if apply_post_processing else ''}): {valid_predicted_sequences_count}/{total_sequences_evaluated} ({(valid_predicted_sequences_count / total_sequences_evaluated) * 100 if total_sequences_evaluated else 0:.2f}%)")

    print("\nStd Classification Report (exact frame-by-frame, non-padded):")
    report_labels, true_labels_flat_all, pred_labels_flat_all = list(range(NUM_CLASSES)), [], []
    target_names_report = [class_names[i] if i < len(class_names) else f"Class_{i}" for i in report_labels]
    for i in range(len(y_original_unpadded_list)):
        sl = len(y_original_unpadded_list[i])
        true_labels_flat_all.extend(y_padded_indices_np[i, :sl]);
        pred_labels_flat_all.extend(y_pred_classes_to_evaluate[i, :sl])
    if true_labels_flat_all:
        unique_labels = np.unique(np.concatenate((true_labels_flat_all, pred_labels_flat_all)))
        current_report_labels = [l for l in report_labels if l in unique_labels]
        current_target_names = [class_names[l] for l in current_report_labels]
        if not current_report_labels:
            print("No common labels for report. Skipping.")
        else:
            print(classification_report(true_labels_flat_all, pred_labels_flat_all, target_names=current_target_names,
                                        labels=current_report_labels, zero_division=0))
        cm = confusion_matrix(true_labels_flat_all, pred_labels_flat_all, labels=report_labels)
        plt.figure(figsize=(8, 6));
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names_report,
                    yticklabels=target_names_report)
        plt.title(f'Std CM - {model_name_str}{"-PostP" if apply_post_processing else ""}');
        plt.ylabel('Actual');
        plt.xlabel('Predicted')
        cm_file = f"{model_name_str}{'_postprocessed' if apply_post_processing else ''}_std_cm.png"
        plt.savefig(MODEL_SAVE_DIR / cm_file);
        plt.close();
        print(f"Std CM saved to {MODEL_SAVE_DIR / cm_file}")
    else:
        print("No non-padded labels for std report/matrix.")


def plot_training_history(train_losses, val_losses, val_accuracies, model_name_suffix):
    model_name_prefix = "Final_PyTorch_OptCRF"  # Updated prefix
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.title(f'{model_name_prefix}_{model_name_suffix} Loss');
    plt.xlabel('Epochs');
    plt.ylabel('Loss');
    plt.legend();
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_accuracies, label='Val Accuracy (Decoded)')
    plt.title(f'{model_name_prefix}_{model_name_suffix} Accuracy');
    plt.xlabel('Epochs');
    plt.ylabel('Accuracy');
    plt.legend();
    plt.grid(True)
    plt.tight_layout()
    hist_path = MODEL_SAVE_DIR / f"{model_name_prefix}_{model_name_suffix}_training_history.png"
    plt.savefig(hist_path);
    plt.close();
    print(f"Training history plot saved to {hist_path}")


def calculate_accuracy(model, dataloader, device):  # Calculate accuracy using model.predict
    model.eval()
    all_true_flat, all_pred_flat = [], []
    with torch.no_grad():
        for features, labels, lengths, mask_batch in dataloader:
            features, labels, lengths, mask_batch = features.to(device), labels.to(device), lengths.to(
                device), mask_batch.to(device)
            predicted_paths = model.predict(features, lengths, mask_batch)
            for i in range(len(predicted_paths)):
                true_len = lengths[i].item()
                true_seq = labels[i, :true_len].cpu().numpy()
                pred_seq = np.array(predicted_paths[i])[:true_len]  # Ensure pred_seq is cut to true_len
                all_true_flat.extend(true_seq)
                all_pred_flat.extend(pred_seq)
    if not all_true_flat: return 0.0
    return np.mean(np.array(all_true_flat) == np.array(all_pred_flat)) if len(all_true_flat) == len(
        all_pred_flat) else 0.0


# --- Main Training and Evaluation Function ---
def train_and_evaluate_model(model_type_str, rnn_units):
    print(f"\n--- Training PyTorch Bi{model_type_str.upper()}-OptimizedCRF Model ---")

    model = RecurrentModelWithCRF(input_dim=NUM_FEATURES,
                                  hidden_dim_rnn_output=rnn_units * 2,  # BiRNN output dim
                                  num_actual_classes=NUM_CLASSES,
                                  recurrent_type=model_type_str,
                                  dropout_rate=DROPOUT_RATE).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=L2_REG_FACTOR)

    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_loss_history, val_loss_history, val_acc_history = [], [], []

    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        for batch_idx, (features, targets, lengths, mask) in enumerate(train_loader):
            features, targets, lengths, mask = features.to(device), targets.to(device), lengths.to(device), mask.to(
                device)
            optimizer.zero_grad()
            loss = model.compute_loss(features, targets, lengths, mask)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss at epoch {epoch + 1}, batch {batch_idx}. Skipping update.")
                loss = torch.nan_to_num(loss, nan=1000.0) if torch.isnan(loss) else torch.tensor(1000.0, device=device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for features, targets, lengths, mask in val_loader:
                features, targets, lengths, mask = features.to(device), targets.to(device), lengths.to(device), mask.to(
                    device)
                loss = model.compute_loss(features, targets, lengths, mask)
                epoch_val_loss += loss.item()
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        val_accuracy = calculate_accuracy(model, val_loader, device)
        val_acc_history.append(val_accuracy)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss, epochs_no_improve = avg_val_loss, 0
            best_model_state = copy.deepcopy(model.state_dict())
            save_path = MODEL_SAVE_DIR / f"bi{model_type_str}_optcrf_best.pt"
            torch.save(best_model_state, save_path)
            print(f"  Val loss improved. Saved best model to {save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE: print(f"Early stopping after {epoch + 1} epochs."); break

    if best_model_state: model.load_state_dict(best_model_state); print(
        f"Loaded best Bi{model_type_str.upper()}-OptCRF state.")
    plot_training_history(train_loss_history, val_loss_history, val_acc_history, f"Bi{model_type_str.upper()}_OptCRF")

    print(f"\nEvaluating Bi{model_type_str.upper()}-OptCRF on Test Set...")
    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for features, targets, lengths, mask in test_loader:
            features, targets, lengths, mask = features.to(device), targets.to(device), lengths.to(device), mask.to(
                device)
            loss = model.compute_loss(features, targets, lengths, mask)
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = calculate_accuracy(model, test_loader, device)
    print(f"Bi{model_type_str.upper()}-OptCRF Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

    analyze_predictions(model, test_loader, y_test_raw, y_test_padded_np,
                        f"Bi{model_type_str.upper()}_OptCRF_Final_Raw", apply_post_processing=False)
    analyze_predictions(model, test_loader, y_test_raw, y_test_padded_np,
                        f"Bi{model_type_str.upper()}_OptCRF_Final_PostP", apply_post_processing=True)


# --- Run for BiLSTM-OptimizedCRF ---
train_and_evaluate_model('lstm', LSTM_UNITS)
# --- Run for BiGRU-OptimizedCRF ---
train_and_evaluate_model('gru', GRU_UNITS)

print("\n--- PyTorch Training with Optimized CRF Finished ---")
print(f"Best models, plots, and CMs saved to: {MODEL_SAVE_DIR}")

# --- END OF FILE train_pytorch_optimized_crf.py ---