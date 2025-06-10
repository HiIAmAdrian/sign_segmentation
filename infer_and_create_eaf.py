import tensorflow as tf
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from pympi.Elan import Eaf
import traceback
import os

# --- Configurații ---
MODEL_PATH = Path("./trained_models_final_prev_label/bigru_best_prev_label.keras")  # Modelul antrenat (stateful=False)
SCALER_PATH = Path("./final_combined_data_for_training_ALL_SIGNERS/final_features_ts_facial_scaler.pkl")
PKL_DATA_FOR_INFERENCE_DIR = Path(
    "./final_combined_data_for_training_ALL_SIGNERS")
PKL_DATA_FOR_INFERENCE_FILE = PKL_DATA_FOR_INFERENCE_DIR / "all_data_final_features_ts_facial.pkl"

SEQUENCE_TO_INFER_INDEX_IN_TEST_SET = 0

OUTPUT_EAF_DIR = Path("./inference_output_eaf")

NUM_CLASSES = 3
LABEL_O, LABEL_B, LABEL_I = 0, 1, 2

# Constante pentru post-procesare (dacă le folosești)
PROB_THRESHOLD_B_WHEN_OI = 0.3  # Ajustează după nevoie
PROB_THRESHOLD_I_WHEN_BO = 0.4  # Ajustează după nevoie


# --- Funcții Ajutătoare ---

def load_trained_model(model_path):
    print(f"Loading trained (stateless) model from: {model_path}")
    # Load_model încarcă și arhitectura și ponderile.
    # Nu e nevoie să fie compilat pentru inferență dacă doar transferăm ponderile.
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


def load_scaler(scaler_path):
    scaler = None
    if scaler_path and scaler_path.exists():
        print(f"Loading scaler from: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    return scaler


# Funcție pentru a construi modelul RECURRENT (similară cu cea din train.py)
# Aceasta va fi folosită pentru a crea modelul de inferență stateful.
def build_recurrent_inference_model(model_type='gru', units=128, batch_input_shape=None, num_classes=3,
                                    dropout_rate=0.0, l2_reg=0.0):
    if batch_input_shape is None: raise ValueError("batch_input_shape must be provided for stateful model")

    # Pentru inferență, dropout-ul ar trebui să fie 0 sau modelul să fie în modul de inferență.
    # Keras gestionează automat dropout-ul (îl dezactivează) în timpul .predict()
    # Deci, dropout_rate aici poate fi cel folosit la antrenament, dar nu va fi activ la inferență.

    input_layer = tf.keras.layers.Input(batch_shape=batch_input_shape, name="Input_Layer_Stateful")
    # Nu e nevoie de Masking layer dacă dăm input cadru cu cadru (timesteps=1)
    # Dacă am da secvențe padate și la modelul stateful, ar fi necesar.

    regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None

    RecurrentLayer = tf.keras.layers.LSTM if model_type == 'lstm' else tf.keras.layers.GRU

    # Layer recurent STATEFUL
    recurrent_layer_stateful = RecurrentLayer(units,
                                              return_sequences=True,
                                              # Chiar și pentru un singur pas, pentru consistența output-ului
                                              stateful=True,  # CRUCIAL
                                              kernel_regularizer=regularizer,
                                              recurrent_regularizer=regularizer,
                                              name=f"{model_type.upper()}_Layer_Stateful")

    bidirectional_layer_stateful = tf.keras.layers.Bidirectional(
        recurrent_layer_stateful,
        name=f"Bi{model_type.upper()}_Layer_Stateful"
    )(input_layer)  # Input-ul merge direct aici dacă nu avem Masking

    # Dropout-ul e aplicat în Keras doar la antrenament. La inferență (predict), e inactiv.
    dropout_output_stateful = tf.keras.layers.Dropout(dropout_rate, name="Dropout_Layer_Stateful")(
        bidirectional_layer_stateful)

    output_layer_stateful = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=regularizer),
        name="Output_Layer_Stateful"
    )(dropout_output_stateful)

    inference_model = tf.keras.Model(inputs=input_layer, outputs=output_layer_stateful)
    return inference_model


def get_sequence_for_inference(pkl_file_path, seq_index_in_test):
    print(f"Loading sequence for inference from: {pkl_file_path}, test index: {seq_index_in_test}")
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)

    if 'X_test_df_indexed' not in data or not data['X_test_df_indexed']:
        raise ValueError("X_test_df_indexed not found or empty in PKL.")
    if seq_index_in_test >= len(data['X_test_df_indexed']):
        raise ValueError(
            f"Sequence index {seq_index_in_test} out of bounds for X_test_df_indexed (len: {len(data['X_test_df_indexed'])}).")

    df_sequence_unscaled = data['X_test_df_indexed'][seq_index_in_test]
    original_id_dict = data['test_ids'][seq_index_in_test] if 'test_ids' in data and len(
        data['test_ids']) > seq_index_in_test else {}
    original_filename_stem_from_id = original_id_dict.get('filename', f"unknown_sequence_{seq_index_in_test}")
    original_filename_stem = Path(original_filename_stem_from_id).stem.split('.')[0]

    if df_sequence_unscaled is None or df_sequence_unscaled.empty:
        raise ValueError(f"Selected sequence (test index {seq_index_in_test}) is None or empty.")

    return df_sequence_unscaled, df_sequence_unscaled.index, original_filename_stem


def scale_data(data_array, scaler):
    if scaler is None:
        print("Warning: Scaler not provided. Using data as is.")
        return data_array
    if data_array.ndim == 1:
        data_array = data_array.reshape(1, -1)
    if data_array.shape[1] != scaler.n_features_in_:
        raise ValueError(
            f"Number of features in data_array ({data_array.shape[1]}) does not match scaler's expected features ({scaler.n_features_in_})")
    return scaler.transform(data_array)


def predict_sequence_stateful(inference_model, initial_feature_sequence_scaled,
                              num_classes, num_original_features):
    seq_len = initial_feature_sequence_scaled.shape[0]

    predicted_probs_autoregressive = np.zeros((seq_len, num_classes), dtype=np.float32)

    # Resetăm stările modelului de inferență la începutul fiecărei secvențe!
    inference_model.reset_states()

    prev_label_one_hot = np.zeros(num_classes, dtype=np.float32)
    prev_label_one_hot[LABEL_O] = 1.0  # Prima etichetă anterioară este O

    print("Starting stateful auto-regressive prediction...")
    for t in range(seq_len):
        current_original_features = initial_feature_sequence_scaled[t, :]
        augmented_frame_features = np.concatenate([current_original_features, prev_label_one_hot])

        # Modelul stateful se așteaptă la input de forma (1, 1, num_augmented_features)
        # (batch_size=1, timesteps=1, features)
        frame_input_for_model = augmented_frame_features.reshape(1, 1, -1)

        # Predicția pentru un singur pas
        pred_probs_current_step_batch = inference_model.predict(frame_input_for_model, verbose=0)
        # Output-ul va fi (1, 1, num_classes)
        pred_probs_current_step = pred_probs_current_step_batch[0, 0, :]

        predicted_probs_autoregressive[t, :] = pred_probs_current_step

        predicted_label_current_step = np.argmax(pred_probs_current_step)
        prev_label_one_hot = tf.keras.utils.to_categorical(predicted_label_current_step,
                                                           num_classes=num_classes).astype(np.float32)

        if (t + 1) % 100 == 0 or t == seq_len - 1:
            print(f"  Predicted frame {t + 1}/{seq_len}")

    print("Stateful auto-regressive prediction finished.")
    final_predicted_labels = np.argmax(predicted_probs_autoregressive, axis=-1)
    return final_predicted_labels, predicted_probs_autoregressive


def post_process_bio_sequence_with_probs(pred_labels_argmax, pred_probs_seq):
    if not isinstance(pred_labels_argmax, (list, np.ndarray)) or len(pred_labels_argmax) == 0:
        return pred_labels_argmax
    if pred_labels_argmax.shape[0] != pred_probs_seq.shape[0]:
        print("Error in post_process: Length mismatch between labels and probabilities.")
        return pred_labels_argmax

    corrected_seq = list(pred_labels_argmax)  # Facem o copie pentru a o modifica
    n = len(corrected_seq)

    # Regula 1: I nu poate fi la începutul unui segment (după O sau la start absolut)
    # Dacă primul non-O este I, schimbă-l în B dacă probabilitatea pentru B este suficient de mare
    first_non_o_idx = -1
    for i in range(n):
        if corrected_seq[i] != LABEL_O:
            first_non_o_idx = i
            break

    if first_non_o_idx != -1 and corrected_seq[first_non_o_idx] == LABEL_I:
        if pred_probs_seq[first_non_o_idx, LABEL_B] > PROB_THRESHOLD_B_WHEN_OI:  # Prag mai relaxat poate
            corrected_seq[first_non_o_idx] = LABEL_B
        # else: # Dacă probabilitatea pentru B e mică, poate e mai bine să fie O
        #     corrected_seq[first_non_o_idx] = LABEL_O

    i = 0
    while i < n:
        current_label = corrected_seq[i]
        prev_label = corrected_seq[i - 1] if i > 0 else LABEL_O  # Asumăm O înainte de începutul secvenței

        # Regula 1.1: O -> I devine O -> B (dacă prob_B e bună) sau O -> O
        if prev_label == LABEL_O and current_label == LABEL_I:
            if pred_probs_seq[i, LABEL_B] > PROB_THRESHOLD_B_WHEN_OI:
                corrected_seq[i] = LABEL_B
            else:  # Dacă B nu e probabil, e mai sigur să fie O
                corrected_seq[i] = LABEL_O

        # Regula 2: B -> B devine B -> I
        if i + 1 < n:
            next_label_pred = corrected_seq[i + 1]
            if current_label == LABEL_B and next_label_pred == LABEL_B:
                corrected_seq[i + 1] = LABEL_I  # Următorul B devine I

        # Regula 3: B -> O (segment de un singur cadru) devine B -> I dacă prob_I e bună, altfel poate B->O devine O->O
        if i + 1 < n:
            next_label_pred = corrected_seq[i + 1]
            if current_label == LABEL_B and next_label_pred == LABEL_O:
                # Dacă probabilitatea pentru I la pasul următor este mai mare decât pentru O și un prag
                if pred_probs_seq[i + 1, LABEL_I] > PROB_THRESHOLD_I_WHEN_BO and \
                        pred_probs_seq[i + 1, LABEL_I] > pred_probs_seq[i + 1, LABEL_O]:
                    corrected_seq[i + 1] = LABEL_I
                    # else: # Dacă I nu e probabil, poate B-ul inițial a fost o greșeală
                    # și ar trebui să fie O. Sau lăsăm B-O scurt.
                    # O altă variantă: dacă B e urmat de O, și B nu e foarte sigur, schimbăm B în O.
                    # if pred_probs_seq[i, LABEL_B] < some_threshold: corrected_seq[i] = LABEL_O
                    pass  # Lăsăm B-O deocamdată, ELAN le va ignora dacă sunt prea scurte oricum.
    i += 1


# O a doua trecere pentru a corecta I-uri care nu urmează un B sau I
# Aceasta e mai agresivă
# for k in range(1, n):
#     if corrected_seq[k] == LABEL_I and corrected_seq[k-1] == LABEL_O:
#         # Am O urmat de I. Acest I ar trebui să fie B.
#         # Verificăm dacă nu cumva cadrul anterior ar fi trebuit să fie B
#         if pred_probs_seq[k-1, LABEL_B] > pred_probs_seq[k-1, LABEL_O] and pred_probs_seq[k-1, LABEL_B] > PROB_THRESHOLD_B_WHEN_OI:
#             corrected_seq[k-1] = LABEL_B
#             # Si cadrul curent ramane I (sau il reevaluam)
#         elif pred_probs_seq[k, LABEL_B] > PROB_THRESHOLD_B_WHEN_OI: # Sau poate actualul I e de fapt B
#             corrected_seq[k] = LABEL_B
#         else: # Nici anteriorul nu pare B, nici actualul. Poate actualul I e O.
#             corrected_seq[k] = LABEL_O


    return np.array(corrected_seq, dtype=int)


def bio_to_segments(bio_labels, timedelta_index_us_values):
    segments = []
    in_segment = False
    start_time_ms = 0
    if len(bio_labels) != len(timedelta_index_us_values):
        print(
            f"Warning: bio_labels length ({len(bio_labels)}) and timedelta_index_us_values length ({len(timedelta_index_us_values)}) mismatch. Truncating to shorter.")
        min_len = min(len(bio_labels), len(timedelta_index_us_values))
        bio_labels = bio_labels[:min_len]
        timedelta_index_us_values = timedelta_index_us_values[:min_len]
        if min_len == 0: return segments
    for i, label in enumerate(bio_labels):
        current_time_ms = int(timedelta_index_us_values[i] / 1000)
        if label == LABEL_B:
            if in_segment:
                end_time_prev_segment_ms = int(timedelta_index_us_values[i - 1] / 1000) if i > 0 else start_time_ms
                if end_time_prev_segment_ms > start_time_ms:
                    segments.append((start_time_ms, end_time_prev_segment_ms, "SIGN"))
            start_time_ms = current_time_ms
            in_segment = True
        elif label == LABEL_I:
            if not in_segment:  # Am întâlnit un I fără un B anterior
                start_time_ms = current_time_ms  # Îl tratăm ca un nou început de segment
                in_segment = True
        elif label == LABEL_O:
            if in_segment:
                end_time_current_segment_ms = int(timedelta_index_us_values[i - 1] / 1000) if i > 0 else start_time_ms
                if end_time_current_segment_ms > start_time_ms:
                    segments.append((start_time_ms, end_time_current_segment_ms, "SIGN"))
                in_segment = False
    if in_segment:
        end_time_final_segment_ms = int(timedelta_index_us_values[-1] / 1000) if len(
            timedelta_index_us_values) > 0 else start_time_ms
        if end_time_final_segment_ms > start_time_ms:
            segments.append((start_time_ms, end_time_final_segment_ms, "SIGN"))
    return segments


def create_eaf_file(output_eaf_path, segments, absolute_media_uri=None, relative_media_path_for_storage=None):
    eafob = Eaf(author="InferenceScript")
    tier_id = "PredictedSigns"
    eafob.add_tier(tier_id)

    if "default-lt" not in eafob.linguistic_types:
        eafob.add_linguistic_type("default-lt", timealignable=True)

    if tier_id in eafob.tiers and (eafob.tiers[tier_id][2] is None or eafob.tiers[tier_id][2] == ''):
        existing_annotations = eafob.tiers[tier_id][0]
        participant = eafob.tiers[tier_id][1]
        default_locale = eafob.tiers[tier_id][3]
        eafob.tiers[tier_id] = (existing_annotations, participant, "default-lt", default_locale)

    if absolute_media_uri and relative_media_path_for_storage:
        mime_type = "video/mp4"
        if ".wav" in relative_media_path_for_storage.lower():
            mime_type = "audio/x-wav"
        print(f"Linking media: URI='{absolute_media_uri}', Relative Path='{relative_media_path_for_storage}'")
        eafob.add_linked_file(
            file_path=absolute_media_uri,
            relpath=relative_media_path_for_storage,
            mimetype=mime_type
        )
    elif relative_media_path_for_storage:
        print(f"Warning: Absolute media URI not available or media file not found. "
              f"Linking with potentially non-existent relative path: {relative_media_path_for_storage}")
        mime_type = "video/mp4"
        if ".wav" in relative_media_path_for_storage.lower(): mime_type = "audio/x-wav"
        eafob.add_linked_file(
            file_path=relative_media_path_for_storage,
            relpath=relative_media_path_for_storage,
            mimetype=mime_type
        )
    else:
        print("No media path information provided. EAF will be created without linked media.")

    for start_ms, end_ms, annotation_value in segments:
        if end_ms <= start_ms:
            print(f"Skipping invalid segment: start_ms={start_ms}, end_ms={end_ms}, value='{annotation_value}'")
            continue
        try:
            eafob.add_annotation(tier_id, int(start_ms), int(end_ms), value=annotation_value)
        except Exception as e_add_ann:
            print(f"Error adding annotation ({start_ms}-{end_ms}, {annotation_value}): {e_add_ann}")
            traceback.print_exc()

    try:
        Path(output_eaf_path).parent.mkdir(parents=True, exist_ok=True)
        eafob.to_file(str(output_eaf_path))
        print(f"EAF file saved to: {output_eaf_path}")
    except Exception as e_save_eaf:
        print(f"Error saving EAF file {output_eaf_path}: {e_save_eaf}")
        traceback.print_exc()


if __name__ == "__main__":
    trained_model_stateless = load_trained_model(MODEL_PATH)
    scaler = load_scaler(SCALER_PATH)

    if trained_model_stateless is None: print("Exiting due to trained model loading failure."); exit()
    if scaler is None: print("Error: Scaler not loaded. Exiting."); exit()

    # Extrage parametrii din modelul antrenat pentru a construi modelul de inferență
    # Presupunem că modelul antrenat are un layer BiGRU sau BiLSTM
    # Numele layer-elor pot varia, ajustează dacă e necesar
    model_type = None
    units = None
    dropout_rate = 0.0  # Default, poate fi extras dacă e stocat în config-ul layer-ului
    l2_reg = 0.0  # Default

    # Încercăm să extragem configurările din modelul încărcat
    # Aceasta este o parte mai heuristică și poate necesita ajustări
    # în funcție de cum sunt numite layer-ele în modelul tău original.
    try:
        for layer in trained_model_stateless.layers:
            if isinstance(layer, tf.keras.layers.Bidirectional):
                if isinstance(layer.forward_layer, tf.keras.layers.GRU):
                    model_type = 'gru'
                    units = layer.forward_layer.units
                    # L2 și dropout pot fi mai greu de extras direct dacă nu sunt în config
                    # Verificăm dacă regularizatorul kernel este L2
                    if hasattr(layer.forward_layer, 'kernel_regularizer') and \
                            isinstance(layer.forward_layer.kernel_regularizer, tf.keras.regularizers.L2):
                        l2_reg = layer.forward_layer.kernel_regularizer.l2
                    break
                elif isinstance(layer.forward_layer, tf.keras.layers.LSTM):
                    model_type = 'lstm'
                    units = layer.forward_layer.units
                    if hasattr(layer.forward_layer, 'kernel_regularizer') and \
                            isinstance(layer.forward_layer.kernel_regularizer, tf.keras.regularizers.L2):
                        l2_reg = layer.forward_layer.kernel_regularizer.l2
                    break
            # Extrage dropout din layer-ul de Dropout, dacă există imediat după BiRNN
            # Acest lucru depinde de structura exactă a modelului.
            # if isinstance(layer, tf.keras.layers.Dropout):
            #    dropout_rate = layer.rate # Acesta poate fi dropout-ul de după BiRNN

        if model_type is None or units is None:
            raise ValueError("Could not determine model_type or units from the loaded model.")

        # Să presupunem că dropout_rate este cel definit global în scriptul de antrenament
        # Dacă ai salvat modelul cu `model.get_config()`, ai putea extrage mai precis.
        # Pentru acum, folosim o valoare hardcodată dacă nu o putem extrage ușor.
        # Acest dropout_rate din build_recurrent_inference_model oricum e inactiv la predict.
        # Important e ca `units` și `model_type` să fie corecte.
        # Extragem DROPOUT_RATE din config-ul modelului antrenat, dacă e posibil
        # Căutăm layer-ul Dropout. Presupunem că e numit "Dropout_Layer" ca în train.py
        dropout_layer_found = trained_model_stateless.get_layer(name="Dropout_Layer")  # Sau cum l-ai numit
        if dropout_layer_found:
            dropout_rate = dropout_layer_found.rate
            print(f"Extracted dropout_rate: {dropout_rate} from trained model.")
        else:  # Folosim o valoare default dacă nu găsim layer-ul
            # Vezi ce valoare ai folosit în train.py pentru DROPOUT_RATE
            dropout_rate = 0.4  # Default din scriptul tău de train
            print(f"Could not find Dropout_Layer, using default dropout_rate: {dropout_rate}")

        print(
            f"Inferred parameters for inference model: type={model_type}, units={units}, dropout={dropout_rate}, l2={l2_reg}")

    except Exception as e:
        print(f"Error inferring parameters from trained model: {e}")
        print(
            "Please ensure model_type, units, and num_augmented_features are correctly set manually if auto-detection fails.")
        exit()

    num_augmented_features_model = trained_model_stateless.input_shape[-1]
    num_original_features_model = num_augmented_features_model - NUM_CLASSES
    print(f"Number of augmented features from trained model: {num_augmented_features_model}")
    print(f"Derived num_original_features from model: {num_original_features_model}")

    # Construiește modelul de inferență stateful
    batch_input_shape_inference = (1, 1, num_augmented_features_model)
    inference_model = build_recurrent_inference_model(
        model_type=model_type,
        units=units,
        batch_input_shape=batch_input_shape_inference,
        num_classes=NUM_CLASSES,
        dropout_rate=dropout_rate,  # Acesta nu va fi activ la inferență
        l2_reg=l2_reg
    )

    # Transferă ponderile
    # Trebuie să ne asigurăm că numele layer-elor corespund sau că structura e identică
    # pentru un transfer direct de ponderi. Cel mai sigur e să iterăm prin layere.
    print("Transferring weights to stateful inference model...")
    for i, layer_trained in enumerate(trained_model_stateless.layers):
        # Asigură-te că layer-ul corespunzător din inference_model există
        if i < len(inference_model.layers):
            layer_inference = inference_model.layers[i]
            # Verifică dacă tipurile de layer se potrivesc și dacă au ponderi
            if type(layer_trained) == type(layer_inference) and layer_trained.get_weights():
                print(f"  Setting weights for layer: {layer_inference.name} from {layer_trained.name}")
                layer_inference.set_weights(layer_trained.get_weights())
            # Caz special pentru Bidirectional: trebuie setate ponderile pentru forward și backward RNN
            elif isinstance(layer_trained, tf.keras.layers.Bidirectional) and \
                    isinstance(layer_inference, tf.keras.layers.Bidirectional):
                print(f"  Setting weights for Bidirectional layer: {layer_inference.name} from {layer_trained.name}")
                # Trebuie să setăm ponderile pentru layer-ele interne (forward și backward)
                # Acest lucru se face de obicei automat dacă structura este aceeași
                # și setăm ponderile pentru layer-ul Bidirectional direct.
                # Dar e bine de verificat. Cel mai simplu, dacă structura e identică, set_weights pe layer-ul BiDi ar trebui să meargă.
                layer_inference.set_weights(layer_trained.get_weights())

    # O metodă mai robustă pentru transferul ponderilor dacă numele layer-elor sunt consistente:
    # for layer_trained in trained_model_stateless.layers:
    #     try:
    #         layer_inference = inference_model.get_layer(name=layer_trained.name + "_Stateful") # Sau cum ai numit layerele în build_recurrent_inference_model
    #         if layer_trained.get_weights():
    #             print(f"  Setting weights for layer: {layer_inference.name} from {layer_trained.name}")
    #             layer_inference.set_weights(layer_trained.get_weights())
    #     except ValueError:
    #         print(f"  Layer {layer_trained.name}_Stateful not found in inference model or no weights for {layer_trained.name}.")

    # Cea mai simplă abordare dacă arhitecturile (fără partea stateful/batch_shape) sunt identice:
    # inference_model.set_weights(trained_model_stateless.get_weights())
    # Dar acest lucru poate eșua dacă input_shape / batch_input_shape diferă chiar și doar prin statefulness.
    # Cel mai sigur este transferul layer cu layer, asigurându-ne că numele/tipurile corespund.
    # Pentru a simplifica, dacă `build_recurrent_inference_model` creează o structură compatibilă
    # (aceleași tipuri de layere în aceeași ordine), transferul direct ar trebui să fie ok.
    # Hai să încercăm un transfer direct, presupunând că structura internă e similară.
    # Aceasta este adesea suficientă dacă diferența principală este stateful și batch_input_shape.

    # Am comentat transferul layer-by-layer de mai sus pentru că set_weights
    # pe modelul întreg e mai simplu dacă funcționează.
    # Cel mai bine este să încerci să încarci ponderile layer cu layer, potrivind după nume.
    # Voi lăsa varianta layer-by-layer mai detaliată, dar va trebui să ajustezi numele layer-elor
    # din modelul de inferență pentru a se potrivi cu cele din modelul antrenat (ex: adăugând un sufix).

    # Să folosim logica de potrivire a ponderilor din exemplul TensorFlow:
    # https://www.tensorflow.org/guide/keras/transfer_learning#transfer_learning_from_a_custom_training_loop
    # Presupunem că numele layerelor relevante sunt aceleași în ambele modele,
    # cu excepția poate a layer-ului de input.
    # Este important ca layer-ele care au ponderi (RNN, Dense) să aibă nume identice sau
    # să fie mapate corect.

    # Revizuire transfer ponderi:
    # Modelul antrenat are layere numite "BiGRU_Layer", "Dropout_Layer", "Output_Layer"
    # Modelul de inferență are sufixul "_Stateful"
    # Trebuie să mapăm corect.

    print("Attempting to transfer weights by matching layer names (potentially with suffix)...")
    skipped_layers = 0
    for layer_trained in trained_model_stateless.layers:
        if not layer_trained.get_weights():  # Skip layers without weights (e.g., Input, Masking, Dropout la inferență)
            continue
        try:
            # Încercăm să găsim un layer corespondent în modelul de inferență
            # Presupunem că numele sunt similare sau identice
            target_layer_name_exact = layer_trained.name
            target_layer_name_stateful_suffix = layer_trained.name + "_Stateful"  # Convenția noastră actuală

            layer_inference = None
            try:
                layer_inference = inference_model.get_layer(name=target_layer_name_exact)
            except ValueError:
                try:
                    layer_inference = inference_model.get_layer(name=target_layer_name_stateful_suffix)
                except ValueError:
                    # O altă convenție comună este ca TF să adauge un prefix la layerele interne ale BiDirectional
                    # ex: modelul antrenat are 'bidirectional_1/forward_gru_1', 'bidirectional_1/backward_gru_1'
                    # Acest lucru devine complex dacă nu avem nume consistente.

                    # Pentru BiGRU_Layer (care e un Bidirectional wrapper):
                    if layer_trained.name == "BiGRU_Layer":  # Numele din train.py
                        layer_inference = inference_model.get_layer(name="BiGRU_Layer_Stateful")
                    elif layer_trained.name == "Output_Layer":  # TimeDistributed(Dense)
                        layer_inference = inference_model.get_layer(name="Output_Layer_Stateful")
                    # Adaugă alte mapări specifice dacă e necesar

            if layer_inference:
                print(f"  Setting weights for '{layer_inference.name}' from '{layer_trained.name}'")
                layer_inference.set_weights(layer_trained.get_weights())
            else:
                print(
                    f"  Skipping weights for '{layer_trained.name}': No corresponding layer found in inference model with expected names.")
                skipped_layers += 1

        except Exception as e_w:
            print(f"  Error setting weights for layer from '{layer_trained.name}': {e_w}")
            skipped_layers += 1

    if skipped_layers > 0:
        print(
            f"Warning: Skipped setting weights for {skipped_layers} layer(s). Check layer names and model structures.")

    inference_model.summary()  # Să vedem structura modelului de inferență

    # --- Start Procesare Secvență ---
    try:
        df_sequence_unscaled, timedelta_index, original_filename_stem = get_sequence_for_inference(
            PKL_DATA_FOR_INFERENCE_FILE,
            SEQUENCE_TO_INFER_INDEX_IN_TEST_SET
        )
        print(f"Loaded sequence with original filename stem: {original_filename_stem}")

        if df_sequence_unscaled.shape[1] != num_original_features_model:
            print(f"FATAL: Number of features in loaded sequence ({df_sequence_unscaled.shape[1]}) "
                  f"does not match num_original_features derived from model ({num_original_features_model}).")
            exit()

        media_file_name_guess = f"{original_filename_stem}_realsense.mp4"
        eaf_file_name = f"{original_filename_stem}_predicted.eaf"
        project_root = Path.cwd()
        absolute_video_path = (project_root / "videos" / media_file_name_guess).resolve()
        output_eaf_full_path = (project_root / OUTPUT_EAF_DIR / eaf_file_name).resolve()
        absolute_video_uri_str = None
        relative_path_str_for_storage = None

        if absolute_video_path.exists():
            absolute_video_uri_str = absolute_video_path.as_uri()
            print(f"Absolute video URI for EAF: {absolute_video_uri_str}")
            try:
                relative_path_os = os.path.relpath(absolute_video_path, output_eaf_full_path.parent)
                relative_path_str_for_storage = Path(relative_path_os).as_posix()
                print(f"Calculated relative video path for EAF storage: {relative_path_str_for_storage}")
            except ValueError as e_relpath:
                print(f"Could not calculate relative path: {e_relpath}")
                relative_path_str_for_storage = (Path("..") / "videos" / media_file_name_guess).as_posix()
                print(f"Using fallback relative video path: {relative_path_str_for_storage}")
        else:
            print(f"Warning: Absolute video path does not exist: {absolute_video_path}")
            relative_path_str_for_storage = (Path("..") / "videos" / media_file_name_guess).as_posix()
            print(
                f"Using expected relative video path for EAF storage (file not found): {relative_path_str_for_storage}")

        sequence_values_scaled = scale_data(df_sequence_unscaled.values, scaler)

        # Folosim noua funcție de predicție stateful
        raw_predicted_labels, raw_predicted_probs = predict_sequence_stateful(
            inference_model,  # Modelul stateful cu ponderile transferate
            sequence_values_scaled,
            NUM_CLASSES,
            num_original_features_model  # Doar pentru concatenare, nu e folosit de model direct
        )

        print("Applying post-processing to predicted sequence...")
        final_predicted_labels = post_process_bio_sequence_with_probs(
            raw_predicted_labels,
            raw_predicted_probs
        )

        timedelta_index_us_values = timedelta_index.to_series().dt.total_seconds() * 1_000_000
        segments = bio_to_segments(final_predicted_labels, timedelta_index_us_values.values)
        print(f"Generated {len(segments)} segments from B-I-O predictions.")
        for seg_idx, seg in enumerate(segments):
            print(f"  Segment {seg_idx}: Start={seg[0]}ms, End={seg[1]}ms, Label='{seg[2]}'")

        create_eaf_file(output_eaf_full_path, segments,
                        absolute_media_uri=absolute_video_uri_str,
                        relative_media_path_for_storage=relative_path_str_for_storage)

    except Exception as e:
        print(f"An error occurred during inference: {e}")
        traceback.print_exc()