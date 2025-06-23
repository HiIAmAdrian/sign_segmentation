import pytest
import numpy as np
import pandas as pd


from infer_logic_functional_fara_fata import (
    scale_data_for_inference,
    predict_sequence_for_inference,
    post_process_oi_labels_functional,
    extract_segments_from_oi_labels,
    run_inference_for_sequence_data,
    DFLT_LABEL_O_2CLASS, DFLT_LABEL_I_2CLASS,
    DFLT_MIN_I_FRAMES_FOR_POSTPROCESS, DFLT_MIN_SEGMENT_DURATION_MS
)


@pytest.fixture
def sample_data_array():
    return np.random.rand(100, 10)


@pytest.fixture
def sample_scaler(sample_data_array):
    class MockScaler:
        def __init__(self):
            self.n_features_in_ = sample_data_array.shape[1]
        def transform(self, data):
            if data.shape[1] != self.n_features_in_:
                raise ValueError("Feature mismatch")
            return data * 0.5
    return MockScaler()


@pytest.fixture
def sample_model():
    class MockTFModel:
        def __init__(self, input_shape_tuple, num_classes):
            self.input_shape = input_shape_tuple
            self._num_classes = num_classes
        @property
        def output_shape(self):
            return (self.input_shape[0], self.input_shape[1], self._num_classes)
        def predict(self, data, verbose=0):
            batch_size, model_max_len, _ = data.shape
            predictions = np.random.rand(batch_size, model_max_len, self._num_classes)
            if DFLT_LABEL_I_2CLASS < self._num_classes and DFLT_LABEL_O_2CLASS < self._num_classes:
                predictions[0, 10:20, DFLT_LABEL_I_2CLASS] = 0.8
                predictions[0, 10:20, DFLT_LABEL_O_2CLASS] = 0.2
                predictions[0, 50:55, DFLT_LABEL_I_2CLASS] = 0.7
                predictions[0, 50:55, DFLT_LABEL_O_2CLASS] = 0.3
            else:
                predictions[0, 10:20, 0] = 0.8
                if self._num_classes > 1:
                    predictions[0, 10:20, 1] = 0.2
                predictions[0, 50:55, 0] = 0.7
                if self._num_classes > 1:
                    predictions[0, 50:55, 1] = 0.3
            return predictions
    return MockTFModel(input_shape_tuple=(None, 128, 10), num_classes=2)


def test_scale_data_for_inference(sample_data_array, sample_scaler):
    scaled_data = scale_data_for_inference(sample_data_array, sample_scaler)
    assert scaled_data.shape == sample_data_array.shape
    assert not np.array_equal(scaled_data, sample_data_array)

    unscaled_data = scale_data_for_inference(sample_data_array, None)
    assert np.array_equal(unscaled_data, sample_data_array)

    wrong_feature_data = np.random.rand(100, 5)
    with pytest.raises(ValueError, match="Feature mismatch"):
        scale_data_for_inference(wrong_feature_data, sample_scaler)


def test_predict_sequence_for_inference(sample_model, sample_data_array):
    model_expected_max_len = sample_model.input_shape[1]
    short_sequence = sample_data_array[:50]
    labels, probs = predict_sequence_for_inference(sample_model, short_sequence, model_expected_max_len)
    assert labels.shape == (50,)
    assert probs.shape == (50, sample_model.output_shape[-1])

    long_sequence = np.vstack([sample_data_array, sample_data_array])
    labels, probs = predict_sequence_for_inference(sample_model, long_sequence, model_expected_max_len)
    assert labels.shape == (200,)
    assert probs.shape == (200, sample_model.output_shape[-1])
    assert np.all(labels[model_expected_max_len:] == DFLT_LABEL_O_2CLASS)

    wrong_feature_data = np.random.rand(50, sample_model.input_shape[-1] + 1)
    with pytest.raises(ValueError, match="Feature count mismatch for prediction"):
        predict_sequence_for_inference(sample_model, wrong_feature_data, model_expected_max_len)


def test_post_process_oi_labels_functional():
    labels1 = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0])
    expected1 = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0])
    assert np.array_equal(post_process_oi_labels_functional(labels1, 5, 0, 1), expected1)
    labels2 = np.array([0, 1, 1, 1, 1, 1, 1, 0])
    expected2 = np.array([0, 1, 1, 1, 1, 1, 1, 0])
    assert np.array_equal(post_process_oi_labels_functional(labels2, 5, 0, 1), expected2)
    labels3 = np.array([0, 0, 0, 0])
    expected3 = np.array([0, 0, 0, 0])
    assert np.array_equal(post_process_oi_labels_functional(labels3, 3, 0, 1), expected3)
    labels4 = np.array([1, 1, 1, 1, 1])
    expected4 = np.array([1, 1, 1, 1, 1])
    assert np.array_equal(post_process_oi_labels_functional(labels4, 3, 0, 1), expected4)
    labels5 = np.array([1, 1])
    expected5 = np.array([0, 0])
    assert np.array_equal(post_process_oi_labels_functional(labels5, 3, 0, 1), expected5)
    assert np.array_equal(post_process_oi_labels_functional(np.array([]), 3, 0, 1), np.array([]))


def test_extract_segments_from_oi_labels():
    labels = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1])
    timestamps_ms = np.arange(len(labels)) * 100
    segments = extract_segments_from_oi_labels(labels, timestamps_ms, 0, 1, 100, 0)
    expected_segments = [
        {"start_ms": 200, "end_ms": 400},
        {"start_ms": 700, "end_ms": 800},
        {"start_ms": 1000, "end_ms": 1400}
    ]
    assert segments == expected_segments

    segments_filtered = extract_segments_from_oi_labels(labels, timestamps_ms, 0, 1, 250, 0)
    expected_filtered = [{"start_ms": 1000, "end_ms": 1400}]
    assert segments_filtered == expected_filtered

    offset = 500
    segments_offset = extract_segments_from_oi_labels(labels, timestamps_ms, 0, 1, 100, offset)
    expected_offset = [
        {"start_ms": 200 + offset, "end_ms": 400 + offset},
        {"start_ms": 700 + offset, "end_ms": 800 + offset},
        {"start_ms": 1000 + offset, "end_ms": 1400 + offset}
    ]
    assert segments_offset == expected_offset

    assert extract_segments_from_oi_labels(np.array([]), np.array([])) == []
    assert extract_segments_from_oi_labels(labels, timestamps_ms[:-1]) == []


def test_run_inference_for_sequence_data(sample_model, sample_data_array):
    scaled_sequence = sample_data_array
    original_td_index = pd.to_timedelta(np.arange(len(sample_data_array)) * 100, unit='ms')
    model_max_len = sample_model.input_shape[1]
    num_model_output_classes = sample_model.output_shape[-1]

    segments = run_inference_for_sequence_data(
        model_to_use=sample_model,
        feature_sequence_scaled_np=scaled_sequence,
        original_timedelta_index=original_td_index,
        model_max_len=model_max_len,
        num_model_output_classes=num_model_output_classes,
        min_i_frames_postprocess=3,
        min_segment_duration_ms_filter=50,
        initial_trim_offset_ms=1000
    )
    assert isinstance(segments, list)
    if segments:
        for seg in segments:
            assert "start_ms" in seg
            assert "end_ms" in seg
            assert seg["start_ms"] >= 1000
            assert seg["end_ms"] >= seg["start_ms"]

    with pytest.raises(ValueError, match="Model not provided"):
        run_inference_for_sequence_data(None, scaled_sequence, original_td_index, model_max_len,
                                        num_model_output_classes)

    with pytest.raises(ValueError, match="Feature sequence is empty"):
        run_inference_for_sequence_data(sample_model, np.array([]), original_td_index, model_max_len,
                                        num_model_output_classes)