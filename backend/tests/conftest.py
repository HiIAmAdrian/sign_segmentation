import pytest
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import sys

BACKEND_DIR = Path(__file__).parent.parent
if str(BACKEND_DIR.resolve()) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR.resolve()))

@pytest.fixture(scope="session")
def test_data_dir():
    return Path(__file__).parent / "test_data"


@pytest.fixture
def dummy_csv_paths(test_data_dir, tmp_path):
    shutil.copy(test_data_dir / "dummy_suit.csv", tmp_path / "suit.csv")
    shutil.copy(test_data_dir / "dummy_glove_r.csv", tmp_path / "glove_r.csv")
    shutil.copy(test_data_dir / "dummy_glove_l.csv", tmp_path / "glove_l.csv")
    shutil.copy(test_data_dir / "dummy.bag", tmp_path / "test.bag")
    return {
        "suit": tmp_path / "suit.csv",
        "glove_r": tmp_path / "glove_r.csv",
        "glove_l": tmp_path / "glove_l.csv",
        "bag": tmp_path / "test.bag",
    }


@pytest.fixture
def mock_model():
    class MockModel:
        def __init__(self, input_shape, output_shape_val):
            self.input_shape = input_shape
            self.output_shape_val = output_shape_val

        @property
        def output_shape(self):
            return (None, self.input_shape[1], self.output_shape_val)

        def predict(self, data, verbose=0):
            num_samples = data.shape[0]
            seq_len = data.shape[1]
            mock_predictions = np.zeros((num_samples, seq_len, self.output_shape_val))
            mid_point = seq_len // 2
            mock_predictions[:, :mid_point, 0] = 0.8
            mock_predictions[:, :mid_point, 1] = 0.2
            mock_predictions[:, mid_point:, 0] = 0.3
            mock_predictions[:, mid_point:, 1] = 0.7
            return mock_predictions
    return MockModel(input_shape=(None, 128, 50), output_shape_val=2)


@pytest.fixture
def mock_scaler():
    class MockScaler:
        def __init__(self, n_features):
            self.n_features_in_ = n_features
            self.mean_ = np.random.rand(n_features)

        def transform(self, data):
            if data.shape[1] != self.n_features_in_:
                raise ValueError(f"MockScaler: Expected {self.n_features_in_} features, got {data.shape[1]}")
            return data * 0.5 + 0.1

        def fit(self, data):
            self.n_features_in_ = data.shape[1]
            return self
    return MockScaler(n_features=50)


@pytest.fixture
def mock_config_data():
    return {
        'feature_names': [f'feature_{i}' for i in range(50)],
    }


@pytest.fixture
def app_instance(mocker, test_data_dir, mock_model, mock_scaler, mock_config_data):
    import backend_without_face

    mocker.patch('backend_without_face.MODEL_PATH_BILSTM', test_data_dir / "dummy_model.keras")
    mocker.patch('backend_without_face.MODEL_PATH_BIGRU', test_data_dir / "dummy_model.keras")
    mocker.patch('backend_without_face.SCALER_PATH', test_data_dir / "dummy_scaler.pkl")
    mocker.patch('backend_without_face.TRAINING_CONFIG_PKL', test_data_dir / "dummy_config.pkl")

    mocker.patch('tensorflow.keras.models.load_model', return_value=mock_model)

    def mock_pickle_load(file_obj):
        if 'scaler.pkl' in str(file_obj.name):
            return mock_scaler
        elif 'config.pkl' in str(file_obj.name) or 'features_ts_gloves_only.pkl' in str(file_obj.name):
            return mock_config_data
        raise ValueError(f"mock_pickle_load: Unhandled file name: {file_obj.name}")
    mocker.patch('pickle.load', side_effect=mock_pickle_load)

    try:
        backend_without_face.load_global_components()
    except SystemExit:
        pytest.fail("load_global_components called exit() during test setup. Mocks might be incomplete.")

    backend_without_face.app.config.update({"TESTING": True})
    return backend_without_face.app


@pytest.fixture
def client(app_instance):
    return app_instance.test_client()