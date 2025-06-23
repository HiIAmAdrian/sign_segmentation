import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from io import BytesIO
import backend_without_face


def test_generate_glove_feature_list_for_backend():
    features_rot_pos_root = backend_without_face.generate_glove_feature_list_for_backend("right", True, True, True)
    assert "right_thumb_proximal.rotation.x" in features_rot_pos_root
    assert "right_index_distal.position.y" in features_rot_pos_root
    assert "right_hand.rotation.w" in features_rot_pos_root
    assert "right_hand.position.z" in features_rot_pos_root

    features_rot_only = backend_without_face.generate_glove_feature_list_for_backend("left", True, False, False)
    assert "left_middle_intermediate.rotation.z" in features_rot_only
    assert "left_hand.position.x" not in features_rot_only
    assert len(features_rot_only) < len(features_rot_pos_root)


def test_populate_ts_feature_lists(app_instance):
    assert len(backend_without_face.TS_SUIT_FEATURES_TO_KEEP) > 0
    assert "hips.position.x" in backend_without_face.TS_SUIT_FEATURES_TO_KEEP
    assert "PelvisTilt.angle" in backend_without_face.TS_SUIT_FEATURES_TO_KEEP

    if any([
        backend_without_face.DFLT_TS_USE_GLOVE_RIGHT_FINGER_ROTATIONS,
        backend_without_face.DFLT_TS_USE_GLOVE_RIGHT_FINGER_POSITIONS,
        backend_without_face.DFLT_TS_USE_GLOVE_RIGHT_HAND_ROOT
    ]):
        assert len(backend_without_face.TS_GLOVE_RIGHT_FEATURES_TO_KEEP) > 0
    else:
        assert len(backend_without_face.TS_GLOVE_RIGHT_FEATURES_TO_KEEP) == 0

    if any([
        backend_without_face.DFLT_TS_USE_GLOVE_LEFT_FINGER_ROTATIONS,
        backend_without_face.DFLT_TS_USE_GLOVE_LEFT_FINGER_POSITIONS,
        backend_without_face.DFLT_TS_USE_GLOVE_LEFT_HAND_ROOT
    ]):
        assert len(backend_without_face.TS_GLOVE_LEFT_FEATURES_TO_KEEP) > 0
    else:
        assert len(backend_without_face.TS_GLOVE_LEFT_FEATURES_TO_KEEP) == 0


def test_adapted_load_and_prepare_df(dummy_csv_paths, capsys):
    suit_path = dummy_csv_paths["suit"]
    selected_features = ["hips.position.x", "PelvisTilt.angle", "non_existent_feature"]

    df, actual_features = backend_without_face.adapted_load_and_prepare_df(
        suit_path, selected_features, "dummy_suit.csv", trim_start_sec=0.5
    )

    assert not df.empty
    assert isinstance(df.index, pd.TimedeltaIndex)
    assert "hips.position.x" in df.columns
    assert "PelvisTilt.angle" in df.columns
    assert "non_existent_feature" not in df.columns
    assert "hips.position.x" in actual_features
    assert "non_existent_feature" not in actual_features

    df_trimmed, _ = backend_without_face.adapted_load_and_prepare_df(
        suit_path, selected_features, "dummy_suit.csv", trim_start_sec=1.5
    )
    assert len(df_trimmed) == 2

    df_non_existent, actual_non_existent = backend_without_face.adapted_load_and_prepare_df(
        "non_existent.csv", selected_features, "ne.csv"
    )
    assert df_non_existent.empty
    assert actual_non_existent == []
    captured = capsys.readouterr()
    assert "Error in adapted_load_and_prepare_df for ne.csv" in captured.out
    assert "ne.csv not found at non_existent.csv" in captured.out

    faulty_csv_content = "colA,colB\n1,2\n3,4"
    faulty_csv_path = Path(suit_path).parent / "faulty.csv"
    with open(faulty_csv_path, "w") as f:
        f.write(faulty_csv_content)
    with pytest.raises(ValueError, match="'frame_timestamp_us' or 'frame_timestamp' missing"):
        backend_without_face.adapted_load_and_prepare_df(faulty_csv_path, ["colA"], "faulty.csv")


def test_adapted_calculate_biomech_derivatives():
    data = {
        'frame_timestamp_us': pd.to_timedelta([100000, 200000, 300000], unit='us'),
        'PelvisTilt.angle': [10.0, 12.0, 15.0],
        'HipFlexExtR.angle': [5.0, 5.5, 6.5]
    }
    df = pd.DataFrame(data).set_index('frame_timestamp_us')
    biomech_joints = ["PelvisTilt", "HipFlexExtR"]

    df_out = backend_without_face.adapted_calculate_biomech_derivatives(df, biomech_joints)

    assert "PelvisTilt.angular_v" in df_out.columns
    assert "PelvisTilt.angular_acc" in df_out.columns
    assert df_out["PelvisTilt.angular_v"].iloc[0] == 0.0
    assert pytest.approx(df_out["PelvisTilt.angular_v"].iloc[1]) == (12.0 - 10.0) / 0.1
    assert df_out["PelvisTilt.angular_acc"].iloc[0] == 0.0
    assert pytest.approx(df_out["PelvisTilt.angular_acc"].iloc[1]) == 200.0
    assert pytest.approx(df_out["PelvisTilt.angular_acc"].iloc[2]) == ((15.0 - 12.0) / 0.1 - (12.0 - 10.0) / 0.1) / 0.1



def test_process_teslasuit_for_demo_backend(dummy_csv_paths, app_instance):
    suit_p, glove_r_p, glove_l_p = dummy_csv_paths["suit"], dummy_csv_paths["glove_r"], dummy_csv_paths["glove_l"]

    df_ts_combined = backend_without_face.process_teslasuit_for_demo_backend(
        suit_p, glove_r_p, glove_l_p, trim_sec_ts=0.0
    )

    assert not df_ts_combined.empty
    assert isinstance(df_ts_combined.index, pd.TimedeltaIndex)
    assert "hips.position.x" in df_ts_combined.columns
    if backend_without_face.TS_GLOVE_RIGHT_FEATURES_TO_KEEP:
        assert "right_hand.position.x" in df_ts_combined.columns
    if backend_without_face.TS_GLOVE_LEFT_FEATURES_TO_KEEP:
        assert "left_hand.position.x" in df_ts_combined.columns
    if backend_without_face.TS_USE_SUIT_BIOMECH and backend_without_face.TS_RELEVANT_BIOMECH_JOINTS:
        assert "PelvisTilt.angular_v" in df_ts_combined.columns


    temp_dir = suit_p.parent
    non_existent_glove = temp_dir / "non_existent_glove.csv"
    df_missing_glove = backend_without_face.process_teslasuit_for_demo_backend(
        suit_p, non_existent_glove, glove_l_p, trim_sec_ts=0.0
    )
    assert not df_missing_glove.empty


def test_combine_ts_and_facial_for_demo(app_instance):
    assert len(backend_without_face.FEATURE_NAMES) >= 2, \
        f"FEATURE_NAMES not populated or too short. Length: {len(backend_without_face.FEATURE_NAMES)}"

    ts_data = {
        'normalized_timestamp_us': pd.to_timedelta(np.arange(0, 500000, 100000), unit='us'),
        backend_without_face.FEATURE_NAMES[0]: np.random.rand(5),
        backend_without_face.FEATURE_NAMES[1]: np.random.rand(5)
    }
    df_ts = pd.DataFrame(ts_data).set_index('normalized_timestamp_us')
    df_facial = None

    df_final = backend_without_face.combine_ts_and_facial_for_demo(
        df_ts, df_facial, backend_without_face.FEATURE_NAMES
    )

    assert not df_final.empty
    assert df_final.shape[1] == len(backend_without_face.FEATURE_NAMES)
    assert backend_without_face.FEATURE_NAMES[0] in df_final.columns
    assert backend_without_face.FEATURE_NAMES[1] in df_final.columns
    for col in backend_without_face.FEATURE_NAMES:
        if col not in df_ts.columns:
            assert np.all(df_final[col] == 0.0)

    with pytest.raises(ValueError, match="TeslaSuit DataFrame is empty"):
        backend_without_face.combine_ts_and_facial_for_demo(
            pd.DataFrame(), df_facial, backend_without_face.FEATURE_NAMES
        )


def test_segment_pipeline_endpoint_success(client, dummy_csv_paths, mocker):
    mock_segments = [{"start_ms": 100, "end_ms": 200}, {"start_ms": 300, "end_ms": 400}]
    mocker.patch('backend_without_face.run_inference_for_sequence_data', return_value=mock_segments)

    data = {
        'suit_file': (BytesIO(open(dummy_csv_paths["suit"], 'rb').read()), 'suit.csv'),
        'glove_right_file': (BytesIO(open(dummy_csv_paths["glove_r"], 'rb').read()), 'glove_r.csv'),
        'glove_left_file': (BytesIO(open(dummy_csv_paths["glove_l"], 'rb').read()), 'glove_l.csv'),
        'bag_file': (BytesIO(open(dummy_csv_paths["bag"], 'rb').read()), 'test.bag')
    }
    response = client.post('/segment_pipeline', data=data, content_type='multipart/form-data')

    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["message"].startswith("Processing successful")
    assert json_data["bilstm_segments"] == mock_segments
    assert json_data["bigru_segments"] == mock_segments
    assert json_data["num_frames_processed"] > 0
    assert json_data["num_features_final"] == len(backend_without_face.FEATURE_NAMES)
    assert json_data["trim_applied_input_ms"] == int(backend_without_face.TRIM_SECONDS_APPLIED_TO_INPUT * 1000)


def test_segment_pipeline_endpoint_missing_files(client):
    data = {'suit_file': (BytesIO(b"dummy suit data"), 'suit.csv')}
    response = client.post('/segment_pipeline', data=data, content_type='multipart/form-data')
    assert response.status_code == 400


def test_segment_pipeline_processing_error(client, dummy_csv_paths, mocker):
    mocker.patch('backend_without_face.process_teslasuit_for_demo_backend',
                 side_effect=ValueError("Test processing error"))
    data = {
        'suit_file': (BytesIO(open(dummy_csv_paths["suit"], 'rb').read()), 'suit.csv'),
        'glove_right_file': (BytesIO(open(dummy_csv_paths["glove_r"], 'rb').read()), 'glove_r.csv'),
        'glove_left_file': (BytesIO(open(dummy_csv_paths["glove_l"], 'rb').read()), 'glove_l.csv'),
        'bag_file': (BytesIO(open(dummy_csv_paths["bag"], 'rb').read()), 'test.bag')
    }
    response = client.post('/segment_pipeline', data=data, content_type='multipart/form-data')
    assert response.status_code == 400


def test_load_global_components_file_not_found(mocker, test_data_dir):
    non_existent_path = test_data_dir / "non_existent_model.keras"

    mocker.patch('backend_without_face.MODEL_PATH_BILSTM', non_existent_path)
    mocker.patch('backend_without_face.MODEL_PATH_BIGRU', test_data_dir / "dummy_model.keras")
    mocker.patch('backend_without_face.SCALER_PATH', test_data_dir / "dummy_scaler.pkl")
    mocker.patch('backend_without_face.TRAINING_CONFIG_PKL', test_data_dir / "dummy_config.pkl")

    mock_config = {'feature_names': ['f1', 'f2']}
    mock_scaler_obj = "dummy_scaler_obj_for_this_test"

    def mock_pickle_load_isolated(file_obj):
        if 'config' in str(file_obj.name) or 'features' in str(file_obj.name):
            return mock_config
        if 'scaler' in str(file_obj.name):
            return mock_scaler_obj
        raise ValueError(f"Isolated mock_pickle_load: Unhandled file {file_obj.name}")

    mocker.patch('pickle.load', side_effect=mock_pickle_load_isolated)
    mocker.patch('tensorflow.keras.models.load_model', side_effect=FileNotFoundError("Mocked load_model error"))

    backend_without_face.populate_ts_feature_lists()

    with pytest.raises(SystemExit):
        backend_without_face.load_global_components()