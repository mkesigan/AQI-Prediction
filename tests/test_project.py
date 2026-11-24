import numpy as np
import pandas as pd
import pytest

from data_preprocessing import CLASSES, FEATURES, category_from_aqi, load_data
from data_preprocessing import prepare_inputs, split_data
from predict import predict_data
from train import evaluate, ranking


@pytest.mark.parametrize("value, label", [
    (0, "Good"), (50, "Good"), (51, "Moderate"), (100, "Moderate"),
    (101, "Unhealthy for Sensitive Groups"), (150, "Unhealthy for Sensitive Groups"),
    (151, "Unhealthy"), (200, "Unhealthy"), (201, "Very Unhealthy"),
    (300, "Very Unhealthy"), (301, "Hazardous"), (1000, "Hazardous"),
    (-1, None), (None, None), (np.nan, None), (np.inf, None),
])
def test_category_boundaries(value, label):
    assert category_from_aqi(value) == label


@pytest.fixture
def inputs():
    return pd.DataFrame([{ "pm25": 35.0, "pm10": 20.0, "lon": 79.8, "lat": 6.9,
                           "no2": 8.0, "co": 2.0, "temp_c": 25.0}], columns=FEATURES)


@pytest.mark.parametrize("name, value", [("pm25", -1), ("lat", 91), ("lon", -181),
                                         ("temp_c", np.inf), ("no2", "invalid")])
def test_invalid_inputs(inputs, name, value):
    frame = inputs.astype(object)
    frame.loc[0, name] = value
    with pytest.raises(ValueError):
        prepare_inputs(frame)


def test_all_missing_pollutants_rejected(inputs):
    inputs[["pm25", "pm10", "no2", "co"]] = np.nan
    with pytest.raises(ValueError, match="at least one"):
        prepare_inputs(inputs)


def test_targets_do_not_enter_features(inputs):
    first = prepare_inputs(inputs.assign(aqi=20, aqi_category="Good", uid=1))
    second = prepare_inputs(inputs.assign(aqi=400, aqi_category="Hazardous", uid=999))
    pd.testing.assert_frame_equal(first, second)


def test_peaks_and_southern_latitudes_are_retained(inputs):
    inputs.loc[0, "pm25"] = 700
    inputs.loc[0, "lat"] = -40
    original = inputs.copy(deep=True)
    prepared = prepare_inputs(inputs)
    assert prepared.loc[0, "pm25"] == 700
    assert prepared.loc[0, "lat"] == -40
    pd.testing.assert_frame_equal(inputs, original)


@pytest.fixture
def dataset_path(tmp_path):
    records = []
    for category_index, aqi in enumerate([20, 75, 125, 175, 250, 350]):
        for station in range(8):
            uid = category_index * 8 + station + 1
            for hour in [8, 9]:
                records.append({"uid": uid, "time": f"2025-09-21 {hour}:00:00", "aqi": aqi,
                                "pm25": aqi, "pm10": 10, "no2": 5, "co": 2,
                                "lat": -30 + station, "lon": 70, "temp_c": 25})
    path = tmp_path / "data.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def test_partitions_have_no_station_overlap(dataset_path):
    data, _ = load_data(dataset_path)
    splits = split_data(data)
    groups = {name: set(data.iloc[ix]["uid"]) for name, ix in splits.items()}
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    assert not groups["validation"] & groups["test"]
    np.testing.assert_array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(len(data)))
    assert set(data.iloc[splits["train"]]["target"]) == set(data["target"])


def test_missing_target_is_excluded(dataset_path):
    raw = pd.read_csv(dataset_path)
    raw.loc[0, "aqi"] = np.nan
    raw.to_csv(dataset_path, index=False)
    data, audit = load_data(dataset_path)
    assert audit["missing_target_rows"] == 1
    assert len(data) == len(raw) - 1
    assert "Unknown" not in set(data["target"])


def test_rare_category_split_repair(dataset_path, monkeypatch):
    data, _ = load_data(dataset_path)
    rare_groups = sorted(set(data.loc[data["target"].eq("Very Unhealthy"), "uid"]))
    assignments = {uid: i % 2 for i, uid in enumerate(rare_groups)}
    assignments.update({uid: i % 5 for i, uid in enumerate(sorted(set(data["uid"]) - set(rare_groups)))})

    class RareHeldOut:
        def __init__(self, **kwargs):
            pass

        def split(self, X, y, groups):
            fold_ids = groups.map(assignments).to_numpy()
            for fold in range(5):
                yield np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)

    monkeypatch.setattr("data_preprocessing.StratifiedGroupKFold", RareHeldOut)
    parts = split_data(data)
    for name in parts:
        assert "Very Unhealthy" in set(data.iloc[parts[name]]["target"])
    groups = [set(data.iloc[ix]["uid"]) for ix in parts.values()]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]


def test_validation_selection_uses_macro_f1():
    high_accuracy = {"model": "A", "accuracy": 0.99, "macro_f1": 0.4}
    better_f1 = {"model": "B", "accuracy": 0.95, "macro_f1": 0.6}
    assert min([high_accuracy, better_f1], key=ranking)["model"] == "B"
    assert evaluate(CLASSES, CLASSES)["macro_f1"] == pytest.approx(1.0)


class FakeClassifier:
    classes_ = np.array(CLASSES[::-1])

    def predict(self, X):
        return np.repeat("Good", len(X))

    def predict_proba(self, X):
        return np.tile([0, 0, 0, 0, 0.1, 0.9], (len(X), 1))


def test_prediction_preserves_rows_and_aligns_scores(inputs):
    original = inputs.copy(deep=True)
    result, scores = predict_data(inputs, {"model": FakeClassifier()})
    assert result["Predicted category"].tolist() == ["Good"]
    assert list(scores) == CLASSES
    np.testing.assert_allclose(scores.iloc[0], [0.9, 0.1, 0, 0, 0, 0])
    pd.testing.assert_frame_equal(inputs, original)
