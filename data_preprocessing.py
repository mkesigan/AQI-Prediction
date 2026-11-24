from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
DATA_PATH = DATA_PATH = ROOT / "data" / "raw" / "waqi_global_dataset_with_categoricals.csv"
MODEL_PATH = ROOT / "models" / "aqi_model.joblib"
REPORTS = ROOT / "reports"
FEATURES = ["pm25", "pm10", "lon", "lat", "no2", "co", "temp_c"]
POLLUTANTS = ["pm25", "pm10", "no2", "co"]
CLASSES = ["Good", "Moderate", "Unhealthy for Sensitive Groups",
           "Unhealthy", "Very Unhealthy", "Hazardous"]


def category_from_aqi(value):
    """Convert a reported AQI to its category. This is not a numeric predictor."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or value < 0:
        return None
    return CLASSES[int(np.searchsorted([50, 100, 150, 200, 300], value, side="left"))]


def prepare_inputs(frame, strict=True):
    """Select the seven numeric inputs. Missing cells are allowed."""
    if frame.empty or not frame.columns.is_unique:
        raise ValueError("Provide a nonempty table with unique column names.")
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    X = frame[FEATURES].copy().reset_index(drop=True)
    for column in FEATURES:
        numeric = pd.to_numeric(X[column], errors="coerce").astype(float)
        invalid = (X[column].notna() & numeric.isna()) | np.isinf(numeric)
        if column in POLLUTANTS:
            invalid |= numeric < 0
        elif column == "lat":
            invalid |= numeric.abs() > 90
        elif column == "lon":
            invalid |= numeric.abs() > 180
        if strict and invalid.any():
            raise ValueError(f"Invalid values in {column}.")
        X[column] = numeric.mask(invalid)
    if strict and X[POLLUTANTS].isna().all(axis=1).any():
        raise ValueError("Each row needs at least one pollutant AQI subindex.")
    return X


def load_data(path=DATA_PATH):
    """Read without changing the CSV; derive labels from its reported AQI."""
    raw = pd.read_csv(path)
    required = set(FEATURES) | {"uid", "aqi"}
    if not required.issubset(raw.columns):
        raise ValueError("Training CSV is missing: " + ", ".join(sorted(required - set(raw.columns))))
    original_rows = len(raw)
    duplicate = raw.duplicated()
    raw["source_row"] = np.arange(len(raw))
    raw = raw.loc[~duplicate].reset_index(drop=True)
    if raw["uid"].isna().any():
        raise ValueError("Station uid cannot be missing.")
    raw["uid"] = raw["uid"].astype(str).str.strip()
    if raw["uid"].eq("").any():
        raise ValueError("Station uid cannot be blank.")
    if "time" in raw and raw.duplicated(["uid", "time"]).any():
        raise ValueError("Resolve conflicting duplicate station/time observations first.")
    X = prepare_inputs(raw, strict=False)
    target = raw["aqi"].map(category_from_aqi)
    has_pollutant = X[POLLUTANTS].notna().any(axis=1)
    keep = target.notna() & has_pollutant
    data = X.loc[keep].copy()
    data["target"] = target.loc[keep]
    data["uid"] = raw.loc[keep, "uid"]
    data["source_row"] = raw.loc[keep, "source_row"]
    audit = {"source_rows": original_rows, "duplicates_removed": int(duplicate.sum()),
             "missing_target_rows": int(target.isna().sum()),
             "labeled_rows_without_input_pollutants": int((target.notna() & ~has_pollutant).sum()),
             "eligible_rows": len(data), "stations": int(data["uid"].nunique())}
    return data.reset_index(drop=True), audit


def split_data(data, seed=42):
    """Approximately 60/20/20, with whole stations and training-label coverage."""
    if data["uid"].nunique() < 5 or data["target"].nunique() < 2:
        raise ValueError("Need at least five stations and two target categories.")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(splitter.split(data[FEATURES], data["target"], data["uid"]))
    held_out = np.union1d(folds[0][1], folds[1][1])
    initial = {"train": np.setdiff1d(np.arange(len(data)), held_out),
               "validation": folds[1][1], "test": folds[0][1]}
    groups = {name: set(data.iloc[positions]["uid"]) for name, positions in initial.items()}
    members = data.groupby("uid", sort=False).indices
    labels = {station: set(data.iloc[positions]["target"]) for station, positions in members.items()}
    category_groups = {label: {station for station in labels if label in labels[station]}
                       for label in set(data["target"])}
    for label in sorted(category_groups):
        if groups["train"] & category_groups[label]:
            continue
        choices = []
        for donor in ("validation", "test"):
            if len(groups[donor]) <= 1:
                continue
            for station in groups[donor] & category_groups[label]:
                lost = sum(len(groups[donor] & category_groups[c]) == 1 for c in labels[station])
                priority = (lost, donor != "validation", len(members[station]), str(station))
                choices.append((priority, donor, station))
        if not choices:
            raise ValueError(f"More station groups are needed to include {label} in training.")
        _, donor, station = min(choices, key=lambda choice: choice[0])
        groups[donor].remove(station)
        groups["train"].add(station)
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if groups[left] & groups[right]:
            raise RuntimeError("Station overlap detected.")
    result = {name: np.flatnonzero(data["uid"].isin(stations).to_numpy())
              for name, stations in groups.items()}
    if not np.array_equal(np.sort(np.concatenate(list(result.values()))), np.arange(len(data))):
        raise RuntimeError("The split lost or repeated observations.")
    if set(data.iloc[result["train"]]["target"]) != set(data["target"]):
        raise RuntimeError("Training is missing a target category.")
    return result


def index_reference(X):
    """Reference from the four available pollutant indices; others can be missing."""
    return X[POLLUTANTS].max(axis=1).map(category_from_aqi).to_numpy()
