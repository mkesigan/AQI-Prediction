import argparse
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score

from data_preprocessing import CLASSES, DATA_PATH, FEATURES, MODEL_PATH, REPORTS
from data_preprocessing import load_data, prepare_inputs


def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Saved model is missing. Run python train.py first.")
    bundle = joblib.load(MODEL_PATH)
    if bundle.get("features") != FEATURES or bundle.get("classes") != CLASSES:
        raise ValueError("The saved model uses a different input contract.")
    if bundle["sklearn_version"] != sklearn.__version__:
        raise ValueError("Use scikit-learn " + bundle["sklearn_version"] + " for this saved model.")
    return bundle


def predict_data(frame, bundle):
    """Return the input rows with estimates, plus a severity-ordered score table."""
    X = prepare_inputs(frame)
    classifier = bundle["model"]
    result = frame.reset_index(drop=True).copy()
    result["Predicted category"] = classifier.predict(X)
    scores = pd.DataFrame(classifier.predict_proba(X), columns=classifier.classes_)
    scores = scores.reindex(columns=CLASSES, fill_value=0.0)
    if not np.isfinite(scores.to_numpy()).all() or not np.allclose(scores.sum(axis=1), 1.0):
        raise RuntimeError("The predictor returned invalid class scores.")
    result["Class score"] = scores.max(axis=1)
    result["Missing inputs"] = X.isna().apply(
        lambda row: ", ".join(row.index[row.to_numpy()]), axis=1)
    return result, scores


def verify_model(data_path=DATA_PATH):
    bundle = load_model()
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if digest != bundle["dataset_sha256"]:
        raise ValueError("Verification needs the exact CSV used for this saved model.")
    data, _ = load_data(data_path)
    report = pd.read_csv(REPORTS / "test_predictions.csv")
    if not np.array_equal(report["source_row"], bundle["test_rows"]):
        raise ValueError("The saved test rows differ from the frozen assignment.")
    test = data.set_index("source_row").loc[bundle["test_rows"]]
    if not np.array_equal(test["target"], report["actual_category"]):
        raise ValueError("Saved target categories differ from the original data.")
    estimates, scores = predict_data(test[FEATURES], bundle)
    if not np.array_equal(estimates["Predicted category"], report["predicted_category"]):
        raise RuntimeError("Saved and reloaded class predictions differ.")
    recorded = report[[f"score_{i}" for i in range(len(CLASSES))]].to_numpy()
    if not np.allclose(recorded, scores.to_numpy(), rtol=1e-8, atol=1e-10):
        raise RuntimeError("Saved and reloaded class scores differ.")
    return {"verification_passed": True, "model": bundle["name"], "test_rows": len(test),
            "maximum_score_difference": float(np.max(np.abs(recorded - scores.to_numpy()))),
            "accuracy": float(accuracy_score(test["target"], estimates["Predicted category"])),
            "macro_f1": float(f1_score(test["target"], estimates["Predicted category"],
                                       labels=CLASSES, average="macro", zero_division=0))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--input", help="CSV or XLSX containing the seven input columns")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_model(), indent=2))
    elif args.input:
        frame = pd.read_excel(args.input) if args.input.lower().endswith(".xlsx") else pd.read_csv(args.input)
        result, _ = predict_data(frame, load_model())
        result.to_csv(args.output, index=False)
        print("Saved:", args.output)
    else:
        parser.error("Use --verify or --input your_inputs.csv.")
