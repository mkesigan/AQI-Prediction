import argparse
import copy
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits

from data_preprocessing import CLASSES, DATA_PATH, FEATURES, MODEL_PATH, REPORTS
from data_preprocessing import index_reference, load_data, split_data
from model import VOTING_WEIGHTS, create_models


def evaluate(actual, predicted):
    return {"accuracy": float(accuracy_score(actual, predicted)),
            "macro_f1": float(f1_score(actual, predicted, labels=CLASSES,
                                      average="macro", zero_division=0))}


def ranking(row):
    return (-row["macro_f1"], -row["accuracy"], row["model"])


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def train(data_path=DATA_PATH, overwrite=False, seed=42):
    # 1. Prepare the CSV and keep each station in exactly one partition.
    if not overwrite and (MODEL_PATH.exists() or (REPORTS / "selection.json").exists()):
        raise FileExistsError("A previous run exists. Use --overwrite only for an intentional rerun.")
    data, audit = load_data(data_path)
    splits = split_data(data, seed)
    training = data.iloc[splits["train"]]
    validation = data.iloc[splits["validation"]]
    print("Data audit:", json.dumps(audit, indent=2))
    counts = {}
    for name, rows in splits.items():
        part = data.iloc[rows]
        counts[name] = {"rows": len(part), "stations": int(part["uid"].nunique()),
                        "categories": {label: int(part["target"].eq(label).sum()) for label in CLASSES}}
    print("Partitions:", json.dumps(counts, indent=2))
    models = create_models(seed)
    records = []

    with threadpool_limits(limits=2):
        # 2. Fit the three individual classifiers on training rows only.
        for name, estimator in models.items():
            print("Training", name, flush=True)
            estimator.fit(training[FEATURES], training["target"])
            row = {"model": name, "kind": "individual",
                   **evaluate(validation["target"], estimator.predict(validation[FEATURES]))}
            records.append(row)
            print("Validation:", row, flush=True)
        best_single = min(records, key=ranking)

        # 3. Compare fixed voting weights using validation rows only.
        # VotingClassifier fits training-only clones of the same three pipelines.
        voter = VotingClassifier(estimators=list(create_models(seed).items()),
                                 voting="soft", n_jobs=1)
        print("Training voting components", flush=True)
        voter.fit(training[FEATURES], training["target"])
        for name, weights in VOTING_WEIGHTS.items():
            candidate = copy.copy(voter)
            candidate.weights = list(weights)
            models[name] = candidate
            row = {"model": name, "kind": "ensemble",
                   **evaluate(validation["target"], candidate.predict(validation[FEATURES]))}
            records.append(row)
            print("Validation:", row, flush=True)
        best_ensemble = min([r for r in records if r["kind"] == "ensemble"], key=ranking)
        selected = min(records, key=ranking)

        # Freeze the learned-model choice before computing test predictions.
        REPORTS.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        selection = {"selected_model": selected["model"], "seed": seed,
                     "best_individual": best_single["model"],
                     "best_ensemble": best_ensemble["model"],
                     "weights": VOTING_WEIGHTS.get(selected["model"]),
                     "selection_metric": "Validation macro F1; accuracy and model name break ties.",
                     "policy": "Select among learned classifiers/ensembles; references are comparison only.",
                     "selection_order": "Frozen before test evaluation."}
        save_json(REPORTS / "selection.json", selection)
        print("Frozen learned predictor:", selected["model"], flush=True)

        # 4. Measure test performance after selection, then save reports/model.
        test = data.iloc[splits["test"]]
        comparison = []
        for row in records:
            measured = evaluate(test["target"], models[row["model"]].predict(test[FEATURES]))
            comparison.append({"model": row["model"], "kind": row["kind"],
                               "validation_accuracy": row["accuracy"],
                               "validation_macro_f1": row["macro_f1"],
                               "test_accuracy": measured["accuracy"],
                               "test_macro_f1": measured["macro_f1"],
                               "selected": row["model"] == selected["model"]})
        majority = training["target"].mode().iloc[0]
        for name, val_pred, test_pred in [
            ("MajorityReference", np.repeat(majority, len(validation)), np.repeat(majority, len(test))),
            ("AvailableIndexReference", index_reference(validation[FEATURES]), index_reference(test[FEATURES])),
        ]:
            val_score = evaluate(validation["target"], val_pred)
            test_score = evaluate(test["target"], test_pred)
            comparison.append({"model": name, "kind": "reference",
                               "validation_accuracy": val_score["accuracy"],
                               "validation_macro_f1": val_score["macro_f1"],
                               "test_accuracy": test_score["accuracy"],
                               "test_macro_f1": test_score["macro_f1"], "selected": False})
        table = pd.DataFrame(comparison)
        table.to_csv(REPORTS / "model_comparison.csv", index=False)

        bundle = {"model": models[selected["model"]], "name": selected["model"],
                  "features": FEATURES, "classes": CLASSES, "sklearn_version": sklearn.__version__,
                  "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                  "test_rows": test["source_row"].astype(int).tolist(), "selection": selection}
        from predict import predict_data
        estimates, scores = predict_data(test[FEATURES], bundle)
        predictions = estimates["Predicted category"].to_numpy()
        saved = pd.DataFrame({"source_row": test["source_row"].to_numpy(),
                              "actual_category": test["target"].to_numpy(),
                              "predicted_category": predictions})
        for i in range(len(CLASSES)):
            saved[f"score_{i}"] = scores.iloc[:, i].to_numpy()
        saved.to_csv(REPORTS / "test_predictions.csv", index=False)
        report = classification_report(test["target"], predictions, labels=CLASSES,
                                       output_dict=True, zero_division=0)
        pd.DataFrame([{ "category": label, **report[label]} for label in CLASSES]).to_csv(
            REPORTS / "classification_report.csv", index=False)
        pd.DataFrame(confusion_matrix(test["target"], predictions, labels=CLASSES),
                     index=CLASSES, columns=CLASSES).rename_axis("actual_category").to_csv(
            REPORTS / "confusion_matrix.csv")
        reference_better = max(r["validation_macro_f1"] for r in comparison if r["kind"] == "reference") > selected["macro_f1"]
        summary = {**selection, "audit": audit, "partitions": counts,
                   "selected_test_metrics": evaluate(test["target"], predictions),
                   "reference_has_higher_validation_macro_f1": reference_better,
                   "note": "Historical snapshots; rare-category results have very limited support. "
                           "The dataset was inspected during earlier development, not fresh external validation."}
        save_json(REPORTS / "summary.json", summary)
        joblib.dump(bundle, MODEL_PATH, compress=3)
    print("\n", table.to_string(index=False))
    if reference_better:
        print("\nA rule/reference outperformed the learned predictors on validation macro F1. "
              "This is recorded; do not claim the learned predictor is best overall.")
    print("\nSaved predictor:", MODEL_PATH)
    print("Saved reports:", REPORTS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(overwrite=args.overwrite, seed=args.seed)
