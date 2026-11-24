# AQI Prediction

Current AQI-category prediction with three classifiers, voting ensembles and a
two-tab Streamlit frontend. This download contains source code only.

## Five application code files

| File | Purpose |
|---|---|
| data_preprocessing.py | Read/validate data and create station-separated partitions |
| model.py | Define Logistic Regression, Random Forest and Gradient Boosting |
| train.py | Train, compare, select an ensemble or individual model and save results |
| predict.py | Load the saved predictor, predict inputs and verify saved outputs |
| app.py | Single and batch prediction interface |

`EDA.ipynb` contains data exploration only. It does not train models, import
training functions or require training switches. `tests/test_project.py` contains
checks that do not fit classifiers. No fitted model is shipped.

## 1. Set up

Extract this project into a fresh folder. Open that folder in VS Code.
In its PowerShell terminal run:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

Put your existing CSV here:

```text
data/waqi_global_dataset_with_categoricals.csv
```

The dataset is not included in the download. Keep its content unchanged.
Open `EDA.ipynb`, select `.venv` as the kernel, and run its cells.

## 2. Train locally

```powershell
python train.py
```

Training uses approximately 60%/20%/20% station-separated partitions. A rare
category missing from training is repaired by moving a whole held-out station
into training, without overlapping stations or using model-performance results.

Each learned pipeline fits its median imputer on training rows only. Logistic
Regression additionally fits a training-only scaler. Random Forest uses
balanced subsampling weights, and Logistic Regression uses balanced class
weights. Valid high pollutant indices and southern latitudes are retained.

The three individual classifiers are compared using validation macro F1, with
validation accuracy and model name as tie breakers. VotingClassifier fits
training-only clones of those pipelines. Four predefined relative weights are
compared: [1,1,1], [1,2,1], [1,1,2], [2,1,1], ordered Logistic Regression,
Random Forest, Gradient Boosting. Weights are normalized during score averaging.

The deployed predictor is selected from these **learned models and ensembles**.
It can remain an individual model if voting does not improve validation.
The choice is written to `reports/selection.json` before test evaluation.

Two additional references are reported: the training-majority category and the
category of the maximum available input pollutant index. They are benchmarks,
not deployment candidates in this workflow. This selection policy matches the
three-model/ensemble experiment; it does not claim the selected ML model beats
every rule. If a reference has higher validation macro F1, the script says so
and records the outcome in the report.

No test score changes the frozen learned-model selection. Macro F1 includes all
six fixed categories, with `zero_division=0`; inspect per-category support.

### Generated files

- `models/aqi_model.joblib`: the complete selected learned predictor and metadata.
- `reports/selection.json`: frozen validation choice.
- `reports/model_comparison.csv`: individual, ensemble and reference comparison.
- `reports/classification_report.csv`: per-category precision, recall, F1/support.
- `reports/confusion_matrix.csv`: actual rows versus predicted columns.
- `reports/test_predictions.csv`: frozen test IDs, labels and selected scores.
- `reports/summary.json`: preparation/split counts and selected metrics.

An existing run is protected. Use `python train.py --overwrite` only when you
intentionally want to replace its model and reports. Training always remains
your explicit action; opening the app or EDA never starts it.

## 3. Verify the saved predictor

```powershell
python predict.py --verify
```

This checks exact dataset bytes, frozen test row IDs, actual labels, predicted
categories and class-score agreement after loading the saved model. It does
not fit anything. Keep the scikit-learn version used to create the artifact.

## 4. Open the app

```powershell
python -m streamlit run app.py
```

Open the local address printed in the terminal, usually `http://localhost:8501`.

The frontend contains only:

- **Single prediction:** seven inputs, estimated category/range, model-score and
  pollutant-index charts, supplied-location map and CSV download.
- **Batch prediction:** CSV/XLSX upload, results and CSV/PDF downloads. PDF exports
  the first 500 rows with repeated page headers; CSV exports all predictions.

There are no experiment selectors, training-score cards, model-comparison tabs,
project-description tabs or login accounts in the app. Training reports remain
in the `reports` folder. The UI has no live data connection.

### Correct input units

| Column | Input |
|---|---|
| pm25, pm10, no2, co | Individual pollutant AQI subindices from WAQI `iaqi` |
| temp_c | Temperature in Celsius |
| lat, lon | Latitude/longitude in degrees |

These pollutant inputs are **not raw concentrations** in micrograms/m3 or
milligrams/m3. At least one pollutant must be present in each row. Other missing
values use the fitted training medians and are listed in the prediction output.
The template contains synthetic example inputs.

The model predicts a category, not a numeric AQI value. The displayed AQI range
describes that category. The app does not invent a weighted numeric AQI, compare
index values with concentration guidelines or claim calibrated confidence.

For terminal batch prediction:

```powershell
python predict.py --input input.csv --output output.csv
```

Extra input columns are ignored by the model and preserved in the CSV result.
Columns named `Predicted category`, `Class score` and `Missing inputs` are
replaced if already present.

## Checks

```powershell
python -m pytest tests/test_project.py -q
```

These checks use synthetic data/fakes and do not train classifiers. Saved-model
verification must be run after you train locally. No new model accuracy is
claimed by the source distribution.

Before delivery, all 29 checks passed, every EDA code cell ran on your CSV,
and Streamlit application checks exercised single prediction, invalid-input
handling, batch CSV prediction, chart/map rendering and CSV/PDF generation
using a fake predictor. These checks do not replace training and saved-model
verification in your local environment.

## Dataset limits

The recording has only 1-3 observations per station. This is current-category
estimation, not future AQI forecasting. Some high-AQI categories have very few
examples; strong aggregate accuracy does not establish reliable performance
for those categories. The historical dataset was already inspected during
earlier development, so this is not a fresh external test.

The categories are derived from pollutant indices; a simple rule can outperform
learned models. The available-index reference uses only the same four input
pollutants, so it can miss an unobserved dominant pollutant. It is not an
official concentration-to-AQI calculation.

Source: World Air Quality Index and originating monitoring agencies.
Documentation: https://aqicn.org/api/
Keep the source CSV local unless you have permission to redistribute it; API
terms restrict redistribution of cached/archived API data. This project does
not contain an API token, collected data or user-account records.
