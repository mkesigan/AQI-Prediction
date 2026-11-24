<div align="center">

# 🌍 AQI Prediction

### Air quality classification · Model comparison · Interactive predictions

Estimate AQI categories from pollutant indices, weather and location using three classifiers and a weighted soft-voting ensemble.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.51-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Ensemble](https://img.shields.io/badge/Ensemble-Soft%20Voting-2463AE?style=flat-square)

[Features](#features) · [Quick start](#quick-start) · [Models and results](#models-and-results) · [Project files](#project-files)

</div>

---

## Overview

AQI Prediction provides a complete workflow from exploratory analysis to an interactive application. It compares Logistic Regression, Random Forest and Gradient Boosting, evaluates four soft-voting combinations, and selects a learned predictor using validation macro F1 before evaluating the test set.

The Streamlit application supports individual observations and CSV/Excel batches, with category results, charts, a location map and downloadable predictions.

**Task:** classify the category of a supplied observation. The project does not forecast future AQI or fetch live sensor readings.

## Features

- **Focused interface:** sidebar inputs, coloured category results and single/batch prediction tabs.
- **Readable visualisations:** category scores, supplied pollutant indices and a light location map.
- **Model comparison:** three individual classifiers and four predefined voting weight combinations.
- **Station separation:** all observations from a station remain in the same data partition.
- **Training-only preprocessing:** fitted median imputation, missing-value indicators and scaling where needed.
- **Reproducible prediction:** saved preprocessing/model pipelines and reloaded-model verification.
- **Useful exports:** single-result CSV and batch CSV/PDF downloads.

## Dataset and inputs

The source CSV was collected using the [World Air Quality Index API](https://aqicn.org/api/). Data attribution belongs to the World Air Quality Index project and the originating monitoring agencies.

| Dataset measure | Observations |
|---|---:|
| Original rows | 12,815 |
| Rows without a valid target | 114 |
| Additional labelled rows without any input pollutant | 98 |
| Eligible rows | **12,603** |
| Eligible monitoring stations | **4,527** |

The target is derived from the reported `aqi`. The reported AQI, supplied category, station ID and timestamp are excluded from model inputs.

| Input | Meaning | Unit |
|---|---|---|
| `pm25` | PM2.5 individual index | AQI subindex |
| `pm10` | PM10 individual index | AQI subindex |
| `no2` | Nitrogen dioxide individual index | AQI subindex |
| `co` | Carbon monoxide individual index | AQI subindex |
| `temp_c` | Temperature | °C |
| `lat` | Latitude | Degrees, −90 to 90 |
| `lon` | Longitude | Degrees, −180 to 180 |

**Input units matter:** the pollutant fields come from WAQI `iaqi` and are individual AQI subindices, not raw concentrations in µg/m³ or mg/m³. Each prediction row requires at least one pollutant. Other missing inputs use preprocessing fitted on training data.

The six categories are **Good**, **Moderate**, **Unhealthy for Sensitive Groups**, **Unhealthy**, **Very Unhealthy** and **Hazardous**.

## Quick start

The recorded training environment used **Python 3.13.3** and **scikit-learn 1.7.2**. The commands below use Windows PowerShell.

### 1. Clone and install

```powershell
git clone https://github.com/mkesigan/AQI-Prediction.git
cd AQI-Prediction

py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` includes the application dependencies, notebook tools and pytest. Use the project `.venv` for both the terminal and notebook kernel.

### 2. Add the dataset

Create the local dataset directory if needed:

```powershell
New-Item -ItemType Directory -Force .\data\raw | Out-Null
```

Place your existing dataset at:

```text
data/raw/waqi_global_dataset_with_categoricals.csv
```

The source dataset and trained model artifacts are kept local and excluded from version control.

### 3. Explore the data

Open [EDA.ipynb](EDA.ipynb), select the project `.venv` kernel, and run the cells. The notebook audits missing values, category balance, station coverage, distributions, correlations and partition support. It does not train or save models.

### 4. Train and compare

```powershell
python train.py
```

The command prepares the data, fits the candidates, selects the learned predictor on validation results, evaluates the test partition and saves the model/reports.

An existing run is protected against accidental replacement. For an intentional rerun, use `python train.py --overwrite`.

### 5. Verify the saved predictor

```powershell
python predict.py --verify
```

Verification checks the dataset fingerprint, frozen test row IDs, actual labels, predicted categories and class scores after reloading the saved artifact. It does not fit a model.

### 6. Launch the application

```powershell
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually [http://localhost:8501](http://localhost:8501). The application loads the saved predictor; it does not trigger training.

## Models and results

### Evaluation approach

The recorded split contains **7,560 training**, **2,521 validation** and **2,522 test** observations, with no station shared across partitions.

If an observed category is absent from training, the split procedure moves a whole held-out station into training to provide category coverage. This fixed repair uses labels and station membership, without model-performance scores.

Each pipeline fits its imputer on training rows only. Logistic Regression also fits a scaler. Logistic Regression uses balanced class weights, while Random Forest uses balanced subsampling weights.

The selection metric is **validation macro F1 across all six categories**. Validation accuracy and model name break ties. The selection is written to `reports/selection.json` before computing test results.

### Recorded experiment

| Candidate | Validation macro F1 | Test accuracy | Test macro F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.5735 | 94.41% | 0.5956 |
| Random Forest | 0.6878 | **98.93%** | 0.7414 |
| Gradient Boosting | 0.8612 | 98.53% | 0.7038 |
| VotingEqual | 0.8532 | 98.69% | 0.7385 |
| VotingForest | 0.8557 | 98.77% | 0.7405 |
| **VotingBoosting — selected learned predictor** | **0.8634** | **98.77%** | **0.7407** |
| VotingLogistic | 0.7307 | 98.41% | 0.7015 |
| Majority reference | 0.1482 | 79.14% | 0.1473 |
| Available-index rule reference | **0.9425** | 98.61% | **0.8853** |

Results describe the recorded historical experiment. Reproducing it requires the same CSV, configuration and compatible dependencies.

### Selected ensemble

`VotingBoosting` combines the three models' class scores with relative weights **1 : 1 : 2**:

```text
Combined score for each category =
    0.25 × Logistic Regression score
  + 0.25 × Random Forest score
  + 0.50 × Gradient Boosting score
```

The category with the highest combined score becomes the prediction. The ensemble improved validation macro F1 modestly, from **0.8612** for Gradient Boosting to **0.8634**. A single model remains eligible for selection when voting does not improve validation.

**Benchmark interpretation:** selection is restricted to learned classifiers and their voting ensembles. The available-index rule is a separate reference and achieved higher macro F1 than every learned candidate. The selected ensemble is therefore not the strongest overall method in this comparison. Test results do not change the frozen validation choice.

Reloaded-model verification passed on all **2,522 test observations**, with a maximum class-score difference of approximately **1.11 × 10⁻¹⁶**.

## Application

| Single prediction | Batch prediction |
|---|---|
| Enter pollutant indices, temperature and coordinates | Upload CSV or XLSX observations |
| View the estimated category and its AQI range | Preview rows and generate category estimates |
| Inspect category scores and supplied pollutant values | Review category counts and the result table |
| Explore the entered location on a map | Download CSV or a PDF of the first 500 rows |
| Download the prediction as CSV | Download an input template |

The displayed AQI range describes the predicted category; it is not a predicted numeric AQI. Model scores are uncalibrated outputs, rather than calibrated confidence estimates. The map shows supplied coordinates and requires internet access for basemap tiles.

### Example batch input

These are synthetic demonstration values, not current monitoring readings. Include all seven headers; leave individual cells blank where a reading is unavailable.

```csv
pm25,pm10,lon,lat,no2,co,temp_c
25,18,79.8,6.9,8,2,28
85,65,79.8,6.9,20,5,28
175,130,79.8,6.9,60,12,28
```

Terminal batch prediction is also supported:

```powershell
python predict.py --input input.csv --output output.csv
```

Extra input columns are preserved in the result but ignored by the model. Existing columns named `Predicted category`, `Class score` and `Missing inputs` are replaced by the new outputs.

## Project files

| Path | Responsibility |
|---|---|
| [EDA.ipynb](EDA.ipynb) | Dataset audit and exploratory analysis |
| [data_preprocessing.py](data_preprocessing.py) | Data validation, label preparation and station-separated splits |
| [model.py](model.py) | Three classifier pipelines and voting weights |
| [train.py](train.py) | Training, validation selection, test evaluation and saving |
| [predict.py](predict.py) | Artifact loading, input prediction and saved-result verification |
| [app.py](app.py) | Single and batch prediction interface |
| [.streamlit/config.toml](.streamlit/config.toml) | Application theme |
| [tests/test_project.py](tests/test_project.py) | Data, partition, score-alignment and selection checks |
| [requirements.txt](requirements.txt) | Runtime dependencies |
| [requirements-dev.txt](requirements-dev.txt) | Runtime, notebook and testing dependencies |
| `data/raw/` | Local source CSV |
| `models/` | Locally generated saved predictor |
| `reports/` | Locally generated evaluation outputs |

### Generated artifacts

| Local file | Contents |
|---|---|
| `models/aqi_model.joblib` | Selected learned predictor, fitted preprocessing and metadata |
| `reports/selection.json` | Frozen validation choice and configuration |
| `reports/model_comparison.csv` | Individual, voting and reference results |
| `reports/classification_report.csv` | Per-category precision, recall, F1 and support |
| `reports/confusion_matrix.csv` | Actual versus predicted categories |
| `reports/test_predictions.csv` | Frozen test row IDs, targets, predictions and scores |
| `reports/summary.json` | Data audit, partitions and selected-model metrics |

## Checks and reproducibility

```powershell
python -m pytest tests/test_project.py -q
```

The test suite checks category boundaries, invalid inputs, exclusion of target fields, preservation of valid peaks, station separation, rare-category split repair, score alignment and selection ordering. These checks do not fit classifiers.

Keep the exact source CSV and dependency versions for saved-model verification. Use the same scikit-learn version to load an artifact as was used to train it.

## Limitations

- **Rare categories have limited evidence.** The recorded training partition contains only one “Very Unhealthy” observation; the test partition contains only two. Aggregate accuracy does not establish reliability for every category.
- **The data are sparse snapshots.** Stations have only 1–3 eligible observations, rather than a sufficiently dense history for future forecasting. Some recorded timestamps are older than the main collection period.
- **The rule baseline is strong.** Categories are derived from pollutant indices, so an available-index rule can outperform learned models. This reference uses only the same four input pollutants and is not an official concentration-to-AQI calculator.
- **Evaluation is historical.** The dataset was inspected during earlier development and is not fresh external validation or evidence of live deployment performance.
- **Model scores are uncalibrated.** Predictions estimate a category from the supplied inputs and are not official monitoring measurements.

---

Built by [Kesigan](https://github.com/mkesigan) · [Repository](https://github.com/mkesigan/AQI-Prediction)
