from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_pipeline(classifier, scale=False):
    steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True,
                                       keep_empty_features=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def create_models(seed=42):
    """Return fresh, unfitted models. Importing this module does not train."""
    return {
        "LogisticRegression": make_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=2500,
                               random_state=seed), scale=True),
        "RandomForest": make_pipeline(
            RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                   max_features=0.8, class_weight="balanced_subsample",
                                   random_state=seed, n_jobs=2)),
        "GradientBoosting": make_pipeline(
            GradientBoostingClassifier(n_estimators=150, learning_rate=0.05,
                                       max_depth=2, random_state=seed)),
    }


# Relative weights: Logistic Regression, Random Forest, Gradient Boosting.
VOTING_WEIGHTS = {
    "VotingEqual": [1, 1, 1],
    "VotingForest": [1, 2, 1],
    "VotingBoosting": [1, 1, 2],
    "VotingLogistic": [2, 1, 1],
}
