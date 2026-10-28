"""
train_models.py
Class-based pipeline for the UCI Spambase dataset.

  1. loads and cleans the data, makes a stratified train/test split
  2. builds five scikit-learn Pipelines (preprocessing + classifier)
  3. tunes EVERY model with the same protocol: stratified k-fold
     GridSearchCV on the training set only, selecting on MCC
  4. evaluates each tuned model on the held-out test set
  5. saves models (model/), test data (test_data.csv at the project root)
     and result tables (results/)

Run:  python train_models.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, make_scorer, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo

# ----------------------------------------------------------------------
# SETTINGS - edit these here; no command-line arguments are needed.
# ----------------------------------------------------------------------
SEED = 7319        # controls the train/test split, CV folds and model randomness
TEST_SIZE = 0.20   # fraction of the data held out for testing
CV_FOLDS = 5       # folds used when tuning on the training set


class SpamModelTrainer:
    """End-to-end training, tuning and evaluation for Spambase."""

    UCI_ID = 94
    TARGET = "is_spam"  # 1 = spam, 0 = not spam

    # Metrics recorded for every candidate during cross-validation.
    SCORING = {
        "accuracy": "accuracy",
        "auc": "roc_auc",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "mcc": make_scorer(matthews_corrcoef),
    }

    # Hyper-parameter grids (searched on the training set only).
    # For the SVM, "estimator" is the SVC inside CalibratedClassifierCV
    # (scikit-learn >= 1.2).
    PARAM_GRIDS = {
        "Logistic Regression": {
            "clf__C": [0.01, 0.1, 1, 10, 100],
        },
        "Decision Tree": {
            "clf__max_depth": [5, 10, 20, None],
            "clf__min_samples_leaf": [1, 3, 5],
        },
        "KNN": {
            "clf__n_neighbors": [3, 5, 7, 11, 15],
            "clf__weights": ["uniform", "distance"],
        },
        "Naive Bayes": {
            "clf__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
        },
        "Support Vector Machine": {
            "clf__estimator__C": [1, 10, 100],
            "clf__estimator__gamma": [0.001, 0.01, 0.1],
        },
    }

    def __init__(self, seed=SEED, test_size=TEST_SIZE, cv_folds=CV_FOLDS):
        self.seed = seed
        self.test_size = test_size
        self.cv_folds = cv_folds

        base = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(base, "model")
        self.base_dir = base
        self.results_dir = os.path.join(base, "results")
        for d in (self.model_dir, self.results_dir):
            os.makedirs(d, exist_ok=True)

        self.X = self.y = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.models = {}
        self.best_params = {}
        self.results = None      # test-set metrics
        self.cv_results = None   # cross-validation metrics (training set)

    # ------------------------------------------------------------ data
    def load_data(self):
        """Fetch Spambase from the UCI repository."""
        ds = fetch_ucirepo(id=self.UCI_ID)
        self.X = ds.data.features.copy()
        self.y = ds.data.targets.iloc[:, 0].astype(int).rename(self.TARGET)
        return self

    def clean(self):
        """Report missing values and drop exact duplicate rows."""
        print(f"scikit-learn version: {sklearn.__version__}")
        print(f"Shape: {self.X.shape} | missing values: {int(self.X.isna().sum().sum())}")
        df = pd.concat([self.X, self.y], axis=1)
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Dropped {before - len(df)} duplicate rows -> {len(df)} remain")
        print("Class balance:\n", df[self.TARGET].value_counts(normalize=True).round(3).to_string())
        self.X, self.y = df.drop(columns=self.TARGET), df[self.TARGET]
        return self

    def split_and_save_test(self):
        """Stratified split; saves the test set (features + target) to the project root."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size,
            stratify=self.y, random_state=self.seed,
        )
        test_df = self.X_test.copy()
        test_df[self.TARGET] = self.y_test.values
        path = os.path.join(self.base_dir, "test_data.csv")
        test_df.to_csv(path, index=False)
        print(f"\nTrain: {self.X_train.shape} | Test: {self.X_test.shape} -> saved {path}")
        self.report_split()
        return self

    def report_split(self):
        """Print split percentages and class distribution after stratification."""
        total = len(self.y)
        rows = []
        for label, part in (("Full data", self.y), ("Train", self.y_train), ("Test", self.y_test)):
            n = len(part)
            ham, spam = int((part == 0).sum()), int((part == 1).sum())
            rows.append({
                "Set": label,
                "Rows": n,
                "% of data": round(100 * n / total, 2),
                "Not spam (0)": ham,
                "Spam (1)": spam,
                "% Not spam": round(100 * ham / n, 2),
                "% Spam": round(100 * spam / n, 2),
            })
        summary = pd.DataFrame(rows)
        print(f"\nStratified split (stratify=y, test_size={self.test_size}, seed={self.seed})")
        print(summary.to_string(index=False))
        summary.to_csv(os.path.join(self.results_dir, "split_summary.csv"), index=False)

    # --------------------------------------------------- preprocessing
    def _preprocessor(self, scale):
        """
        Numeric -> log1p (tames the skewed capital_run_length / word-frequency
        features, all >= 0) then StandardScaler, when scale=True.
        Categorical -> OneHotEncoder. Spambase is fully numeric, so the
        one-hot branch is empty here but keeps the pipeline reusable.
        """
        cat = self.X.select_dtypes(include=["object", "category"]).columns.tolist()
        num = [c for c in self.X.columns if c not in cat]
        if scale:
            num_tf = Pipeline([
                ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ])
        else:
            num_tf = "passthrough"
        return ColumnTransformer([
            ("num", num_tf, num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ])

    def build_models(self):
        """Create the 5 pipelines (scaling for LR, KNN, SVM)."""
        s = self.seed
        self.models = {
            "Logistic Regression": Pipeline([
                ("prep", self._preprocessor(True)),
                ("clf", LogisticRegression(max_iter=3000, random_state=s))]),
            "Decision Tree": Pipeline([
                ("prep", self._preprocessor(False)),
                ("clf", DecisionTreeClassifier(random_state=s))]),
            "KNN": Pipeline([
                ("prep", self._preprocessor(True)),
                ("clf", KNeighborsClassifier())]),
            "Naive Bayes": Pipeline([
                ("prep", self._preprocessor(False)),
                ("clf", GaussianNB())]),
            "Support Vector Machine": Pipeline([
                ("prep", self._preprocessor(True)),
                ("clf", CalibratedClassifierCV(SVC(random_state=s), ensemble=False))]),
        }
        return self

    # ------------------------------------------------------ evaluation
    def evaluate(self, model):
        """Return the six required metrics on the held-out test set."""
        pred = model.predict(self.X_test)
        score = model.predict_proba(self.X_test)[:, 1]
        return {
            "Accuracy": accuracy_score(self.y_test, pred),
            "AUC": roc_auc_score(self.y_test, score),
            "Precision": precision_score(self.y_test, pred),
            "Recall": recall_score(self.y_test, pred),
            "F1": f1_score(self.y_test, pred),
            "MCC": matthews_corrcoef(self.y_test, pred),
        }

    @staticmethod
    def filename(name):
        return name.lower().replace(" ", "_") + ".joblib"

    def tune(self, name, pipe):
        """Grid search with stratified k-fold CV on the training set, selecting on MCC."""
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        search = GridSearchCV(
            pipe, self.PARAM_GRIDS[name], scoring=self.SCORING,
            refit="mcc", cv=cv, n_jobs=-1,
        )
        search.fit(self.X_train, self.y_train)
        i = search.best_index_
        cv_row = {f"CV {m.upper() if m in ('auc', 'mcc') else m.capitalize()}":
                  search.cv_results_[f"mean_test_{m}"][i] for m in self.SCORING}
        cv_row["CV MCC std"] = search.cv_results_["std_test_mcc"][i]
        return search.best_estimator_, search.best_params_, cv_row

    def train_and_evaluate(self):
        """Tune each model, evaluate it on the test set, print and save it."""
        test_rows, cv_rows = [], []
        for name, pipe in self.models.items():
            print(f"\n=== {name} ===")
            best, params, cv_row = self.tune(name, pipe)
            self.best_params[name] = params
            print(f"Best params : {params}")
            print(f"CV MCC      : {cv_row['CV MCC']:.4f} (+/- {cv_row['CV MCC std']:.4f}, "
                  f"{self.cv_folds}-fold on training set)")
            metrics = self.evaluate(best)
            print("Test-set metrics:")
            for k, v in metrics.items():
                print(f"  {k:<10}: {v:.4f}")
            joblib.dump(best, os.path.join(self.model_dir, self.filename(name)))
            test_rows.append({"ML Model Name": name, **metrics})
            cv_rows.append({"ML Model Name": name, **cv_row})
        self.results = pd.DataFrame(test_rows).round(4)
        self.cv_results = pd.DataFrame(cv_rows).round(4)
        return self

    def save_results(self):
        self.results.to_csv(os.path.join(self.results_dir, "metrics.csv"), index=False)
        self.cv_results.to_csv(os.path.join(self.results_dir, "cv_metrics.csv"), index=False)
        with open(os.path.join(self.results_dir, "best_params.json"), "w") as f:
            json.dump({"seed": self.seed, "sklearn": sklearn.__version__,
                       "best_params": self.best_params}, f, indent=2, default=str)
        print("\n================ TEST-SET COMPARISON TABLE ================")
        print(self.results.to_string(index=False))
        print("\n================ CROSS-VALIDATION (TRAINING SET) ==========")
        print(self.cv_results.to_string(index=False))
        best = self.results.loc[self.results["MCC"].idxmax(), "ML Model Name"]
        print(f"\nBest on test set by MCC: {best}")
        return self

    # ------------------------------------------------------------- run
    def run(self):
        (self.load_data().clean().split_and_save_test()
             .build_models().train_and_evaluate().save_results())


if __name__ == "__main__":
    SpamModelTrainer().run()