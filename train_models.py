"""
train_models.py
Class-based pipeline: loads UCI Spambase, preprocesses, trains 5 classifiers,
evaluates them, saves models to model/ and the held-out test set to data/.

Run:  python train_models.py
      python train_models.py --seed 7 --test-size 0.25
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo


class SpamModelTrainer:
    """End-to-end training and evaluation for the Spambase dataset."""

    UCI_ID = 94
    TARGET = "is_spam"  # 1 = spam, 0 = not spam

    def __init__(self, seed=2026, test_size=0.20):
        self.seed = seed
        self.test_size = test_size

        base = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(base, "model")
        self.data_dir = os.path.join(base, "data")
        self.results_dir = os.path.join(base, "results")
        for d in (self.model_dir, self.data_dir, self.results_dir):
            os.makedirs(d, exist_ok=True)

        self.X = self.y = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.models = {}
        self.results = None

    # ------------------------------------------------------------ data
    def load_data(self):
        """Fetch Spambase from the UCI repository."""
        ds = fetch_ucirepo(id=self.UCI_ID)
        self.X = ds.data.features.copy()
        self.y = ds.data.targets.iloc[:, 0].astype(int).rename(self.TARGET)
        return self

    def clean(self):
        """Report missing values and drop exact duplicate rows."""
        print(f"Shape: {self.X.shape} | missing values: {int(self.X.isna().sum().sum())}")
        df = pd.concat([self.X, self.y], axis=1)
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Dropped {before - len(df)} duplicate rows -> {len(df)} remain")
        print("Class balance:\n", df[self.TARGET].value_counts(normalize=True).round(3).to_string())
        self.X, self.y = df.drop(columns=self.TARGET), df[self.TARGET]
        return self

    def split_and_save_test(self):
        """Stratified split; saves the test set (features + target) to data/."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size,
            stratify=self.y, random_state=self.seed,
        )
        test_df = self.X_test.copy()
        test_df[self.TARGET] = self.y_test.values
        path = os.path.join(self.data_dir, "test_data.csv")
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
        features, all of which are >= 0) then StandardScaler, when scale=True.
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
                ("clf", LogisticRegression(max_iter=2000, random_state=s))]),
            "Decision Tree": Pipeline([
                ("prep", self._preprocessor(False)),
                ("clf", DecisionTreeClassifier(random_state=s))]),
            "KNN": Pipeline([
                ("prep", self._preprocessor(True)),
                ("clf", KNeighborsClassifier(n_neighbors=5))]),
            "Naive Bayes": Pipeline([
                ("prep", self._preprocessor(False)),
                ("clf", GaussianNB())]),
            "Support Vector Machine": Pipeline([
                ("prep", self._preprocessor(True)),
                ("clf", CalibratedClassifierCV(
                    SVC(random_state=s),
                    ensemble=False))]),
        }
        return self

    # SVM hyper-parameter grid (searched on the training set only).
    # "estimator" is the SVC inside CalibratedClassifierCV (scikit-learn >= 1.2).
    PARAM_GRIDS = {
        "Support Vector Machine": {
            "clf__estimator__C": [0.1, 1, 10],
            "clf__estimator__gamma": ["scale", 0.01, 0.1],
        },
    }

    # ------------------------------------------------------ evaluation
    def evaluate(self, model):
        """Return the six required metrics on the test set."""
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

    def train_and_evaluate(self):
        """Train each model, print its metrics right away, and save it."""
        rows = []
        for name, pipe in self.models.items():
            print(f"\n=== {name} ===")
            if name in self.PARAM_GRIDS:
                search = GridSearchCV(pipe, self.PARAM_GRIDS[name], cv=3,
                                      scoring="f1", n_jobs=-1)
                search.fit(self.X_train, self.y_train)
                pipe = search.best_estimator_
                print(f"Best params: {search.best_params_}")
            else:
                pipe.fit(self.X_train, self.y_train)
            metrics = self.evaluate(pipe)
            for k, v in metrics.items():
                print(f"{k:<10}: {v:.4f}")
            joblib.dump(pipe, os.path.join(self.model_dir, self.filename(name)))
            rows.append({"ML Model Name": name, **metrics})
        self.results = pd.DataFrame(rows).round(4)
        return self

    def save_results(self):
        self.results.to_csv(os.path.join(self.results_dir, "metrics.csv"), index=False)
        print("\n================ COMPARISON TABLE ================")
        print(self.results.to_string(index=False))
        best = self.results.loc[self.results["MCC"].idxmax(), "ML Model Name"]
        print(f"\nBest by MCC: {best}")
        return self

    # ------------------------------------------------------------- run
    def run(self):
        (self.load_data().clean().split_and_save_test()
             .build_models().train_and_evaluate().save_results())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Spambase classifiers")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-size", type=float, default=0.20)
    args = parser.parse_args()
    SpamModelTrainer(seed=args.seed, test_size=args.test_size).run()