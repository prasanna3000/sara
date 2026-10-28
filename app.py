"""
app.py - SARA Spam Classifier Evaluation Dashboard

Loads five pre-trained Spambase classifiers from model/
and evaluates them against a held-out test CSV.

Nothing is retrained in this application.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "model"

SAMPLE_CSV = (
    BASE_DIR
    / "data"
    / "test_data.csv"
)

TARGET = "is_spam"

LABELS = [
    "Not spam",
    "Spam",
]

MODEL_FILES = {
    "Logistic Regression":
        "logistic_regression.joblib",

    "Decision Tree":
        "decision_tree.joblib",

    "KNN":
        "knn.joblib",

    "Naive Bayes":
        "naive_bayes.joblib",

    "Support Vector Machine":
        "support_vector_machine.joblib",
}


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SARA",
    page_icon=":material/mark_email_unread:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model(model_name):
    """
    Load a previously trained model.
    """

    model_path = (
        MODEL_DIR
        / MODEL_FILES[model_name]
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return joblib.load(model_path)


# ============================================================
# MODEL FEATURES
# ============================================================

def get_model_features(model):
    """
    Return the feature names stored by the trained model.
    """

    if not hasattr(
        model,
        "feature_names_in_",
    ):

        raise ValueError(
            "The saved model does not contain "
            "feature_names_in_."
        )

    return list(
        model.feature_names_in_
    )


# ============================================================
# MODEL SCORES
# ============================================================

def get_model_scores(model, X):
    """
    Get continuous prediction scores.

    predict_proba() is preferred.
    decision_function() is used as a fallback.
    """

    if hasattr(
        model,
        "predict_proba",
    ):

        probabilities = (
            model.predict_proba(X)
        )

        if probabilities.shape[1] != 2:

            raise ValueError(
                "The model does not provide "
                "two-class probabilities."
            )

        return probabilities[:, 1]

    if hasattr(
        model,
        "decision_function",
    ):

        return model.decision_function(X)

    raise ValueError(
        "The model provides neither "
        "predict_proba nor decision_function."
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
):
    """
    Evaluate one model.

    Returns:
        global_metrics
        class_metrics
        predictions
        scores
    """

    predictions = model.predict(X)

    scores = get_model_scores(
        model,
        X,
    )

    # --------------------------------------------------------
    # GLOBAL METRICS
    # --------------------------------------------------------

    global_metrics = {

        "Accuracy": accuracy_score(
            y,
            predictions,
        ),

        "AUC": (
            roc_auc_score(
                y,
                scores,
            )
            if y.nunique() == 2
            else float("nan")
        ),

        "MCC": matthews_corrcoef(
            y,
            predictions,
        ),
    }

    # --------------------------------------------------------
    # SPAM METRICS
    # --------------------------------------------------------

    spam_metrics = {

        "Precision": precision_score(
            y,
            predictions,
            pos_label=1,
            zero_division=0,
        ),

        "Recall": recall_score(
            y,
            predictions,
            pos_label=1,
            zero_division=0,
        ),

        "F1": f1_score(
            y,
            predictions,
            pos_label=1,
            zero_division=0,
        ),

        "Support": int(
            (y == 1).sum()
        ),
    }

    # --------------------------------------------------------
    # NOT SPAM METRICS
    # --------------------------------------------------------

    not_spam_metrics = {

        "Precision": precision_score(
            y,
            predictions,
            pos_label=0,
            zero_division=0,
        ),

        "Recall": recall_score(
            y,
            predictions,
            pos_label=0,
            zero_division=0,
        ),

        "F1": f1_score(
            y,
            predictions,
            pos_label=0,
            zero_division=0,
        ),

        "Support": int(
            (y == 0).sum()
        ),
    }

    class_metrics = {

        "Spam": spam_metrics,

        "Not spam": not_spam_metrics,
    }

    return (
        global_metrics,
        class_metrics,
        predictions,
        scores,
    )


# ============================================================
# EVALUATE ALL MODELS
# ============================================================

def evaluate_all_models(
    X,
    y,
):
    """
    Evaluate all five saved models on
    exactly the same test data.
    """

    rows = []

    for model_name in MODEL_FILES:

        try:

            model = load_model(
                model_name
            )

            metrics, _, _, _ = evaluate_model(
                model,
                X,
                y,
            )

            rows.append(
                {
                    "ML Model": model_name,

                    "Accuracy":
                        metrics["Accuracy"],

                    "AUC":
                        metrics["AUC"],

                    "MCC":
                        metrics["MCC"],
                }
            )

        except Exception as exc:

            st.warning(
                f"Skipped {model_name}: {exc}"
            )

    return pd.DataFrame(rows)


# ============================================================
# SARA APPLICATION
# ============================================================

class SARAApp:

    def __init__(self):

        self.df = None

        self.model = None

        self.model_name = None

        self.X = None

        self.y = None

        self.global_metrics = None

        self.class_metrics = None

        self.predictions = None

        self.scores = None

    # ========================================================
    # SIDEBAR
    # ========================================================

    def render_sidebar(self):

        with st.sidebar:

            st.title(
                ":material/mark_email_unread: SARA"
            )

            st.caption(
                "Spam Analysis & Risk Assistant"
            )

            st.divider()

            uploaded_file = st.file_uploader(
                "Upload test CSV",
                type=["csv"],
                help=(
                    "Upload the held-out test dataset "
                    "containing the 57 Spambase features "
                    "and the is_spam target column."
                ),
            )

            self.model_name = st.selectbox(
                "Choose a model",
                list(
                    MODEL_FILES.keys()
                ),
            )

            st.divider()

            if SAMPLE_CSV.exists():

                st.download_button(
                    label=(
                        ":material/download: "
                        "Download sample test data"
                    ),
                    data=SAMPLE_CSV.read_bytes(),
                    file_name="test_data.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            st.info(
                "The CSV must contain the "
                "57 feature columns and "
                "`is_spam` "
                "(1 = spam, 0 = not spam)."
            )

        return uploaded_file

    # ========================================================
    # LOAD TEST DATA
    # ========================================================

    def load_test_data(
        self,
        uploaded_file,
    ):

        # ----------------------------------------------------
        # READ CSV
        # ----------------------------------------------------

        try:

            self.df = pd.read_csv(
                uploaded_file
            )

        except Exception as exc:

            st.error(
                f"Could not read the CSV: {exc}"
            )

            return False

        # ----------------------------------------------------
        # TARGET CHECK
        # ----------------------------------------------------

        if TARGET not in self.df.columns:

            st.error(
                f"Target column `{TARGET}` "
                "is missing from the uploaded file."
            )

            return False

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        try:

            self.model = load_model(
                self.model_name
            )

        except Exception as exc:

            st.error(
                f"Could not load the selected model: "
                f"{exc}"
            )

            return False

        # ----------------------------------------------------
        # GET FEATURES
        # ----------------------------------------------------

        try:

            features = get_model_features(
                self.model
            )

        except Exception as exc:

            st.error(
                str(exc)
            )

            return False

        # ----------------------------------------------------
        # CHECK MISSING FEATURES
        # ----------------------------------------------------

        missing_features = [
            feature
            for feature in features
            if feature not in self.df.columns
        ]

        if missing_features:

            st.error(
                f"{len(missing_features)} required "
                "feature column(s) are missing."
            )

            st.write(
                "Missing columns:",
                missing_features,
            )

            return False

        # ----------------------------------------------------
        # CREATE X AND Y
        # ----------------------------------------------------

        try:

            self.X = self.df[
                features
            ]

            self.y = self.df[
                TARGET
            ].astype(int)

        except Exception as exc:

            st.error(
                f"Could not prepare the test data: "
                f"{exc}"
            )

            return False

        # ----------------------------------------------------
        # CHECK TARGET VALUES
        # ----------------------------------------------------

        unique_values = set(
            self.y.unique()
        )

        invalid_values = (
            unique_values - {0, 1}
        )

        if invalid_values:

            st.error(
                "`is_spam` must contain only "
                "0 and 1."
            )

            st.write(
                "Invalid values:",
                sorted(
                    invalid_values
                ),
            )

            return False

        # ----------------------------------------------------
        # EVALUATE SELECTED MODEL
        # ----------------------------------------------------

        try:

            (
                self.global_metrics,
                self.class_metrics,
                self.predictions,
                self.scores,
            ) = evaluate_model(
                self.model,
                self.X,
                self.y,
            )

        except Exception as exc:

            st.error(
                f"Could not evaluate the model: "
                f"{exc}"
            )

            return False

        return True

    # ========================================================
    # PAGE HEADER
    # ========================================================

    def render_header(self):

        st.title(
            ":material/mark_email_unread: SARA"
        )

        st.subheader(
            "Spam Analysis & Risk Assistant"
        )

        st.caption(
            "Machine-learning based evaluation "
            "of pre-trained spam classification models."
        )

    # ========================================================
    # DATA PREVIEW
    # ========================================================

    def render_data_preview(self):

        with st.expander(
            ":material/table_view: "
            f"Test Data Preview ({len(self.df)} rows)"
        ):

            st.dataframe(
                self.df.head(10),
                use_container_width=True,
                hide_index=True,
            )

            not_spam_count = int(
                (self.y == 0).sum()
            )

            spam_count = int(
                (self.y == 1).sum()
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Not Spam Samples",
                    f"{not_spam_count:,}",
                )

            with col2:

                st.metric(
                    "Spam Samples",
                    f"{spam_count:,}",
                )

    # ========================================================
    # TOP PERFORMING MODEL
    # ========================================================

    def render_top_model(
        self,
        comparison,
    ):

        if comparison.empty:
            return

        best_index = comparison[
            "MCC"
        ].idxmax()

        best_model = comparison.loc[
            best_index,
            "ML Model",
        ]

        best_mcc = comparison.loc[
            best_index,
            "MCC",
        ]

        if best_model == self.model_name:

            st.success(
                f":material/emoji_events: "
                f"Top Performing Model — "
                f"{best_model} "
                f"(MCC: {best_mcc:.4f})"
            )

        else:

            st.info(
                f":material/leaderboard: "
                f"Top Performing Model — "
                f"{best_model} "
                f"(MCC: {best_mcc:.4f})"
            )

    # ========================================================
    # GLOBAL RESULTS
    # ========================================================

    def render_global_results(self):

        with st.container(
            border=True
        ):

            st.subheader(
                ":material/analytics: "
                "Global Results"
            )

            st.caption(
                "Overall metrics for the selected model."
            )

            col1, col2, col3 = st.columns(3)

            # ------------------------------------------------
            # ACCURACY
            # ------------------------------------------------

            with col1:

                st.metric(
                    label="Accuracy",
                    value=(
                        f"{self.global_metrics['Accuracy']:.4f}"
                    ),
                    help=(
                        "Percentage of all predictions "
                        "that were correct."
                    ),
                )

            # ------------------------------------------------
            # AUC
            # ------------------------------------------------

            with col2:

                auc_value = (
                    self.global_metrics["AUC"]
                )

                if pd.isna(auc_value):

                    auc_display = "N/A"

                else:

                    auc_display = (
                        f"{auc_value:.4f}"
                    )

                st.metric(
                    label="AUC",
                    value=auc_display,
                    help=(
                        "Area Under the ROC Curve. "
                        "Measures how well the model "
                        "separates spam from not spam."
                    ),
                )

            # ------------------------------------------------
            # MCC
            # ------------------------------------------------

            with col3:

                st.metric(
                    label="MCC",
                    value=(
                        f"{self.global_metrics['MCC']:.4f}"
                    ),
                    help=(
                        "Matthews Correlation Coefficient. "
                        "A balanced binary classification "
                        "performance measure."
                    ),
                )

    # ========================================================
    # CLASS METRICS
    # ========================================================

    def render_class_results(self):

        with st.container(
            border=True
        ):

            st.subheader(
                ":material/category: "
                "Class Metrics"
            )

            st.caption(
                "Metrics calculated separately "
                "for each class."
            )

            spam_tab, not_spam_tab = st.tabs(
                [
                    ":material/report: Spam",
                    ":material/inbox: Not Spam",
                ]
            )

            # =================================================
            # SPAM TAB
            # =================================================

            with spam_tab:

                metrics = (
                    self.class_metrics[
                        "Spam"
                    ]
                )

                col1, col2, col3, col4 = (
                    st.columns(4)
                )

                with col1:

                    st.metric(
                        label="Precision",
                        value=(
                            f"{metrics['Precision']:.4f}"
                        ),
                        help=(
                            "Of all emails predicted "
                            "as spam, how many were "
                            "actually spam."
                        ),
                    )

                with col2:

                    st.metric(
                        label="Recall",
                        value=(
                            f"{metrics['Recall']:.4f}"
                        ),
                        help=(
                            "Of all actual spam emails, "
                            "how many were correctly "
                            "identified."
                        ),
                    )

                with col3:

                    st.metric(
                        label="F1",
                        value=(
                            f"{metrics['F1']:.4f}"
                        ),
                        help=(
                            "Harmonic mean of precision "
                            "and recall for spam."
                        ),
                    )

                with col4:

                    st.metric(
                        label="Support",
                        value=(
                            f"{metrics['Support']:,}"
                        ),
                        help=(
                            "Number of actual spam "
                            "samples."
                        ),
                    )

            # =================================================
            # NOT SPAM TAB
            # =================================================

            with not_spam_tab:

                metrics = (
                    self.class_metrics[
                        "Not spam"
                    ]
                )

                col1, col2, col3, col4 = (
                    st.columns(4)
                )

                with col1:

                    st.metric(
                        label="Precision",
                        value=(
                            f"{metrics['Precision']:.4f}"
                        ),
                        help=(
                            "Of all emails predicted "
                            "as not spam, how many were "
                            "actually not spam."
                        ),
                    )

                with col2:

                    st.metric(
                        label="Recall",
                        value=(
                            f"{metrics['Recall']:.4f}"
                        ),
                        help=(
                            "Of all actual not-spam emails, "
                            "how many were correctly "
                            "identified."
                        ),
                    )

                with col3:

                    st.metric(
                        label="F1",
                        value=(
                            f"{metrics['F1']:.4f}"
                        ),
                        help=(
                            "Harmonic mean of precision "
                            "and recall for not spam."
                        ),
                    )

                with col4:

                    st.metric(
                        label="Support",
                        value=(
                            f"{metrics['Support']:,}"
                        ),
                        help=(
                            "Number of actual not-spam "
                            "samples."
                        ),
                    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    def create_confusion_matrix(self):

        fig, ax = plt.subplots(
            figsize=(6, 4.5)
        )

        ConfusionMatrixDisplay.from_predictions(
            self.y,
            self.predictions,
            display_labels=LABELS,
            cmap="Blues",
            colorbar=False,
            values_format="d",
            ax=ax,
        )

        ax.set_title(
            "Confusion Matrix"
        )

        ax.set_xlabel(
            "Predicted Label"
        )

        ax.set_ylabel(
            "Actual Label"
        )

        fig.tight_layout()

        return fig

    # ========================================================
    # ROC CURVE
    # ========================================================

    def create_roc_curve(self):

        fig, ax = plt.subplots(
            figsize=(6, 4.5)
        )

        RocCurveDisplay.from_estimator(
            self.model,
            self.X,
            self.y,
            name=self.model_name,
            ax=ax,
        )

        ax.plot(
            [0, 1],
            [0, 1],
            "k--",
            linewidth=0.8,
            label="Random classifier",
        )

        ax.set_title(
            "ROC Curve"
        )

        ax.set_xlabel(
            "False Positive Rate"
        )

        ax.set_ylabel(
            "True Positive Rate"
        )

        ax.legend(
            loc="lower right"
        )

        fig.tight_layout()

        return fig

    # ========================================================
    # MODEL DIAGNOSTICS
    # ========================================================

    def render_diagnostics(self):

        with st.container(
            border=True
        ):

            st.subheader(
                ":material/monitoring: "
                "Model Diagnostics"
            )

            st.caption(
                "Visual evaluation of the selected model."
            )

            left, right = st.columns(2)

            # ------------------------------------------------
            # CONFUSION MATRIX
            # ------------------------------------------------

            with left:

                st.write(
                    ":material/grid_view: "
                    "Confusion Matrix"
                )

                fig = (
                    self.create_confusion_matrix()
                )

                st.pyplot(
                    fig,
                    use_container_width=True,
                )

                plt.close(fig)

            # ------------------------------------------------
            # ROC CURVE
            # ------------------------------------------------

            with right:

                st.write(
                    ":material/show_chart: "
                    "ROC Curve"
                )

                if self.y.nunique() == 2:

                    fig = (
                        self.create_roc_curve()
                    )

                    st.pyplot(
                        fig,
                        use_container_width=True,
                    )

                    plt.close(fig)

                else:

                    st.info(
                        "The ROC curve requires "
                        "both classes to be present "
                        "in the test dataset."
                    )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    def render_model_comparison(
        self,
        comparison,
    ):

        with st.container(
            border=True
        ):

            st.subheader(
                ":material/leaderboard: "
                "Model Comparison"
            )

            st.caption(
                "All saved models evaluated "
                "on the same test dataset."
            )

            if comparison.empty:

                st.warning(
                    "No models could be evaluated."
                )

                return

            display_df = (
                comparison.copy()
            )

            display_df[
                "Accuracy"
            ] = display_df[
                "Accuracy"
            ].round(4)

            display_df[
                "AUC"
            ] = display_df[
                "AUC"
            ].round(4)

            display_df[
                "MCC"
            ] = display_df[
                "MCC"
            ].round(4)

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

            best_index = comparison[
                "MCC"
            ].idxmax()

            best_model = comparison.loc[
                best_index,
                "ML Model",
            ]

            best_mcc = comparison.loc[
                best_index,
                "MCC",
            ]

            st.success(
                f":material/emoji_events: "
                f"Best MCC: {best_model} "
                f"({best_mcc:.4f})"
            )

    # ========================================================
    # MAIN FLOW
    # ========================================================

    def run(self):

        # ----------------------------------------------------
        # SIDEBAR
        # ----------------------------------------------------

        uploaded_file = (
            self.render_sidebar()
        )

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        self.render_header()

        # ----------------------------------------------------
        # WAIT FOR USER DATA
        # ----------------------------------------------------

        if uploaded_file is None:

            st.info(
                ":material/upload_file: "
                "Upload the held-out test CSV "
                "from the sidebar to begin."
            )

            return

        # ----------------------------------------------------
        # LOAD DATA
        # ----------------------------------------------------

        if not self.load_test_data(
            uploaded_file
        ):

            return

        # ----------------------------------------------------
        # EVALUATE ALL MODELS
        # ----------------------------------------------------

        comparison = (
            evaluate_all_models(
                self.X,
                self.y,
            )
        )

        # ----------------------------------------------------
        # TOP MODEL
        # ----------------------------------------------------

        self.render_top_model(
            comparison
        )

        # ----------------------------------------------------
        # SELECTED MODEL
        # ----------------------------------------------------

        st.header(
            ":material/model_training: "
            f"Results: {self.model_name}"
        )

        # ----------------------------------------------------
        # DATA PREVIEW
        # ----------------------------------------------------

        self.render_data_preview()

        # ----------------------------------------------------
        # GLOBAL METRICS
        # ----------------------------------------------------

        self.render_global_results()

        # ----------------------------------------------------
        # CLASS METRICS
        # ----------------------------------------------------

        self.render_class_results()

        # ----------------------------------------------------
        # DIAGNOSTICS
        # ----------------------------------------------------

        self.render_diagnostics()

        # ----------------------------------------------------
        # MODEL COMPARISON
        # ----------------------------------------------------

        self.render_model_comparison(
            comparison
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    app = SARAApp()

    app.run()