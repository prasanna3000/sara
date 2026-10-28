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
DATA_DIR = BASE_DIR / "data"

TARGET = "is_spam"

LABELS = [
    "Not Spam",
    "Spam",
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


# ============================================================
# PAGE CONFIGURATION
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

    model_path = MODEL_DIR / MODEL_FILES[model_name]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return joblib.load(model_path)


# ============================================================
# MODEL FEATURES
# ============================================================

def get_model_features(model):

    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "The saved model does not contain "
            "feature_names_in_."
        )

    return list(model.feature_names_in_)


# ============================================================
# MODEL SCORES
# ============================================================

def get_model_scores(model, X):

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X)

        if probabilities.shape[1] != 2:
            raise ValueError(
                "The model does not provide "
                "two-class probabilities."
            )

        return probabilities[:, 1]

    if hasattr(model, "decision_function"):

        return model.decision_function(X)

    raise ValueError(
        "The model provides neither "
        "predict_proba nor decision_function."
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model, X, y):

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
        "Not Spam": not_spam_metrics,
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

def evaluate_all_models(X, y):

    rows = []

    for model_name in MODEL_FILES:

        try:

            model = load_model(
                model_name
            )

            (
                metrics,
                _,
                _,
                _,
            ) = evaluate_model(
                model,
                X,
                y,
            )

            rows.append(
                {
                    "ML Model Name": model_name,
                    "Accuracy": metrics["Accuracy"],
                    "AUC": metrics["AUC"],
                    "MCC": metrics["MCC"],
                }
            )

        except Exception as exc:

            st.warning(
                f"Could not evaluate "
                f"{model_name}: {exc}"
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
                "Test CSV",
                type=["csv"],
                help=(
                    "Upload the test dataset containing "
                    "the required model features and "
                    "the is_spam target column."
                ),
            )

            self.model_name = st.selectbox(
                "Select Model",
                list(MODEL_FILES.keys()),
            )

            st.divider()

            st.caption(
                ":material/info: is_spam: "
                "0 = Not Spam • 1 = Spam"
            )

            st.caption(
                "SARA • Spam Analysis & Risk Assistant"
            )

        return uploaded_file
    # ========================================================
    # DASHBOARD STEPPER
    # ========================================================

    def render_dashboard_stepper(
        self,
        current_step,
    ):

        st.subheader(
            ":material/dashboard: SARA Dashboard"
        )

        step1, step2, step3 = st.columns(3)

        # ----------------------------------------------------
        # STEP 1
        # ----------------------------------------------------

        with step1:

            with st.container(
                border=True
            ):

                if current_step > 1:

                    st.success(
                        ":material/check_circle: "
                        "Step 1"
                    )

                elif current_step == 1:

                    st.info(
                        ":material/radio_button_checked: "
                        "Step 1"
                    )

                else:

                    st.write(
                        ":material/radio_button_unchecked: "
                        "Step 1"
                    )

                st.write(
                    "Upload Test Data"
                )

                st.caption(
                    "Upload the CSV containing "
                    "the test data."
                )

        # ----------------------------------------------------
        # STEP 2
        # ----------------------------------------------------

        with step2:

            with st.container(
                border=True
            ):

                if current_step > 2:

                    st.success(
                        ":material/check_circle: "
                        "Step 2"
                    )

                elif current_step == 2:

                    st.info(
                        ":material/radio_button_checked: "
                        "Step 2"
                    )

                else:

                    st.write(
                        ":material/radio_button_unchecked: "
                        "Step 2"
                    )

                st.write(
                    "Select Model"
                )

                st.caption(
                    "Choose one of the five "
                    "machine learning models."
                )

        # ----------------------------------------------------
        # STEP 3
        # ----------------------------------------------------

        with step3:

            with st.container(
                border=True
            ):

                if current_step >= 3:

                    st.success(
                        ":material/check_circle: "
                        "Step 3"
                    )

                else:

                    st.write(
                        ":material/radio_button_unchecked: "
                        "Step 3"
                    )

                st.write(
                    "View Results"
                )

                st.caption(
                    "Review metrics, class performance "
                    "and visual analytics."
                )

        st.divider()

    # ========================================================
    # HEADER
    # ========================================================

    def render_header(self):

        title_col, model_col = st.columns(
            [3, 1]
        )

        with title_col:

            st.title(
                ":material/mark_email_unread: SARA"
            )

            st.caption(
                "Spam Analysis & Risk Assistant"
            )

        with model_col:

            st.metric(
                "Selected Model",
                self.model_name,
            )

        st.divider()

    # ========================================================
    # LOAD TEST DATA
    # ========================================================

    def load_test_data(
        self,
        uploaded_file,
    ):

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
        # CHECK FEATURES
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

            with st.expander(
                "Show missing columns"
            ):

                st.write(
                    missing_features
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
        # TARGET VALIDATION
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
            "ML Model Name",
        ]

        best_mcc = comparison.loc[
            best_index,
            "MCC",
        ]

        if best_model == self.model_name:

            st.success(
                f":material/emoji_events: "
                f"Top Performing Model — "
                f"{best_model} | "
                f"MCC: {best_mcc:.4f}"
            )

        else:

            st.info(
                f":material/leaderboard: "
                f"Top Performing Model — "
                f"{best_model} | "
                f"MCC: {best_mcc:.4f}"
            )

    # ========================================================
    # GLOBAL METRICS
    # ========================================================

    def render_global_metrics(self):

        st.subheader(
            ":material/analytics: "
            "Global Performance"
        )

        st.caption(
            "Overall performance of the selected model."
        )

        col1, col2, col3 = st.columns(3)

        # ----------------------------------------------------
        # ACCURACY
        # ----------------------------------------------------

        with col1:

            st.metric(
                label="Accuracy",
                value=(
                    f"{self.global_metrics['Accuracy'] * 100:.2f}%"
                ),
                help=(
                    "Percentage of all predictions "
                    "that were correct."
                ),
            )

        # ----------------------------------------------------
        # AUC
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # MCC
        # ----------------------------------------------------

        with col3:

            st.metric(
                label="MCC",
                value=(
                    f"{self.global_metrics['MCC']:.4f}"
                ),
                help=(
                    "Matthews Correlation Coefficient. "
                    "A balanced binary classification "
                    "metric ranging from -1 to +1."
                ),
            )

    # ========================================================
    # CLASS METRICS
    # ========================================================

    def render_class_metrics(self):

        st.subheader(
            ":material/category: "
            "Class Performance"
        )

        st.caption(
            "Detailed performance for each class."
        )

        spam_tab, not_spam_tab = st.tabs(
            [
                ":material/report: Spam",
                ":material/inbox: Not Spam",
            ]
        )

        # ----------------------------------------------------
        # SPAM
        # ----------------------------------------------------

        with spam_tab:

            metrics = (
                self.class_metrics["Spam"]
            )

            col1, col2, col3, col4 = (
                st.columns(4)
            )

            with col1:

                st.metric(
                    "Precision",
                    f"{metrics['Precision']:.4f}",
                )

            with col2:

                st.metric(
                    "Recall",
                    f"{metrics['Recall']:.4f}",
                )

            with col3:

                st.metric(
                    "F1",
                    f"{metrics['F1']:.4f}",
                )

            with col4:

                st.metric(
                    "Support",
                    f"{metrics['Support']:,}",
                )

        # ----------------------------------------------------
        # NOT SPAM
        # ----------------------------------------------------

        with not_spam_tab:

            metrics = (
                self.class_metrics["Not Spam"]
            )

            col1, col2, col3, col4 = (
                st.columns(4)
            )

            with col1:

                st.metric(
                    "Precision",
                    f"{metrics['Precision']:.4f}",
                )

            with col2:

                st.metric(
                    "Recall",
                    f"{metrics['Recall']:.4f}",
                )

            with col3:

                st.metric(
                    "F1",
                    f"{metrics['F1']:.4f}",
                )

            with col4:

                st.metric(
                    "Support",
                    f"{metrics['Support']:,}",
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

        RocCurveDisplay.from_predictions(
            self.y,
            self.scores,
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
    # VISUAL ANALYTICS
    # ========================================================

    def render_visualizations(self):

        st.subheader(
            ":material/monitoring: "
            "Visual Analytics"
        )

        left, right = st.columns(2)

        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        with left:

            with st.container(
                border=True
            ):

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

        # ----------------------------------------------------
        # ROC CURVE
        # ----------------------------------------------------

        with right:

            with st.container(
                border=True
            ):

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
                        "ROC curve requires both "
                        "classes to be present."
                    )

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    def render_dataset_summary(self):

        with st.expander(
            ":material/database: "
            "Dataset Information"
        ):

            col1, col2, col3, col4 = (
                st.columns(4)
            )

            with col1:

                st.metric(
                    "Test Samples",
                    f"{len(self.df):,}",
                )

            with col2:

                st.metric(
                    "Features",
                    f"{self.X.shape[1]:,}",
                )

            with col3:

                st.metric(
                    "Not Spam",
                    f"{(self.y == 0).sum():,}",
                )

            with col4:

                st.metric(
                    "Spam",
                    f"{(self.y == 1).sum():,}",
                )

            st.divider()

            col1, col2 = st.columns(2)

            with col1:

                st.caption(
                    "Class Distribution"
                )

                distribution = pd.DataFrame(
                    {
                        "Class": [
                            "Not Spam",
                            "Spam",
                        ],
                        "Count": [
                            int(
                                (self.y == 0).sum()
                            ),
                            int(
                                (self.y == 1).sum()
                            ),
                        ],
                    }
                )

                st.dataframe(
                    distribution,
                    use_container_width=True,
                    hide_index=True,
                )

            with col2:

                st.caption(
                    "Test Data Preview"
                )

                st.dataframe(
                    self.df.head(5),
                    use_container_width=True,
                    hide_index=True,
                )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    def render_model_comparison(
        self,
        comparison,
    ):

        st.subheader(
            ":material/leaderboard: "
            "Model Comparison"
        )

        st.caption(
            "Performance of all available models "
            "on the same test dataset."
        )

        if comparison.empty:

            st.warning(
                "No models could be evaluated."
            )

            return

        display_df = comparison.copy()

        display_df["Accuracy"] = (
            display_df["Accuracy"] * 100
        ).round(2)

        display_df["AUC"] = (
            display_df["AUC"]
        ).round(4)

        display_df["MCC"] = (
            display_df["MCC"]
        ).round(4)

        display_df = display_df.rename(
            columns={
                "Accuracy": "Accuracy (%)",
            }
        )

        best_index = comparison[
            "MCC"
        ].idxmax()

        best_model = comparison.loc[
            best_index,
            "ML Model Name",
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ML Model Name": st.column_config.TextColumn(
                    "ML Model",
                    width="large",
                ),
                "Accuracy (%)": st.column_config.NumberColumn(
                    "Accuracy",
                    format="%.2f%%",
                ),
                "AUC": st.column_config.NumberColumn(
                    "AUC",
                    format="%.4f",
                ),
                "MCC": st.column_config.NumberColumn(
                    "MCC",
                    format="%.4f",
                ),
            },
        )

        st.success(
            f":material/emoji_events: "
            f"Best Performing Model: "
            f"{best_model}"
        )

    # ========================================================
    # SELECTED MODEL SUMMARY
    # ========================================================

    def render_selected_model_summary(self):

        spam = self.class_metrics[
            "Spam"
        ]

        not_spam = self.class_metrics[
            "Not Spam"
        ]

        with st.expander(
            ":material/description: "
            "Selected Model Summary"
        ):

            st.write(
                f"**Model:** {self.model_name}"
            )

            st.write(
                "The selected model was evaluated "
                "against the uploaded test dataset."
            )

            st.divider()

            st.write(
                "**Spam Performance**"
            )

            st.write(
                f"Precision: {spam['Precision']:.4f}"
            )

            st.write(
                f"Recall: {spam['Recall']:.4f}"
            )

            st.write(
                f"F1: {spam['F1']:.4f}"
            )

            st.divider()

            st.write(
                "**Not Spam Performance**"
            )

            st.write(
                f"Precision: "
                f"{not_spam['Precision']:.4f}"
            )

            st.write(
                f"Recall: "
                f"{not_spam['Recall']:.4f}"
            )

            st.write(
                f"F1: "
                f"{not_spam['F1']:.4f}"
            )

    # ========================================================
    # RUN
    # ========================================================

    def run(self):

        # ----------------------------------------------------
        # SIDEBAR
        # ----------------------------------------------------

        uploaded_file = self.render_sidebar()

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        self.render_header()

        # ----------------------------------------------------
        # INITIAL DASHBOARD
        # ----------------------------------------------------

        if uploaded_file is None:

            self.render_dashboard_stepper(
                current_step=1
            )

            st.info(
                ":material/upload_file: "
                "Upload the test CSV from the sidebar "
                "to begin the analysis."
            )

            return

        # ----------------------------------------------------
        # LOAD DATA AND EVALUATE SELECTED MODEL
        # ----------------------------------------------------

        if not self.load_test_data(
            uploaded_file
        ):
            return

        # ----------------------------------------------------
        # EVALUATE ALL MODELS
        # ----------------------------------------------------

        comparison = evaluate_all_models(
            self.X,
            self.y,
        )

        # ----------------------------------------------------
        # TOP MODEL
        # ----------------------------------------------------

        self.render_top_model(
            comparison
        )

        st.divider()

        # ----------------------------------------------------
        # SELECTED MODEL
        # ----------------------------------------------------

        st.header(
            ":material/model_training: "
            f"{self.model_name} Results"
        )

        # ----------------------------------------------------
        # DATASET SUMMARY
        # ----------------------------------------------------

        self.render_dataset_summary()

        # ----------------------------------------------------
        # GLOBAL METRICS
        # ----------------------------------------------------

        self.render_global_metrics()

        st.divider()

        # ----------------------------------------------------
        # CLASS METRICS
        # ----------------------------------------------------

        self.render_class_metrics()

        st.divider()

        # ----------------------------------------------------
        # VISUAL ANALYTICS
        # ----------------------------------------------------

        self.render_visualizations()

        st.divider()

        # ----------------------------------------------------
        # MODEL COMPARISON
        # ----------------------------------------------------

        self.render_model_comparison(
            comparison
        )

        # ----------------------------------------------------
        # MODEL SUMMARY
        # ----------------------------------------------------

        self.render_selected_model_summary()

        # ----------------------------------------------------
        # FOOTER
        # ----------------------------------------------------

        st.divider()

        st.caption(
            "SARA — Spam Analysis & Risk Assistant"
        )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    SARAApp().run()