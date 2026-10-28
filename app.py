"""
SARA - Spam Analysis & Risk Assistant (Streamlit).

Loads the five saved scikit-learn pipelines from model/ and evaluates them on a
test CSV uploaded by the user. Nothing is retrained here.
"""

import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"

TARGET = "is_spam"
LABELS = ["Not Spam", "Spam"]
METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

# One fixed colour per model, shared by every chart (matches chartCategoricalColors
# in .streamlit/config.toml).
MODEL_COLORS = {
    "Logistic Regression": "#0473ea",
    "Decision Tree": "#f59e0b",
    "KNN": "#7c3aed",
    "Naive Bayes": "#e63b44",
    "Support Vector Machine": "#0f9d74",
}
TEXT_COLOR = "#0f2137"
GRID_COLOR = "#dce4f0"

METRIC_HELP = {
    "Accuracy": "Share of all emails classified correctly.",
    "AUC": "How well the model ranks spam above non-spam (1.0 is perfect).",
    "Precision": "Of the emails flagged as spam, the share that really are spam.",
    "Recall": "Of the real spam emails, the share that were caught.",
    "F1": "Harmonic mean of precision and recall.",
    "MCC": "Balanced score using all four confusion-matrix cells (-1 to 1).",
}

st.set_page_config(
    page_title="SARA",
    page_icon=":material/mark_email_unread:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# MODEL LOADING AND EVALUATION
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model(model_name):
    path = MODEL_DIR / MODEL_FILES[model_name]
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path.name}")
    return joblib.load(path)


@st.cache_data(show_spinner="Evaluating all five models...")
def evaluate_all(file_bytes):
    """Validate the uploaded CSV and evaluate every saved model on it."""
    df = pd.read_csv(io.BytesIO(file_bytes))

    if TARGET not in df.columns:
        raise ValueError(f"Target column `{TARGET}` is missing from the uploaded file.")

    features = list(load_model(next(iter(MODEL_FILES))).feature_names_in_)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} required feature column(s) are missing, "
            f"for example: {', '.join(missing[:6])}"
        )

    y = df[TARGET].astype(int)
    if not set(y.unique()) <= {0, 1}:
        raise ValueError(f"`{TARGET}` must contain only 0 (Not Spam) and 1 (Spam).")

    X = df[features]
    two_classes = y.nunique() == 2
    results = {}

    for name in MODEL_FILES:
        model = load_model(name)
        pred = model.predict(X)
        prob = model.predict_proba(X)[:, 1]

        metrics = {
            "Accuracy": accuracy_score(y, pred),
            "AUC": roc_auc_score(y, prob) if two_classes else float("nan"),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "MCC": matthews_corrcoef(y, pred),
        }
        fpr, tpr = (roc_curve(y, prob)[:2]) if two_classes else (None, None)
        report = pd.DataFrame(
            classification_report(
                y, pred, labels=[0, 1], target_names=LABELS,
                output_dict=True, zero_division=0,
            )
        ).T.drop(index="accuracy", errors="ignore")
        report["support"] = report["support"].astype(int)

        results[name] = {
            "metrics": metrics,
            "cm": confusion_matrix(y, pred, labels=[0, 1]),
            "fpr": fpr,
            "tpr": tpr,
            "report": report,
            "pred": pred,
            "prob": prob,
        }

    return df, results


# ============================================================
# UI PIECES
# ============================================================

def render_sidebar():
    with st.sidebar:
        st.title(":blue[:material/mark_email_unread:] SARA")
        st.caption("Spam Analysis & Risk Assistant")
        st.divider()

        uploaded = st.file_uploader(
            "Test CSV",
            type=["csv"],
            help=f"CSV with the 57 model features and the `{TARGET}` target column.",
        )
        model_name = st.selectbox("Select model", list(MODEL_FILES))

        sample = BASE_DIR / "test_data.csv"
        if sample.exists():
            st.download_button(
                "Sample test data",
                data=sample.read_bytes(),
                file_name="test_data.csv",
                mime="text/csv",
                icon=":material/download:",
                width="stretch",
            )

        st.divider()
        st.caption(f":material/info: `{TARGET}`: 0 = Not Spam, 1 = Spam")
        st.caption("Models are loaded from disk and never retrained on uploaded data.")
    return uploaded, model_name


def render_stepper(step):
    steps = [
        ("Upload test data", "A CSV with the features and the target."),
        ("Select a model", "Choose one of the five trained models."),
        ("Review results", "Metrics, charts and model comparison."),
    ]
    for i, (col, (title, text)) in enumerate(zip(st.columns(3), steps), start=1):
        icon = (":green[:material/check_circle:]" if i < step
                else ":blue[:material/radio_button_checked:]" if i == step
                else ":gray[:material/radio_button_unchecked:]")
        with col, st.container(border=True):
            st.markdown(f"{icon} **Step {i}: {title}**")
            st.caption(text)


def fmt(value):
    return "N/A" if pd.isna(value) else f"{value:.4f}"


def render_overview(df, y):
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}", border=True)
    c2.metric("Features", f"{df.shape[1] - 1}", border=True)
    c3.metric("Spam share", f"{(y == 1).mean() * 100:.1f}%", border=True)

    left, right = st.columns([1, 2])
    with left:
        st.caption("Class distribution")
        counts = pd.DataFrame({
            "Class": LABELS,
            "Count": [int((y == 0).sum()), int((y == 1).sum())],
        })
        st.dataframe(counts, hide_index=True, width="stretch")
        st.bar_chart(counts.set_index("Class"), height=180)
    with right:
        st.caption("Test data preview (first 8 rows)")
        st.dataframe(df.head(8), hide_index=True, width="stretch")


def confusion_figure(cm, normalise):
    data = cm / cm.sum(axis=1, keepdims=True) if normalise else cm
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ConfusionMatrixDisplay(data, display_labels=LABELS).plot(
        ax=ax, cmap="Blues", colorbar=False,
        values_format=".1%" if normalise else "d",
    )
    ax.grid(False)
    fig.tight_layout()
    return fig


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)


def roc_figure(results, highlight=None):
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for name, r in results.items():
        if r["fpr"] is None:
            continue
        selected = highlight is None or name == highlight
        ax.plot(
            r["fpr"], r["tpr"],
            color=MODEL_COLORS[name],
            linewidth=2.6 if selected else 1.2,
            alpha=1.0 if selected else 0.5,
            label=f"{name} (AUC {r['metrics']['AUC']:.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#7a8fa6", linewidth=0.8)
    ax.grid(color=GRID_COLOR, linewidth=0.6)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    style_axes(ax)
    fig.tight_layout()
    return fig


def mcc_figure(comp, highlight):
    mcc = comp["MCC"].sort_values()
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    bars = ax.barh(mcc.index, mcc.values, color=[MODEL_COLORS[n] for n in mcc.index])
    for bar, name in zip(bars, mcc.index):
        bar.set_alpha(1.0 if name == highlight else 0.55)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8, color=TEXT_COLOR)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("MCC")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    style_axes(ax)
    fig.tight_layout()
    return fig


def render_performance(model_name, results):
    res = results[model_name]
    st.subheader(f":blue[:material/model_training:] {model_name}")

    cols = st.columns(6)
    for col, metric in zip(cols, METRICS):
        value = res["metrics"][metric]
        best = np.nanmax([r["metrics"][metric] for r in results.values()])
        gap = value - best
        delta = "Best" if abs(gap) < 1e-12 else f"{gap:+.4f} vs best"
        col.metric(metric, fmt(value), delta=delta, delta_color="off",
                   help=METRIC_HELP[metric], border=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        normalise = st.toggle("Show percentages per true class")
        fig = confusion_figure(res["cm"], normalise)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**ROC curve**")
        if res["fpr"] is None:
            st.info("ROC needs both classes in the uploaded data.")
        else:
            fig = roc_figure(results, highlight=model_name)
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("**Classification report**")
    st.dataframe(
        res["report"].style.format(
            {"precision": "{:.4f}", "recall": "{:.4f}", "f1-score": "{:.4f}", "support": "{:d}"}
        ),
        width="stretch",
    )
    st.caption("Precision, Recall and F1 in the tiles above are for the Spam (positive) class.")


def render_comparison(results, model_name):
    comp = pd.DataFrame(
        [{"ML Model Name": n, **r["metrics"]} for n, r in results.items()]
    ).set_index("ML Model Name")

    st.subheader(":blue[:material/leaderboard:] All models on this data")
    st.dataframe(
        comp.style.format("{:.4f}").highlight_max(
            axis=0, props="background-color:#edf8e8;color:#207e00;font-weight:bold"
        ),
        width="stretch",
    )
    st.caption("Best value in each column is highlighted.")

    left, right = st.columns(2)
    with left:
        st.markdown("**MCC by model**")
        fig = mcc_figure(comp, highlight=model_name)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**ROC curves**")
        if all(r["fpr"] is None for r in results.values()):
            st.info("ROC needs both classes in the uploaded data.")
        else:
            fig = roc_figure(results, highlight=model_name)
            st.pyplot(fig)
            plt.close(fig)

    cv_path = RESULTS_DIR / "cv_metrics.csv"
    params_path = RESULTS_DIR / "best_params.json"
    if cv_path.exists():
        with st.expander(":material/science: Training-time cross-validation and tuned parameters"):
            st.caption("Computed during training on the training split only (not on your upload).")
            st.dataframe(pd.read_csv(cv_path), hide_index=True, width="stretch")
            if params_path.exists():
                st.json(json.loads(params_path.read_text()).get("best_params", {}), expanded=False)


def render_predictions(df, results, model_name):
    res = results[model_name]
    out = df.copy()
    out["predicted"] = res["pred"]
    out["spam_probability"] = res["prob"].round(4)
    out["correct"] = out[TARGET].astype(int) == out["predicted"]
    first = [TARGET, "predicted", "spam_probability", "correct"]
    out = out[first + [c for c in out.columns if c not in first]]

    st.subheader(f":blue[:material/fact_check:] Predictions by {model_name}")
    errors = int((~out["correct"]).sum())
    view = st.radio(
        "Rows to show", ["All rows", f"Misclassified only ({errors})"], horizontal=True,
    )
    shown = out if view == "All rows" else out[~out["correct"]]
    st.dataframe(shown, hide_index=True, width="stretch", height=380)
    st.download_button(
        "Download predictions",
        data=out.to_csv(index=False).encode(),
        file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        icon=":material/download:",
    )


# ============================================================
# MAIN
# ============================================================

def main():
    uploaded, model_name = render_sidebar()

    title_col, model_col = st.columns([3, 1])
    with title_col:
        st.title(":blue[:material/mark_email_unread:] SARA")
        st.caption("Spam Analysis & Risk Assistant")
    with model_col:
        st.metric("Selected model", model_name)
    st.divider()

    if uploaded is None:
        render_stepper(1)
        st.info("Upload the test CSV from the sidebar to begin the analysis.",
                icon=":material/upload_file:")
        return

    try:
        df, results = evaluate_all(uploaded.getvalue())
    except Exception as exc:
        render_stepper(1)
        st.error(str(exc), icon=":material/error:")
        return

    render_stepper(3)

    best_model = max(results, key=lambda n: results[n]["metrics"]["MCC"])
    best_mcc = results[best_model]["metrics"]["MCC"]
    banner = f"Top performing model on this data: **{best_model}** (MCC {best_mcc:.4f})"
    if best_model == model_name:
        st.success(banner, icon=":material/emoji_events:")
    else:
        st.info(banner, icon=":material/leaderboard:")

    y = df[TARGET].astype(int)
    tab_perf, tab_compare, tab_pred, tab_data = st.tabs([
        ":material/analytics: Performance",
        ":material/leaderboard: Compare models",
        ":material/fact_check: Predictions",
        ":material/dataset: Test data",
    ])
    with tab_perf:
        render_performance(model_name, results)
    with tab_compare:
        render_comparison(results, model_name)
    with tab_pred:
        render_predictions(df, results, model_name)
    with tab_data:
        render_overview(df, y)

    st.divider()
    st.caption("SARA: Spam Analysis & Risk Assistant")


main()