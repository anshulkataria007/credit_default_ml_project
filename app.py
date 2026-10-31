import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Default Classifier", page_icon="💳", layout="wide")
st.title("Credit Card Default Prediction")
st.write("Upload labelled test data, select a saved classifier, and view evaluation metrics. Models are loaded from disk; no retraining occurs.")

TARGET = "default.payment.next.month"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model_name = st.selectbox("Select a classification model", list(MODEL_FILES))
uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"])
st.caption(f"CSV must include all model input features and the target column '{TARGET}'.")

if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)
        if TARGET not in test_df.columns:
            st.error(f"Target column '{TARGET}' is missing.")
            st.stop()
        model_path = os.path.join("model", MODEL_FILES[model_name])
        if not os.path.exists(model_path):
            st.error(f"Saved model not found: {model_path}. Ensure the model folder is committed to GitHub.")
            st.stop()
        model = load_model(model_path)
        y_true = test_df[TARGET].astype(int)
        X_test = test_df.drop(columns=[TARGET])
        expected = list(model.feature_names_in_)
        missing = [c for c in expected if c not in X_test.columns]
        if missing:
            st.error("Missing required features: " + ", ".join(missing))
            st.stop()
        X_test = X_test[expected]
        if y_true.nunique() < 2:
            st.error("AUC requires both target classes (0 and 1) in the uploaded test data.")
            st.stop()
        y_pred = model.predict(X_test)
        scores = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": roc_auc_score(y_true, scores),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred),
        }
        st.subheader(f"Results: {model_name}")
        cols = st.columns(3)
        for i, (label, value) in enumerate(metrics.items()):
            cols[i % 3].metric(label, f"{value:.4f}")
        st.write(f"**Rows evaluated:** {len(test_df):,}")
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm, display_labels=["No default (0)", "Default (1)"]).plot(
            ax=ax, cmap="Blues", colorbar=False
        )
        st.pyplot(fig)
        plt.close(fig)
        st.subheader("Classification Report")
        report = classification_report(
            y_true, y_pred, labels=[0, 1],
            target_names=["No default (0)", "Default (1)"],
            output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)
    except Exception as exc:
        st.error(f"Could not evaluate this CSV: {exc}")
else:
    st.info("Upload the provided test_data.csv to view model evaluation results.")
