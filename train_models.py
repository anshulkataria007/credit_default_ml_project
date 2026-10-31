import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

DATA_PATH = "UCI_Credit_Card.csv"
TARGET = "default.payment.next.month"
RANDOM_STATE = 42

def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ]),
        "Naive Bayes": GaussianNB(),
        # LinearSVC gives a decision score for ROC-AUC without costly probability calibration.
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LinearSVC(max_iter=2000, random_state=RANDOM_STATE))
        ])
    }

def get_auc_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)

def evaluate(model, X_test, y_test):
    predicted = model.predict(X_test)
    scores = get_auc_scores(model, X_test)
    return {
        "Accuracy": accuracy_score(y_test, predicted),
        "AUC": roc_auc_score(y_test, scores),
        "Precision": precision_score(y_test, predicted, zero_division=0),
        "Recall": recall_score(y_test, predicted, zero_division=0),
        "F1": f1_score(y_test, predicted, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predicted)
    }

def main():
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError("Download the Kaggle CSV and save it as UCI_Credit_Card.csv beside this script.")
    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Required target column '{TARGET}' was not found.")
    y = df[TARGET].astype(int)
    # ID is a row identifier, not a meaningful predictive feature.
    X = df.drop(columns=[TARGET] + (["ID"] if "ID" in df.columns else []))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    os.makedirs("model", exist_ok=True)
    rows = []
    for name, model in build_models().items():
        print(f"Training {name}...", flush=True)
        model.fit(X_train, y_train)
        scores = evaluate(model, X_test, y_test)
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, os.path.join("model", filename))
        rows.append({"Model": name, **scores})
        print(name, {key: round(value, 4) for key, value in scores.items()}, flush=True)

    pd.DataFrame(rows).to_csv("model_metrics.csv", index=False)
    test_set = X_test.copy()
    test_set[TARGET] = y_test
    test_set.to_csv("test_data.csv", index=False)
    print("Complete: saved all five models, model_metrics.csv and test_data.csv.")

if __name__ == "__main__":
    main()
