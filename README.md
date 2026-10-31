# Credit Card Default Prediction – ML Classification Assignment

## a. Problem statement

The objective is to predict whether a credit-card client will default on payment in the next month. This is a binary classification task. Five classification algorithms are trained and compared using the same dataset and the same stratified 80:20 train/test split.

## b. Dataset description

- **Dataset:** Default of Credit Card Clients
- **Source:** [Kaggle – UCI Credit Card Default Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
- **Rows:** 30,000
- **Original columns:** 25 (including the ID and target columns)
- **Predictor features used:** 23 (the `ID` column is excluded because it is only an identifier)
- **Target:** `default.payment.next.month`
- **Target meaning:** `0` = no default; `1` = default
- **Class distribution:** 23,364 no-default records and 6,636 default records.
- **Split:** 80% training and 20% testing, stratified by target, with `random_state=42`.

The supplied `test_data.csv` contains the held-out test rows, all 23 input features, and the target column. The original Kaggle CSV is used by `train_models.py` and should be downloaded separately for retraining.

## c. GitHub Repository Link

**Repository:** (https://github.com/anshulkataria007/credit_default_ml_projec)

## d. Models used and evaluation

### Evaluation metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8078 | 0.7076 | 0.6883 | 0.2396 | 0.3555 | 0.3251 |
| Decision Tree | 0.7145 | 0.6075 | 0.3694 | 0.4115 | 0.3893 | 0.2042 |
| KNN | 0.7928 | 0.7014 | 0.5487 | 0.3564 | 0.4322 | 0.3233 |
| Naive Bayes (Gaussian) | 0.4160 | 0.6516 | 0.2496 | 0.8176 | 0.3824 | 0.1111 |
| Support Vector Machine (LinearSVC) | 0.8043 | 0.7042 | 0.7311 | 0.1824 | 0.2919 | 0.2969 |

*The SVM AUC is calculated from its continuous decision-function scores, which are valid scores for ROC-AUC. The other models use predicted probabilities.*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest accuracy (0.8078), AUC (0.7076), and MCC (0.3251). It has relatively low recall, so it misses many actual defaults at the default 0.5 threshold. |
| Decision Tree | Lowest AUC (0.6075) and moderate recall (0.4115). Its performance is weaker overall and may be sensitive to the training data. |
| KNN | Highest F1 score (0.4322), giving the best precision-recall balance among these five models on this test split. |
| Naive Bayes | Highest recall (0.8176), meaning it identifies most default cases, but precision is low (0.2496), so it also produces many false alarms. |
| Support Vector Machine | Highest precision (0.7311), but lowest recall (0.1824). Its positive predictions are relatively precise, but it misses many default cases. |

**Overall winner for this dataset:** KNN based on the highest F1 score. However, if the main objective is to catch as many potential defaulters as possible, Gaussian Naive Bayes has the highest recall. The preferred model therefore depends on the cost of false negatives versus false positives.

## Streamlit app

**Live Streamlit App:** https://creditdefaultmlproject-svk2zmm2dncfz3vr57rzvi.streamlit.app/

The app supports:
1. Uploading a labelled test CSV.
2. Selecting one of the five saved models.
3. Viewing Accuracy, AUC, Precision, Recall, F1, and MCC.
4. Viewing a confusion matrix and classification report.

The app loads saved models from `model/` and does not retrain on uploaded test data.
<img width="959" height="526" alt="Screenshot 2026-10-31 105142" src="https://github.com/user-attachments/assets/3a1d8b2d-db2f-4704-93e0-7c9b599b58a1" />


## How to run locally

1. Install the dependencies: `pip install -r requirements.txt`
2. Download the original Kaggle CSV and place it beside `train_models.py` as `UCI_Credit_Card.csv`.
3. To reproduce training and regenerate the saved models and test split, run: `python train_models.py`
4. Start the app: `streamlit run app.py`
5. Upload `test_data.csv` in the app and select a model.

## Repository structure

```text
credit_default_ml_project/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_metrics.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── support_vector_machine.joblib
```

## BITS Virtual Lab evidence

Insert the required screenshot of the assignment execution on BITS Virtual Lab into the final submission PDF. Do not use a screenshot of the local Streamlit app as a substitute for the BITS Virtual Lab proof.
