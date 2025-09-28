import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

# Import preprocessing & models
from src.data_preprocessing import preprocess_data
from src.models import logistic_regression_model, random_forest_model, xgboost_model, lightgbm_model


# Load & Preprocess Data
df = pd.read_csv("data/fraud_data - Sheet 1.csv")
X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

# Transform entire dataset
X = preprocessor.fit_transform(pd.concat([X_train, X_test]))
y = pd.concat([y_train, y_test])


# Handle Class Imbalance with SMOTE

smote = SMOTE(sampling_strategy='minority', random_state=42)
X, y = smote.fit_resample(X, y)


# Define Models with Balanced Parameters
models = {
    "Logistic Regression": logistic_regression_model(),
    "Random Forest": random_forest_model(),
    "XGBoost": xgboost_model(),
    "LightGBM": lightgbm_model()
}


# Cross-Validation with Custom Threshold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(model, X, y, cv, threshold=0.3):
    precision_scores, recall_scores, f1_scores, roc_scores = [], [], [], []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)  # custom threshold

        precision_scores.append(precision_score(y_val, y_pred, zero_division=0))
        recall_scores.append(recall_score(y_val, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_val, y_pred, zero_division=0))
        roc_scores.append(roc_auc_score(y_val, y_prob))

    return {
        "Precision": (np.mean(precision_scores), np.std(precision_scores)),
        "Recall": (np.mean(recall_scores), np.std(recall_scores)),
        "F1-Score": (np.mean(f1_scores), np.std(f1_scores)),
        "ROC-AUC": (np.mean(roc_scores), np.std(roc_scores))
    }


# Train & Report Metrics
for name, model in models.items():
    print(f"\n===== {name} =====")
    scores = evaluate_model(model, X, y, skf, threshold=0.3)
    for metric, (mean, std) in scores.items():
        print(f"{metric}: {mean:.3f} ± {std:.3f}")
