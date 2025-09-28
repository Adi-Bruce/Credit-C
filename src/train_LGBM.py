import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from src.data_preprocessing import preprocess_data
from src.models import lightgbm_model


# Load & Preprocess Data
df = pd.read_csv("data/fraud_data - Sheet 1.csv")
X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

# Combine full data for CV
X = preprocessor.fit_transform(pd.concat([X_train, X_test]))
y = pd.concat([y_train, y_test])


# Handle Class Imbalance (SMOTE)
smote = SMOTE(sampling_strategy='minority', random_state=42)
X, y = smote.fit_resample(X, y)


# LightGBM Model with better params
model = lightgbm_model(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,             # No depth limit
    num_leaves=31,            # More leaves allow more splits
    min_data_in_leaf=1,        # Allow small leaves
    min_gain_to_split=0        # Allow zero-gain splits
)


# Cross-Validation Function
def evaluate_lightgbm(model, X, y, cv, threshold=0.3):
    precision_scores, recall_scores, f1_scores, roc_scores = [], [], [], []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

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


# Run Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = evaluate_lightgbm(model, X, y, skf, threshold=0.3)

print("\n===== LightGBM Model Performance =====")
for metric, (mean, std) in scores.items():
    print(f"{metric}: {mean:.3f} ± {std:.3f}")


# Retrain on full data and save
model.fit(X, y)

# Make sure folder exists
os.makedirs("models", exist_ok=True)
os.makedirs("preprocessor", exist_ok=True)

# Save model and preprocessor
joblib.dump(model, "models/lightgbm_model.pkl")
joblib.dump(preprocessor, "preprocessor/preprocessor.pkl")

print("\nModel and preprocessor saved successfully!")
print("\n===== LightGBM Model Performance =====")
for metric, (mean, std) in scores.items():
    print(f"{metric}: {mean:.3f} ± {std:.3f}")
