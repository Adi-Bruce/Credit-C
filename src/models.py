"""
models.py
Defines baseline and ensemble models for credit card fraud detection.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier



# Baseline Models


def logistic_regression_model():
    """Logistic Regression with balanced class weights."""
    return LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
        n_jobs=-1
    )


def random_forest_model():
    """Random Forest Classifier with class weights balanced."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )



# Boosting Models


def xgboost_model():
    """XGBoost Classifier tuned for imbalance."""
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        scale_pos_weight=1,  # adjust if severe imbalance
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )


def lightgbm_model(**kwargs):
    """Returns a LightGBM model with default or user-specified params"""
    default_params = {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": -1,
        "num_leaves": 31,
        "random_state": 42,
        "class_weight": "balanced"
    }
    default_params.update(kwargs)
    return LGBMClassifier(**default_params)



# Ensemble: Voting Classifier

def voting_classifier():
    """Soft voting ensemble using Logistic Regression, RF, XGB, LGBM."""
    lr = logistic_regression_model()
    rf = random_forest_model()
    xgb = xgboost_model()
    lgb = lightgbm_model()

    return VotingClassifier(
        estimators=[('lr', lr), ('rf', rf), ('xgb', xgb), ('lgb', lgb)],
        voting='soft',  # use probabilities instead of class labels
        n_jobs=-1
    )
