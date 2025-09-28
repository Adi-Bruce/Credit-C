# Credit Card Fraud Detection  

This project focuses on detecting fraudulent credit card transactions using various machine learning models.  
I tested multiple algorithms (Logistic Regression, Random Forest, XGBoost, and LightGBM) with preprocessing, cross-validation, and handling of class imbalance using **SMOTE**.  

The **LightGBM model** gave the best performance with high precision, recall, and ROC-AUC.

## Why We Avoided Ensembling  

Initially, I considered **ensemble methods** like Voting Classifiers or Stacking.  
However, after experiments, **LightGBM alone consistently outperformed all other models** in terms of precision, recall, and ROC-AUC.  

Adding ensembling:  
- Did **not significantly improve metrics**.  
- **Increased computational complexity** without meaningful gains.  
- Made deployment heavier and harder to maintain.  

Thus, we finalized **LightGBM as a single strong learner** rather than combining multiple models unnecessarily.

| Model                | Precision | Recall  | F1-Score | ROC-AUC |
|----------------------|-----------|---------|----------|---------|
| Logistic Regression   | 0.524     | 0.981   | 0.683    | 0.628   |
| Random Forest         | 0.918     | 0.960   | 0.938    | 0.991   |
| XGBoost               | 0.948     | 0.947   | 0.947    | 0.981   |
| **LightGBM**          | **0.972** | **0.949** | **0.960** | **0.987** |


---

## Project Structure

cred_fraud/
│
├── data/ # Raw dataset (CSV files)
│ └── fraud_data - Sheet 1.csv
│
├── models/ # Saved trained models & preprocessors
│ ├── lightgbm_model.pkl
│ └── preprocessor.pkl
│
├── src/
│ ├── data_preprocessing.py # Data cleaning, encoding, scaling
│ ├── models.py # Model definitions (Logistic, RF, XGB, LGBM)
│ ├── training.py # Cross-validation on all models
│ └── train_LGBM.py # Separate training & saving for LightGBM
│
├── requirements.txt # Python dependencies
└── README.md # Project documentation