import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_data(df, test_size=0.2, random_state=42):

    
    # Define features
    num_features = ['Amount', 'Time', 'CardHolderAge']
    cat_features = ['Location', 'MerchantCategory']

    # Define imputers
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    # Build preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ('imputer', num_imputer),
                ('scaler', StandardScaler())
            ]), num_features),
            
            ("cat", Pipeline([
                ('imputer', cat_imputer),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), cat_features)
        ]
    )

    # Separate features & target
    X = df.drop(columns=["TransactionID", "IsFraud"])
    y = df["IsFraud"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, preprocessor
