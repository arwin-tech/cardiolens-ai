import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(data_path="data/cardiovascular_disease.csv"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Download it from Kaggle first.")
    
    try:
        df = pd.read_csv(data_path, sep=";")
        if df.shape[1] == 1:
            df = pd.read_csv(data_path, sep=",")
    except Exception:
        df = pd.read_csv(data_path)

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    if "age" in df.columns:
        df["age"] = (df["age"] / 365.25).round().astype(int)

    X = df.drop(columns=["cardio"])
    y = df["cardio"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, list(X.columns)

def main():
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names
    }

if __name__ == "__main__":
    main()