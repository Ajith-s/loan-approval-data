import numpy as np
import pandas as pd 
import lightgbm as lgb
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import joblib
import json


def load_data(file_path):
    # Load your dataset
    # file_path = '/Users/ajithsharma/Documents/projects/loan-data/loan_data.csv'
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        return data
    else:
        print(f"File not found: {file_path}")

def data_preprocessing(data, target_column='loan_status'):
    # Handle missing values, encode categorical variables, etc.
    lgb_data = data.copy()
    cat_features = ['person_gender','person_education','person_home_ownership','loan_intent','previous_loan_defaults_on_file']
    lgb_data[cat_features] = lgb_data[cat_features].astype('category')
    X = lgb_data.drop(columns = target_column)
    y = lgb_data[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42,stratify = y)

def train_model(X_train, y_train):
    model = LGBMClassifier(objective='binary', metric='binary_logloss', random_state=42)
    cat_features = ['person_gender','person_education','person_home_ownership','loan_intent','previous_loan_defaults_on_file']
    model.fit(X_train, y_train, categorical_feature = cat_features)
    return model

def save_model(model, file_path="loan_approval_model.pkl"):
    """Save the trained model to a file."""
    joblib.dump(model, file_path)
    print(f"Model saved successfully at {file_path}")

def save_metrics(metrics, file_path="evaluation_metrics.json"):
    """Save evaluation metrics to a JSON file."""
    with open(file_path, "w") as file:
        json.dump(metrics, file)
    print(f"Metrics saved successfully at {file_path}")

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    return metrics


if __name__ == "__main__":
    # Replace 'path_to_your_file.csv' with your actual file path
    data_file = "/Users/ajithsharma/Documents/projects/loan-data/loan_data.csv"
    model_file = "loan_approval_model.pkl"
    metrics_file = "evaluation_metrics.json"
    target_column = "loan_status"  # Replace with your target column

    df = load_data(data_file)
    X_train, X_test, y_train, y_test = data_preprocessing(df, target_column)
     # Train the model
    model = train_model(X_train, y_train)

    # Save the trained model
    save_model(model, model_file)

    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)
    print("Evaluation Metrics:", metrics)

    # Save evaluation metrics to a file
    save_metrics(metrics, metrics_file)
