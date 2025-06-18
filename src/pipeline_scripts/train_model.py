
import pandas as pd 
import joblib 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import StandardScaler 
import argparse
import os
import glob
from sklearn.preprocessing import LabelEncoder
import json
import time
import mlflow
from azureml.core import Run
import numpy as np

def find_csv_file(path):
    if os.path.isdir(path):
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        return csv_files[0]
    return path

def convert_time_to_seconds(time_str):
    """Convert time string (HH:MM:SS,fff) to total seconds"""
    try:
        hh_mm_ss, millis = time_str.split(',')
        h, m, s = hh_mm_ss.split(':')
        return float(h) * 3600 + float(m) * 60 + float(s) + (float(millis)/1000)
    except:
        return np.nan

def preprocess_data(train_path, test_path): 
    train_file = find_csv_file(train_path)
    test_file = find_csv_file(test_path)

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Identify time columns and convert to seconds
    time_cols = [col for col in train_df.columns 
                if any(x in str(train_df[col].dtype) for x in ['time', 'object'])]

    for col in time_cols:
        if train_df[col].astype(str).str.match(r'\d{2}:\d{2}:\d{2},\d{3}').any():
            train_df[col] = train_df[col].astype(str).apply(convert_time_to_seconds)
            test_df[col] = test_df[col].astype(str).apply(convert_time_to_seconds)

    target_columns = ['target', 'Target', 'label', 'Label', 'emotion', 'Emotion']
    target_col = next((col for col in target_columns if col in train_df.columns), None)

    if target_col is None:
        raise ValueError(f"No target column found. Available columns: {train_df.columns.tolist()}")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df[target_col])
    y_test = label_encoder.transform(test_df[target_col])

    X_train = train_df.drop(columns=[target_col])
    X_test = test_df.drop(columns=[target_col])

    # Convert remaining string columns to numeric or categorical
    for col in X_train.select_dtypes(include=['object']).columns:
        try:
            X_train[col] = pd.to_numeric(X_train[col], errors='raise')
            X_test[col] = pd.to_numeric(X_test[col], errors='raise')
        except:
            X_train[col] = LabelEncoder().fit_transform(X_train[col].astype(str))
            X_test[col] = LabelEncoder().fit_transform(X_test[col].astype(str))

    # Handle missing values
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs('outputs', exist_ok=True)
    joblib.dump(scaler, 'outputs/scaler.pkl')
    joblib.dump(label_encoder, 'outputs/label_encoder.pkl')

    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate(args):
    run = Run.get_context()
    mlflow.start_run()

    try:
        X_train, X_test, y_train, y_test = preprocess_data(args.train_data, args.test_data)

        if args.model_name == "random_forest":
            params = {
                "n_estimators": args.n_estimators, 
                "max_depth": args.max_depth,
                "min_samples_split": args.min_samples_split,
                "random_state": 42
            }
            model = RandomForestClassifier(**params)
        elif args.model_name == "logistic_regression":
            params = {
                "C": args.C,
                "max_iter": 1000,
                "random_state": 42
            }
            model = LogisticRegression(**params)
        else:
            raise ValueError(f"Unsupported model: {args.model_name}")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        os.makedirs(args.model_output, exist_ok=True)
        joblib.dump(model, os.path.join(args.model_output, 'model.pkl'))

        mlflow.log_metric("accuracy", acc)
        mlflow.log_params(params)
        mlflow.log_param("model_name", args.model_name)

        return acc

    finally:
        mlflow.end_run()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--model_output', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=20)
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--C', type=float, default=1.0)
    args = parser.parse_args()

    accuracy = train_and_evaluate(args)
    print(f"Model trained with accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
