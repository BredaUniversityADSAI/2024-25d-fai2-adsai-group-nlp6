
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

def find_csv_file(path):
    """Find CSV file in directory or return path if it's a file"""
    if os.path.isdir(path):
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        print(f"Found CSV file: {csv_files[0]}")
        return csv_files[0]
    return path

def preprocess_data(train_path, test_path): 
    print("Loading data...")
    print(f"Train path: {train_path}")
    print(f"Test path: {test_path}")

    # Find CSV files in directories or use direct file paths
    train_file = find_csv_file(train_path)
    test_file = find_csv_file(test_path)

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    print("Dataset Info:")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print("Column data types:")
    print(train_df.dtypes)

    # Find target column
    target_columns = ['target', 'Target', 'label', 'Label', 'emotion', 'Emotion']
    target_col = None
    for col in target_columns:
        if col in train_df.columns:
            target_col = col
            break

    if target_col is None:
        raise ValueError(f"No target column found. Available columns: {train_df.columns.tolist()}")

    print(f"Using target column: {target_col}")

    # Extract and encode target variables
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df[target_col])
    y_test = label_encoder.transform(test_df[target_col])

    print(f"Unique target values: {label_encoder.classes_}")

    # Drop target column and process features
    X_train = train_df.drop(target_col, axis=1)
    X_test = test_df.drop(target_col, axis=1)

    feature_columns = []
    for column in X_train.columns:
        print(f"Processing column: {column}")
        try:
            # Try numeric conversion
            X_train[column] = pd.to_numeric(X_train[column], errors='raise')
            X_test[column] = pd.to_numeric(X_test[column], errors='raise')
            feature_columns.append(column)
            print(f"Converted to numeric: {column}")
        except (ValueError, TypeError):
            # For non-numeric columns, try encoding
            try:
                label_enc = LabelEncoder()
                X_train[column] = label_enc.fit_transform(X_train[column].astype(str))
                X_test[column] = label_enc.transform(X_test[column].astype(str))
                feature_columns.append(column)
                print(f"Encoded categorical: {column}")
            except Exception as e:
                print(f"Skipping column {column}: {str(e)}")

    if not feature_columns:
        print("Column details:")
        for col in X_train.columns:
            print(f"{col}: {X_train[col].dtype}")
            print(f"Sample values: {X_train[col].head()}")
        raise ValueError("No usable features found for training!")

    X_train = X_train[feature_columns]
    X_test = X_test[feature_columns]

    print(f"Selected features: {feature_columns}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler and label encoder for model registration
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(scaler, 'outputs/scaler.pkl')
    joblib.dump(label_encoder, 'outputs/label_encoder.pkl')

    return X_train_scaled, X_test_scaled, y_train, y_test

def get_model(name, params=None): 
    if name == "random_forest": 
        return RandomForestClassifier(**(params or {})) 
    elif name == "logistic_regression": 
        return LogisticRegression(**(params or {})) 
    else: 
        raise ValueError("Model not supported.") 

def log_metrics_to_file(metrics, params):
    """Log metrics and parameters to files instead of MLflow"""
    os.makedirs('outputs', exist_ok=True)

    # Save metrics
    with open('outputs/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save parameters  
    with open('outputs/params.json', 'w') as f:
        json.dump(params, f, indent=2)

    print("Metrics and parameters saved to outputs/")

def safe_mlflow_logging(metrics, params):
    """Safely attempt MLflow logging with fallback"""
    try:
        import mlflow
        # Try to set a simple tracking URI to avoid Azure ML registry issues
        mlflow.set_tracking_uri("file:./mlruns")

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        for key, value in params.items():
            mlflow.log_param(key, value)

        print("Successfully logged to MLflow")
        return True
    except Exception as e:
        print(f"MLflow logging failed: {str(e)}")
        print("Falling back to file-based logging...")
        log_metrics_to_file(metrics, params)
        return False

def register_model_locally(model, accuracy, threshold, model_name, hyperparams):
    """Register model locally if it passes the threshold"""
    if accuracy >= threshold:
        print(f"Model passed evaluation with accuracy {accuracy:.4f} >= {threshold}")
        print("Registering model locally...")

        # Create model metadata
        model_metadata = {
            "accuracy": accuracy,
            "threshold": threshold,
            "model_type": model_name,
            "hyperparameters": hyperparams,
            "training_timestamp": time.time(),
            "passed_evaluation": True
        }

        # Save model metadata
        with open('outputs/model_metadata.json', 'w') as f:
            json.dump(model_metadata, f, indent=2)

        # Save the trained model
        joblib.dump(model, 'outputs/model.pkl')

        # Create a model registration file
        with open('outputs/model_registered.txt', 'w') as f:
            f.write(f"Model registered successfully\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Model Type: {model_name}\n")
            f.write(f"Hyperparameters: {hyperparams}\n")
            f.write(f"Registration Time: {time.ctime()}\n")

        print("[SUCCESS] Model registered locally in outputs/ directory")
        print(f"   - Model file: outputs/model.pkl")
        print(f"   - Metadata: outputs/model_metadata.json")
        print(f"   - Registration info: outputs/model_registered.txt")
        return True
    else:
        print(f"[FAILED] Model did not pass evaluation: {accuracy:.4f} < {threshold}")
        print("Model will not be registered.")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--C', type=float, default=1.0)  # For logistic regression
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = preprocess_data(args.train_data, args.test_data)

    # Prepare hyperparameters based on model type
    if args.model_name == "random_forest":
        params = {
            "n_estimators": args.n_estimators, 
            "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split,
            "random_state": 42
        }
    elif args.model_name == "logistic_regression":
        params = {
            "C": args.C,
            "max_iter": 1000,
            "random_state": 42
        }
    else:
        params = {"n_estimators": args.n_estimators, "max_depth": args.max_depth}

    model = get_model(args.model_name, params)

    print("Training model...")
    print(f"Model: {args.model_name}")
    print(f"Hyperparameters: {params}")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Prepare metrics and parameters for logging
    metrics = {
        "accuracy": acc,
        "evaluation_passed": 1 if acc >= args.threshold else 0
    }

    log_params = {
        "model_name": args.model_name,
        "threshold": args.threshold,
        **params  # Include all model-specific parameters
    }

    # Try MLflow logging with fallback
    safe_mlflow_logging(metrics, log_params)

    # Register model locally if it passes the threshold
    model_registered = register_model_locally(model, acc, args.threshold, args.model_name, params)

    print(f"\n=== Training Results ===")
    print(f"Model: {args.model_name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Threshold: {args.threshold}")
    print(f"Model Registered: {'Yes' if model_registered else 'No'}")
    print(f"Hyperparameters: {params}")

if __name__ == "__main__":
    main()
