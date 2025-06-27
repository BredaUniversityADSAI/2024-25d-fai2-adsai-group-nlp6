
import pandas as pd 
import joblib 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler 
import argparse
import os
import glob
from sklearn.preprocessing import LabelEncoder
import json
import time
import numpy as np

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

    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder.classes_

def get_model(name, params=None): 
    if name == "random_forest": 
        return RandomForestClassifier(**(params or {})) 
    elif name == "logistic_regression": 
        return LogisticRegression(**(params or {})) 
    else: 
        raise ValueError("Model not supported.") 

def calculate_comprehensive_metrics(y_true, y_pred, target_classes):
    """Calculate comprehensive metrics for model evaluation"""
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Weighted metrics (better for imbalanced datasets)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    for i, class_name in enumerate(target_classes):
        metrics[f'precision_{class_name}'] = precision_per_class[i] if i < len(precision_per_class) else 0
        metrics[f'recall_{class_name}'] = recall_per_class[i] if i < len(recall_per_class) else 0
        metrics[f'f1_{class_name}'] = f1_per_class[i] if i < len(f1_per_class) else 0

    # Confusion matrix statistics
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix_trace'] = np.trace(cm)  # Sum of diagonal elements
    metrics['total_predictions'] = len(y_true)

    return metrics

def azure_ml_logging(metrics, params):
    """Proper Azure ML MLflow logging that will show in the UI"""
    try:
        import mlflow

        # Don't set tracking URI - let Azure ML handle it automatically
        print("Starting MLflow logging for Azure ML...")

        # Start an MLflow run (Azure ML will handle the tracking)
        with mlflow.start_run() as run:
            # Log all parameters
            print("Logging parameters...")
            for key, value in params.items():
                mlflow.log_param(key, value)
                print(f"  Logged param: {key} = {value}")

            # Log all metrics
            print("Logging metrics...")
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    mlflow.log_metric(key, float(value))
                    print(f"  Logged metric: {key} = {value}")
                else:
                    # Convert non-numeric values to string and log as param
                    mlflow.log_param(f"info_{key}", str(value))
                    print(f"  Logged info: {key} = {value}")

            print(f"MLflow run completed successfully! Run ID: {run.info.run_id}")
            return True

    except ImportError:
        print("MLflow not available - this is expected in some environments")
        return False
    except Exception as e:
        print(f"Azure ML MLflow logging encountered an error: {str(e)}")
        print("This might be due to environment setup, but the model will still be saved locally")
        return False

def log_metrics_to_file(metrics, params):
    """Fallback: Log metrics and parameters to files"""
    os.makedirs('outputs', exist_ok=True)

    # Save metrics
    with open('outputs/metrics.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                json_metrics[key] = float(value)
            else:
                json_metrics[key] = value
        json.dump(json_metrics, f, indent=2)

    # Save parameters  
    with open('outputs/params.json', 'w') as f:
        json.dump(params, f, indent=2)

    print("Metrics and parameters saved to outputs/ directory")

def register_model_locally(model, metrics, threshold, model_name, hyperparams):
    """Register model locally if it passes the threshold"""
    accuracy = metrics['accuracy']

    if accuracy >= threshold:
        print(f"Model passed evaluation with accuracy {accuracy:.4f} >= {threshold}")
        print("Registering model locally...")

        # Create comprehensive model metadata
        model_metadata = {
            "model_performance": metrics,
            "threshold": threshold,
            "model_type": model_name,
            "hyperparameters": hyperparams,
            "training_timestamp": time.time(),
            "passed_evaluation": True,
            "evaluation_summary": {
                "accuracy": accuracy,
                "f1_macro": metrics.get('f1_macro', 0),
                "precision_macro": metrics.get('precision_macro', 0),
                "recall_macro": metrics.get('recall_macro', 0)
            }
        }

        # Save model metadata
        with open('outputs/model_metadata.json', 'w') as f:
            # Convert numpy types for JSON serialization
            json_metadata = {}
            for key, value in model_metadata.items():
                if isinstance(value, dict):
                    json_metadata[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                        for k, v in value.items()}
                elif isinstance(value, (np.integer, np.floating)):
                    json_metadata[key] = float(value)
                else:
                    json_metadata[key] = value
            json.dump(json_metadata, f, indent=2)

        # Save the trained model
        joblib.dump(model, 'outputs/model.pkl')

        # Create a detailed model registration report
        with open('outputs/model_registered.txt', 'w') as f:
            f.write(f"=== MODEL REGISTRATION REPORT ===\n")
            f.write(f"Registration Time: {time.ctime()}\n")
            f.write(f"Model Type: {model_name}\n")
            f.write(f"\n=== PERFORMANCE METRICS ===\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"F1 Score (Macro): {metrics.get('f1_macro', 0):.4f}\n")
            f.write(f"Precision (Macro): {metrics.get('precision_macro', 0):.4f}\n")
            f.write(f"Recall (Macro): {metrics.get('recall_macro', 0):.4f}\n")
            f.write(f"\n=== HYPERPARAMETERS ===\n")
            for key, value in hyperparams.items():
                f.write(f"{key}: {value}\n")
            f.write(f"\n=== EVALUATION ===\n")
            f.write(f"Threshold: {threshold}\n")
            f.write(f"Status: PASSED (Accuracy {accuracy:.4f} >= {threshold})\n")

        print("[SUCCESS] Model registered locally in outputs/ directory")
        print(f"   - Model file: outputs/model.pkl")
        print(f"   - Comprehensive metadata: outputs/model_metadata.json")
        print(f"   - Registration report: outputs/model_registered.txt")
        return True
    else:
        print(f"[FAILED] Model did not pass evaluation: {accuracy:.4f} < {threshold}")
        print("Model will not be registered.")

        # Still save performance report for analysis
        with open('outputs/failed_model_report.txt', 'w') as f:
            f.write(f"=== FAILED MODEL REPORT ===\n")
            f.write(f"Evaluation Time: {time.ctime()}\n")
            f.write(f"Model Type: {model_name}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Threshold: {threshold}\n")
            f.write(f"Status: FAILED (Accuracy {accuracy:.4f} < {threshold})\n")

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

    print("=== STARTING TRAINING PROCESS ===")
    print(f"Model: {args.model_name}")
    print(f"Threshold: {args.threshold}")

    X_train, X_test, y_train, y_test, target_classes = preprocess_data(args.train_data, args.test_data)

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

    print("\n=== TRAINING MODEL ===")
    print(f"Hyperparameters: {params}")
    model.fit(X_train, y_train)

    print("\n=== EVALUATING MODEL ===")
    preds = model.predict(X_test)

    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(y_test, preds, target_classes)

    # Add threshold-based evaluation
    metrics['evaluation_passed'] = 1 if metrics['accuracy'] >= args.threshold else 0
    metrics['threshold_used'] = args.threshold

    # Prepare parameters for logging
    log_params = {
        "model_name": args.model_name,
        "threshold": args.threshold,
        "num_features": X_train.shape[1],
        "train_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
        "num_classes": len(target_classes),
        **params  # Include all model-specific parameters
    }

    # Primary logging: Try Azure ML MLflow integration
    print("\n=== LOGGING METRICS TO AZURE ML ===")
    azure_ml_success = azure_ml_logging(metrics, log_params)

    # Fallback logging: Always save to files as well
    log_metrics_to_file(metrics, log_params)

    # Register model locally if it passes the threshold
    print("\n=== MODEL REGISTRATION ===")
    model_registered = register_model_locally(model, metrics, args.threshold, args.model_name, params)

    # Print comprehensive results
    print(f"\n=== TRAINING RESULTS SUMMARY ===")
    print(f"Model: {args.model_name}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
    print(f"Threshold: {args.threshold}")
    print(f"Azure ML Logging: {'Success' if azure_ml_success else 'Failed (fallback used)'}")
    print(f"Model Registered: {'Yes' if model_registered else 'No'}")
    print(f"Hyperparameters: {params}")

    if azure_ml_success:
        print("\n✅ Metrics should now be visible in Azure ML Studio!")
        print("   Go to: Azure ML Studio > Experiments > Your Experiment > Run Details")
    else:
        print("\n⚠️  Azure ML logging failed, but metrics are saved locally in outputs/")

if __name__ == "__main__":
    main()
