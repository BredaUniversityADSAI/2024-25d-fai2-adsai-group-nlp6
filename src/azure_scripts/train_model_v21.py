
import pandas as pd 
import joblib 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import StandardScaler 
import argparse
import os

# NO azureml.core imports - they're causing the problem!

def preprocess_data(train_path, test_path): 
    """Load and preprocess data from file paths"""
    print(f"Loading training data from: {train_path}")
    print(f"Loading test data from: {test_path}")

    # Load data directly from CSV files
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")

    X_train = train_df.drop('target', axis=1) 
    y_train = train_df['target'] 
    X_test = test_df.drop('target', axis=1) 
    y_test = test_df['target'] 

    scaler = StandardScaler() 
    X_train_scaled = scaler.fit_transform(X_train) 
    X_test_scaled = scaler.transform(X_test) 

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def get_model(name, params=None): 
    """Get model instance"""
    if name == "random_forest": 
        return RandomForestClassifier(**(params or {})) 
    elif name == "logistic_regression": 
        return LogisticRegression(**(params or {})) 
    else: 
        raise ValueError(f"Model {name} not supported.") 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=0.8)
    args = parser.parse_args()

    print("="*50)
    print("STARTING TRAINING")
    print("="*50)
    print(f"Model: {args.model_name}")
    print(f"N_estimators: {args.n_estimators}")
    print(f"Max_depth: {args.max_depth}")
    print(f"Threshold: {args.threshold}")

    try:
        # Preprocess data
        X_train, X_test, y_train, y_test, scaler = preprocess_data(args.train_data, args.test_data)

        # Set up model parameters
        params = {"n_estimators": args.n_estimators, "max_depth": args.max_depth}

        # Train model
        print("Training model...")
        model = get_model(args.model_name, params)
        model.fit(X_train, y_train)

        # Make predictions and calculate accuracy
        print("Making predictions...")
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log metrics to stdout (Azure ML will capture these)
        print("="*50)
        print("TRAINING RESULTS")
        print("="*50)
        print(f"METRIC: accuracy = {acc}")
        print(f"METRIC: n_estimators = {args.n_estimators}")
        print(f"METRIC: max_depth = {args.max_depth}")
        print(f"METRIC: model_name = {args.model_name}")

        # Save model and scaler
        os.makedirs('outputs', exist_ok=True)
        model_path = 'outputs/model.pkl'
        scaler_path = 'outputs/scaler.pkl'
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")

        # Evaluate model
        passed = acc >= args.threshold
        print(f"METRIC: evaluation_passed = {passed}")

        if passed:
            print("✅ Model passed evaluation and is ready for use.")
        else:
            print("❌ Model did not meet performance threshold.")

        print("="*50)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*50)

        return acc

    except Exception as e:
        print(f"ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
