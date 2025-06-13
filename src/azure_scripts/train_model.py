
import pandas as pd 
import joblib 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import StandardScaler 
from azureml.core import Dataset, Run
import argparse
import os

def preprocess_data(train_dataset, test_dataset): 
    # Convert to pandas DataFrames
    train_df = train_dataset.to_pandas_dataframe()
    test_df = test_dataset.to_pandas_dataframe()

    X_train = train_df.drop('target', axis=1) 
    y_train = train_df['target'] 
    X_test = test_df.drop('target', axis=1) 
    y_test = test_df['target'] 

    scaler = StandardScaler() 
    X_train_scaled = scaler.fit_transform(X_train) 
    X_test_scaled = scaler.transform(X_test) 

    return X_train_scaled, X_test_scaled, y_train, y_test 

def get_model(name, params=None): 
    if name == "random_forest": 
        return RandomForestClassifier(**(params or {})) 
    elif name == "logistic_regression": 
        return LogisticRegression(**(params or {})) 
    else: 
        raise ValueError("Model not supported.") 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_name', type=str, required=True)
    parser.add_argument('--test_dataset_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=0.8)
    args = parser.parse_args()

    # Get the run context
    run = Run.get_context()

    # Get datasets
    train_dataset = Dataset.get_by_name(run.experiment.workspace, args.train_dataset_name)
    test_dataset = Dataset.get_by_name(run.experiment.workspace, args.test_dataset_name)

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(train_dataset, test_dataset)

    # Set up model parameters
    params = {"n_estimators": args.n_estimators, "max_depth": args.max_depth}

    # Train model
    model = get_model(args.model_name, params)
    model.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Log metrics
    run.log("accuracy", acc)
    run.log("n_estimators", args.n_estimators)
    run.log("max_depth", args.max_depth)

    # Save model
    os.makedirs('outputs', exist_ok=True)
    model_path = 'outputs/model.pkl'
    joblib.dump(model, model_path)

    # Evaluate model
    passed = acc >= args.threshold
    run.log("evaluation_passed", passed)

    print(f"Training accuracy: {acc}")
    if passed:
        print("CORRECT Model passed evaluation and is ready for use.")
    else:
        print("ERROR Model did not meet performance threshold.")

    return acc

if __name__ == "__main__":
    main()
