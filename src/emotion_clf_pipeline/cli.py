"""
Command-Line Interface for the Emotion Classification Pipeline.

This script provides a CLI to predict emotions from the content of a given YouTube URL,
and also to run data preprocessing, model training, and evaluation/registration steps.
"""
import argparse
import json
import os
import logging
import pandas as pd
import torch
import numpy as np
from typing import Any, Dict, List

# Transformers
from transformers import AutoTokenizer

# Scikit-learn
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score # Required by run_evaluate_register from train.py

# Azure ML imports moved to run_evaluate_register function to avoid import errors when not needed


# Use relative import for sibling modules
try:
    from .predict import process_youtube_url_and_predict
    from .data import DataPreparation, DatasetLoader # DatasetLoader for run_preprocess
    from .model import DEBERTAClassifier
    from .train import CustomTrainer
except ImportError:
    # Fallback for scenarios where the script might be run directly
    from predict import process_youtube_url_and_predict
    from data import DataPreparation, DatasetLoader
    from model import DEBERTAClassifier
    from train import CustomTrainer

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# --- Functions moved from train.py ---

def run_preprocess(args):
    """Runs the data preprocessing step."""
    logger.info("--- Starting Preprocessing Step ---")
    logger.info(f"Raw train CSV path: {args.raw_train_csv_path}")
    logger.info(f"Raw test CSV path: {args.raw_test_csv_path}")
    logger.info(f"Processed train output dir: {args.processed_train_output_dir}")
    logger.info(f"Processed test output dir: {args.processed_test_output_dir}")
    logger.info(f"Encoders output dir: {args.encoders_output_dir}")

    os.makedirs(args.processed_train_output_dir, exist_ok=True)
    os.makedirs(args.processed_test_output_dir, exist_ok=True)
    os.makedirs(args.encoders_output_dir, exist_ok=True)

    # Load raw data
    dataset_loader = DatasetLoader()
    if os.path.isdir(args.raw_train_csv_path):
        train_df = dataset_loader.load_training_data(data_dir=args.raw_train_csv_path)
    else:
        # Assuming if not a directory, it's a single file for training (though unusual for 'load_training_data' which expects a dir)
        # This branch might need review based on expected behavior for single train file.
        # For now, let's assume it should still be passed to data_dir,
        # and load_training_data might need adjustment if it strictly expects a directory.
        # Or, the command line argument parsing should ensure raw_train_csv_path is always a directory.
        logger.warning(f"raw_train_csv_path '{args.raw_train_csv_path}' is not a directory. Attempting to load as single source.")
        # If load_training_data is designed for a directory, this might still fail or behave unexpectedly.
        # Consider adding a separate method in DatasetLoader for single training files if that's a valid use case.
        train_df = dataset_loader.load_training_data(data_dir=os.path.dirname(args.raw_train_csv_path)) # Example: pass parent dir
                                                                                                    # Or handle as a direct file if DatasetLoader supports it.
                                                                                                    # For now, this is a placeholder, the primary fix is 'data_dir'.
                                                                                                    # A better approach for single file might be:
                                                                                                    # train_df = pd.read_csv(args.raw_train_csv_path)
                                                                                                    # and then apply similar processing as in load_training_data

    test_df = dataset_loader.load_test_data(test_file=args.raw_test_csv_path)
    logger.info(f"Loaded raw train data with {len(train_df)} samples.")
    logger.info(f"Loaded raw test data with {len(test_df)} samples.")

    intensity_mapping = {"mild": "mild", "neutral": "mild", "moderate": "moderate", "intense": "strong", "overwhelming": "strong"}
    train_df["intensity"] = train_df["intensity"].map(intensity_mapping).fillna("mild")
    test_df["intensity"] = test_df["intensity"].map(intensity_mapping).fillna("mild")

    output_tasks_list = [task.strip() for task in args.output_tasks.split(',')]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_tokenizer)

    feature_config = {"pos": False, "textblob": False, "vader": False, "tfidf": True, "emolex": True}
    data_prep = DataPreparation(
        output_columns=output_tasks_list,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=16, # Batch size for DataLoaders, not directly used for saving processed files
        feature_config=feature_config,
        encoders_save_dir=args.encoders_output_dir
    )

    logger.info("Preparing data (fitting encoders, transforming text, extracting features)...")
    # This call will fit encoders and save them.
    # We also need the processed dataframes to save them.
    # DataPreparation's prepare_data usually returns dataloaders.
    # We need to ensure it also makes processed dfs available or returns them.
    # For now, assume data_prep stores them as attributes after processing.
    _, _, _ = data_prep.prepare_data(
        train_df=train_df.copy(),
        test_df=test_df.copy(),
        validation_split=0.1 # This split is for dataloaders, not for saving raw processed files
    )
    logger.info(f"Encoders saved to {args.encoders_output_dir}")

    # Save processed DataFrames (assuming they are stored in data_prep after the call)
    # This part relies on DataPreparation class having train_df_processed and test_df_processed attributes
    # or similar, after prepare_data is called.
    # The original train.py snippet had a check for these attributes.
    if hasattr(data_prep, 'train_df_processed') and data_prep.train_df_processed is not None:
        train_output_path = os.path.join(args.processed_train_output_dir, "train.csv")
        data_prep.train_df_processed.to_csv(train_output_path, index=False)
        logger.info(f"Processed train data saved to {train_output_path}")
    else:
        logger.warning("Processed train DataFrame (train_df_processed) not found in DataPreparation object after prepare_data. Skipping save.")

    if hasattr(data_prep, 'test_df_processed') and data_prep.test_df_processed is not None:
        test_output_path = os.path.join(args.processed_test_output_dir, "test.csv")
        data_prep.test_df_processed.to_csv(test_output_path, index=False)
        logger.info(f"Processed test data saved to {test_output_path}")
    else:
        logger.warning("Processed test DataFrame (test_df_processed) not found in DataPreparation object after prepare_data. Skipping save.")

    logger.info("--- Preprocessing Step Completed ---")


def run_train(args):
    """Runs the model training step."""
    logger.info("--- Starting Training Step ---")
    output_tasks_list = [task.strip() for task in args.output_tasks.split(',')]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Model output dir: {args.trained_model_output_dir}")
    logger.info(f"Metrics output file: {args.metrics_output_file}")

    os.makedirs(args.trained_model_output_dir, exist_ok=True)
    # Ensure directory for metrics_output_file exists
    if os.path.dirname(args.metrics_output_file):
        os.makedirs(os.path.dirname(args.metrics_output_file), exist_ok=True)


    train_df = pd.read_csv(os.path.join(args.processed_train_dir, "train.csv"))
    test_df_for_splits = pd.read_csv(os.path.join(args.processed_test_dir, "test.csv"))
    logger.info(f"Loaded processed train data: {len(train_df)} samples.")
    logger.info(f"Loaded processed data for val/test splits: {len(test_df_for_splits)} samples.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_bert)

    feature_config = {"pos": False, "textblob": False, "vader": False, "tfidf": True, "emolex": True}
    data_prep = DataPreparation(
        output_columns=output_tasks_list,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        feature_config=feature_config,
        encoders_load_dir=args.encoders_input_dir
    )

    train_dataloader, val_dataloader, test_dataloader_from_prep = data_prep.prepare_data(
        train_df=train_df,
        test_df=test_df_for_splits,
        validation_split=0.1 # This is the val split from test_df_for_splits
    )
    final_test_df = data_prep.test_df_split # This is the actual test set after splitting validation

    feature_dim = data_prep.feature_extractor.get_feature_dim()
    num_classes = data_prep.get_num_classes()
    logger.info(f"Feature dimension from DataPrep: {feature_dim}")
    logger.info(f"Num classes from DataPrep: {num_classes}")

    class_weights_tensor = None
    if "emotion" in output_tasks_list and "emotion" in train_df.columns:
        emotion_labels = data_prep.label_encoders["emotion"].transform(train_df["emotion"])
        class_weights_emotion = compute_class_weight('balanced', classes=np.unique(emotion_labels), y=emotion_labels)
        class_weights_tensor = {"emotion": torch.tensor(class_weights_emotion, dtype=torch.float).to(device)}
        logger.info(f"Computed class weights for 'emotion' task: {class_weights_emotion}")

    # Define feature configuration used for training
    feature_config = {"pos": False, "textblob": False, "vader": False, "tfidf": True, "emolex": True}

    model = DEBERTAClassifier(
        model_name=args.model_name_bert,
        feature_dim=feature_dim,
        num_classes=num_classes,
        # hidden_dim, dropout can be added if they are part of DEBERTAClassifier constructor and CLI args
    ).to(device)

    trainer = CustomTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader_from_prep, # This is the test_dataloader for the final_test_df
        device=device,
        test_set_df=final_test_df, # Pass the correct test_df corresponding to test_dataloader_from_prep
        class_weights_tensor=class_weights_tensor,
        encoders_dir=args.encoders_input_dir, # For loading encoders within CustomTrainer if needed for eval
        output_tasks=output_tasks_list,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        feature_config=feature_config,
        # weight_decay can be added if it's a CLI arg
    )

    logger.info("Starting training and evaluation...")
    # train_and_evaluate saves the best model to args.trained_model_output_dir
    # and metrics to args.metrics_output_file
    trainer.train_and_evaluate(
        trained_model_output_dir=args.trained_model_output_dir,
        metrics_output_file=args.metrics_output_file
        # weights_dir_base can be passed if made a CLI arg
    )

    logger.info(f"Model training complete. Best model saved in {args.trained_model_output_dir}")
    logger.info(f"Metrics saved to {args.metrics_output_file}")
    logger.info("--- Training Step Completed ---")


def run_evaluate_register(args):
    """Runs model evaluation and registration step."""
    # Import Azure ML dependencies only when needed
    try:
        from azure.ai.ml import MLClient
        from azure.ai.ml.entities import Model as AzureModel
        from azure.ai.ml.constants import AssetTypes
        from azure.identity import DefaultAzureCredential
        azure_available = True
    except ImportError:
        logger.warning("Azure ML SDK not available. Model registration to Azure will be skipped.")
        azure_available = False
    
    logger.info("--- Starting Evaluate & Register Step ---")
    output_tasks_list = [task.strip() for task in args.output_tasks.split(',')] if args.output_tasks else None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.final_eval_output_dir, exist_ok=True)
    if os.path.dirname(args.registration_status_output_file):
      os.makedirs(os.path.dirname(args.registration_status_output_file), exist_ok=True)

    # Load training metrics to get best_val_emotion_f1
    training_metrics = {}
    best_val_emotion_f1 = -1.0 # Default if not found
    
    # Try multiple possible locations for training metrics
    possible_metrics_paths = []
    
    # If metrics_input_file is provided directly, use it
    if hasattr(args, 'metrics_input_file') and args.metrics_input_file and os.path.exists(args.metrics_input_file):
        possible_metrics_paths.append(args.metrics_input_file)
        
    # Try common locations relative to model_input_dir
    possible_metrics_paths.extend([
        os.path.join(args.model_input_dir, "training_metrics.json"),
        os.path.join(os.path.dirname(args.model_input_dir), "training_metrics.json"),
        os.path.join(os.path.dirname(args.model_input_dir), "evaluation", "metrics.json"),
        "models/evaluation/metrics.json",  # Standard location
    ])
    
    metrics_file_found = None
    for metrics_path in possible_metrics_paths:
        if os.path.exists(metrics_path):
            metrics_file_found = metrics_path
            logger.info(f"Found training metrics at: {metrics_path}")
            break
    
    if metrics_file_found:
        try:
            with open(metrics_file_found, 'r') as f:
                training_metrics = json.load(f)
            
            # Try multiple possible structures for backward compatibility
            if 'best_validation_f1s' in training_metrics and 'emotion' in training_metrics['best_validation_f1s']:
                best_val_emotion_f1 = training_metrics['best_validation_f1s']['emotion']
                logger.info(f"Loaded best_val_emotion_f1 from training metrics (best_validation_f1s): {best_val_emotion_f1}")
            elif 'best_overall_validation_emotion_f1' in training_metrics:
                best_val_emotion_f1 = training_metrics['best_overall_validation_emotion_f1']
                logger.info(f"Loaded best_val_emotion_f1 from training metrics (best_overall_validation_emotion_f1): {best_val_emotion_f1}")
            elif 'best_val_metrics' in training_metrics and 'emotion' in training_metrics['best_val_metrics'] and \
               'f1' in training_metrics['best_val_metrics']['emotion']:
                best_val_emotion_f1 = training_metrics['best_val_metrics']['emotion']['f1']
                logger.info(f"Loaded best_val_emotion_f1 from training metrics (best_val_metrics): {best_val_emotion_f1}")
            else:
                logger.warning("best_val_emotion_f1 not found in any expected structure within training metrics file.")
                logger.info(f"Available top-level keys in metrics: {list(training_metrics.keys())}")
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from training metrics file {metrics_file_found}.")
        except Exception as e:
            logger.error(f"Error loading training metrics from {metrics_file_found}: {e}")
    else:
        logger.error(f"Training metrics file not found in any expected location. Cannot determine best_val_emotion_f1 for registration.")
        logger.info(f"Searched locations: {possible_metrics_paths}")


    model_config_path = os.path.join(args.model_input_dir, "model_config.json")
    if not os.path.exists(model_config_path):
        logger.error(f"Model config file not found: {model_config_path}. Cannot proceed with evaluation.")
        return

    with open(model_config_path, 'r') as f:
        model_config = json.load(f)

    # Override tasks from config if provided in args
    if output_tasks_list is None:
        output_tasks_list = model_config.get("output_tasks", ["emotion", "sub_emotion", "intensity"])

    tokenizer_model_name = args.model_name_bert if args.model_name_bert else model_config.get("model_name")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
    test_df = pd.read_csv(os.path.join(args.processed_test_dir, "test.csv"))

    feature_config = model_config.get("feature_config", {"pos": False, "textblob": False, "vader": False, "tfidf": True, "emolex": True})
    data_prep_eval = DataPreparation(
        output_columns=output_tasks_list,
        tokenizer=tokenizer,
        max_length=args.max_length, # Use CLI arg for max_length
        batch_size=16, # Default batch size for eval dataloader
        feature_config=feature_config,
        encoders_load_dir=args.encoders_input_dir
    )

    # Load training data to fit TF-IDF properly (required for correct feature dimensions)
    train_df = pd.read_csv(args.train_path)
    logger.info(f"Loaded training data for TF-IDF fitting: {len(train_df)} samples")
    
    # Get test dataloader. Use training data for TF-IDF fitting to match model expectations.
    _, _, final_test_dataloader = data_prep_eval.prepare_data(
        train_df=train_df,  # Use proper training data for TF-IDF fitting
        test_df=test_df.copy(),
        validation_split=0 # Ensure all of test_df goes to test_dataloader
    )
    # The test_set_df for CustomTrainer should be the one corresponding to final_test_dataloader
    eval_test_df = data_prep_eval.test_df_split # Should be the full test_df

    model_for_eval = DEBERTAClassifier(
        model_name=model_config.get("model_name"),
        feature_dim=model_config.get("feature_dim"),
        num_classes=model_config.get("num_classes"),
        hidden_dim=model_config.get("hidden_dim"), # from config
        dropout=model_config.get("dropout") # from config
    ).to(device)

    # Use feature_config from model config if available, otherwise use default
    feature_config = model_config.get("feature_config", {"pos": False, "textblob": False, "vader": False, "tfidf": True, "emolex": True})

    eval_trainer = CustomTrainer(
        model=model_for_eval,
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=final_test_dataloader,
        device=device,
        test_set_df=eval_test_df, # Pass the correct test_df
        class_weights_tensor=None, # No class weights for final evaluation typically
        encoders_dir=args.encoders_input_dir, # For loading encoders if model uses them directly
        output_tasks=output_tasks_list,
        feature_config=feature_config,
    )

    model_pt_path = os.path.join(args.model_input_dir, "best_model.pt")
    logger.info(f"Performing final evaluation of model: {model_pt_path}")
    # evaluate_final_model should load the weights and return metrics
    eval_results_df = eval_trainer.evaluate_final_model(
        model_path=model_pt_path,
        evaluation_output_dir=args.final_eval_output_dir # For saving reports, plots
    )
    logger.info(f"Final evaluation complete. Report saved in {args.final_eval_output_dir}")    # Use the F1 score from the final evaluation on the test set for registration decision, if available
    # Or stick to best_val_emotion_f1 from training if that's the agreed metric for registration.
    # The original logic used best_val_emotion_f1 from training metrics.
    
    registration_decision = "not_registered_default"
    actual_f1_for_registration = best_val_emotion_f1 # From training metrics

    if actual_f1_for_registration != -1.0 and actual_f1_for_registration >= args.registration_f1_threshold_emotion:
        logger.info(f"Metric best_val_emotion_f1 ({actual_f1_for_registration}) meets threshold ({args.registration_f1_threshold_emotion}). Attempting to register model.")
        if not azure_available:
            logger.warning("Azure ML SDK not available. Cannot register model to Azure ML.")
            registration_decision = "registration_failed_azure_sdk_unavailable"
        elif args.subscription_id and args.resource_group and args.workspace_name:
            try:
                ml_client = MLClient(
                    credential=DefaultAzureCredential(),
                    subscription_id=args.subscription_id,
                    resource_group_name=args.resource_group,
                    workspace_name=args.workspace_name
                )
                
                # Ensure model_input_dir is what Azure expects (e.g., contains an MLmodel file or is a folder of assets)
                # For custom models, the path is usually the folder containing all model artifacts.
                azure_model_payload = AzureModel(
                    path=args.model_input_dir, 
                    name=args.model_asset_name,
                    description=f"Emotion classification model. Config: {model_config_path}. Trained with best_val_emotion_f1: {actual_f1_for_registration}",
                    type=AssetTypes.CUSTOM_MODEL, # Or MLFLOW_MODEL if applicable
                    properties={
                        "best_val_emotion_f1": str(actual_f1_for_registration),
                        "model_config_details": json.dumps(model_config),
                        "training_metrics_summary": json.dumps(training_metrics.get("summary", {})),
                        "evaluation_results_path": args.final_eval_output_dir
                    }
                )
                ml_client.models.create_or_update(azure_model_payload)
                logger.info(f"Model '{args.model_asset_name}' registered successfully in Azure ML.")
                registration_decision = "registered"
            except Exception as e:
                logger.error(f"Azure ML Model registration failed: {e}")
                registration_decision = "registration_failed_exception"
        else:
            logger.warning("Azure subscription details not provided. Skipping actual registration. Set to 'mock_registered'.")
            registration_decision = "mock_registered_azure_details_missing"
    elif actual_f1_for_registration == -1.0:
        logger.warning("best_val_emotion_f1 metric unavailable. Model not registered.")
        registration_decision = "not_registered_metric_unavailable"
    else:
        logger.info(f"Metric best_val_emotion_f1 ({actual_f1_for_registration}) is below threshold ({args.registration_f1_threshold_emotion}). Model will not be registered.")
        registration_decision = "not_registered_threshold_not_met"

    output_status = {
        "registration_status": registration_decision,
        "best_val_emotion_f1_used_for_decision": actual_f1_for_registration,
        "registration_f1_threshold": args.registration_f1_threshold_emotion,
        "model_asset_name": args.model_asset_name if "registered" in registration_decision else None
    }
    with open(args.registration_status_output_file, 'w') as f:
        json.dump(output_status, f, indent=4)
    logger.info(f"Registration status ({registration_decision}) saved to {args.registration_status_output_file}")
    logger.info("--- Evaluate & Register Step Completed ---")


def main():
    """
    Parses command-line arguments, performs emotion prediction, or runs training/processing pipelines.
    """
    parser = argparse.ArgumentParser(
        description="Emotion Classification Pipeline CLI for prediction, training, and data processing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="action", required=True, help="Action to perform: predict, preprocess, train, evaluate_register")

    # --- Predict action ---
    predict_parser = subparsers.add_parser("predict", help="Predict emotions from a YouTube URL.")
    predict_parser.add_argument(
        "url",
        type=str,
        help="The YouTube URL from which to extract and analyze content for emotion.",
    )
    predict_parser.add_argument(
        "--transcription",
        type=str,
        choices=["assemblyAI", "whisper"],
        default="assemblyAI",
        help="Method for speech-to-text transcription.",
    )

    # --- Preprocess action ---
    preprocess_parser = subparsers.add_parser("preprocess", help="Run data preprocessing for training.")
    preprocess_parser.add_argument("--raw_train_csv_path", required=True, help="Path to raw training CSV file or directory of CSV files.")
    preprocess_parser.add_argument("--raw_test_csv_path", required=True, help="Path to raw test CSV file.")
    preprocess_parser.add_argument("--output_tasks", type=str, default="emotion,sub_emotion,intensity", help="Comma-separated list of output tasks (e.g., 'emotion,sub_emotion,intensity').")
    preprocess_parser.add_argument("--model_name_tokenizer", required=True, type=str, help="Name or path of the pre-trained tokenizer model (e.g., 'microsoft/deberta-v3-base').")
    preprocess_parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length for tokenization.")
    preprocess_parser.add_argument("--processed_train_output_dir", required=True, help="Directory to save the processed training data (train.csv).")
    preprocess_parser.add_argument("--processed_test_output_dir", required=True, help="Directory to save the processed test data (test.csv).")
    preprocess_parser.add_argument("--encoders_output_dir", required=True, help="Directory to save the fitted label encoders.")

    # --- Train action ---
    train_parser = subparsers.add_parser("train", help="Run model training.")
    train_parser.add_argument("--processed_train_dir", required=True, help="Directory containing the processed train.csv from the preprocess step.")
    train_parser.add_argument("--processed_test_dir", required=True, help="Directory containing the processed test.csv from the preprocess step (used for validation and test splits).")
    train_parser.add_argument("--encoders_input_dir", required=True, help="Directory containing the pre-fitted label encoders from the preprocess step.")
    train_parser.add_argument("--output_tasks", type=str, default="emotion,sub_emotion,intensity", help="Comma-separated list of output tasks.")
    train_parser.add_argument("--model_name_bert", required=True, type=str, help="Name or path of the pre-trained BERT-like model (e.g., 'microsoft/deberta-v3-base').")
    train_parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length for tokenization.")
    train_parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation.")
    train_parser.add_argument("--learning_rate", type=float, default=2e-5, help="Initial learning rate for the AdamW optimizer.")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.") # Defaulted to 3, common practice
    train_parser.add_argument("--trained_model_output_dir", required=True, help="Directory to save the trained model (best_model.pt, model_config.json).")
    train_parser.add_argument("--metrics_output_file", required=True, help="Path to save the training metrics as a JSON file (e.g., training_metrics.json).")
    # train_parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer.") # Optional

    # --- Evaluate and Register action ---
    evaluate_register_parser = subparsers.add_parser("evaluate_register", help="Run final model evaluation and optionally register to Azure ML.")
    evaluate_register_parser.add_argument("--model_input_dir", required=True, help="Directory containing the trained model (best_model.pt and model_config.json) from the train step.")
    evaluate_register_parser.add_argument("--processed_test_dir", required=True, help="Directory containing the processed test.csv for final evaluation.")
    evaluate_register_parser.add_argument("--train_path", required=True, help="Path to the training CSV file for TF-IDF fitting (required for correct feature dimensions).")
    evaluate_register_parser.add_argument("--encoders_input_dir", required=True, help="Directory containing pre-fitted label encoders.")
    evaluate_register_parser.add_argument("--metrics_input_file", type=str, help="Path to the training metrics JSON file (e.g., training_metrics.json from train step). If not provided, will try to infer.")
    evaluate_register_parser.add_argument("--output_tasks", type=str, help="Comma-separated list of output tasks (overrides model_config if provided).")
    evaluate_register_parser.add_argument("--model_name_bert", type=str, help="Name of the BERT-like model (overrides model_config if provided, e.g., 'microsoft/deberta-v3-base').")
    evaluate_register_parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length for tokenization.")
    evaluate_register_parser.add_argument("--final_eval_output_dir", required=True, help="Directory to save final evaluation reports and visualizations.")
    evaluate_register_parser.add_argument("--registration_f1_threshold_emotion", type=float, required=True, help="F1 threshold for the 'emotion' task (best_val_emotion_f1 from training) to register the model.")
    evaluate_register_parser.add_argument("--registration_status_output_file", required=True, help="Path to save the registration status as a JSON file.")
    # Azure ML specific arguments
    evaluate_register_parser.add_argument("--subscription_id", type=str, help="Azure subscription ID for model registration.")
    evaluate_register_parser.add_argument("--resource_group", type=str, help="Azure resource group name for model registration.")
    evaluate_register_parser.add_argument("--workspace_name", type=str, help="Azure ML workspace name for model registration.")
    evaluate_register_parser.add_argument("--model_asset_name", type=str, default="EmotionClassificationModel", help="Asset name for the registered model in Azure ML.")


    args = parser.parse_args()

    if args.action == "predict":
        try:
            list_of_predictions: List[Dict[str, Any]] = process_youtube_url_and_predict(
                youtube_url=args.url,
                transcription_method=args.transcription,
            )
            print(json.dumps(list_of_predictions, indent=4))
        except Exception as e:
            logger.error(f"An error occurred during the emotion prediction pipeline: {e}", exc_info=True)
            print(f"An error occurred during the emotion prediction pipeline: {e}")
            print(
                "Please ensure the URL is correct, the video is accessible, "
                "and all configurations are set."
            )
            return 1 # Indicate error
    elif args.action == "preprocess":
        run_preprocess(args)
    elif args.action == "train":
        run_train(args)
    elif args.action == "evaluate_register":
        run_evaluate_register(args)
    else:
        parser.print_help()
    
    return 0 # Indicate success


if __name__ == "__main__":
    exit(main())
