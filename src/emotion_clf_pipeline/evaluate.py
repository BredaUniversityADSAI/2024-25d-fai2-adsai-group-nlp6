import argparse
import json
import logging
import os
from pathlib import Path

import mlflow
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.metrics import classification_report
from transformers import AutoTokenizer

from .azure_pipeline import get_ml_client
from .model import DEBERTAClassifier  # Assuming this is your model class
from .data import DataPreparation  # Assuming this is your data prep class

logger = logging.getLogger(__name__)


def evaluate_and_register(args):
    """
    Evaluates a trained model and registers it with Azure ML if it meets
    the F1-score threshold.
    """
    load_dotenv()
    logger.info("Starting model evaluation and registration process...")

    try:
        # --- 1. Load Dependencies ---
        logger.info("Loading model, data, and encoders...")

        # Load test data
        if not os.path.exists(args.processed_test_path):
            raise FileNotFoundError(
                f"Test data not found at: {args.processed_test_path}"
            )
        test_df = pd.read_csv(args.processed_test_path)
        logger.info(f"Loaded {len(test_df)} test samples.")

        # Load model configuration and weights
        output_tasks = ["emotion", "sub_emotion", "intensity"]
        model_path = Path(args.model_input_dir)
        
        # Load model configuration from the saved config file
        model_config_path = model_path / "model_config.json"
        if not model_config_path.exists():
            raise FileNotFoundError(f"Model config not found at: {model_config_path}")
        
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)
        
        logger.info(f"Loaded model config: {model_config}")
        
        # Extract configuration parameters
        model_name = model_config["model_name"]
        feature_dim = model_config["feature_dim"]
        num_classes = model_config["num_classes"]
        hidden_dim = model_config["hidden_dim"]
        dropout = model_config["dropout"]
        
        # Load tokenizer from the original pretrained model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Loaded tokenizer from pretrained model: {model_name}")
        
        # Initialize the model with the correct architecture
        model = DEBERTAClassifier(
            model_name=model_name,
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Load the trained weights
        weights_path = model_path / "dynamic_weights.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found at: {weights_path}")
        
        # Load state dict and handle potential key remapping (bert -> deberta)
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # Create a new state_dict with corrected keys if needed
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("bert."):
                new_key = "deberta." + k[len("bert."):]
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        
        # Load the state dict into the model
        model.load_state_dict(new_state_dict)
        logger.info(f"Loaded model weights from: {weights_path}")

        # --- 2. Prepare Data for Evaluation ---
        # We need a DataPreparation object to create the test dataloader
        data_prep = DataPreparation(
            output_columns=output_tasks,
            tokenizer=tokenizer,
            max_length=256,  # This should ideally be passed as an arg
            batch_size=args.batch_size,
            feature_config={
                "pos": False, "textblob": False, "vader": False,
                "tfidf": True, "emolex": True
            },
            encoders_load_dir=args.encoders_dir  # Load encoders from directory
        )
        _, _, test_dataloader = data_prep.prepare_data(
            train_df=None, test_df=test_df, validation_split=0
        )
        logger.info("Prepared test dataloader.")

        # --- 3. Run Evaluation ---
        logger.info("Running evaluation on the test set...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        all_preds = {task: [] for task in output_tasks}
        all_labels = {task: [] for task in output_tasks}

        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)

                for task in output_tasks:
                    labels = batch[task].to(device)
                    preds = torch.argmax(outputs[task], dim=1)
                    all_preds[task].extend(preds.cpu().numpy())
                    all_labels[task].extend(labels.cpu().numpy())
        
        logger.info("Evaluation complete.")

        # --- 4. Calculate and Save Metrics ---
        os.makedirs(args.final_eval_output_dir, exist_ok=True)
        metrics_summary = {}

        for task in output_tasks:
            report = classification_report(
                all_labels[task],
                all_preds[task],
                target_names=data_prep.label_encoders[task].classes_,
                output_dict=True,
                zero_division=0
            )
            metrics_summary[task] = report
            logger.info(f"\nClassification Report for '{task}':\n"
                        f"{json.dumps(report, indent=2)}")

        metrics_file = os.path.join(
            args.final_eval_output_dir, "evaluation_metrics.json"
        )
        with open(metrics_file, 'w') as f:
            json.dump(metrics_summary, f, indent=4)
        logger.info(f"Evaluation metrics saved to {metrics_file}")
        
        # --- 5. Register Model if Threshold is Met ---
        primary_metric = "f1-score"
        primary_task = "emotion"
        avg_type = "weighted avg"
        f1_score = metrics_summary.get(primary_task, {}).get(
            avg_type, {}
        ).get(primary_metric, 0.0)
        
        logger.info(
            f"F1-score for primary task '{primary_task}' ({avg_type}): "
            f"{f1_score:.4f}"
        )
        logger.info(f"Registration F1 threshold: {args.registration_f1_threshold}")

        registration_status = {
            "status": "completed",
            "f1_score": f1_score,
            "threshold": args.registration_f1_threshold
        }

        if f1_score >= args.registration_f1_threshold:
            logger.info("F1 score meets threshold. Registering model...")
            try:
                ml_client = get_ml_client()
                workspace = ml_client.workspaces.get(ml_client.workspace_name)
                mlflow.set_tracking_uri(workspace.mlflow_tracking_uri)
                
                # Register the model using the MLFlow model format
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path="model",  # folder name in artifact store
                    registered_model_name="emotion-classifier-deberta",  # registry name
                    pip_requirements=["torch", "transformers", "scikit-learn"]
                )
                logger.info("Model registered successfully in Azure ML!")
                registration_status["registered"] = True
                registration_status["reason"] = "F1 score met threshold."

            except Exception as e:
                logger.error(f"Model registration failed: {e}", exc_info=True)
                registration_status["registered"] = False
                registration_status["reason"] = (
                    f"Registration failed with exception: {e}"
                )
        else:
            logger.warning(
                "F1 score below threshold. Model will not be registered."
            )
            registration_status["registered"] = False
            registration_status["reason"] = "F1 score did not meet threshold."

        # Write final status file for the pipeline
        with open(args.registration_status_output_file, 'w') as f:
            json.dump(registration_status, f, indent=4)

    except Exception as e:
        logger.error(f"Evaluation and registration failed: {e}", exc_info=True)
        # Write a failure status
        with open(args.registration_status_output_file, 'w') as f:
            json.dump({"status": "failed", "error": str(e)}, f, indent=4)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # This should match add_evaluate_register_args in cli.py
    # and args passed from pipeline
    parser.add_argument("--model-input-dir", type=str, required=True)
    parser.add_argument("--processed-test-path", type=str, required=True)
    parser.add_argument("--encoders-dir", type=str, required=True)
    parser.add_argument(
        "--final-eval-output-dir", type=str, default="results/evaluation"
    )
    parser.add_argument("--registration-f1-threshold", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--registration-status-output-file", type=str, required=True)
    
    cli_args = parser.parse_args()
    evaluate_and_register(cli_args)
