import argparse
import json
import logging
import os
import pickle
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoConfig

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
            raise FileNotFoundError(f"Test data not found at: {args.processed_test_path}")
        test_df = pd.read_csv(args.processed_test_path)
        logger.info(f"Loaded {len(test_df)} test samples.")

        # Load encoders
        output_tasks = ["emotion", "sub_emotion", "intensity"]
        label_encoders = {}
        for task in output_tasks:
            encoder_path = os.path.join(args.encoders_dir, f"{task}_encoder.pkl")
            if not os.path.exists(encoder_path):
                raise FileNotFoundError(f"Encoder not found for task '{task}' at: {encoder_path}")
            with open(encoder_path, "rb") as f:
                label_encoders[task] = pickle.load(f)
        logger.info("Loaded label encoders for all tasks.")

        # Load model and tokenizer from the directory provided by the pipeline
        model_path = Path(args.model_input_dir)
        
        # Load config first to get model parameters
        config = AutoConfig.from_pretrained(model_path)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Create model with config
        model = DEBERTAClassifier(
            model_name=config._name_or_path,
            feature_dim=config.feature_dim,
            num_classes=config.num_classes,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        
        # Load state dict
        model.load_state_dict(torch.load(model_path / 'pytorch_model.bin', map_location='cpu'))
        logger.info(f"Loaded model and tokenizer from: {model_path}")

        # --- 2. Prepare Data for Evaluation ---
        # We need a DataPreparation object to create the test dataloader
        data_prep = DataPreparation(
            output_columns=output_tasks,
            tokenizer=tokenizer,
            max_length=256, # This should ideally be passed as an arg
            batch_size=args.batch_size,
            feature_config={"pos": False, "textblob": False, "vader": False, "tfidf": True, "emolex": True},
            encoders_save_dir=args.encoders_dir,
            label_encoders=label_encoders # Pass the loaded encoders
        )
        _ , _, test_dataloader = data_prep.prepare_data(train_df=None, test_df=test_df, validation_split=0)
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
                target_names=label_encoders[task].classes_,
                output_dict=True,
                zero_division=0
            )
            metrics_summary[task] = report
            logger.info(f"\nClassification Report for '{task}':\n"
                        f"{json.dumps(report, indent=2)}")

        metrics_file = os.path.join(args.final_eval_output_dir, "evaluation_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics_summary, f, indent=4)
        logger.info(f"Evaluation metrics saved to {metrics_file}")
        
        # --- 5. Register Model if Threshold is Met ---
        primary_metric = "f1-score"
        primary_task = "emotion"
        avg_type = "weighted avg"
        f1_score = metrics_summary.get(primary_task, {}).get(avg_type, {}).get(primary_metric, 0.0)
        
        logger.info(f"F1-score for primary task '{primary_task}' ({avg_type}): {f1_score:.4f}")
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
                mlflow.set_tracking_uri(ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri)
                
                # Register the model using the MLFlow model format
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path="model", # a name for the folder in the artifact store
                    registered_model_name="emotion-classifier-deberta", # name in model registry
                    pip_requirements=["torch", "transformers", "scikit-learn"]
                )
                logger.info("Model registered successfully in Azure ML!")
                registration_status["registered"] = True
                registration_status["reason"] = "F1 score met threshold."

            except Exception as e:
                logger.error(f"Model registration failed: {e}", exc_info=True)
                registration_status["registered"] = False
                registration_status["reason"] = f"Registration failed with exception: {e}"
        else:
            logger.warning("F1 score below threshold. Model will not be registered.")
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
    # This should match add_evaluate_register_args in cli.py and args passed from pipeline
    parser.add_argument("--model-input-dir", type=str, required=True)
    parser.add_argument("--processed-test-path", type=str, required=True)
    parser.add_argument("--encoders-dir", type=str, required=True)
    parser.add_argument("--final-eval-output-dir", type=str, default="results/evaluation")
    parser.add_argument("--registration-f1-threshold", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--registration-status-output-file", type=str, required=True)
    
    cli_args = parser.parse_args()
    evaluate_and_register(cli_args) 