"""
Custom trainer class for DeBERTa-based emotion classification model.
This module provides a comprehensive training and evaluation framework for
multi-task emotion classification, including emotion, sub-emotion, and intensity
prediction with flexible output options.
It also includes CLI actions for Azure ML Pipelines.
"""

import json
import logging
import os
import pickle
import shutil
import sys
import time
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import mlflow
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tabulate import tabulate
from termcolor import colored
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

# Import the local modules
from .data import DataPreparation, DatasetLoader, FeatureExtractor  # Added FeatureExtractor
from .model import DEBERTAClassifier

# Azure ML specific imports for model registration
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model as AzureModel  # Renamed to avoid conflict
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class CustomTrainer:
    """
    A custom trainer class for BERT-based emotion classification model
    with flexible outputs.
    """

    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        device,
        test_set_df,  # Changed from test_set to test_set_df (DataFrame)
        class_weights_tensor,
        encoders_dir,  # Directory to load encoders
        output_tasks=None,
        learning_rate=2e-5,
        weight_decay=0.01,
        epochs=1,
        feature_config=None,  # Add feature_config parameter
    ):
        """
        Initialize the CustomTrainer.
        Args:
            test_set_df (pd.DataFrame): Test dataframe.
            encoders_dir (str): Directory containing encoder pickle files.
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.test_set_df = test_set_df  # Store the DataFrame
        self.class_weights_tensor = class_weights_tensor
        self.output_tasks = output_tasks or ["emotion", "sub_emotion", "intensity"]
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.feature_config = feature_config or {"pos": False, "textblob": False, "vader": False, "tfidf": True, "emolex": True}

        self._load_encoders(encoders_dir)

        # Feature dim determination might need adjustment if train_dataloader is None during init for eval
        if self.train_dataloader:
            self.feature_dim = self._get_feature_dim()
        else:
            # If no train_dataloader (e.g. during eval-only), try to get from model config or test_dataloader
            # This part might need a more robust way if model's feature_dim isn't directly accessible
            # or if test_dataloader isn't guaranteed to be similar.
            # For now, assuming model has feature_dim or it's passed/set differently for eval.
            if hasattr(model, "feature_dim"):
                self.feature_dim = model.feature_dim
            elif self.test_dataloader:  # Fallback to test_dataloader
                first_batch = next(iter(self.test_dataloader))
                if "features" in first_batch:
                    self.feature_dim = first_batch["features"].shape[-1]
                else:
                    logger.warning(
                        "Cannot determine feature_dim for evaluation without train_dataloader or model.feature_dim or features in test_dataloader."
                    )
                    self.feature_dim = 0  # Placeholder
            else:
                logger.warning(
                    "Cannot determine feature_dim for evaluation without train_dataloader or model.feature_dim."
                )
                self.feature_dim = 0  # Placeholder

        self.task_weights = {
            "emotion": 1.0 if "emotion" in self.output_tasks else 0.0,
            "sub_emotion": 0.8 if "sub_emotion" in self.output_tasks else 0.0,
            "intensity": 0.2 if "intensity" in self.output_tasks else 0.0,
        }
        logger.info(f"CustomTrainer initialized with tasks: {self.output_tasks}, device: {self.device}")
        logger.info(f"Encoders loaded from: {encoders_dir}")
        if self.feature_dim > 0:
            logger.info(f"Feature dimension: {self.feature_dim}")

        self._validate_model_dimensions()

    def _get_feature_dim(self):
        """Determine feature dimension from the first batch of training data."""
        if not self.train_dataloader:
            logger.error("Train dataloader is not available to determine feature dimension.")
            # This should ideally not happen if called appropriately.
            # Fallback or raise error. For now, returning a placeholder.
            return 0  # Or raise an error
        try:
            first_batch = next(iter(self.train_dataloader))
            if "features" not in first_batch or first_batch["features"] is None:
                logger.warning(
                    "'features' key not found or is None in the first batch of train_dataloader. Assuming 0 feature_dim or check data prep."
                )
                return 0
            feature_dim = first_batch["features"].shape[-1]
            return feature_dim
        except Exception as e:
            logger.error(f"Error getting feature dimension from train_dataloader: {e}")
            return 0  # Or raise

    def _load_encoders(self, encoders_dir):
        """Load label encoders from pickle files."""
        logger.info(f"Loading encoders from {encoders_dir}")
        try:
            if "emotion" in self.output_tasks:
                with open(os.path.join(encoders_dir, "emotion_encoder.pkl"), "rb") as f:
                    self.emotion_encoder = pickle.load(f)
                logger.info("Loaded emotion_encoder.pkl")
            if "sub_emotion" in self.output_tasks:
                with open(os.path.join(encoders_dir, "sub_emotion_encoder.pkl"), "rb") as f:
                    self.sub_emotion_encoder = pickle.load(f)
                logger.info("Loaded sub_emotion_encoder.pkl")
            if "intensity" in self.output_tasks:
                with open(os.path.join(encoders_dir, "intensity_encoder.pkl"), "rb") as f:
                    self.intensity_encoder = pickle.load(f)
                logger.info("Loaded intensity_encoder.pkl")
        except FileNotFoundError as e:
            logger.error(f"Encoder file not found in {encoders_dir}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading encoders: {e}")
            raise

    def _validate_model_dimensions(self):
        """Validate that model output dimensions match encoder classes."""
        logger.info("Validating model dimensions against encoders...")
        
        if hasattr(self, 'emotion_encoder') and hasattr(self.model, 'num_classes') and "emotion" in self.model.num_classes:
            expected_emotion_classes = len(self.emotion_encoder.classes_)
            if self.model.num_classes["emotion"] != expected_emotion_classes:
                logger.warning(
                    f"Emotion dimension mismatch: Model expects {self.model.num_classes['emotion']} "
                    f"classes but encoder has {expected_emotion_classes} classes"
                )
                # Update model to match encoder
                self.model.num_classes["emotion"] = expected_emotion_classes
                logger.info(f"Updated emotion classes to {expected_emotion_classes}")
        
        if hasattr(self, 'sub_emotion_encoder') and hasattr(self.model, 'num_classes') and "sub_emotion" in self.model.num_classes:
            expected_sub_emotion_classes = len(self.sub_emotion_encoder.classes_)
            if self.model.num_classes["sub_emotion"] != expected_sub_emotion_classes:
                logger.warning(
                    f"Sub-emotion dimension mismatch: Model expects {self.model.num_classes['sub_emotion']} "
                    f"classes but encoder has {expected_sub_emotion_classes} classes"
                )
                # Update model to match encoder
                self.model.num_classes["sub_emotion"] = expected_sub_emotion_classes
                logger.info(f"Updated sub-emotion classes to {expected_sub_emotion_classes}")
        
        if hasattr(self, 'intensity_encoder') and hasattr(self.model, 'num_classes') and "intensity" in self.model.num_classes:
            expected_intensity_classes = len(self.intensity_encoder.classes_)
            if self.model.num_classes["intensity"] != expected_intensity_classes:
                logger.warning(
                    f"Intensity dimension mismatch: Model expects {self.model.num_classes['intensity']} "
                    f"classes but encoder has {expected_intensity_classes} classes"
                )
                # Update model to match encoder
                self.model.num_classes["intensity"] = expected_intensity_classes
                logger.info(f"Updated intensity classes to {expected_intensity_classes}")

    def setup_training(self):
        """
        Set up training components including loss function, optimizer,
        and learning rate scheduler.
        """
        criterion_dict = {}
        if "emotion" in self.output_tasks:
            actual_emotion_weights = None
            if self.class_weights_tensor is not None:
                if isinstance(self.class_weights_tensor, dict):
                    # If class_weights_tensor is a dictionary, get the tensor for "emotion"
                    tensor_for_emotion = self.class_weights_tensor.get("emotion")
                    if tensor_for_emotion is not None and hasattr(tensor_for_emotion, 'to'):
                        actual_emotion_weights = tensor_for_emotion.to(self.device)
                elif hasattr(self.class_weights_tensor, 'to'): 
                    # If class_weights_tensor is directly a tensor
                    actual_emotion_weights = self.class_weights_tensor.to(self.device)
                # If self.class_weights_tensor is None or an unexpected type, actual_emotion_weights remains None
            criterion_dict["emotion"] = nn.CrossEntropyLoss(weight=actual_emotion_weights)
        if "sub_emotion" in self.output_tasks:
            criterion_dict["sub_emotion"] = nn.CrossEntropyLoss()
        if "intensity" in self.output_tasks:
            criterion_dict["intensity"] = nn.CrossEntropyLoss()

        optimizer = AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        total_steps = len(self.train_dataloader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps
        )
        logger.info("Training setup complete: criterion, optimizer, scheduler initialized.")
        return criterion_dict, optimizer, scheduler

    def train_epoch(self, criterion_dict, optimizer, scheduler):
        """Train the model for one epoch."""
        self.model.train()
        train_loss = 0
        all_preds_train = {task: [] for task in self.output_tasks}
        all_labels_train = {task: [] for task in self.output_tasks}

        for batch_idx, batch in enumerate(tqdm(self.train_dataloader, desc="Training", ncols=120, colour="green")):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            features = batch.get("features") # Use .get() for safety
            if features is not None and self.feature_dim > 0:
                features = features.to(self.device)
            else:
                features = None # Ensure features is None if not used or not present

            outputs = self.model(input_ids, attention_mask=attention_mask, features=features)

            # Prepare labels for each task
            labels = {}
            # Correctly access labels from batch keys directly
            for task in self.output_tasks:
                task_label_key = f"{task}_label" # Construct the task-specific label key
                if task_label_key in batch:
                    labels[task] = batch[task_label_key].to(self.device)
                    # Collect labels for metrics calculation
                    all_labels_train[task].extend(batch[task_label_key].cpu().numpy())
                else:
                    logger.error(f"Task label key '{task_label_key}' not found in batch. Available keys: {list(batch.keys())}")
                    continue
            
            # Collect predictions for metrics calculation
            for task in self.output_tasks:
                if isinstance(outputs, dict) and task in outputs and outputs[task] is not None:
                    preds = torch.argmax(outputs[task], dim=1).cpu().numpy()
                    all_preds_train[task].extend(preds)
                elif not (isinstance(outputs, dict) and task in outputs):
                    logger.warning(f"Task '{task}' not in model outputs during training metrics collection or outputs is not a dict.")


            # Calculate loss for each task
            current_loss = torch.tensor(0.0, device=self.device, requires_grad=True) # Initialize as a tensor
            valid_task_loss_calculated = False
            for task in self.output_tasks:
                if (isinstance(outputs, dict) and task in outputs and
                    isinstance(labels, dict) and task in labels):
                    # Ensure outputs[task] and labels[task] are not None before passing to criterion
                    if outputs[task] is not None and labels[task] is not None:
                        task_loss = criterion_dict[task](outputs[task], labels[task])
                        current_loss = current_loss + (self.task_weights[task] * task_loss) 
                        valid_task_loss_calculated = True
            
            if valid_task_loss_calculated: 
                current_loss.backward() # Use current_loss for backward pass
                optimizer.step()
                scheduler.step()
                train_loss += current_loss.item()
            else:
                logger.warning("No valid task loss calculated for a batch. Skipping backward pass.")

        avg_train_loss = train_loss / len(self.train_dataloader) if len(self.train_dataloader) > 0 else 0
        logger.debug(f"Epoch training loss: {avg_train_loss}")

        # Calculate training metrics for the epoch
        train_metrics_epoch = {}
        for task in self.output_tasks:
            if all_labels_train[task] and all_preds_train[task]:
                train_metrics_epoch[task] = self.calculate_metrics(
                    all_preds_train[task], all_labels_train[task], task_name=f"Train {task}"
                )
            else:
                logger.warning(f"No training data/predictions collected for task '{task}' in epoch. Metrics will be zero.")
                train_metrics_epoch[task] = {"acc": 0, "f1": 0, "prec": 0, "rec": 0, "report": "No data for training metrics"}
        
        return avg_train_loss, train_metrics_epoch

    def evaluate(self, dataloader, criterion_dict, is_test=False):
        """Evaluate the model on validation or test data."""
        self.model.eval()
        eval_loss = 0
        all_preds = {task: [] for task in self.output_tasks}
        all_labels = {task: [] for task in self.output_tasks}

        with torch.no_grad():
            for batch in tqdm(
                dataloader,
                desc="Testing" if is_test else "Validation",
                ncols=120,
                colour="yellow" if is_test else "blue",
            ):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                features = batch["features"].to(self.device) if "features" in batch and self.feature_dim > 0 else None
                
                true_labels_batch = {}
                for task in self.output_tasks:
                    task_label_key = f"{task}_label" # Construct the task-specific label key
                    if task_label_key in batch:
                        true_labels_batch[task] = batch[task_label_key].to(self.device)
                    else:
                        logger.error(f"Task label key '{task_label_key}' not found in validation/test batch. Available keys: {list(batch.keys())}")
                        # Handle missing task label, e.g., skip task or batch, or raise error
                        # For now, let's ensure it doesn't crash if a label is unexpectedly missing,
                        # though this indicates a data problem.
                        true_labels_batch[task] = torch.empty(0, device=self.device) # Placeholder to avoid crash, but metrics will be affected

                model_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, features=features)

                if len(self.output_tasks) == 1 and not isinstance(model_outputs, (list, tuple)):
                    task_key = self.output_tasks[0]
                    model_outputs = {task_key: model_outputs}


                loss = 0
                for task_idx, task in enumerate(self.output_tasks):
                    task_output = model_outputs[task]
                    task_labels = true_labels_batch[task]
                    task_loss = criterion_dict[task](task_output, task_labels)
                    loss += self.task_weights[task] * task_loss
                    
                    preds = torch.argmax(task_output, dim=1).cpu().numpy()
                    all_preds[task].extend(preds)
                    all_labels[task].extend(task_labels.cpu().numpy())
                
                eval_loss += loss.item()
        
        avg_eval_loss = eval_loss / len(dataloader)
        logger.debug(f"{'Test' if is_test else 'Validation'} loss: {avg_eval_loss}")
        return avg_eval_loss, all_preds, all_labels

    def train_and_evaluate(self, trained_model_output_dir, metrics_output_file, weights_dir_base="models/weights"):
        """
        Main training and evaluation loop. Saves best model and metrics.
        Args:
            trained_model_output_dir (str): Directory to save the final best model.
            metrics_output_file (str): File to save training metrics (JSON).
            weights_dir_base (str): Base directory for temporary epoch weights.
        """
        criterion_dict, optimizer, scheduler = self.setup_training()
        best_val_f1s = {task: 0.0 for task in self.output_tasks}
        best_overall_val_f1 = 0.0 # Using emotion F1 for overall best model saving
        best_model_epoch_path = None

        # Ensure the temporary weights directory for this run exists and is clean
        # This is for epoch-wise saving before picking the best one for trained_model_output_dir
        run_weights_dir = os.path.join(weights_dir_base, "current_run_temp_weights")
        if os.path.exists(run_weights_dir):
            shutil.rmtree(run_weights_dir)
        os.makedirs(run_weights_dir, exist_ok=True)

        logger.info(f"Starting training for {self.epochs} epochs.")
        final_metrics_to_save = {"epochs": []}

        with mlflow.start_run(nested=True) as run: # Use nested if called from another MLflow run
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_param("weight_decay", self.weight_decay)
            mlflow.log_param("epochs", self.epochs)
            mlflow.log_params({f"task_weight_{task}": weight for task, weight in self.task_weights.items() if task in self.output_tasks})
            mlflow.log_param("output_tasks", str(self.output_tasks))

            for epoch in range(self.epochs):
                logger.info(f"Epoch {epoch + 1}/{self.epochs}")
                train_loss, train_metrics_for_epoch = self.train_epoch(criterion_dict, optimizer, scheduler)
                val_loss, val_preds, val_labels = self.evaluate(self.val_dataloader, criterion_dict)

                epoch_metrics = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_tasks_metrics": train_metrics_for_epoch,
                    "val_loss": val_loss,
                    "val_tasks_metrics": {} # For validation metrics
                }
                current_epoch_val_f1s = {}

                for task in self.output_tasks:
                    task_val_metrics = self.calculate_metrics(val_preds[task], val_labels[task], task_name=f"Val {task}")
                    current_epoch_val_f1s[task] = task_val_metrics["f1"]
                    logger.info(f"Epoch {epoch+1} Val {task.capitalize()} - F1: {task_val_metrics['f1']:.4f}, Acc: {task_val_metrics['acc']:.4f}")
                    mlflow.log_metric(f"val_{task}_f1", task_val_metrics["f1"], step=epoch)
                    mlflow.log_metric(f"val_{task}_acc", task_val_metrics["acc"], step=epoch)
                    epoch_metrics["val_tasks_metrics"][task] = task_val_metrics

                self.print_metrics(train_metrics_for_epoch, "Train", loss=train_loss)
                self.print_metrics(epoch_metrics["val_tasks_metrics"], "Val", loss=val_loss)

                final_metrics_to_save["epochs"].append(epoch_metrics)

                # Save model if current emotion F1 is better than overall best
                current_emotion_val_f1 = current_epoch_val_f1s.get("emotion", 0.0)
                if current_emotion_val_f1 > best_overall_val_f1:
                    best_overall_val_f1 = current_emotion_val_f1
                    best_val_f1s = current_epoch_val_f1s.copy() # Store all task F1s for this best model
                    
                    # Save to temp path first
                    temp_model_path = os.path.join(run_weights_dir, f"best_model_epoch_{epoch+1}.pt")
                    torch.save(self.model.state_dict(), temp_model_path)
                    if best_model_epoch_path and os.path.exists(best_model_epoch_path):
                        os.remove(best_model_epoch_path) # Remove previous best temp model
                    best_model_epoch_path = temp_model_path
                    logger.info(f"New best validation model (Emotion F1: {best_overall_val_f1:.4f}) saved to {best_model_epoch_path} (epoch {epoch+1})")

            # After all epochs, copy the best model to the final output directory
            if best_model_epoch_path:
                os.makedirs(trained_model_output_dir, exist_ok=True)
                # Save as dynamic_weights.pt (auto-updating model)
                dynamic_model_path = os.path.join(trained_model_output_dir, "dynamic_weights.pt")
                shutil.copy(best_model_epoch_path, dynamic_model_path)
                logger.info(f"Dynamic model saved to: {dynamic_model_path}")
                
                # Save model config (like num_classes, feature_dim) alongside the model
                model_config = {
                    "model_name": self.model.model_name, # Assuming model has this attribute
                    "feature_dim": self.feature_dim,
                    "num_classes": self.model.num_classes, # Assuming model has this attribute
                    "hidden_dim": self.model.hidden_dim, # Assuming model has this attribute
                    "dropout": self.model.dropout, # Assuming model has this attribute
                    "output_tasks": self.output_tasks,
                    "feature_config": self.feature_config
                }
                config_path = os.path.join(trained_model_output_dir, "model_config.json")
                with open(config_path, 'w') as f:
                    json.dump(model_config, f, indent=4)
                logger.info(f"Model config saved to {config_path}")

                # Upload dynamic model to Azure ML with auto-promotion
                try:
                    from .azure_model_sync import AzureMLModelManager
                    manager = AzureMLModelManager(weights_dir=trained_model_output_dir)
                    
                    upload_metadata = {
                        "epoch": str(self.epochs),
                        "output_tasks": ",".join(self.output_tasks),
                        "feature_config": str(self.feature_config),
                        "training_time": str(time.time() - training_start_time) if 'training_start_time' in locals() else None
                    }
                    
                    # Auto-upload with optional auto-promotion based on F1 threshold
                    sync_results = manager.auto_upload_after_training(
                        f1_score=best_overall_val_f1,
                        auto_promote_threshold=0.85,  # Configurable threshold
                        metadata=upload_metadata
                    )
                    
                    if sync_results["uploaded"]:
                        logger.info("✓ Dynamic model uploaded to Azure ML successfully")
                        if sync_results["promoted"]:
                            logger.info("✓ Model auto-promoted to baseline (F1 >= 0.85)")
                    else:
                        logger.warning("✗ Failed to upload dynamic model to Azure ML")
                except Exception as e:
                    logger.warning(f"Azure ML upload failed: {e}")

            else:
                logger.warning("No best model was saved during training.")
                # As a fallback, save the last epoch model if no improvement was seen
                # This might not be desired, depends on strategy.
                # For now, we only save if there was a best_model_epoch_path.

            # Clean up temp weights directory
            if os.path.exists(run_weights_dir):
                shutil.rmtree(run_weights_dir)

            # Save all metrics to the output file
            final_metrics_to_save["best_validation_f1s"] = best_val_f1s
            final_metrics_to_save["best_overall_validation_emotion_f1"] = best_overall_val_f1
            with open(metrics_output_file, 'w') as f:
                json.dump(final_metrics_to_save, f, indent=4)
            logger.info(f"Training metrics saved to {metrics_output_file}")
            mlflow.log_artifact(metrics_output_file)

        return best_val_f1s # Return F1s of the best model based on validation

    def evaluate_final_model(self, model_path, evaluation_output_dir):
        """
        Evaluate a given model, save results and visualizations.
        Args:
            model_path (str): Path to the .pt model file.
            evaluation_output_dir (str): Directory to save evaluation.csv and plots.
        Returns:
            pd.DataFrame: DataFrame containing predictions and true labels.
        """
        logger.info(f"Loading model for final evaluation from: {model_path}")
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")

        try:
            # Attempt to load model config if it exists alongside the model
            model_config_path = os.path.join(os.path.dirname(model_path), "model_config.json")
            if os.path.exists(model_config_path):
                with open(model_config_path, 'r') as f:
                    model_config = json.load(f)
                logger.info(f"Loaded model config from {model_config_path}")
                # Re-initialize model based on config for safety, or ensure current model matches
                # This is crucial if evaluate_final_model is called in a new context
                # For simplicity here, we assume self.model is already the correct architecture
                # and we are just loading weights. A more robust way would be to reconstruct
                # the model here based on model_config.
                if self.model.model_name != model_config.get("model_name") or \
                   self.feature_dim != model_config.get("feature_dim") or \
                   self.model.num_classes != model_config.get("num_classes"):
                    logger.warning("Model architecture from config seems different from current model. "
                                   "Ensure model is correctly initialized before loading state_dict.")
            else:
                logger.warning(f"Model config file not found at {model_config_path}. "
                               "Ensure model is correctly initialized before loading state_dict.")

            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device) # Ensure model is on the correct device
            self.model.eval()
            logger.info("Model loaded and set to evaluation mode.")
        except Exception as e:
            logger.error(f"Error loading model state_dict from {model_path}: {e}")
            raise

        # Initialize lists for predictions and labels
        predictions = {task: [] for task in self.output_tasks}
        labels = {task: [] for task in self.output_tasks}

        logger.info("Starting final evaluation on the test dataloader.")
        # Generate predictions using the test_dataloader
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc="Final Testing", ncols=120, colour="green"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                features = batch["features"].to(self.device) if "features" in batch and self.feature_dim > 0 else None
                
                true_labels_batch = {}
                for task in self.output_tasks:
                    task_label_key = f"{task}_label"  # Construct the task-specific label key
                    if task_label_key in batch:
                        labels[task].extend(batch[task_label_key].cpu().numpy())  # Store original labels
                        true_labels_batch[task] = batch[task_label_key].to(self.device)
                    else:
                        logger.error(f"Task label key '{task_label_key}' not found in test batch. Available keys: {list(batch.keys())}")
                        # Handle missing task label gracefully
                        continue

                # Only proceed with model prediction if we have at least one valid task
                if not true_labels_batch:
                    logger.warning("No valid task labels found in batch. Skipping this batch.")
                    continue


                model_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, features=features)

                if len(self.output_tasks) == 1 and not isinstance(model_outputs, (list, tuple)):
                    task_key = self.output_tasks[0]
                    model_outputs = {task_key: model_outputs}

                for task_idx, task in enumerate(self.output_tasks):
                    task_output = model_outputs[task]
                    preds = torch.argmax(task_output, dim=1).cpu().numpy()
                    predictions[task].extend(preds)
        
        logger.info("Predictions generated.")
        # Convert predictions and labels to original format
        # Ensure self.test_set_df has the 'text' column and matches the order of test_dataloader
        if 'text' not in self.test_set_df.columns:
            logger.warning("'text' column not found in test_set_df. Results will not include original text.")
            # Create a placeholder text column if necessary, matching the length of predictions
            num_test_samples = len(predictions[self.output_tasks[0]]) if self.output_tasks else 0
            results = {"text": [f"Sample_{i}" for i in range(num_test_samples)]}
        else:
            # Ensure the test_set_df is correctly aligned with dataloader output
            # This might require passing the original test_df to prepare_data and getting it back
            # or ensuring test_dataloader preserves order and length.
            # For now, assume test_set_df is correctly aligned and has sufficient rows.
            num_predicted_samples = len(predictions[self.output_tasks[0]])
            if len(self.test_set_df) < num_predicted_samples:
                logger.warning(f"Test DataFrame has {len(self.test_set_df)} rows, but "
                               f"{num_predicted_samples} predictions were made. Text column might be misaligned.")
                # Truncate or pad self.test_set_df['text'] if necessary, or raise error
                # For now, we'll use what's available, which might lead to errors if lengths mismatch.
                results = {"text": self.test_set_df["text"][:num_predicted_samples].tolist()}

            else:
                 results = {"text": self.test_set_df["text"][:num_predicted_samples].tolist()}


        for task in self.output_tasks:
            encoder = getattr(self, f"{task}_encoder", None)
            if encoder:
                # Ensure labels[task] are integers before inverse_transform
                labels_for_inverse = [int(lbl) for lbl in labels[task]]
                predictions_for_inverse = [int(pred) for pred in predictions[task]]

                results[f"true_{task}"] = encoder.inverse_transform(labels_for_inverse)
                results[f"pred_{task}"] = encoder.inverse_transform(predictions_for_inverse)
            else:
                logger.warning(f"Encoder for task {task} not found. Skipping inverse transform.")
                results[f"true_{task}"] = labels[task]
                results[f"pred_{task}"] = predictions[task]


        results_df = pd.DataFrame(results)
        for task in self.output_tasks:
            results_df[f"{task}_correct"] = (results_df[f"true_{task}"] == results_df[f"pred_{task}"])

        if len(self.output_tasks) > 1:
            all_correct_col = pd.Series([True] * len(results_df))
            for task in self.output_tasks:
                all_correct_col &= results_df[f"{task}_correct"]
            results_df["all_correct"] = all_correct_col
        
        logger.info("Results DataFrame created.")
        os.makedirs(evaluation_output_dir, exist_ok=True)
        eval_csv_path = os.path.join(evaluation_output_dir, "evaluation_report.csv")
        results_df.to_csv(eval_csv_path, index=False)
        logger.info(f"Evaluation report saved to {eval_csv_path}")

        # Generate and save visualizations
        # self._generate_visualizations(results_df, output_dir=evaluation_output_dir)
        # logger.info(f"Visualizations saved to {evaluation_output_dir}")

        return results_df
    # ... (calculate_metrics, print_metrics, _generate_visualizations etc. remain mostly same) ...
    # _generate_visualizations should be updated to save plots to a specified output_dir

    @staticmethod
    def calculate_metrics(preds, labels, task_name=""):
        """Calculate performance metrics."""
        # Ensure labels and preds are 1D arrays of the same length
        preds = np.array(preds).flatten()
        labels = np.array(labels).flatten()

        if len(preds) != len(labels):
            logger.error(f"Task {task_name}: preds length ({len(preds)}) and labels length ({len(labels)}) mismatch. Cannot calculate metrics.")
            return {"acc": 0, "f1": 0, "prec": 0, "rec": 0, "report": "Length mismatch"}

        if len(labels) == 0: # No samples
            logger.warning(f"Task {task_name}: Empty labels/preds. Returning zero metrics.")
            return {"acc": 0, "f1": 0, "prec": 0, "rec": 0, "report": "Empty labels/preds"}
        
        # Get unique labels present in true labels and predictions to pass to classification_report
        # This avoids warnings if some classes in the encoder are not present in this specific batch/dataset split
        unique_labels_in_data = np.unique(np.concatenate((labels, preds)))


        # It's possible that after a split, not all original classes are present.
        # We should use the labels known to the encoder for a full report,
        # but for calculation, ensure `labels` arg in classification_report matches what's in y_true, y_pred
        # For f1_score, precision_score, recall_score, `average='weighted'` handles this.
        # `zero_division=0` handles cases where a class has no predictions or no true labels.
        
        report_str = classification_report(labels, preds, zero_division=0, labels=unique_labels_in_data, target_names=[str(x) for x in unique_labels_in_data])


        return {
            "acc": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted", zero_division=0),
            "prec": precision_score(labels, preds, average="weighted", zero_division=0),
            "rec": recall_score(labels, preds, average="weighted", zero_division=0),
            "report": report_str # Adding full report
        }

    @staticmethod
    def print_metrics(metrics_dict, split, loss=None):
        """Print formatted metrics."""
        split_colors = {"Train": "cyan", "Val": "yellow", "Test": "green"}
        color = split_colors.get(split, "white")
        header = f" {split} Metrics "
        print(colored(f"\n{'='*20} {header} {'='*20}", color, attrs=["bold"]))
        if loss is not None:
            print(colored(f"Loss: {loss:.4f}", color))
        table_data = []
        headers = ["Task", "Accuracy", "F1 Score", "Precision", "Recall"]
        for task, metrics in metrics_dict.items():
            if isinstance(metrics, dict):  # Ensure metrics is a dict
                table_data.append([
                    task.capitalize(),
                    f"{metrics.get('acc', 0):.4f}",
                    f"{metrics.get('f1', 0):.4f}",
                    f"{metrics.get('prec', 0):.4f}",
                    f"{metrics.get('rec', 0):.4f}",
                ])
        if table_data:
            print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
        else:
            print(colored("No metrics to display for this split/task.", "red"))
        print(colored(f"{'='* (40 + len(header))}", color))

    @staticmethod
    def promote_dynamic_to_baseline(weights_dir: str = "models/weights") -> bool:
        """
        Simple function to promote dynamic_weights.pt to baseline_weights.pt
        Returns True if successful, False otherwise
        """
        dynamic_path = os.path.join(weights_dir, "dynamic_weights.pt")
        baseline_path = os.path.join(weights_dir, "baseline_weights.pt")
        
        if os.path.exists(dynamic_path):
            shutil.copy(dynamic_path, baseline_path)
            logging.info(f"Promoted dynamic model to baseline: {baseline_path}")
            return True
        else:
            logging.error(f"Dynamic weights not found at: {dynamic_path}")
            return False

    def should_promote_to_baseline(self, dynamic_f1, baseline_f1, threshold=0.01):
        """
        Determine if the dynamic model should be promoted to baseline.
        
        Args:
            dynamic_f1 (float): F1 score of the dynamic model
            baseline_f1 (float): F1 score of the current baseline
            threshold (float): Minimum improvement required for promotion
        
        Returns:
            bool: True if dynamic model should be promoted
        """
        return dynamic_f1 > baseline_f1 + threshold

