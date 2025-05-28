"""
Custom trainer class for BERT-based emotion classification model.
This module provides a comprehensive training and evaluation framework for
multi-task emotion classification, including emotion, sub-emotion, and intensity
prediction with flexible output options.
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
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
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import the local modules
from .data import DataPreparation, DatasetLoader
from .model import DEBERTAClassifier


class CustomTrainer:
    """
    A custom trainer class for BERT-based emotion classification model
    with flexible outputs.

    This class handles the complete training pipeline including:
    - Model training with customizable multi-task learning
    - Validation and testing with flexible output options
    - Performance metrics calculation and visualization
    - Feature importance analysis

    Attributes:
        model (nn.Module): The BERT-based emotion classification model
        train_dataloader (DataLoader): DataLoader for training data
        val_dataloader (DataLoader): DataLoader for validation data
        test_dataloader (DataLoader): DataLoader for test data
        device (torch.device): Device to run the model on (CPU/GPU)
        test_set (Dataset): Test dataset
        feature_dim (int): Dimension of input features (automatically determined
                           from data)
        class_weights_tensor (torch.Tensor): Class weights for handling class
                                             imbalance
        output_tasks (list): List of tasks to output
                            (e.g., ['emotion', 'sub_emotion', 'intensity'])
    """

    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        device,
        test_set,
        class_weights_tensor,
        encoders_dir,
        output_tasks=None,
    ):
        """
        Initialize the CustomTrainer with model and data components.

        Args:
            model (nn.Module): The BERT-based emotion classification model
            train_dataloader (DataLoader): DataLoader for training data
            val_dataloader (DataLoader): DataLoader for validation data
            test_dataloader (DataLoader): DataLoader for test data
            device (torch.device): Device to run the model on (CPU/GPU)
            test_set (Dataset): Test dataset
            class_weights_tensor (torch.Tensor): Class weights for handling
                                                 class imbalance
            encoders_dir (str): Directory containing encoder pickle files
            output_tasks (list, optional): List of tasks to output
                (e.g., ['emotion', 'sub_emotion', 'intensity']). Defaults to this list.
        """
        # Store model and data components
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.test_set = test_set
        self.class_weights_tensor = class_weights_tensor

        # Set output tasks
        self.output_tasks = output_tasks or ["emotion", "sub_emotion", "intensity"]

        # Load encoders
        self._load_encoders(encoders_dir)

        # Automatically determine feature dimension from the first batch
        self.feature_dim = self._get_feature_dim()

        # Training hyperparameters
        self.learning_rate = 2e-5  # Learning rate for AdamW optimizer
        self.weight_decay = 0.01  # Weight decay for regularization
        self.epochs = 1  # Number of training epochs

        # Task weights for multi-task learning
        self.task_weights = {
            "emotion": 1.0 if "emotion" in self.output_tasks else 0.0,
            "sub_emotion": 0.8 if "sub_emotion" in self.output_tasks else 0.0,
            "intensity": 0.2 if "intensity" in self.output_tasks else 0.0,
        }

    def _get_feature_dim(self):
        """
        Determine the feature dimension from the first batch of training data.

        Returns:
            int: Dimension of the feature vector
        """
        # Get the first batch
        first_batch = next(iter(self.train_dataloader))

        # Get the feature dimension from the features tensor
        feature_dim = first_batch["features"].shape[-1]

        return feature_dim

    def _load_encoders(self, encoders_dir):
        """
        Load label encoders from pickle files.

        Args:
            encoders_dir (str): Directory containing the encoder pickle files
        """
        # Load emotion encoder
        with open(f"{encoders_dir}/emotion_encoder.pkl", "rb") as f:
            self.emotion_encoder = pickle.load(f)

        # Load sub-emotion encoder
        with open(f"{encoders_dir}/sub_emotion_encoder.pkl", "rb") as f:
            self.sub_emotion_encoder = pickle.load(f)

        # Load intensity encoder
        with open(f"{encoders_dir}/intensity_encoder.pkl", "rb") as f:
            self.intensity_encoder = pickle.load(f)

    def setup_training(self):
        """
        Set up training components including loss function, optimizer,
        and learning rate scheduler.

        Returns:
            tuple: (criterion_dict, optimizer, scheduler)
                - criterion_dict: Dict of loss functions for each task
                - optimizer: AdamW optimizer with weight decay
                - scheduler: Linear learning rate scheduler with warmup
        """
        # Initialize loss functions with appropriate class weights for each task
        criterion_dict = {}

        # For emotion task - use the provided class weights
        if "emotion" in self.output_tasks:
            criterion_dict["emotion"] = nn.CrossEntropyLoss(
                weight=self.class_weights_tensor
            )

        # For sub-emotion and intensity - use regular CrossEntropyLoss without weights
        if "sub_emotion" in self.output_tasks:
            criterion_dict["sub_emotion"] = nn.CrossEntropyLoss()

        if "intensity" in self.output_tasks:
            criterion_dict["intensity"] = nn.CrossEntropyLoss()

        # Initialize AdamW optimizer with weight decay for regularization
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Calculate total training steps for scheduler
        total_steps = len(self.train_dataloader) * self.epochs

        # Initialize learning rate scheduler with warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0.1 * total_steps,  # 10% warmup
            num_training_steps=total_steps,
        )

        return criterion_dict, optimizer, scheduler

    def train_epoch(self, criterion_dict, optimizer, scheduler):
        """
        Train the model for one epoch.

        Args:
            criterion_dict (dict): Dictionary of loss functions for each task
            optimizer (torch.optim.Optimizer): Optimizer
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler

        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()  # Set model to training mode
        train_loss = 0

        # Iterate over training batches
        for batch in tqdm(
            self.train_dataloader, desc="Training", ncols=120, colour="green"
        ):
            # Move batch data to device (CPU/GPU)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            features = batch["features"].to(self.device)

            # Get labels for selected tasks
            labels = {}
            for task in self.output_tasks:
                labels[task] = batch[f"{task}_label"].to(self.device)

            # Forward pass through the model
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, features=features
            )

            # Handle single output vs multiple outputs
            if len(self.output_tasks) == 1 and not isinstance(outputs, (list, tuple)):
                outputs = [outputs]

            # Calculate losses for selected tasks
            loss = 0
            for i, task in enumerate(self.output_tasks):
                # Ensure output has proper batch dimension
                output = outputs[i]
                if output.dim() == 1:
                    output = output.unsqueeze(0)

                task_loss = criterion_dict[task](output, labels[task])
                loss += self.task_weights[task] * task_loss

            # Backward pass and optimization
            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Compute gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()  # Update parameters
            scheduler.step()  # Update learning rate

            train_loss += loss.item()

        return train_loss / len(self.train_dataloader)  # Return average loss

    def evaluate(self, dataloader, criterion_dict, is_test=False):
        """
        Evaluate the model on validation or test data.

        Args:
            dataloader (DataLoader): DataLoader for evaluation data
            criterion_dict (dict): Dictionary of loss functions for each task
            is_test (bool): Whether this is test set evaluation

        Returns:
            tuple: (average_loss, predictions_dict, labels_dict)
                - average_loss: Average evaluation loss
                - predictions_dict: Dictionary containing predictions for selected tasks
                - labels_dict: Dictionary containing true labels for selected tasks
        """
        self.model.eval()  # Set model to evaluation mode
        val_loss = 0
        all_preds = {task: [] for task in self.output_tasks}
        all_labels = {task: [] for task in self.output_tasks}

        with torch.no_grad():  # Disable gradient computation
            for batch in tqdm(
                dataloader,
                desc="Testing" if is_test else "Validation",
                ncols=120,
                colour="orange" if is_test else "blue",
            ):
                # Move batch data to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                features = batch["features"].to(self.device)

                # Get labels for selected tasks
                labels = {}
                for task in self.output_tasks:
                    labels[task] = batch[f"{task}_label"].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    features=features,
                )

                # Handle single output vs multiple outputs
                if len(self.output_tasks) == 1 and not isinstance(
                    outputs, (list, tuple)
                ):
                    outputs = [outputs]

                # Calculate losses for selected tasks
                loss = 0
                for i, task in enumerate(self.output_tasks):
                    # Ensure output has proper batch dimension
                    output = outputs[i]
                    if output.dim() == 1:
                        output = output.unsqueeze(0)

                    task_loss = criterion_dict[task](output, labels[task])
                    loss += self.task_weights[task] * task_loss

                    # Collect predictions and labels
                    all_preds[task].extend(
                        torch.argmax(outputs[i], dim=1).cpu().numpy()
                    )
                    all_labels[task].extend(labels[task].cpu().numpy())

                val_loss += loss.item()

        return val_loss / len(dataloader), all_preds, all_labels

    def train_and_evaluate(self):
        """
        Main training and evaluation loop.

        This method:
        1. Sets up training components
        2. Trains the model for multiple epochs
        3. Evaluates on validation and test sets
        4. Saves best models based on F1 scores for each task
        5. Prints metrics for each epoch
        """
        criterion_dict, optimizer, scheduler = self.setup_training()

        # Initialize best model tracking for each task
        best_val_f1s = {task: 0.0 for task in self.output_tasks}
        best_test_f1s = {task: 0.0 for task in self.output_tasks}

        # Training loop
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")

            # Training phase
            train_loss = self.train_epoch(criterion_dict, optimizer, scheduler)

            # Validation phase
            val_loss, val_preds, val_labels = self.evaluate(
                self.val_dataloader, criterion_dict
            )

            # Test phase
            _, test_preds, test_labels = self.evaluate(
                self.test_dataloader, criterion_dict, is_test=True
            )

            # Calculate metrics for selected tasks
            val_metrics = {}
            test_metrics = {}
            for task in self.output_tasks:
                val_metrics[task] = self.calculate_metrics(
                    val_preds[task], val_labels[task]
                )
                test_metrics[task] = self.calculate_metrics(
                    test_preds[task], test_labels[task]
                )

            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            self.print_metrics(val_metrics, "Val", val_loss)
            self.print_metrics(test_metrics, "Test")

            # Save best models for each task based on F1 scores
            for task in self.output_tasks:
                # Save based on validation F1
                if val_metrics[task]["f1"] > best_val_f1s[task]:
                    best_val_f1s[task] = val_metrics[task]["f1"]
                    f1_val_score = val_metrics[task]["f1"]
                    model_save_path = (
                        f"/app/models/weights/best_val_in_{task}_f1_{f1_val_score:.4f}.pt"
                    )
                    torch.save(self.model.state_dict(), model_save_path)
                    print(f"Model saved based on best validation F1 for {task}!")

                # Save based on test F1
                if test_metrics[task]["f1"] > best_test_f1s[task]:
                    best_test_f1s[task] = test_metrics[task]["f1"]
                    f1_test_score = test_metrics[task]["f1"]
                    model_save_path = f"/app/models/weights/best_test_in_{task}_\
                        f1_{f1_test_score:.4f}.pt"
                    torch.save(self.model.state_dict(), model_save_path)
                    print(f"Model saved based on best test F1 for {task}!")

                # TODO: Remove all weights except the best ones
                # Check f1 score based on the filename
                # weights_dir = './models/weights'
                # best_f1 = {}
                # for filename in os.listdir(weights_dir):
                #     if 'test' in filename:
                #         f1_score = float(filename.split('f1_')[1].split('.pt')[0])
                #         if f1_score < best_test_f1s[task]:
                #             os.remove(os.path.join(weights_dir, filename))
                #     if 'val' in filename:
                #         f1_score = float(filename.split('f1_')[1].split('.pt')[0])
                #         if f1_score < best_val_f1s[task]:
                #             os.remove(os.path.join(weights_dir, filename))

    def evaluate_final_model(self):
        """
        Evaluate the final model and generate comprehensive visualizations.

        This method:
        1. Finds and loads the best model based on test F1 score
        2. Makes predictions on the test set
        3. Converts predictions to original labels
        4. Creates a results DataFrame
        5. Generates visualizations and analysis
        6. Deletes suboptimal model weights.

        Returns:
            pd.DataFrame: DataFrame containing predictions and true labels
        """
        weights_dir = "/app/models/weights"
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)  # Ensure directory exists

        best_f1_test = {"emotion": 0.0, "sub_emotion": 0.0, "intensity": 0.0}
        best_f1_val = {"emotion": 0.0, "sub_emotion": 0.0, "intensity": 0.0}
        best_emotion_test_f1 = 0.0
        best_model_path = None
        model_files_info = []  # Store (filepath, type, task, f1_score)

        # First pass: find all best F1 scores and the best model for emotion_test
        for filename in os.listdir(weights_dir):
            if filename.endswith(".pt"):
                try:
                    # Expected format: best_val_in_emotion_f1_0.1234.pt
                    # or best_test_in_sub_emotion_f1_0.1234.pt
                    if "_f1_" not in filename:
                        print(
                            f"Skipping malformed filename (missing '_f1_'): {filename}"
                        )
                        continue

                    name_parts = filename.split("_f1_")
                    if len(name_parts) != 2:
                        print(
                            f"Skipping malformed filename (unexpected format \
                                around '_f1_'): "
                            f"{filename}"
                        )
                        continue

                    prefix_parts = name_parts[0].split("_")
                    if (
                        len(prefix_parts) < 4
                        or prefix_parts[0] != "best"
                        or prefix_parts[2] != "in"
                    ):
                        print(
                            f"Skipping malformed filename (prefix incorrect): \
                                {filename}"
                        )
                        continue

                    type_str = prefix_parts[1]  # 'val' or 'test'
                    # handles multi-word task names like sub_emotion
                    task_str = "_".join(prefix_parts[3:])
                    f1_score_str = name_parts[1].replace(".pt", "")
                    current_f1 = float(f1_score_str)

                    filepath = os.path.join(weights_dir, filename)
                    model_files_info.append((filepath, type_str, task_str, current_f1))

                    if task_str not in self.output_tasks:  # Ensure task is valid
                        print(
                            f"Skipping file with unknown task '{task_str}': {filename}"
                        )
                        continue

                    if type_str == "test":
                        # Use .get for safety
                        if current_f1 > best_f1_test.get(task_str, -1.0):
                            best_f1_test[task_str] = current_f1
                        if task_str == "emotion" and current_f1 > best_emotion_test_f1:
                            best_emotion_test_f1 = current_f1
                            best_model_path = filepath
                    elif type_str == "val":
                        # Use .get for safety
                        if current_f1 > best_f1_val.get(task_str, -1.0):
                            best_f1_val[task_str] = current_f1
                except ValueError as e:
                    print(f"Error converting F1 score to float for {filename}: {e}")
                    continue
                except Exception as e:
                    print(f"Error parsing filename {filename}: {e}")
                    continue

        if best_model_path is None:
            # Try to find any model if best_emotion_test_f1 is 0
            # (e.g. only val models saved or only other tasks were saved for test)
            # Prioritize any test model, then any val model.
            test_models = [info for info in model_files_info if info[1] == "test"]
            if test_models:
                # Select the one with highest F1 among any test task
                test_models.sort(key=lambda x: x[3], reverse=True)
                best_model_path = test_models[0][0]
                best_emotion_test_f1 = test_models[0][3]  # This might not be emotion
                print(
                    f"Warning: No 'emotion' test model found. Using best overall "
                    f"test model: {os.path.basename(best_model_path)} with F1: "
                    f"{best_emotion_test_f1:.4f}"
                )
            elif model_files_info:  # if no test models, use best val model
                model_files_info.sort(key=lambda x: x[3], reverse=True)
                best_model_path = model_files_info[0][0]
                # best_emotion_test_f1 will remain 0 or its previous value
                print(
                    f"Warning: No test models found. Using best overall "
                    f"validation model: {os.path.basename(best_model_path)}"
                )
            else:
                raise FileNotFoundError(
                    "No model files found in the weights directory that match "
                    "the expected naming convention."
                )

        print(
            f"Loading best model (based on emotion test F1 or best available): "
            f"{os.path.basename(best_model_path)} "
            f"with F1 score: {best_emotion_test_f1:.4f} (for emotion if applicable)"
        )

        # Load state dict with potential remapping and filtering for shape mismatches
        checkpoint = torch.load(best_model_path, map_location=self.device)
        remapped_state_dict = {}
        has_bert_prefix = any(k.startswith("bert.") for k in checkpoint.keys())
        has_deberta_prefix = any(k.startswith("deberta.") for k in checkpoint.keys())

        if has_bert_prefix and not has_deberta_prefix:
            print("Attempting to remap state_dict keys from 'bert.' to 'deberta.'")
            for key, value in checkpoint.items():
                if key.startswith("bert."):
                    remapped_state_dict[key.replace("bert.", "deberta.", 1)] = value
                else:
                    remapped_state_dict[key] = value
        else:
            remapped_state_dict = checkpoint

        # Filter the remapped_state_dict to include only keys that exist in the
        # current model and have matching shapes.
        current_model_state_dict = self.model.state_dict()
        final_loadable_state_dict = {}
        skipped_shape_mismatch_keys = []
        skipped_missing_in_model_keys = []

        for key_ckpt, value_ckpt in remapped_state_dict.items():
            if key_ckpt in current_model_state_dict:
                if current_model_state_dict[key_ckpt].shape == value_ckpt.shape:
                    final_loadable_state_dict[key_ckpt] = value_ckpt
                else:
                    skipped_shape_mismatch_keys.append(key_ckpt)
                    print(
                        f"Shape Mismatch: Skipping key '{key_ckpt}'. "
                        f"Model shape: {current_model_state_dict[key_ckpt].shape}, "
                        f"Checkpoint shape: {value_ckpt.shape}"
                    )
            else:
                skipped_missing_in_model_keys.append(key_ckpt)

        if skipped_missing_in_model_keys:
            print(
                f"Warning: Checkpoint keys not found in current model structure "
                f"(and thus skipped): {skipped_missing_in_model_keys}"
            )
        if skipped_shape_mismatch_keys:
            print(
                f"Warning: Due to shape mismatches, the following checkpoint keys "
                f"were not loaded: {skipped_shape_mismatch_keys}"
            )

        # Load the filtered state_dict.
        # strict=False will report layers in self.model not in final_loadable_state_dict
        try:
            incompatible_keys = self.model.load_state_dict(
                final_loadable_state_dict, strict=False
            )

            if incompatible_keys.missing_keys:
                print(
                    f"Warning: Some layers of the current model were not found in the "
                    f"checkpoint (or were skipped due to shape/key mismatch) and "
                    f"thus retain their initial weights: \
                        {incompatible_keys.missing_keys}"
                )
            if incompatible_keys.unexpected_keys:  # Should be empty by now
                print(
                    f"Warning: Checkpoint contained unexpected keys that were \
                        not loaded "
                    f"(should be empty if filtering worked): \
                        {incompatible_keys.unexpected_keys}"
                )

            if (
                not skipped_shape_mismatch_keys
                and not skipped_missing_in_model_keys
                and not incompatible_keys.missing_keys
                and not incompatible_keys.unexpected_keys
            ):
                print(
                    "State_dict loaded successfully and matched current model \
                        structure (after potential prefix remapping and filtering)."
                )
            else:
                print(
                    "State_dict loaded with some incompatibilities (see \
                        warnings above). Some layers may not be from the checkpoint."
                )

        except Exception as e:
            print(f"Critical error during final load_state_dict: {e}")
            raise

        self.model.eval()

        # Second pass: delete suboptimal models
        for file_path, type_str, task_str, f1_val in model_files_info:
            if file_path == best_model_path:  # Don't delete the loaded best model
                continue

            delete_file_flag = False
            if type_str == "test":
                if task_str in best_f1_test and f1_val < best_f1_test[task_str]:
                    delete_file_flag = True
            elif type_str == "val":
                if task_str in best_f1_val and f1_val < best_f1_val[task_str]:
                    delete_file_flag = True

            if delete_file_flag:
                try:
                    os.remove(file_path)
                    print(f"Deleted suboptimal model: {os.path.basename(file_path)}")
                except OSError as e:
                    print(f"Error deleting file {file_path}: {e}")

        # Initialize lists for predictions and labels
        predictions = {task: [] for task in self.output_tasks}
        labels = {task: [] for task in self.output_tasks}

        # Generate predictions
        with torch.no_grad():
            for batch in tqdm(
                self.test_dataloader, desc="Testing", ncols=120, colour="green"
            ):
                # Move batch data to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                features = batch["features"].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    features=features,
                )

                # Handle single output vs multiple outputs
                if len(self.output_tasks) == 1 and not isinstance(
                    outputs, (list, tuple)
                ):
                    outputs = [outputs]

                # Get predictions and labels for selected tasks
                for i, task in enumerate(self.output_tasks):
                    pred = torch.argmax(outputs[i], dim=1).cpu().numpy()
                    label = batch[f"{task}_label"].cpu().numpy()

                    predictions[task].extend(pred)
                    labels[task].extend(label)

        # Convert predictions and labels to original format
        results = {"text": self.test_set["text"]}
        for task in self.output_tasks:
            encoder = getattr(self, f"{task}_encoder")
            results[f"true_{task}"] = encoder.inverse_transform(labels[task])
            results[f"pred_{task}"] = encoder.inverse_transform(predictions[task])

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Add correctness columns
        for task in self.output_tasks:
            results_df[f"{task}_correct"] = (
                results_df[f"true_{task}"] == results_df[f"pred_{task}"]
            )

        if len(self.output_tasks) > 1:
            results_df["all_correct"] = True
            for task in self.output_tasks:
                results_df["all_correct"] &= results_df[f"{task}_correct"]

        # Generate visualizations
        self._generate_visualizations(results_df)

        return results_df

    @staticmethod
    def calculate_metrics(preds, labels):
        """
        Calculate performance metrics for predictions.

        Args:
            preds (np.ndarray): Model predictions
            labels (np.ndarray): True labels

        Returns:
            dict: Dictionary containing accuracy, F1 score, precision, and recall
        """
        return {
            "acc": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted"),
            "prec": precision_score(labels, preds, average="weighted"),
            "rec": recall_score(labels, preds, average="weighted"),
        }

    @staticmethod
    def print_metrics(metrics_dict, split, loss=None):
        """
        Print formatted metrics with visual bars for better readability.

        Args:
            metrics_dict (dict): Dictionary containing metrics for each task
            split (str): Data split name (Train/Val/Test)
            loss (float, optional): Loss value to display
        """
        # Define colors for different splits
        split_colors = {"Train": "cyan", "Val": "yellow", "Test": "green"}

        color = split_colors.get(split, "white")
        header = f" {split} Metrics "
        print(colored(f"\n{'='*20} {header} {'='*20}", color, attrs=["bold"]))

        if loss is not None:
            print(colored(f"Loss: {loss:.4f}", color))

        # Prepare table data with visual bars
        table_data = []
        headers = ["Task", "Accuracy", "F1 Score", "Precision", "Recall"]

        for task, metrics in metrics_dict.items():
            # Create visual bars for metrics (scaled to 20 chars)
            acc_bar = "█" * int(metrics["acc"] * 20)
            f1_bar = "█" * int(metrics["f1"] * 20)
            prec_bar = "█" * int(metrics["prec"] * 20)
            rec_bar = "█" * int(metrics["rec"] * 20)

            table_data.append(
                [
                    task,
                    f"{metrics['acc']:.4f} {acc_bar}",
                    f"{metrics['f1']:.4f} {f1_bar}",
                    f"{metrics['prec']:.4f} {prec_bar}",
                    f"{metrics['rec']:.4f} {rec_bar}",
                ]
            )

        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
        print(colored(f"{'='*50}", color))

    def _generate_visualizations(self, results_df):
        """
        Generate comprehensive visualizations for model evaluation.

        This method creates:
        1. Classification reports
        2. Confusion matrices for each task
        3. Performance comparison charts
        4. Misclassification examples

        Args:
            results_df (pd.DataFrame): DataFrame containing results
        """
        # Set up visualization style
        plt.style.use("fivethirtyeight")
        sns.set(font_scale=1.2)
        sns.set_style("whitegrid", {"axes.grid": False})

        # Generate classification reports
        self._print_styled_report(
            classification_report(
                results_df["true_emotion"], results_df["pred_emotion"]
            ),
            "EMOTION CLASSIFICATION REPORT",
        )
        self._print_styled_report(
            classification_report(
                results_df["true_sub_emotion"], results_df["pred_sub_emotion"]
            ),
            "SUB-EMOTION CLASSIFICATION REPORT",
        )
        self._print_styled_report(
            classification_report(
                results_df["true_intensity"], results_df["pred_intensity"]
            ),
            "INTENSITY CLASSIFICATION REPORT",
        )

        # Generate confusion matrices
        self._plot_enhanced_confusion_matrix(
            results_df["true_emotion"],
            results_df["pred_emotion"],
            self.emotion_encoder.classes_,
            "Emotion Confusion Matrix",
            cmap="YlGnBu",
        )

        self._plot_enhanced_confusion_matrix(
            results_df["true_intensity"],
            results_df["pred_intensity"],
            self.intensity_encoder.classes_,
            "Intensity Confusion Matrix",
            cmap="RdPu",
            figsize=(10, 8),
        )

        # Generate top sub-emotions confusion matrix
        top_sub_emotions = (
            pd.Series(results_df["true_sub_emotion"])
            .value_counts()
            .nlargest(10)
            .index.tolist()
        )
        mask = np.isin(results_df["true_sub_emotion"], top_sub_emotions) & np.isin(
            results_df["pred_sub_emotion"], top_sub_emotions
        )
        self._plot_enhanced_confusion_matrix(
            results_df["true_sub_emotion"][mask],
            results_df["pred_sub_emotion"][mask],
            top_sub_emotions,
            "Top 10 Sub-Emotions Confusion Matrix",
            cmap="PuBuGn",
            figsize=(14, 12),
        )

        # Generate performance comparison charts
        self._plot_performance_comparison(results_df)

        # Show misclassified examples
        self._show_misclassified_examples(results_df)

    @staticmethod
    def _print_styled_report(report, title):
        """
        Print a styled classification report with color coding.

        Args:
            report (str): Classification report text
            title (str): Report title
        """
        report_lines = report.split("\n")
        print(f"\n{'='*80}")
        print(f"{title.center(80)}")
        print(f"{'='*80}")

        for line in report_lines:
            if not line.strip():
                continue
            if "precision" in line or "accuracy" in line:
                print(colored(line, "cyan"))
            elif "avg" in line:
                print(colored(line, "yellow", attrs=["bold"]))
            else:
                print(line)

    @staticmethod
    def _plot_enhanced_confusion_matrix(
        true_labels, pred_labels, classes, title, cmap="Blues", figsize=(12, 10)
    ):
        """
        Plot an enhanced confusion matrix with normalized values and annotations.

        Args:
            true_labels (np.ndarray): True labels
            pred_labels (np.ndarray): Predicted labels
            classes (list): List of class names
            title (str): Plot title
            cmap (str): Color map for the heatmap
            figsize (tuple): Figure size (width, height)
        """
        # Get unique classes that actually appear in the data
        actual_classes = sorted(
            list(set(np.unique(true_labels)) | set(np.unique(pred_labels)))
        )

        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Create plot
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm_norm,
            annot=cm,
            fmt="d",
            cmap=cmap,
            linewidths=0.5,
            cbar=True,
            square=True,
            xticklabels=actual_classes,
            yticklabels=actual_classes,
            annot_kws={"size": 10},
        )

        f1 = f1_score(true_labels, pred_labels, average="weighted")
        plt.title(f"{title}\nF1 Score: {f1:.2%}", fontsize=16, fontweight="bold")

        plt.ylabel("True Label", fontsize=14)
        plt.xlabel("Predicted Label", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def _plot_performance_comparison(self, results_df):
        """
        Plot performance comparison charts for different tasks and metrics.

        Args:
            results_df (pd.DataFrame): DataFrame containing results
        """
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))

        # Plot emotion accuracy by class
        emotion_accuracy = {}
        for emotion in self.emotion_encoder.classes_:
            mask = results_df["true_emotion"] == emotion
            if mask.sum() > 0:
                emotion_accuracy[emotion] = np.mean(
                    results_df.loc[mask, "emotion_correct"]
                )

        emotion_df = pd.DataFrame({"Accuracy": emotion_accuracy}).sort_values(
            "Accuracy", ascending=False
        )
        sns.barplot(
            data=emotion_df,
            x=emotion_df.index,
            y="Accuracy",
            palette="viridis",
            ax=axes[0],
        )
        axes[0].set_title("Accuracy by Emotion Category", fontsize=15)
        axes[0].set_ylim(0, 1)
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")
        for i, v in enumerate(emotion_df["Accuracy"]):
            axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center")

        # Plot intensity accuracy by class
        intensity_accuracy = {}
        for intensity in self.intensity_encoder.classes_:
            mask = results_df["true_intensity"] == intensity
            if mask.sum() > 0:
                intensity_accuracy[intensity] = np.mean(
                    results_df.loc[mask, "intensity_correct"]
                )

        intensity_df = pd.DataFrame({"Accuracy": intensity_accuracy}).sort_values(
            "Accuracy", ascending=False
        )
        sns.barplot(
            data=intensity_df,
            x=intensity_df.index,
            y="Accuracy",
            palette="plasma",
            ax=axes[1],
        )
        axes[1].set_title("Accuracy by Intensity Level", fontsize=15)
        axes[1].set_ylim(0, 1)
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
        for i, v in enumerate(intensity_df["Accuracy"]):
            axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center")

        # Plot overall metrics
        overall_metrics = {
            "Emotion": np.mean(results_df["emotion_correct"]),
            "Sub_Emotion": np.mean(results_df["sub_emotion_correct"]),
            "Intensity": np.mean(results_df["intensity_correct"]),
            "All": np.mean(results_df["all_correct"]),
        }

        overall_df = pd.DataFrame(
            {
                "Task": list(overall_metrics.keys()),
                "Accuracy": list(overall_metrics.values()),
            }
        )
        sns.barplot(
            data=overall_df, x="Task", y="Accuracy", palette="magma", ax=axes[2]
        )
        axes[2].set_title("Overall Performance by Task", fontsize=15)
        axes[2].set_ylim(0, 1)
        for i, v in enumerate(overall_metrics.values()):
            axes[2].text(i, v + 0.02, f"{v:.2f}", ha="center")

        plt.tight_layout()
        plt.show()

    def _show_misclassified_examples(self, results_df):
        """
        Display examples of misclassified emotions for error analysis.

        Args:
            results_df (pd.DataFrame): DataFrame containing results
        """
        print("\n" + "=" * 80)
        print("MISCLASSIFICATION EXAMPLES".center(80))
        print("=" * 80)

        # Find most problematic emotion
        emotion_misclass = results_df[~results_df["emotion_correct"]]
        most_problematic = emotion_misclass["true_emotion"].value_counts().idxmax()

        print(
            f"\nMost problematic emotion: \
            {colored(most_problematic, 'red', attrs=['bold'])}"
        )
        print(f"Examples of '{most_problematic}' misclassified:")

        # Show examples of misclassifications
        problematic_examples = emotion_misclass[
            emotion_misclass["true_emotion"] == most_problematic
        ].sample(min(5, len(emotion_misclass)))
        for i, (_, row) in enumerate(problematic_examples.iterrows()):
            print(f"\n{i+1}. Text: {colored(row['text'][:100] + '...', 'cyan')}")
            print(
                f"   True: {colored(row['true_emotion'], 'green')} \
                  → Predicted: {colored(row['pred_emotion'], 'red')}"
            )
            print(
                f"   Sub_emotion: {row['true_sub_emotion']} \
                  → {row['pred_sub_emotion']}"
            )
            print(f"   Intensity: {row['true_intensity']} → {row['pred_intensity']}")


if __name__ == "__main__":
    # Hyperparameters
    MODEL_NAME = "microsoft/deberta-v3-xsmall"
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_TASKS = ["emotion", "sub_emotion", "intensity"]

    # Load the tokenizer
    # Note: AutoModel.from_pretrained(MODEL_NAME) is not directly used here
    # as the model architecture is defined in DEBERTAClassifier
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load the training and test data
    dataset_loader = DatasetLoader()
    df = dataset_loader.load_training_data(data_dir="data/raw/train")
    test_df = dataset_loader.load_test_data(test_file="data/raw/test/group 21_url1.csv")

    # Change intensity to mild, moderate and strong
    intensity_mapping = {
        "mild": "mild",
        "neutral": "mild",
        "moderate": "moderate",
        "intense": "strong",
        "overwhelming": "strong",
    }
    df["intensity"] = df["intensity"].map(intensity_mapping)
    test_df["intensity"] = test_df["intensity"].map(intensity_mapping)

    # Calculate or load class_weights_tensor
    # For example, if you have labels and want to compute weights:
    emotion_labels = df["emotion"].unique()
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(emotion_labels), y=df["emotion"]
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    # Features to use
    feature_config = {
        "pos": False,
        "textblob": False,
        "vader": False,
        "tfidf": True,
        "emolex": True,
    }

    # Initialize the data preparation
    # Ensure encoders_dir exists or is created by DataPreparation/elsewhere
    encoders_output_dir = "/app/models/encoders"
    os.makedirs(encoders_output_dir, exist_ok=True)

    data_prep = DataPreparation(
        output_columns=OUTPUT_TASKS,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        feature_config=feature_config,
    )

    # Prepare the data
    # The prepare_data method should save the encoders
    train_dataloader, val_dataloader, test_dataloader = data_prep.prepare_data(
        train_df=df, test_df=test_df, validation_split=0.1
    )

    # Get feature dimension from the feature extractor
    feature_dim = data_prep.feature_extractor.get_feature_dim()

    # Get number of classes for each output
    num_classes = data_prep.get_num_classes()

    # Initialize model
    model = DEBERTAClassifier(
        model_name=MODEL_NAME,
        feature_dim=feature_dim,
        num_classes=num_classes,
        hidden_dim=256,
        dropout=0.1,
    ).to(DEVICE)

    # Ensure results/weights directory exists
    weights_output_dir = "/app/models/weights"
    os.makedirs(weights_output_dir, exist_ok=True)

    # Initialize CustomTrainer
    trainer = CustomTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        device=DEVICE,
        test_set=test_df,
        class_weights_tensor=class_weights_tensor,
        encoders_dir=encoders_output_dir,
        output_tasks=OUTPUT_TASKS,
    )

    print("[INFO] Starting training and evaluation...")
    trainer.train_and_evaluate()

    print("[INFO] Starting final model evaluation...")
    final_results_df = trainer.evaluate_final_model()
    print("[INFO] Final evaluation complete.")
    print("Sample of final results:")
    print(final_results_df.head())

    # Example: Save final_results_df to a CSV
    results_output_dir = "/app/models/evaluation_outputs"
    os.makedirs(results_output_dir, exist_ok=True)
    final_results_df.to_csv(
        f"{results_output_dir}/final_evaluation_results.csv", index=False
    )
    print(f"Final results saved to {results_output_dir}/final_evaluation_results.csv")
