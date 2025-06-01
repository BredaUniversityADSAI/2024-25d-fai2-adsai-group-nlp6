"""
Prediction module for emotion classification model.
This module provides functionality for loading trained models and making predictions,
including post-processing to map sub-emotions to main emotions.
"""

import glob
import io
import logging
import os
import pickle
import shutil

import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, DebertaV2Tokenizer

# Import necessary classes from .data explicitly
from .data import EmotionDataset, FeatureExtractor  # Corrected name, moved up

logger = logging.getLogger(__name__)

# Ensure NLTK knows where to find its data, especially in Docker
if os.path.exists("/app/nltk_data"):
    nltk.data.path.append("/app/nltk_data")
    logger.info("NLTK data path /app/nltk_data appended in model.py")
elif not any("nltk_data" in p for p in nltk.data.path):
    try:
        logger.info("NLTK data path not set, trying to download punkt for model.py")
        nltk.download("punkt", quiet=True)
        nltk.download("averaged_perceptron_tagger", quiet=True)
        nltk.download("vader_lexicon", quiet=True)
    except Exception as e:
        logger.warning(f"NLTK download in model.py failed: {e}")


class DEBERTAClassifier(nn.Module):
    """
    Multi-task DEBERTA classifier for emotion classification.

    This model performs three classification tasks:
    1. Main emotion classification
    2. Sub-emotion classification
    3. Intensity classification

    The model combines BERT embeddings with additional features through
    projection layers.
    """

    def __init__(
        self, model_name, feature_dim, num_classes, hidden_dim=256, dropout=0.1
    ):
        """
        Initialize the DEBERTAClassifier model.

        Args:
            model_name (str): Name of the pretrained model to use
            feature_dim (int): Dimension of additional features
            num_classes (dict): Dictionary containing number of classes for each task
            hidden_dim (int): Dimension of hidden layers
            dropout (float): Dropout probability
        """
        super().__init__()
        self.model_name = model_name  # Store model_name
        self.num_classes = num_classes # Store num_classes
        self.hidden_dim = hidden_dim # Store hidden_dim
        self.dropout = dropout # Store dropout

        # Load base DEBERTA model
        self.deberta = AutoModel.from_pretrained(model_name)

        # Get DEBERTA embedding dimension
        deberta_dim = self.deberta.config.hidden_size

        # Feature projection layer
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )

        # Combine DEBERTA and feature embeddings
        combined_dim = deberta_dim + hidden_dim

        # Task-specific layers
        self.emotion_classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes["emotion"]),
        )

        self.sub_emotion_classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes["sub_emotion"]),
        )

        self.intensity_classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes["intensity"]),
        )

    def forward(self, input_ids, attention_mask, features):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask
            features (torch.Tensor): Additional features

        Returns:
            dict: A dictionary mapping task names to their logits.
                  Example: {'emotion': emotion_logits, 'sub_emotion': sub_emotion_logits, 'intensity': intensity_logits}
        """
        # Get DEBERTA embeddings
        deberta_output = self.deberta(
            input_ids=input_ids, attention_mask=attention_mask
        )
        deberta_embeddings = deberta_output.last_hidden_state[
            :, 0, :
        ]  # Use [CLS] token

        # Project additional features
        projected_features = self.feature_projection(features)

        # Combine embeddings
        combined = torch.cat([deberta_embeddings, projected_features], dim=1)

        # Task-specific predictions
        emotion_logits = self.emotion_classifier(combined)
        sub_emotion_logits = self.sub_emotion_classifier(combined)
        intensity_logits = self.intensity_classifier(combined)

        return {
            "emotion": emotion_logits,
            "sub_emotion": sub_emotion_logits,
            "intensity": intensity_logits,
        }


class ModelLoader:
    """
    A class to handle loading of the model and tokenizer.

    This class handles:
    - Loading the DEBERTA model and tokenizer
    - Setting up the device (CPU/GPU)
    - Loading model weights
    """

    def __init__(self, model_name="microsoft/deberta-v3-xsmall", device=None):
        """
        Initialize the ModelLoader.

        Args:
            model_name (str): Name of the pretrained model to use
            device (torch.device, optional): Device to use. If None, will use
            GPU if available
        """
        self.model_name = model_name
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.info(f"Using device: {self.device} in ModelLoader")
        logger.info(f"Loading tokenizer from: {self.model_name}")

        # Load tokenizer
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_name)

    def load_model(
        self, feature_dim, num_classes, weights_path=None, hidden_dim=256, dropout=0.1
    ):
        """
        Create and load the model.

        Args:
            feature_dim (int): Dimension of additional features
            num_classes (dict): Dictionary containing number of classes for each task
            weights_path (str, optional): Path to model weights
            hidden_dim (int): Dimension of hidden layers
            dropout (float): Dropout probability

        Returns:
            DEBERTAClassifier: Loaded model
        """
        # Create model instance
        model = DEBERTAClassifier(
            model_name=self.model_name,
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Load weights if provided
        if weights_path is not None:
            logger.info(f"Attempting to load weights from: {weights_path}")
            logger.info(f"Attempting to load weights from (absolute path check): {os.path.abspath(weights_path)}")
            logger.info(f"Does weights_path ({weights_path}) exist? {os.path.exists(weights_path)}")
            logger.info(f"Is weights_path ({weights_path}) a file? {os.path.isfile(weights_path)}")

            expected_weights_dir = os.path.dirname(weights_path)  
            logger.info(f"Expected weights directory: {expected_weights_dir}")
            logger.info(f"Does expected weights directory ({expected_weights_dir}) exist? {os.path.exists(expected_weights_dir)}")
            if os.path.exists(expected_weights_dir):
                try:
                    logger.info(f"Contents of expected weights directory ({expected_weights_dir}): {os.listdir(expected_weights_dir)}")
                except Exception as e_list_weights:
                    logger.error(f"Could not list contents of {expected_weights_dir}: {e_list_weights}")
            
            models_base_dir = os.path.dirname(expected_weights_dir) # Should be /models
            logger.info(f"Expected base models directory: {models_base_dir}")
            logger.info(f"Does base models directory ({models_base_dir}) exist? {os.path.exists(models_base_dir)}")
            if os.path.exists(models_base_dir):
                try:
                    logger.info(f"Contents of base models directory ({models_base_dir}): {os.listdir(models_base_dir)}")
                except Exception as e_list_models:
                    logger.error(f"Could not list contents of {models_base_dir}: {e_list_models}")
            
            # Check root directory contents if /models doesn't exist or is empty
            if not os.path.exists(models_base_dir) or (os.path.exists(models_base_dir) and not os.listdir(models_base_dir)):
                logger.info("Listing contents of root directory '/' for debugging:")
                try:
                    logger.info(f"Contents of '/': {os.listdir('/')}")
                except Exception as e_list_root:
                    logger.error(f"Could not list contents of '/': {e_list_root}")

            try:
                # Load weights using BytesIO to handle seekable file requirement
                with open(weights_path, "rb") as f:
                    buffer = io.BytesIO(f.read())
                    state_dict = torch.load(buffer, map_location=self.device)

                # Create a new state_dict with corrected keys
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("bert."):
                        new_key = "deberta." + k[len("bert.") :]
                        new_state_dict[new_key] = v
                    else:
                        new_state_dict[k] = v

                model.load_state_dict(new_state_dict)
                logger.info(f"Successfully loaded model weights from {weights_path}")
            except FileNotFoundError:
                logger.error(f"FileNotFoundError: Weight file not found at {weights_path} by open().")
                # Detailed path traversal logging
                logger.info(f"Detailed path check for {weights_path}:")
                path_parts = weights_path.split(os.sep)
                current_path_check = "/"
                # Ensure the first part of an absolute path is handled correctly
                if path_parts[0] == '': # Handles paths starting with '/'
                    current_path_check = os.sep 
                    start_index = 1
                else: # Should not happen for our absolute path
                    start_index = 0

                for i in range(start_index, len(path_parts)):
                    part = path_parts[i]
                    if not part: continue # Skip empty parts if any from multiple slashes

                    parent_of_current_check = os.path.dirname(current_path_check.rstrip(os.sep))
                    if not parent_of_current_check: parent_of_current_check = os.sep
                    
                    logger.info(f"Checking existence of directory component: {parent_of_current_check}")
                    if os.path.exists(parent_of_current_check):
                        try:
                            logger.info(f"Contents of {parent_of_current_check}: {os.listdir(parent_of_current_check)}")
                        except Exception as e_list_parent:
                            logger.error(f"Could not list contents of {parent_of_current_check}: {e_list_parent}")
                    else:
                        logger.error(f"Parent directory component {parent_of_current_check} does NOT exist. Stopping path traversal check.")
                        break
                    
                    if i < len(path_parts) -1 : # It's a directory in the path
                        current_path_check = os.path.join(current_path_check, part)
                        logger.info(f"Checking specifically for directory: {current_path_check}")
                        if not os.path.exists(current_path_check) or not os.path.isdir(current_path_check):
                             logger.error(f"Directory {current_path_check} does NOT exist or is not a directory. Stopping path traversal check.")
                             break
                    else: # It's the file itself
                         current_path_check = os.path.join(current_path_check, part)


                raise  # Re-raise the original exception
            except Exception as e:
                logger.error(f"An unexpected error occurred while loading weights from {weights_path}: {e}", exc_info=True)
                raise

        # Move model to device
        model = model.to(self.device)

        return model

    def create_predictor(
        self, model, encoders_dir="models/encoders", feature_config=None
    ):
        """
        Create a CustomPredictor instance with the loaded model and tokenizer.

        Args:
            model (nn.Module): The model instance
            encoders_dir (str): Directory containing the encoder pickle files
            feature_config (dict, optional): Configuration for feature extraction

        Returns:
            CustomPredictor: Predictor instance ready for making predictions
        """
        return CustomPredictor(
            model=model,            tokenizer=self.tokenizer,
            device=self.device,
            encoders_dir=encoders_dir,
            feature_config=feature_config,
        )
    def load_baseline_model(self, weights_dir="models/weights", sync_azure=True):
        """Load the baseline (stable production) model with Azure ML sync."""
        baseline_path = os.path.join(weights_dir, "baseline_weights.pt")
        
        # Enhanced Azure ML sync with update checking
        if sync_azure:
            try:
                from .azure_model_sync import AzureMLModelManager
                manager = AzureMLModelManager(weights_dir)
                sync_results = manager.auto_sync_on_startup(check_for_updates=True)
                
                if sync_results["baseline_downloaded"]:
                    logger.info("✓ Baseline model downloaded from Azure ML")
                if sync_results["baseline_updated"]:
                    logger.info("✓ Baseline model updated from Azure ML")
                    
            except Exception as e:
                logger.warning(f"Azure ML sync failed: {e}")
        
        if not os.path.exists(baseline_path):
            raise FileNotFoundError(f"Baseline model not found: {baseline_path}")
        logger.info(f"Loading baseline model from: {baseline_path}")
        self.model.load_state_dict(torch.load(baseline_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
    def load_dynamic_model(self, weights_dir="models/weights", sync_azure=True):
        """Load the dynamic (latest trained) model with Azure ML sync."""
        dynamic_path = os.path.join(weights_dir, "dynamic_weights.pt")
        
        # Enhanced Azure ML sync with update checking
        if sync_azure:
            try:
                from .azure_model_sync import AzureMLModelManager
                manager = AzureMLModelManager(weights_dir)
                sync_results = manager.auto_sync_on_startup(check_for_updates=True)
                
                if sync_results["dynamic_downloaded"]:
                    logger.info("✓ Dynamic model downloaded from Azure ML")
                if sync_results["dynamic_updated"]:
                    logger.info("✓ Dynamic model updated from Azure ML")
                    
            except Exception as e:
                logger.warning(f"Azure ML sync failed: {e}")
        
        if not os.path.exists(dynamic_path):
            raise FileNotFoundError(f"Dynamic model not found: {dynamic_path}")
        logger.info(f"Loading dynamic model from: {dynamic_path}")
        self.model.load_state_dict(torch.load(dynamic_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
    def promote_dynamic_to_baseline(self, weights_dir="models/weights", sync_azure=True):
        """Promote the current dynamic model to become the new baseline with Azure ML sync."""
        if sync_azure:
            try:
                from .azure_model_sync import promote_to_baseline_with_azure
                success = promote_to_baseline_with_azure(weights_dir)
                if success:
                    logger.info("Dynamic model promoted to baseline (local + Azure ML)")
                    return
            except Exception as e:
                logger.warning(f"Azure ML promotion failed, falling back to local: {e}")
        
        # Fallback to local promotion
        dynamic_path = os.path.join(weights_dir, "dynamic_weights.pt")
        baseline_path = os.path.join(weights_dir, "baseline_weights.pt")
        
        if not os.path.exists(dynamic_path):
            raise FileNotFoundError(f"Dynamic model not found: {dynamic_path}")
        
        # Copy dynamic to baseline
        shutil.copy(dynamic_path, baseline_path)
        logger.info(f"Promoted dynamic model to baseline: {baseline_path}")


class CustomPredictor:
    """
    A class for making predictions with trained emotion classification models.

    This class handles:
    - Loading the best model based on test F1 scores
    - Making predictions on new data
    - Post-processing predictions to map sub-emotions to main emotions

    Attributes:
        model (nn.Module): The loaded emotion classification model
        tokenizer: Tokenizer for text preprocessing
        device (torch.device): Device to run the model on (CPU/GPU)
        emotion_mapping (dict): Dictionary mapping sub-emotions to main emotions
        encoders (dict): Dictionary of label encoders for each task
        feature_extractor (FeatureExtractor): Feature extractor for additional features
    """

    def __init__(
        self,
        model,
        tokenizer,
        device,
        encoders_dir="models/encoders",
        feature_config=None,
    ):
        """
        Initialize the EmotionPredictor.

        Args:
            model (nn.Module): The base model architecture (unloaded)
            tokenizer: Tokenizer for text preprocessing
            device (torch.device): Device to run the model on
            encoders_dir (str): Directory containing the encoder pickle files
            feature_config (dict, optional): Configuration for feature extraction
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.expected_feature_dim = model.feature_projection[0].in_features

        if feature_config is None:
            self.feature_config = {
                "pos": False,
                "textblob": False,
                "vader": False,
                "tfidf": False,
                "emolex": False,
            }
        else:
            self.feature_config = feature_config

        _current_file_path_cp = os.path.abspath(__file__)
        _project_root_dir_cp = os.path.dirname(
            os.path.dirname(os.path.dirname(_current_file_path_cp))
        )
        emolex_path = os.path.join(
            _project_root_dir_cp,
            "models",
            "features",
            "EmoLex",
            "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
        )

        self.feature_extractor = FeatureExtractor(
            feature_config=self.feature_config, lexicon_path=emolex_path
        )        # TF-IDF fitting for CustomPredictor should happen here if tfidf is
        # enabled in its config
        if (
            self.feature_config.get("tfidf", False)
            and self.feature_extractor.tfidf_vectorizer is None
        ):
            logger.info("CustomPredictor: TF-IDF enabled, loading training data for fitting.")
            # Load actual training data to fit TF-IDF properly
            training_data_path = os.path.join(
                _project_root_dir_cp, "data", "processed", "train.csv"
            )
            try:
                import pandas as pd
                if os.path.exists(training_data_path):
                    train_df = pd.read_csv(training_data_path)
                    if "text" in train_df.columns:
                        training_texts = train_df["text"].tolist()
                        logger.info(f"Loaded {len(training_texts)} training texts for TF-IDF fitting.")
                        self.feature_extractor.fit_tfidf(training_texts)
                    else:
                        logger.warning("No 'text' column in training data, using dummy document.")
                        self.feature_extractor.fit_tfidf(
                            ["dummy document for tfidf initialization in predictor"]
                        )
                else:
                    logger.warning(f"Training data not found at {training_data_path}, using dummy document.")
                    self.feature_extractor.fit_tfidf(
                        ["dummy document for tfidf initialization in predictor"]
                    )
            except Exception as e:
                logger.warning(f"Error loading training data for TF-IDF: {e}, using dummy document.")
                self.feature_extractor.fit_tfidf(
                    ["dummy document for tfidf initialization in predictor"]
                )
            
            # Verify dimension
            calculated_dim_after_fit = self.feature_extractor.get_feature_dim()
            if calculated_dim_after_fit != self.expected_feature_dim:
                logger.warning(
                    f"Predictor: TF-IDF dim {calculated_dim_after_fit} \
                        vs model {self.expected_feature_dim}. "
                    "Check config."
                )
            else:
                logger.info(
                    f"Predictor: TF-IDF init. Dim {calculated_dim_after_fit} \
                        matches."
                )

        self.emotion_mapping = {
            "curiosity": "happiness",
            "neutral": "neutral",
            "annoyance": "anger",
            "confusion": "surprise",
            "disappointment": "sadness",
            "excitement": "happiness",
            "surprise": "surprise",
            "realization": "surprise",
            "desire": "happiness",
            "amusement": "happiness",
            "caring": "happiness",
            "approval": "happiness",
            "disapproval": "disgust",
            "nervousness": "fear",
            "embarrassment": "fear",
            "admiration": "happiness",
            "pride": "happiness",
            "anger": "anger",
            "optimism": "happiness",
            "sadness": "sadness",
            "joy": "happiness",
            "fear": "fear",
            "remorse": "sadness",
            "gratitude": "happiness",
            "disgust": "disgust",
            "love": "happiness",
            "relief": "happiness",
            "grief": "sadness",
        }

        self.encoders = self._load_encoders(encoders_dir)
        self.output_tasks = ["emotion", "sub_emotion", "intensity"]

        logger.info(
            f"CustomPredictor init. Model expects {self.expected_feature_dim} features."
        )
        calculated_dim = self.feature_extractor.get_feature_dim()
        if calculated_dim != self.expected_feature_dim:
            logger.warning(
                f"Extractor feats: {calculated_dim}, model expects: \
                    {self.expected_feature_dim}. "
                "Padding/truncation needed."
            )

    def _load_encoders(self, encoders_dir):
        """
        Load label encoders from pickle files.

        Args:
            encoders_dir (str): Directory containing the encoder pickle files

        Returns:
            dict: Dictionary of loaded encoders
        """
        logger.info(f"Attempting to load encoders from directory: {encoders_dir}")
        logger.info(f"Absolute path of encoders_dir: {os.path.abspath(encoders_dir)}")
        logger.info(f"Does encoders_dir ({encoders_dir}) exist? {os.path.exists(encoders_dir)}")
        if os.path.exists(encoders_dir):
            try:
                logger.info(f"Contents of encoders_dir ({encoders_dir}): {os.listdir(encoders_dir)}")
            except Exception as e:
                logger.error(f"Could not list contents of {encoders_dir}: {e}")
        else:
            # Check parent directory if encoders_dir doesn't exist
            parent_encoders_dir = os.path.dirname(encoders_dir.rstrip(os.sep))
            if not parent_encoders_dir: parent_encoders_dir = os.sep
            logger.info(f"Encoders_dir ({encoders_dir}) does not exist. Checking parent: {parent_encoders_dir}")
            if os.path.exists(parent_encoders_dir):
                 try:
                    logger.info(f"Contents of parent directory ({parent_encoders_dir}): {os.listdir(parent_encoders_dir)}")
                 except Exception as e_list_parent_encoder:
                    logger.error(f"Could not list contents of {parent_encoders_dir}: {e_list_parent_encoder}")
            else:
                logger.info(f"Parent directory {parent_encoders_dir} also does not exist.")


        encoders = {}
        for task in ["emotion", "sub_emotion", "intensity"]:
            encoder_file_path = os.path.join(encoders_dir, f"{task}_encoder.pkl")
            logger.info(f"Attempting to load encoder file: {encoder_file_path}")
            logger.info(f"Does encoder file ({encoder_file_path}) exist? {os.path.exists(encoder_file_path)}")
            try:
                with open(encoder_file_path, "rb") as f:
                    encoders[task] = pickle.load(f)
                logger.info(f"Successfully loaded encoder: {encoder_file_path}")
            except FileNotFoundError:
                logger.error(f"FileNotFoundError: Encoder file not found at {encoder_file_path}.")
                # If one encoder is missing, we should probably raise or handle it more explicitly
                # For now, this will cause a KeyError later if an encoder is missing but the dir exists.
                # If encoders_dir itself was not found, the earlier logs would indicate that.
                raise # Re-raise to ensure failure is propagated
            except Exception as e:
                logger.error(f"Error loading encoder {encoder_file_path}: {e}", exc_info=True)
                raise
        return encoders

    def load_best_model(self, weights_dir="models/weights", task="sub_emotion"):
        """
        Load the best model based on test F1 scores.

        Args:
            weights_dir (str): Directory containing model weights
            task (str): Task to optimize for ('emotion', 'sub_emotion', or 'intensity')

        Returns:
            float: F1 score of the loaded model
        """
        # Find all model files for the specified task
        pattern = f"best_test_in_{task}_f1_*.pt"

        model_files = glob.glob(os.path.join(weights_dir, pattern))

        if not model_files:
            raise FileNotFoundError(f"No model files found matching pattern: {pattern}")

        # Find the model with highest F1 score
        best_f1 = 0.0
        best_model_path = None

        for model_file in model_files:
            # Extract F1 score from filename
            f1_score = float(model_file.split("f1_")[1].split("_")[0])
            if f1_score > best_f1:
                best_f1 = f1_score
                best_model_path = model_file

        logger.info(f"Loading best model: {os.path.basename(best_model_path)}")
        logger.info(f"Best {task} F1 score: {best_f1:.4f}")        # Load the model weights
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        return best_f1

    def predict(self, texts, batch_size=16):
        """
        Make predictions on new texts.

        Args:
            texts (list): List of texts to predict on
            batch_size (int): Batch size for predictions

        Returns:
            pd.DataFrame: DataFrame containing predictions and mapped emotions
        """
        # Prepare features for all texts, ensuring they match expected_feature_dim
        all_features_for_dataset = []
        for text_item in texts:
            # Pass expected_feature_dim so FeatureExtractor handles padding/truncation
            features_for_item = self.feature_extractor.extract_all_features(
                text_item, expected_dim=self.expected_feature_dim
            )
            all_features_for_dataset.append(features_for_item)

        # Convert to numpy array before passing to dataset
        features_np_array = np.array(all_features_for_dataset, dtype=np.float32)

        dataset = EmotionDataset(
            texts,
            self.tokenizer,
            features=features_np_array,  # Pass the processed features
            # feature_extractor=self.feature_extractor,
            max_length=128
            # removed: expected_feature_dim=self.expected_feature_dim
        )

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Initialize lists for predictions
        predictions = {task: [] for task in self.output_tasks}
        all_sub_emotion_logits = []  # To store sub_emotion logits

        # Generate predictions
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating predictions", ncols=120):
                # Move batch data to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                features = batch["features"].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    features=features,
                )                # Get predictions for each task
                for i, task in enumerate(self.output_tasks):
                    if task == "sub_emotion":
                        # Store raw logits for sub_emotion
                        all_sub_emotion_logits.extend(outputs[task].cpu().detach())
                    pred = torch.argmax(outputs[task], dim=1).cpu().numpy()
                    predictions[task].extend(pred)

        # Create results DataFrame
        results = pd.DataFrame({"text": texts})
        # Add sub_emotion_logits to the DataFrame
        # Convert tensors to lists of floats for DataFrame compatibility
        results["sub_emotion_logits"] = [
            logit_tensor.tolist() for logit_tensor in all_sub_emotion_logits
        ]

        # Convert predictions to original labels
        for task in self.output_tasks:
            task_encoder = self.encoders[task]
            num_known_labels = len(task_encoder.classes_)

            # Cap predictions to the range known by the encoder
            # Ensure predictions are integers for min function and list indexing
            current_task_predictions = [int(p) for p in predictions[task]]
            capped_predictions = [
                min(p, num_known_labels - 1) for p in current_task_predictions
            ]

            try:
                results[f"predicted_{task}"] = task_encoder.inverse_transform(
                    capped_predictions
                )
            except ValueError as e:
                logger.error(f"Inverse_transform error for task '{task}': {e}")
                logger.error(f"Original '{task}' preds: " f"{predictions[task]}")
                logger.error(f"Capped '{task}' preds: " f"{capped_predictions}")
                logger.error(f"Encoder classes for '{task}': {task_encoder.classes_}")
                # Fallback: fill with a default "unknown" string
                results[f"predicted_{task}"] = ["unknown_error"] * len(
                    predictions[task]
                )

        # Add mapped emotions
        results = self.post_process(results)

        return results

    def post_process(self, df):
        """
        Post-process predictions to add mapped emotions and refine sub-emotions.

        Args:
            df (pd.DataFrame): DataFrame containing predictions,
                               including 'predicted_emotion'
                               and 'sub_emotion_logits'.

        Returns:
            pd.DataFrame: DataFrame with added mapped emotions and refined sub-emotions.
        """
        refined_sub_emotions = []
        sub_emotion_encoder = self.encoders["sub_emotion"]
        sub_emotion_classes = sub_emotion_encoder.classes_

        for index, row in df.iterrows():
            main_emotion_predicted = row["predicted_emotion"]
            # Convert list back to tensor
            logits = torch.tensor(row["sub_emotion_logits"])
            probabilities = torch.softmax(logits, dim=-1)            # Create a list of (sub_emotion_label, probability)
            sub_emotion_probs = []
            
            # Ensure we don't go beyond the bounds of either probabilities or classes
            max_idx = min(len(probabilities), len(sub_emotion_classes))
            
            for i in range(max_idx):
                sub_emotion_probs.append((sub_emotion_classes[i], probabilities[i].item()))
                
            # If there's a mismatch, log it for debugging
            if len(probabilities) != len(sub_emotion_classes):
                logger.warning(
                    f"Dimension mismatch: Model outputs {len(probabilities)} logits "
                    f"but encoder has {len(sub_emotion_classes)} classes. "
                    f"Using first {max_idx} predictions."
                )

            # Sort by probability in descending order
            sub_emotion_probs.sort(key=lambda x: x[1], reverse=True)

            # Default to original prediction
            chosen_sub_emotion = row["predicted_sub_emotion"]
            found_consistent_sub_emotion = False
            for sub_emotion_label, prob in sub_emotion_probs:
                mapped_emotion = self.emotion_mapping.get(sub_emotion_label)
                if mapped_emotion == main_emotion_predicted:
                    chosen_sub_emotion = sub_emotion_label
                    found_consistent_sub_emotion = True
                    break

            if not found_consistent_sub_emotion:
                logger.warning(
                    f"Could not find a consistent sub-emotion for main emotion "
                    f"'{main_emotion_predicted}' from the probability distribution for "
                    f"text: '{row['text']}'. Falling back to originally predicted "
                    f"sub-emotion: '{chosen_sub_emotion}'."
                )
            refined_sub_emotions.append(chosen_sub_emotion)

        df["predicted_sub_emotion"] = refined_sub_emotions

        # Map refined predicted sub-emotions to main emotions for verification
        df["emotion_pred_post_processed"] = df["predicted_sub_emotion"].map(
            self.emotion_mapping
        )

        return df


class EmotionPredictor:
    def __init__(self):
        self._model = None
        self._predictor = None

        # Ensure NLTK's vader_lexicon is available
        try:
            # Check if vader_lexicon is found by trying to load it indirectly
            # A more direct way like nltk.data.find might be preferable if
            # available for specific resources
            # For vader, it's typically located at 'sentiment/vader_lexicon.zip'
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            # If not found, download it
            nltk.download("vader_lexicon")

    def predict(self, texts, feature_config=None, reload_model=False):
        """
        Predicts emotions for the given text(s)

        Args:
            texts (str or list): Text or list of texts to analyze
            feature_config (dict, optional): Configuration for features to use in
            prediction
            reload_model (bool): Force reload the model even if cached

        Returns:
            dict or list: Dictionary with emotion predictions for a single text or
                        list of dictionaries for multiple texts
        """
        # Convert single text to list for uniform processing
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        # Default feature configuration if not provided
        if feature_config is None:
            feature_config = {
                "pos": False,
                "textblob": False,
                "vader": False,
                "tfidf": True,
                "emolex": True,
            }        # Load the model and create predictor if not already loaded or if
        # reload is requested
        if self._model is None or self._predictor is None or reload_model:
            _current_file_path_ep = os.path.abspath(__file__)
            _project_root_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(_current_file_path_ep))
            )            
            model_weights_filename = "baseline_weights.pt"
            # Construct paths relative to the project root
            model_path = os.path.join(
                _project_root_dir, "models", "weights", model_weights_filename
            )
            encoders_path = os.path.join(_project_root_dir, "models", "encoders")
            weights_dir = os.path.join(_project_root_dir, "models", "weights")
            
            # Auto-sync with Azure ML before loading model
            try:
                from .azure_model_sync import AzureMLModelManager
                logger.info("Attempting auto-sync with Azure ML before model loading...")
                manager = AzureMLModelManager(weights_dir=weights_dir)
                baseline_synced, dynamic_synced = manager.sync_on_startup()
                
                if baseline_synced:
                    logger.info("✓ Baseline model auto-downloaded from Azure ML")
                elif dynamic_synced:
                    logger.info("✓ Dynamic model auto-downloaded from Azure ML")
                else:
                    logger.info("Local models are up to date with Azure ML")
                    
            except Exception as e:
                logger.warning(f"Azure ML auto-sync failed, continuing with local models: {e}")

            # Initialize model loader
            loader = ModelLoader("microsoft/deberta-v3-xsmall")

            # Tokenizer
            # tokenizer = loader.tokenizer # Already part of loader instance
            feature_dim = 121 # This should be consistent with the model training

            # Load model
            num_classes = {"emotion": 7, "sub_emotion": 28, "intensity": 3}

            # The previous try-except block for /app vs ./ paths is removed.
            # loader.load_model and create_predictor will use the resolved paths.
            # Error handling for file not found is within loader.load_model.
            self._model = loader.load_model(
                feature_dim=feature_dim,
                num_classes=num_classes,
                weights_path=model_path,
            )
            # Create predictor with feature configuration
            self._predictor = loader.create_predictor(
                model=self._model,
                encoders_dir=encoders_path, # Use the resolved path
                feature_config=feature_config,
            )

        # Make predictions
        results = self._predictor.predict(texts)

        # Format the output
        output = []
        for i, text in enumerate(texts):
            prediction = {
                "text": text,
                "emotion": results.loc[i, "predicted_emotion"],
                "sub_emotion": results.loc[i, "predicted_sub_emotion"],
                "intensity": results.loc[i, "predicted_intensity"],
            }
            output.append(prediction)

        # Return single dictionary if input was a single text
        if single_input:
            return output[0]

        return output
