"""
Emotion classification model components.

Provides multi-task DEBERTA-based emotion classification with sub-emotion mapping
and intensity prediction. Supports both local and Azure ML model synchronization.
"""

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

# Configure NLTK data path for Docker environments
if os.path.exists("/app/nltk_data"):
    nltk.data.path.append("/app/nltk_data")
elif not any("nltk_data" in p for p in nltk.data.path):
    # Download essential NLTK data
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("vader_lexicon", quiet=True)


class DEBERTAClassifier(nn.Module):
    """
    Multi-task DEBERTA-based emotion classifier.

    Performs simultaneous classification for:
    - Main emotions (7 categories)
    - Sub-emotions (28 categories)
    - Emotion intensity (3 levels)

    Combines DEBERTA embeddings with engineered features through projection layers.
    """

    def __init__(
        self, model_name, feature_dim, num_classes, hidden_dim=256, dropout=0.1
    ):
        """
        Initialize the multi-task emotion classifier.

        Args:
            model_name (str): Pretrained DEBERTA model identifier
            feature_dim (int): Dimension of engineered features
            num_classes (dict): Class counts for each task
                               (emotion, sub_emotion, intensity)
            hidden_dim (int): Hidden layer dimension. Defaults to 256.
            dropout (float): Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        # Store configuration for potential serialization
        self.model_name = model_name
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Initialize pretrained DEBERTA backbone
        self.deberta = AutoModel.from_pretrained(model_name)
        deberta_dim = self.deberta.config.hidden_size

        # Project engineered features to match hidden dimension
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )

        combined_dim = deberta_dim + hidden_dim

        # Independent classification heads for each task
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
        Compute multi-task emotion predictions.

        Args:
            input_ids (torch.Tensor): Tokenized input text
            attention_mask (torch.Tensor): Attention mask for input
            features (torch.Tensor): Engineered features

        Returns:
            dict: Logits for each classification task
        """
        # Extract [CLS] token representation from DEBERTA
        deberta_output = self.deberta(
            input_ids=input_ids, attention_mask=attention_mask
        )
        deberta_embeddings = deberta_output.last_hidden_state[:, 0, :]

        # Project and combine embeddings
        projected_features = self.feature_projection(features)
        combined = torch.cat([deberta_embeddings, projected_features], dim=1)

        # Generate predictions for each task
        return {
            "emotion": self.emotion_classifier(combined),
            "sub_emotion": self.sub_emotion_classifier(combined),
            "intensity": self.intensity_classifier(combined),
        }


class ModelLoader:
    """
    Handles DEBERTA model and tokenizer loading with device management.

    Supports loading pretrained models, applying custom weights, and creating
    predictor instances. Provides automatic device selection (GPU/CPU).
    """

    def __init__(self, model_name="microsoft/deberta-v3-xsmall", device=None):
        """
        Initialize model loader with tokenizer.

        Args:
            model_name (str): Pretrained model identifier.
                             Defaults to 'microsoft/deberta-v3-xsmall'.
            device (torch.device, optional): Target device. Auto-detects if None.
        """
        self.model_name = model_name
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.info(f"Using device: {self.device} in ModelLoader")
        logger.info(f"Loading tokenizer from: {self.model_name}")

        self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_name)

    def load_model(
        self, feature_dim, num_classes, weights_path=None, hidden_dim=256, dropout=0.1
    ):
        """
        Create and optionally load pretrained model weights.

        Args:
            feature_dim (int): Dimension of engineered features
            num_classes (dict): Class counts for each classification task
            weights_path (str, optional): Path to saved model weights
            hidden_dim (int): Hidden layer dimension. Defaults to 256.
            dropout (float): Dropout probability. Defaults to 0.1.

        Returns:
            DEBERTAClassifier: Loaded model ready for inference or training

        Raises:
            FileNotFoundError: If weights_path doesn't exist
            RuntimeError: If weight loading fails
        """

        # Error handling
        try:

            # Initialize the DEBERTA-based classifier model
            model = DEBERTAClassifier(
                model_name=self.model_name,
                feature_dim=feature_dim,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )

            # Load weights if provided
            if weights_path is not None:

                # Load the weights (using BytesIO to handle seekable file requirement)
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

                # Load the state_dict into the model
                model.load_state_dict(new_state_dict)

            # Move model to the specified device
            model = model.to(self.device)

            return model

        # Error handling for file not found or loading issues
        except FileNotFoundError as e:
            logger.error(f"FileNotFoundError: Weights not found at {weights_path}.")
            raise e

    def create_predictor(
        self, model, encoders_dir="models/encoders", feature_config=None
    ):
        """
        Create predictor instance for emotion classification.

        Args:
            model (nn.Module): Trained emotion classification model
            encoders_dir (str): Directory containing label encoder files
            feature_config (dict, optional): Feature extraction configuration

        Returns:
            CustomPredictor: Ready-to-use predictor instance
        """
        return CustomPredictor(
            model=model,
            tokenizer=self.tokenizer,
            device=self.device,
            encoders_dir=encoders_dir,
            feature_config=feature_config,
        )

    def load_baseline_model(self, weights_dir="models/weights", sync_azure=True):
        """
        Load stable production model with optional Azure ML sync.

        Args:
            weights_dir (str): Directory containing model weights
            sync_azure (bool): Whether to sync with Azure ML on startup
        """

        # Error handling
        try:

            # Path to baseline model weights
            baseline_path = os.path.join(weights_dir, "baseline_weights.pt")

            # IF sync_azure is set to true
            if sync_azure:

                # Sync with Azure ML
                from .azure_sync import AzureMLModelManager
                manager = AzureMLModelManager(weights_dir)
                manager.auto_sync_on_startup(check_for_updates=True)

            # Load the baseline model weights
            self.model.load_state_dict(
                torch.load(baseline_path, map_location=self.device)
            )

            # Move model to the specified device
            self.model.to(self.device)

            # Set model to evaluation mode
            self.model.eval()

        # Error handling for file not found or loading issues
        except FileNotFoundError as e:
            logger.error(f"Baseline model not found at {baseline_path}.")
            raise e

    def load_dynamic_model(self, weights_dir="models/weights", sync_azure=True):
        """
        Load latest trained model with optional Azure ML sync.

        Args:
            weights_dir (str): Directory containing model weights
            sync_azure (bool): Whether to sync with Azure ML on startup
        """

        # Error handling
        try:

            # Path to dynamic model weights
            dynamic_path = os.path.join(weights_dir, "dynamic_weights.pt")

            # If sync_azure is set to true
            if sync_azure:

                # Sync with Azure ML
                from .azure_sync import AzureMLModelManager
                manager = AzureMLModelManager(weights_dir)
                manager.auto_sync_on_startup(check_for_updates=True)

            # Load the dynamic model weights
            self.model.load_state_dict(
                torch.load(dynamic_path, map_location=self.device)
            )

            # Move model to the specified device
            self.model.to(self.device)

            # Set model to evaluation mode
            self.model.eval()

        # Error handling for file not found or loading issues
        except FileNotFoundError as e:
            logger.error(f"Dynamic model not found at {dynamic_path}.")
            raise e

    def promote_dynamic_to_baseline(
        self, weights_dir="models/weights", sync_azure=True
    ):
        """
        Copies dynamic weights to baseline location, optionally syncing with Azure ML.

        Args:
            weights_dir (str): Directory containing model weights
            sync_azure (bool): Whether to sync with Azure ML on startup
        """

        # Error handling
        try:

            # If sync_azure is set to true
            if sync_azure:

                # Copy dynamic weights to baseline location
                from .azure_sync import promote_to_baseline_with_azure
                success = promote_to_baseline_with_azure(weights_dir)

                # If promotion was successful, return
                if success:
                    return

            # Fallback to local promotion
            dynamic_path = os.path.join(weights_dir, "dynamic_weights.pt")
            baseline_path = os.path.join(weights_dir, "baseline_weights.pt")

            # Copy dynamic to baseline
            shutil.copy(dynamic_path, baseline_path)

        except FileNotFoundError as e:
            logger.error(f"Could not promote dynamic to baseline: {e}")
            raise e

    def ensure_best_baseline_model(self):
        """
        Ensure we have the best available baseline model from Azure ML.

        This method checks Azure ML for models with better F1 scores than the
        current local baseline and downloads them if found. It forces a reload
        of the prediction model to use the updated baseline.

        Returns:
            bool: True if a better model was downloaded and loaded, False otherwise
        """
        try:
            # Get current project paths
            _current_file_path_ep = os.path.abspath(__file__)
            _project_root_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(_current_file_path_ep))
            )

            # Add /app if we are inside Docker container
            if _project_root_dir == "/" and os.path.exists("/app/models"):
                _project_root_dir = "/app"

            weights_dir = os.path.join(_project_root_dir, "models", "weights")

            # Check and download best baseline model
            from .azure_sync import AzureMLModelManager
            manager = AzureMLModelManager(weights_dir=weights_dir)

            logger.info("Checking for best baseline model in Azure ML...")
            best_baseline_updated = manager.download_best_baseline_model()

            if best_baseline_updated:
                logger.info("Better baseline model found and downloaded from Azure ML")
                # Force reload of the model on next prediction
                self._model = None
                self._predictor = None
                return True
            else:
                logger.info("Local baseline model is already the best available")
                return False

        except Exception as e:
            logger.error(f"Failed to check for best baseline model: {e}")
            return False


class CustomPredictor:
    """
    Multi-task emotion prediction engine.

    Handles emotion classification inference by combining the trained model
    with feature engineering and post-processing. Maps sub-emotions to
    main emotions for consistent predictions.
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
        Initialize emotion predictor with model and supporting components.

        Args:
            model (nn.Module): Trained emotion classification model
            tokenizer: Tokenizer for text preprocessing
            device (torch.device): Target device for inference
            encoders_dir (str): Directory containing label encoder files
            feature_config (dict, optional): Feature extraction configuration
        """

        # Error handling
        try:

            # Initializations
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            self.expected_feature_dim = model.feature_projection[0].in_features

            # Feature configuration
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

            # Paths
            _current_file_path_cp = os.path.abspath(__file__)
            _project_root_dir_cp = os.path.dirname(
                os.path.dirname(os.path.dirname(_current_file_path_cp))
            )

            # Fix for Docker container: if we're in /app, use /app as project root
            if _project_root_dir_cp == "/" and os.path.exists("/app/models"):
                _project_root_dir_cp = "/app"
            emolex_path = os.path.join(
                _project_root_dir_cp,
                "models",
                "features",
                "EmoLex",
                "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
            )

            # Initialize feature extractor with configuration and lexicon path
            self.feature_extractor = FeatureExtractor(
                feature_config=self.feature_config, lexicon_path=emolex_path
            )

            # If TF-IDF is enabled, fit the vectorizer if not already done
            if (
                self.feature_config.get("tfidf", False)
                and self.feature_extractor.tfidf_vectorizer is None
            ):
                # Load actual training data to fit TF-IDF properly
                training_data_path = os.path.join(
                    _project_root_dir_cp, "data", "processed", "train.csv"
                )

                # Load the training data if it exists
                if os.path.exists(training_data_path):
                    train_df = pd.read_csv(training_data_path)
                    training_texts = train_df["text"].tolist()
                    self.feature_extractor.fit_tfidf(training_texts)

            # Emotion mapping dictionary to align sub-emotions with main emotions
            self.emotion_mapping = {
                "curiosity": "happiness", "neutral": "neutral", "annoyance": "anger",
                "confusion": "surprise", "disappointment": "sadness",
                "excitement": "happiness", "surprise": "surprise",
                "realization": "surprise", "desire": "happiness",
                "approval": "happiness", "disapproval": "disgust",
                "embarrassment": "fear", "admiration": "happiness",
                "anger": "anger", "optimism": "happiness", "sadness": "sadness",
                "joy": "happiness", "fear": "fear", "remorse": "sadness",
                "gratitude": "happiness", "disgust": "disgust", "love": "happiness",
                "relief": "happiness", "grief": "sadness", "amusement": "happiness",
                "caring": "happiness", "nervousness": "fear", "pride": "happiness",
            }

            # Load label encoders
            self.encoders = self._load_encoders(encoders_dir)

            # Output tasks for predictions
            self.output_tasks = ["emotion", "sub_emotion", "intensity"]

        except Exception as e:
            logger.error(f"Error initializing CustomPredictor: {e}", exc_info=True)
            raise e

    def _load_encoders(self, encoders_dir):
        """
        Load label encoders for emotion classification tasks.

        Args:
            encoders_dir (str): Directory containing encoder pickle files

        Returns:
            dict: Loaded encoders keyed by task name

        Raises:
            FileNotFoundError: If encoder files are missing
        """

        # Error handling
        try:

            # Initialize encoders dictionary
            encoders = {}

            # Loop over the tasks
            for task in ["emotion", "sub_emotion", "intensity"]:

                # Construct the file path for each encoder
                encoder_file_path = os.path.join(encoders_dir, f"{task}_encoder.pkl")

                # Load the encoder from the specified file
                with open(encoder_file_path, "rb") as f:
                    encoders[task] = pickle.load(f)

            return encoders

        # Error handling if error
        except FileNotFoundError as e:
            logger.error(f"Encoder files not found in {encoders_dir}.")
            raise e

    def predict(self, texts, batch_size=16):
        """
        Generate emotion predictions for text inputs.

        Processes texts through feature extraction, model inference, and
        post-processing to produce final emotion classifications.

        Args:
            texts (list): List of text strings to classify
            batch_size (int): Batch size for inference. Defaults to 16.

        Returns:
            pd.DataFrame: Predictions with mapped emotions and confidence scores
        """

        # Error handling
        try:

            # Extract features from texts
            all_features_for_dataset = []
            for text_item in texts:
                features_for_item = self.feature_extractor.extract_all_features(
                    text_item,
                    expected_dim=self.expected_feature_dim
                )
                all_features_for_dataset.append(features_for_item)

            # Convert features into array
            features_np_array = np.array(all_features_for_dataset, dtype=np.float32)

            # Create dataset for DataLoader
            dataset = EmotionDataset(
                texts,
                self.tokenizer,
                features=features_np_array,
                max_length=128
                # feature_extractor=self.feature_extractor,
                # expected_feature_dim=self.expected_feature_dim
            )

            # Create DataLoader for batching
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            # Initialize lists for predictions
            predictions = {task: [] for task in self.output_tasks}
            all_sub_emotion_logits = []

            # Set model to evaluation mode
            self.model.eval()

            # Disable gradient calculation for inference
            with torch.no_grad():

                # Loop through DataLoader batches
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
                    )

                    # Loop over each task's outputs
                    for i, task in enumerate(self.output_tasks):

                        # If task is sub_emotion, collect logits
                        if task == "sub_emotion":
                            all_sub_emotion_logits.extend(outputs[task].cpu().detach())

                        # Get the prediction (highest logit) and save it
                        pred = torch.argmax(outputs[task], dim=1).cpu().numpy()
                        predictions[task].extend(pred)

            # Convert texts into a DataFrame
            results = pd.DataFrame({"text": texts})

            # Add sub_emotion_logits to the DataFrame
            results["sub_emotion_logits"] = [
                logit_tensor.tolist() for logit_tensor in all_sub_emotion_logits
            ]

            # Loop over each task
            for task in self.output_tasks:

                # Get the encoder for the task
                task_encoder = self.encoders[task]

                # Get number of known labels for the task
                num_known_labels = len(task_encoder.classes_)

                # Convert predictions to integers
                current_task_predictions = [int(p) for p in predictions[task]]

                # Cap predictions to the range known by the encoder
                capped_predictions = [
                    min(p, num_known_labels - 1) for p in current_task_predictions
                ]

                # Get the inverse transform to get the predicted labels
                try:
                    results[f"predicted_{task}"] = task_encoder.inverse_transform(
                        capped_predictions
                    )

                # If inverse transform fails, handle the error
                except ValueError as e:
                    results[f"predicted_{task}"] = ["unknown_error"] * len(
                        predictions[task]
                    )
                    logger.error(
                        f"Error during inverse transform for task '{task}': {e}",
                        exc_info=True
                    )

            # Add mapped emotions
            results = self.post_process(results)

            return results

        # If error occurs during prediction
        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            raise e

    def post_process(self, df):
        """
        Refine predictions by aligning sub-emotions with main emotions.

        Uses probability distributions to select sub-emotions that are
        consistent with predicted main emotions, improving classification
        coherence.

        Args:
            df (pd.DataFrame): Predictions with sub_emotion_logits column

        Returns:
            pd.DataFrame: Refined predictions with emotion_pred_post_processed
        """

        # Error handling
        try:

            # Initialize empty list for refined sub-emotions
            refined_sub_emotions = []

            # Encoder for sub-emotions
            sub_emotion_encoder = self.encoders["sub_emotion"]

            # Get the sub-emotion classes from the encoder
            sub_emotion_classes = sub_emotion_encoder.classes_

            # Loop through each row in the DataFrame
            for index, row in df.iterrows():

                # Get the main emotion predicted by the model
                main_emotion_predicted = row["predicted_emotion"]

                # Convert list back to tensor
                logits = torch.tensor(row["sub_emotion_logits"])

                # Apply softmax to get probabilities
                probabilities = torch.softmax(logits, dim=-1)

                # Initialize a list of (sub_emotion_label, probability)
                sub_emotion_probs = []

                # Get the minimum length to avoid index errors
                # Ensure we don't go beyond bounds of either probabilities or classes
                max_idx = min(len(probabilities), len(sub_emotion_classes))

                # Loop through the probabilities and sub-emotion classes
                for i in range(max_idx):
                    sub_emotion_probs.append(
                        (sub_emotion_classes[i], probabilities[i].item())
                    )

                # Sort by probability in descending order
                sub_emotion_probs.sort(key=lambda x: x[1], reverse=True)

                # Get the predicted sub-emotion
                chosen_sub_emotion = row["predicted_sub_emotion"]

                # Loop through sorted sub-emotion probabilities
                for sub_emotion_label, prob in sub_emotion_probs:

                    # Map the sub-emotion label to its main emotion
                    mapped_emotion = self.emotion_mapping.get(sub_emotion_label)

                    # If mapped emotion matches prediction update chosen sub-emotion
                    if mapped_emotion == main_emotion_predicted:
                        chosen_sub_emotion = sub_emotion_label
                        break

                # Append the chosen sub-emotion to the refined list
                refined_sub_emotions.append(chosen_sub_emotion)

            # Add refined sub-emotions to the DataFrame
            df["predicted_sub_emotion"] = refined_sub_emotions

            # Map refined predicted sub-emotions to main emotions for verification
            df["emotion_pred_post_processed"] = df["predicted_sub_emotion"].map(
                self.emotion_mapping
            )

            return df

        # If error occurs during post-processing
        except Exception as e:
            logger.error(f"Error during post-processing: {e}", exc_info=True)
            raise e


class EmotionPredictor:
    """
    High-level interface for emotion classification.

    Provides a simple API for predicting emotions from text with automatic
    model loading, Azure ML synchronization, and feature configuration.
    Handles single texts or batches transparently.
    """

    def __init__(self):
        """Initialize predictor with lazy model loading."""

        # Initializations
        self._model = None
        self._predictor = None

        # Download vader_lexicon if not found
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon")

    def predict(self, texts, feature_config=None, reload_model=False):
        """
        Predict emotions for single text or batch of texts.

        Automatically handles model loading, feature extraction, and result
        formatting. Returns structured predictions with emotion, sub-emotion,
        and intensity classifications.

        Args:
            texts (str or list): Text(s) to classify
            feature_config (dict, optional): Feature extraction settings.
                Defaults to tfidf=True, emolex=True, others=False.
            reload_model (bool): Force model reload. Defaults to False.

        Returns:
            dict or list: Prediction dict for single text, list for batch
        """

        # Determine if input is a single text or a list of texts
        if isinstance(texts, str):
            texts = [texts]       # Convert single text to list
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
            }

        # If model or predictor is not loaded, or if reload_model is True
        if self._model is None or self._predictor is None or reload_model:

            # Set the paths
            _current_file_path_ep = os.path.abspath(__file__)
            _project_root_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(_current_file_path_ep))
            )

            # Add /app if we are inside Docker container
            if _project_root_dir == "/" and os.path.exists("/app/models"):
                _project_root_dir = "/app"            # Set the model path
            model_path = os.path.join(
                _project_root_dir, "models", "weights", "baseline_weights.pt"
            )

            # Path to encoders and weights directories
            encoders_path = os.path.join(_project_root_dir, "models", "encoders")
            weights_dir = os.path.join(_project_root_dir, "models", "weights")

            # Auto-sync with Azure ML before loading model - check for best baseline
            try:
                from .azure_sync import AzureMLModelManager
                manager = AzureMLModelManager(weights_dir=weights_dir)

                # First perform regular sync
                baseline_synced, dynamic_synced = manager.sync_on_startup()

                # Then check for best baseline model based on F1 score
                logger.info(
                    "Checking Azure ML for best baseline model based on F1 score..."
                )
                best_baseline_updated = manager.download_best_baseline_model()

                if best_baseline_updated:
                    logger.info("Downloaded better baseline model from Azure ML")
                else:
                    logger.info("Local baseline model is already the best available")

            except Exception as e:
                logger.warning(
                    f"Azure ML auto-sync failed, continuing with local models: {e}"
                )

            # Initialize model loader
            loader = ModelLoader("microsoft/deberta-v3-xsmall")

            # Initialize model arguments
            feature_dim = 121
            num_classes = {"emotion": 7, "sub_emotion": 28, "intensity": 3}

            # Load the model
            self._model = loader.load_model(
                feature_dim=feature_dim,
                num_classes=num_classes,
                weights_path=model_path,
            )

            # Create predictor with feature configuration
            self._predictor = loader.create_predictor(
                model=self._model,
                encoders_dir=encoders_path,
                feature_config=feature_config,
            )

        # Make predictions
        results = self._predictor.predict(texts)

        # Format the output into list of dictionaries
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

    def ensure_best_baseline(self):
        """
        Ensure we have the best available baseline model from Azure ML.

        This is an alias for ensure_best_baseline_model() for backward compatibility.
        Checks Azure ML for models with better F1 scores than the current local
        baseline and downloads them if found.

        Returns:
            bool: True if a better model was downloaded and loaded, False otherwise
        """
        # Delegate to the ModelLoader instance method
        loader = ModelLoader("microsoft/deberta-v3-xsmall")
        return loader.ensure_best_baseline_model()

    def ensure_best_baseline_model(self):
        """
        Ensure we have the best available baseline model from Azure ML.

        This method checks Azure ML for models with better F1 scores than the
        current local baseline and downloads them if found. It forces a reload
        of the prediction model to use the updated baseline.

        Returns:
            bool: True if a better model was downloaded and loaded, False otherwise
        """
        try:
            # Get current project paths
            _current_file_path_ep = os.path.abspath(__file__)
            _project_root_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(_current_file_path_ep))
            )

            # Add /app if we are inside Docker container
            if _project_root_dir == "/" and os.path.exists("/app/models"):
                _project_root_dir = "/app"

            weights_dir = os.path.join(_project_root_dir, "models", "weights")

            # Check and download best baseline model
            from .azure_sync import AzureMLModelManager
            manager = AzureMLModelManager(weights_dir=weights_dir)

            logger.info("Checking for best baseline model in Azure ML...")
            best_baseline_updated = manager.download_best_baseline_model()

            if best_baseline_updated:
                logger.info("Better baseline model found and downloaded from Azure ML")
                # Force reload of the model on next prediction
                self._model = None
                self._predictor = None
                return True
            else:
                logger.info("Local baseline model is already the best available")
                return False

        except Exception as e:
            logger.error(f"Failed to check for best baseline model: {e}")
            return False
