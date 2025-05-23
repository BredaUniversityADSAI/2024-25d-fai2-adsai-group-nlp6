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
            tuple: (emotion_logits, sub_emotion_logits, intensity_logits)
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

        return emotion_logits, sub_emotion_logits, intensity_logits


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
            logger.info(f"Loading weights from: {weights_path}")
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
            logger.info("Successfully loaded model weights")

        # Move model to device
        model = model.to(self.device)

        return model

    def create_predictor(
        self, model, encoders_dir="./models/encoders", feature_config=None
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
            model=model,
            tokenizer=self.tokenizer,
            device=self.device,
            encoders_dir=encoders_dir,
            feature_config=feature_config,
        )


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
        encoders_dir="./models/encoders",
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
        )

        # TF-IDF fitting for CustomPredictor should happen here if tfidf is
        # enabled in its config
        if (
            self.feature_config.get("tfidf", False)
            and self.feature_extractor.tfidf_vectorizer is None
        ):
            logger.info("CustomPredictor: TF-IDF enabled, fitting dummy doc for init.")
            # Fit with a dummy document to initialize the vocabulary for
            # consistent TF-IDF dimension
            self.feature_extractor.fit_tfidf(
                ["dummy document for tfidf initialization in predictor"]
            )
            # Verify dimension
            calculated_dim_after_dummy_fit = self.feature_extractor.get_feature_dim()
            if calculated_dim_after_dummy_fit != self.expected_feature_dim:
                logger.warning(
                    f"Predictor: TF-IDF dim {calculated_dim_after_dummy_fit} \
                        vs model {self.expected_feature_dim}. "
                    "Check config."
                )
            else:
                logger.info(
                    f"Predictor: TF-IDF init. Dim {calculated_dim_after_dummy_fit} \
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
        encoders = {}
        for task in ["emotion", "sub_emotion", "intensity"]:
            with open(f"{encoders_dir}/{task}_encoder.pkl", "rb") as f:
                encoders[task] = pickle.load(f)
        return encoders

    def load_best_model(self, weights_dir="./models/weights", task="sub_emotion"):
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
        logger.info(f"Best {task} F1 score: {best_f1:.4f}")

        # Load the model weights
        self.model.load_state_dict(torch.load(best_model_path))
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
                )

                # Get predictions for each task
                for i, task in enumerate(self.output_tasks):
                    if task == "sub_emotion":
                        # Store raw logits for sub_emotion
                        all_sub_emotion_logits.extend(outputs[i].cpu().detach())
                    pred = torch.argmax(outputs[i], dim=1).cpu().numpy()
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
            probabilities = torch.softmax(logits, dim=-1)

            # Create a list of (sub_emotion_label, probability)
            sub_emotion_probs = []
            for i, prob in enumerate(probabilities):
                if i < len(sub_emotion_classes):  # Ensure index is within bounds
                    sub_emotion_probs.append((sub_emotion_classes[i], prob.item()))
                else:
                    # This case should ideally not happen
                    logger.warning(
                        f"Index {i} for sub-emotion probabilities is out of "
                        f"bounds for encoder classes (len: "
                        f"{len(sub_emotion_classes)}). Skipping."
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
            }

        # Load the model and create predictor if not already loaded or if
        # reload is requested
        if self._model is None or self._predictor is None or reload_model:
            # Determine project root directory
            _current_file_path_ep = os.path.abspath(__file__)
            # base_dir will now point to the project root
            base_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(_current_file_path_ep))
            )

            # Initialize model loader
            loader = ModelLoader("microsoft/deberta-v3-xsmall")

            # Tokenizer
            # tokenizer = loader.tokenizer
            feature_dim = 121

            # Load model
            num_classes = {"emotion": 7, "sub_emotion": 28, "intensity": 3}

            model_path = os.path.join(
                base_dir, "models", "weights", "best_test_in_emotion_f1_0.7851.pt"
            )
            self._model = loader.load_model(
                feature_dim=feature_dim,
                num_classes=num_classes,
                weights_path=model_path,
            )

            # Create predictor with feature configuration
            encoders_dir = os.path.join(base_dir, "models", "encoders")
            self._predictor = loader.create_predictor(
                model=self._model,
                encoders_dir=encoders_dir,
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
