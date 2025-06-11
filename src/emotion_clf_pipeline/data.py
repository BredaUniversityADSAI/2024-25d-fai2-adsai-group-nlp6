import glob  # Add glob import
import logging
import os
import pickle  # Add pickle import

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_class_weight
from textblob import TextBlob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import FeatureExtractor from .features
from .features import (
    EmolexFeatureExtractor,
    FeatureExtractor,
    POSFeatureExtractor,
    TextBlobFeatureExtractor,
    VaderFeatureExtractor,
)

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    A class to handle loading and preprocessing of emotion classification datasets.

    This class handles:
    - Loading training and test data from CSV files
    - Cleaning and preprocessing the data
    - Mapping emotions to standardized categories
    - Visualizing data distributions

    Attributes:
        emotion_mapping (dict): Dictionary mapping sub-emotions to standardized emotions
        train_df (pd.DataFrame): Processed training data
        test_df (pd.DataFrame): Processed test data
    """

    def __init__(self):

        # Initialize the DataLoader with emotion mapping.
        self.train_df = None
        self.test_df = None

    def load_training_data(self, data_dir="./../../data/raw/all groups"):
        """
        Load and preprocess training data from multiple CSV files.

        Args:
            data_dir (str): Directory containing training data CSV files

        Returns:
            pd.DataFrame: Processed training data
        """

        # Load the dataset (contains train_data-0001.csv, train_data-0002.csv, etc.)
        self.train_df = pd.DataFrame()

        # Loop over all files in the data directory
        for i_file in os.listdir(data_dir):

            # If the file is not a CSV, skip it
            if not i_file.endswith(".csv"):
                logger.warning(f"Skipping non-CSV file: {i_file}")
                continue

            # Read the current CSV file and select specific columns
            try:
                df_ = pd.read_csv(os.path.join(data_dir, i_file))[
                    [
                        "start_time",
                        "end_time",
                        "text",
                        "emotion",
                        "sub-emotion",
                        "intensity",
                    ]
                ]
            except Exception as e:
                logger.error(f"Error reading {i_file}: {e}")
                continue

            # Handle column name variations (sub-emotion vs sub_emotion)
            if "sub-emotion" in df_.columns:
                df_ = df_.rename(columns={"sub-emotion": "sub_emotion"})

            # Concatenate the current file's data with the main DataFrame
            self.train_df = pd.concat([self.train_df, df_])

        # Drop null and duplicate rows
        self.train_df = self.train_df.dropna()
        self.train_df = self.train_df.drop_duplicates()

        # Reset index of the combined DataFrame
        self.train_df = self.train_df.reset_index(drop=True)

        return self.train_df

    def load_test_data(self, test_file="./../../data/test_data-0001.csv"):
        """
        Load and preprocess test data from a CSV file.

        Args:
            test_file (str): Path to the test data CSV file

        Returns:
            pd.DataFrame: Processed test data
        """

        # Read the test data CSV file
        try:
            self.test_df = pd.read_csv(test_file)[
                [
                    "start_time",
                    "end_time",
                    "text",
                    "emotion",
                    "sub-emotion",
                    "intensity",
                ]
            ]
        except Exception as e:
            logger.error(f"Error reading test file {test_file}: {e}")
            return None

        # Handle column name variations (sub-emotion vs sub_emotion)
        if "sub-emotion" in self.test_df.columns:
            self.test_df = self.test_df.rename(columns={"sub-emotion": "sub_emotion"})

        # Drop null and duplicate rows
        self.test_df = self.test_df.dropna()
        self.test_df = self.test_df.drop_duplicates()

        # Reset index of the test DataFrame
        self.test_df = self.test_df.reset_index(drop=True)

        return self.test_df

    def plot_distributions(self):
        """Plot distributions of emotions, sub-emotions, and intensities \
            for both training and test sets."""
        # Distribution of emotions in the training set
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        for i, col in enumerate(["emotion", "sub_emotion", "intensity"]):
            sns.countplot(data=self.train_df, x=col, palette="Set2", ax=axes[i])
            axes[i].set_title(f"'{col.capitalize()}' Distribution in Train/Val Set")
            axes[i].set_xlabel(col.capitalize())
            axes[i].set_ylabel("Count")
            axes[i].tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.show()

        # Distribution of emotions in the test set
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        for i, col in enumerate(["emotion", "sub_emotion", "intensity"]):
            sns.countplot(data=self.test_df, x=col, palette="Set2", ax=axes[i])
            axes[i].set_title(f"'{col.capitalize()}' Distribution in Test Set")
            axes[i].set_xlabel(col.capitalize())
            axes[i].set_ylabel("Count")
            axes[i].tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.show()


class DataPreparation:
    """
    A class to handle data preparation for emotion classification tasks.

    This class handles:
    - Label encoding for target variables
    - Dataset creation
    - Dataloader setup

    Args:
        output_columns (list): List of output column names to encode
        model_name (str): Name of the pretrained model to use for tokenization
        max_length (int): Maximum sequence length for tokenization
        batch_size (int): Batch size for dataloaders
        feature_config (dict, optional): Configuration for feature extraction
    """

    def __init__(
        self,
        output_columns,
        tokenizer,
        max_length=128,
        batch_size=16,
        feature_config=None,
        encoders_save_dir=None,  # Add encoders_save_dir
        encoders_load_dir=None,  # Add encoders_load_dir
    ):
        self.output_columns = output_columns
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

        # Initialize label encoders
        self.label_encoders = {col: LabelEncoder() for col in output_columns}

        # Determine project root directory to construct lexicon path
        _current_file_path_dp = os.path.abspath(__file__)
        # Assuming data.py is in src/emotion_clf_pipeline/
        _project_root_dir_dp = os.path.dirname(
            os.path.dirname(os.path.dirname(_current_file_path_dp))
        )

        # Fix for Docker container: if we're in /app, use /app as project root
        if _project_root_dir_dp == "/" and os.path.exists("/app/models"):
            _project_root_dir_dp = "/app"

        emolex_lexicon_path = os.path.join(
            _project_root_dir_dp,
            "models",
            "features",
            "EmoLex",
            "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
        )
        # Use provided encoders_save_dir or default
        self.encoders_output_dir = (
            encoders_save_dir
            if encoders_save_dir
            else os.path.join(_project_root_dir_dp, "models", "encoders")
        )
        # Store encoders_load_dir
        self.encoders_input_dir = encoders_load_dir

        # Attempt to load encoders if encoders_input_dir is provided
        self.encoders_loaded = self._load_encoders()

        # Initialize feature extractor with configuration and lexicon path
        self.feature_extractor = FeatureExtractor(
            feature_config=feature_config, lexicon_path=emolex_lexicon_path
        )

    def _load_encoders(self):
        """Load label encoders from disk if encoders_input_dir is set."""
        if not self.encoders_input_dir:
            logger.info(
                "Encoder input directory not provided. Will fit new encoders if training."
            )
            return False

        loaded_all = True
        for col in self.output_columns:
            encoder_path = os.path.join(self.encoders_input_dir, f"{col}_encoder.pkl")
            if os.path.exists(encoder_path):
                try:
                    with open(encoder_path, "rb") as f:
                        self.label_encoders[col] = pickle.load(f)
                    logger.info(f"Loaded encoder for {col} from {encoder_path}")
                except Exception as e:
                    logger.error(
                        f"Error loading encoder for {col} from {encoder_path}: {e}. A new encoder will be used."
                    )
                    self.label_encoders[col] = LabelEncoder()  # Revert to new encoder
                    loaded_all = False
            else:
                logger.warning(
                    f"Encoder file not found for {col} at {encoder_path}. A new encoder will be used and fitted if training data is provided."
                )
                self.label_encoders[col] = (
                    LabelEncoder()
                )  # Ensure it's a new encoder if not found
                loaded_all = False

        if loaded_all:
            logger.info("All encoders loaded successfully.")
        else:
            logger.warning(
                "One or more encoders failed to load or were not found. New encoders will be fitted for these if training data is provided."
            )
        return loaded_all

    def apply_data_augmentation(
        self,
        train_df,
        balance_strategy="equal",
        samples_per_class=None,
        augmentation_ratio=2,
        random_state=42,
    ):
        """
        Apply text augmentation to balance the training data.

        Args:
            train_df (pd.DataFrame): Training dataframe
            balance_strategy (str, optional): Strategy for balancing. Options:
            'equal', 'majority', 'target'. Defaults to 'equal'.
            samples_per_class (int, optional): Number of samples per class for
            'equal' or 'target' strategy. Defaults to None.
            augmentation_ratio (int, optional): Maximum ratio of augmented to
            original samples. Defaults to 2.
            random_state (int, optional): Random seed. Defaults to 42.

        Returns:
            pd.DataFrame: Balanced training dataframe
        """
        # Import the TextAugmentor
        from .augmentation import TextAugmentor

        logger.info(f"Applying data augmentation with strategy: {balance_strategy}")
        original_class_dist = train_df["emotion"].value_counts()
        logger.info("Original class distribution:")
        for emotion, count in original_class_dist.items():
            logger.info(f"  {emotion}: {count}")

        # Create an instance of TextAugmentor
        augmentor = TextAugmentor(random_state=random_state)

        # Apply the appropriate balancing strategy
        if balance_strategy == "equal":
            # Generate exactly equal samples per class
            if samples_per_class is None:
                # If not specified, use the average count
                samples_per_class = int(
                    len(train_df) / len(train_df["emotion"].unique())
                )

            balanced_df = augmentor.generate_equal_samples(
                train_df,
                text_column="text",
                emotion_column="emotion",
                samples_per_class=samples_per_class,
                random_state=random_state,
            )

        elif balance_strategy == "majority":
            # Balance up to the majority class
            balanced_df = augmentor.balance_dataset(
                train_df,
                text_column="text",
                emotion_column="emotion",
                target_count=None,  # Use majority class count
                augmentation_ratio=augmentation_ratio,
                random_state=random_state,
            )

        elif balance_strategy == "target":
            # Balance to a target count
            if samples_per_class is None:
                # If not specified, use the median count
                samples_per_class = int(train_df["emotion"].value_counts().median())

            balanced_df = augmentor.balance_dataset(
                train_df,
                text_column="text",
                emotion_column="emotion",
                target_count=samples_per_class,
                augmentation_ratio=augmentation_ratio,
                random_state=random_state,
            )

        else:
            raise ValueError(f"Unknown balance strategy: {balance_strategy}")

        # Apply additional sub-emotion balancing if needed
        if "sub_emotion" in self.output_columns:
            logger.info("After emotion balancing, checking sub-emotion distribution:")
            sub_emotion_dist = balanced_df["sub_emotion"].value_counts()
            logger.info(f"Sub-emotion classes: {len(sub_emotion_dist)}")
            logger.info(
                f"Min class size: {sub_emotion_dist.min()}, "
                f"Max class size: {sub_emotion_dist.max()}"
            )

            # If sub-emotion is highly imbalanced, apply additional balancing
            imbalance_ratio = sub_emotion_dist.max() / sub_emotion_dist.min()
            if imbalance_ratio > 5:  # If max/min ratio is greater than 5
                logger.info(
                    f"Sub-emotion imbalance ratio: {imbalance_ratio:.1f}, "
                    "applying additional balancing"
                )

                # Apply augmentation for sub-emotions with extreme imbalance
                sub_balanced_df = augmentor.balance_dataset(
                    balanced_df,
                    text_column="text",
                    emotion_column="sub_emotion",
                    target_count=max(
                        50, sub_emotion_dist.median() // 2
                    ),  # Target at least 50 samples or half median
                    augmentation_ratio=1,  # Keep augmentation minimal
                    random_state=random_state,
                )
                balanced_df = sub_balanced_df

        return balanced_df

    def prepare_data(
        self,
        train_df,
        test_df=None,
        validation_split=0.2,
        apply_augmentation=False,
        balance_strategy="equal",
        samples_per_class=None,
        augmentation_ratio=2,
    ):
        """
        Prepare data for training emotion classification models.

        Args:
            train_df (pd.DataFrame): Training dataframe
            test_df (pd.DataFrame, optional): Test dataframe. Defaults to None.
            validation_split (float, optional): Fraction of training data to use
            for validation. Defaults to 0.2.
            apply_augmentation (bool, optional): Whether to apply data
            augmentation. Defaults to False.
            balance_strategy (str, optional): Strategy for balancing if
            augmentation is applied. Options: 'equal', 'majority', 'target'.
            Defaults to 'equal'.
            samples_per_class (int, optional): Number of samples per class for
            balancing. Defaults to None.
            augmentation_ratio (int, optional): Maximum ratio of augmented to
            original samples. Defaults to 2.

        Returns:
            tuple: (train_dataset, val_dataset, test_dataset, train_dataloader,
            val_dataloader, test_dataloader, class_weights_tensor)
        """
        # Create output directory for encoders if it doesn't exist and we plan to save
        if not self.encoders_loaded and self.encoders_output_dir:
            os.makedirs(self.encoders_output_dir, exist_ok=True)
            logger.info(
                f"Ensured encoder output directory exists: {self.encoders_output_dir}"
            )

        # Fit label encoders on training data ONLY IF NOT LOADED
        if not self.encoders_loaded:
            logger.info(
                "Fitting new label encoders as they were not loaded or load failed."
            )
            for col in self.output_columns:
                if col in train_df.columns:
                    # Ensure the column is treated as string for consistent fitting
                    self.label_encoders[col].fit(train_df[col].astype(str))
                    logger.info(f"Fitted encoder for column: {col}")
                else:
                    logger.warning(
                        f"Column {col} not found in train_df for fitting encoder."
                    )
            # Save label encoders if they were just fitted and a save directory is provided
            if self.encoders_output_dir:
                self._save_encoders()
        else:
            logger.info("Using pre-loaded label encoders.")

        # Transform training data labels
        for col in self.output_columns:
            if col in train_df.columns:
                try:
                    # Ensure the column is treated as string for consistent transformation
                    train_df[f"{col}_encoded"] = self.label_encoders[col].transform(
                        train_df[col].astype(str)
                    )
                except ValueError as e:
                    logger.error(
                        f"Error transforming column {col} in training data: {e}"
                    )
                    logger.error(
                        f"Classes known to encoder for {col}: {list(self.label_encoders[col].classes_) if hasattr(self.label_encoders[col], 'classes_') else 'Encoder not fitted or classes_ not available'}"
                    )
                    raise e  # Or handle more gracefully
            else:
                logger.warning(f"Column {col} (for encoding) not found in train_df.")

        # Split into train and validation sets
        if validation_split == 0.0:  # Handle the case causing the error
            train_indices = list(range(len(train_df)))
            val_indices = []
            logger.info(
                "validation_split is 0.0, using all train_df for train_indices."
            )
        elif validation_split > 0 and validation_split < 1:  # Standard case
            stratify_on = None
            if self.output_columns and self.output_columns[0] in train_df:
                # sklearn's train_test_split handles cases with single class for stratification
                # by not stratifying if it's not possible.
                stratify_on = train_df[self.output_columns[0]]

            train_indices, val_indices = train_test_split(
                range(len(train_df)),
                test_size=validation_split,
                random_state=42,
                stratify=stratify_on,
            )
        else:
            # If validation_split is not 0.0 and not in (0.0, 1.0)
            # This case should ideally not be hit with current CLI usage (0.0 or 0.1).
            raise ValueError(
                f"Unsupported validation_split value: {validation_split}. Must be 0.0 or in (0.0, 1.0)."
            )

        # Fit TF-IDF vectorizer on training texts
        logger.info("Fitting TF-IDF vectorizer...")
        self.feature_extractor.fit_tfidf(train_df["text"].values)

        # Extract features for all texts
        logger.info("Extracting features for training data...")
        train_features = []
        for text in tqdm(
            train_df["text"],
            desc="Processing training texts",
            ncols=120,
            colour="green",
        ):
            train_features.append(self.feature_extractor.extract_all_features(text))
        train_features = np.array(train_features)

        # Create train and validation datasets
        train_dataset = EmotionDataset(
            texts=train_df["text"].values[train_indices],
            labels=train_df[[f"{col}_encoded" for col in self.output_columns]].values[
                train_indices
            ],
            features=train_features[train_indices],
            tokenizer=self.tokenizer,
            feature_extractor=self.feature_extractor,
            max_length=self.max_length,
            output_tasks=self.output_columns,
        )

        val_dataset = EmotionDataset(
            texts=train_df["text"].values[val_indices],
            labels=train_df[[f"{col}_encoded" for col in self.output_columns]].values[
                val_indices
            ],
            features=train_features[val_indices],
            tokenizer=self.tokenizer,
            feature_extractor=self.feature_extractor,
            max_length=self.max_length,
            output_tasks=self.output_columns,
        )

        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Create test dataloader if test data is provided
        test_dataloader = None
        if test_df is not None:
            # Transform test data labels
            for col in self.output_columns:
                if col in test_df:  # Check if column exists before transforming
                    # Ensure consistency with fitting: apply .astype(str)
                    test_df[f"{col}_encoded"] = self.label_encoders[col].transform(
                        test_df[col].astype(str)  # Added .astype(str)
                    )

            # Extract features for test texts
            logger.info("Extracting features for test data...")
            test_features = []
            for text in tqdm(
                test_df["text"],
                desc="Processing test texts",
                ncols=120,
                colour="blue",
            ):
                test_features.append(self.feature_extractor.extract_all_features(text))
            test_features = np.array(test_features)

            # Transform test labels
            for col in self.output_columns:
                if col in test_df.columns:
                    try:
                        # Ensure the column is treated as string for consistent transformation
                        test_df[f"{col}_encoded"] = self.label_encoders[col].transform(
                            test_df[col].astype(str)
                        )
                    except ValueError as e:
                        logger.error(
                            f"Error transforming column {col} in test data: {e}"
                        )
                        logger.error(
                            f"Value causing error: {test_df[col][~test_df[col].isin(self.label_encoders[col].classes_)].unique() if hasattr(self.label_encoders[col], 'classes_') else 'unknown'}"
                        )
                        logger.error(
                            f"Classes known to encoder for {col}: {list(self.label_encoders[col].classes_) if hasattr(self.label_encoders[col], 'classes_') else 'Encoder not fitted or classes_ not available'}"
                        )
                        raise e
                else:
                    logger.warning(f"Column {col} (for encoding) not found in test_df.")

            test_dataset = EmotionDataset(
                texts=test_df["text"].values,
                labels=(
                    test_df[[f"{col}_encoded" for col in self.output_columns]].values
                    if all(
                        f"{col}_encoded" in test_df.columns
                        for col in self.output_columns
                    )
                    else None
                ),
                features=test_features,
                tokenizer=self.tokenizer,
                feature_extractor=self.feature_extractor,
                max_length=self.max_length,
                output_tasks=self.output_columns,
            )
            test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size)

        # Make a copy of the dataframes to avoid modifying the originals
        # Store processed dataframes as attributes
        self.train_df_processed = train_df.copy()
        if test_df is not None:
            self.test_df_processed = test_df.copy()
            self.test_df_split = test_df.copy()  # Assign self.test_df_split
        else:
            self.test_df_processed = None
            self.test_df_split = None  # Assign self.test_df_split

        # Apply data augmentation if requested
        if apply_augmentation:
            # Assuming augmentation logic might be added here or called
            # For now, if it was empty, it remains effectively so.
            # If self.apply_data_augmentation was intended:
            # train_df = self.apply_data_augmentation(train_df, balance_strategy, samples_per_class, augmentation_ratio)
            # And then train_dataset/val_dataset would need to be recreated or updated.
            # This is a potential latent issue if augmentation is used.
            pass

        return train_dataloader, val_dataloader, test_dataloader

    def _save_encoders(self):
        """Save label encoders to disk."""
        if not self.encoders_output_dir:
            logger.warning(
                "Encoders output directory not set. Skipping saving encoders."
            )
            return
        os.makedirs(self.encoders_output_dir, exist_ok=True)  # Ensure dir exists
        for col, encoder in self.label_encoders.items():
            encoder_path = os.path.join(self.encoders_output_dir, f"{col}_encoder.pkl")
            with open(encoder_path, "wb") as f:
                pickle.dump(encoder, f)

    def get_num_classes(self):
        """Get the number of classes for each output column."""
        num_classes = {}
        for col in self.output_columns:
            if hasattr(self.label_encoders[col], "classes_"):
                num_classes[col] = len(self.label_encoders[col].classes_)
            else:
                # This case should ideally not happen if encoders are always fitted or loaded before this call
                logger.warning(
                    f"Label encoder for column {col} does not have classes_ attribute. It might not have been fitted or loaded correctly."
                )
                num_classes[col] = 0  # Or raise an error, or handle as appropriate
        return num_classes


class EmotionDataset(Dataset):
    """Custom Dataset for emotion classification."""

    def __init__(
        self,
        texts,
        tokenizer,
        features,
        labels=None,
        feature_extractor=None,
        max_length=128,
        output_tasks=None,
    ):
        """
        Initialize the dataset.

        Args:
            texts (list): List of text samples
            tokenizer: BERT tokenizer
            features (np.ndarray): Pre-extracted features
            labels (list, optional): List of label tuples (emotion, sub_emotion,
            intensity). None for prediction.
            feature_extractor (FeatureExtractor, optional): Feature extractor
            instance. Not strictly needed if features are pre-computed.
            max_length (int): Maximum sequence length for BERT
            output_tasks (list, optional): List of tasks to output. Used only if
            labels are provided.
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.features = features  # Should be pre-calculated and correctly dimensioned
        self.labels = labels
        # self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.output_tasks = output_tasks

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
        }

        # Add labels if they are available (i.e., during training/evaluation)
        if self.labels is not None and self.output_tasks is not None:
            current_labels = self.labels[idx]
            for i, task in enumerate(self.output_tasks):
                item[f"{task}_label"] = torch.tensor(
                    current_labels[i], dtype=torch.long
                )

        return item
