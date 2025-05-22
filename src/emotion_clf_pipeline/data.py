import logging
import os
import pickle
from collections import Counter

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

logger = logging.getLogger(__name__)

##############################
#        LOAD DATASET        #
##############################


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
        """Initialize the DataLoader with emotion mapping."""
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
        self.train_df = None
        self.test_df = None

    def load_training_data(self, data_dir="./../../Data/raw/all groups"):
        """
        Load and preprocess training data from multiple CSV files.

        Args:
            data_dir (str): Directory containing training data CSV files

        Returns:
            pd.DataFrame: Processed training data
        """
        # Initialize an empty DataFrame to store the combined data
        self.train_df = pd.DataFrame()

        # Loop over all files in the data directory
        for i_file in os.listdir(data_dir):
            # Read the current CSV file and select specific columns
            df_ = pd.read_csv(os.path.join(data_dir, i_file))[
                ["Translation", "Emotion", "Intensity"]
            ]

            # Concatenate the current file's data with the main DataFrame
            self.train_df = pd.concat([self.train_df, df_])

        # Rename columns to standardize the DataFrame
        self.train_df = self.train_df.rename(
            columns={
                "Translation": "text",
                "Emotion": "sub_emotion",
                "Intensity": "intensity",
            }
        )

        # Clean the data
        self.train_df.drop_duplicates(inplace=True)
        self.train_df.dropna(inplace=True)
        self.train_df.reset_index(drop=True, inplace=True)

        # Map the emotions
        self.train_df["emotion"] = self.train_df["sub_emotion"].map(
            self.emotion_mapping
        )

        return self.train_df

    def load_test_data(self, test_file="./../../Data/group 21_url1.csv"):
        """
        Load and preprocess test data from a CSV file.

        Args:
            test_file (str): Path to the test data CSV file

        Returns:
            pd.DataFrame: Processed test data
        """
        # Load the test set
        self.test_df = pd.read_csv(test_file)[
            ["Corrected Sentence", "Emotion", "Intensity"]
        ]

        # Rename columns to standardize the test set DataFrame
        self.test_df = self.test_df.rename(
            columns={
                "Corrected Sentence": "text",
                "Emotion": "sub_emotion",
                "Intensity": "intensity",
            }
        )

        # Clean the data
        self.test_df.drop_duplicates(inplace=True)
        self.test_df.dropna(inplace=True)
        self.test_df.reset_index(drop=True, inplace=True)

        # Map the emotions
        self.test_df["emotion"] = self.test_df["sub_emotion"].map(self.emotion_mapping)

        # Drop null and duplicate rows
        self.test_df = self.test_df.dropna()
        self.test_df = self.test_df.drop_duplicates()
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

        ####################################
        #        FEATURE EXTRACTION        #
        ####################################


class POSFeatureExtractor:
    """Feature extractor for Part-of-Speech tagging."""

    def extract_features(self, text):
        """
        Extract part-of-speech features from text.

        Args:
            text (str): Input text to analyze

        Returns:
            list: List of POS features including normalized counts
        """
        if not text or pd.isna(text):
            return [0] * 10

        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)

        # Count POS tags
        pos_counts = Counter(tag for word, tag in pos_tags)

        # Calculate features (normalized by total tokens)
        total = len(tokens) if tokens else 1
        features = [
            pos_counts.get("NN", 0) / total,  # Nouns
            pos_counts.get("NNS", 0) / total,  # Plural nouns
            pos_counts.get("VB", 0) / total,  # Verbs
            pos_counts.get("VBD", 0) / total,  # Past tense verbs
            pos_counts.get("JJ", 0) / total,  # Adjectives
            pos_counts.get("RB", 0) / total,  # Adverbs
            pos_counts.get("PRP", 0) / total,  # Personal pronouns
            pos_counts.get("IN", 0) / total,  # Prepositions
            pos_counts.get("DT", 0) / total,  # Determiners
            len(tokens) / 30,  # Text length (normalized)
        ]

        return features


class TextBlobFeatureExtractor:
    """Feature extractor for TextBlob sentiment analysis."""

    def extract_features(self, text):
        """
        Extract TextBlob sentiment features.

        Args:
            text (str): Input text to analyze

        Returns:
            list: List containing [polarity, subjectivity] scores
        """
        if not text or pd.isna(text):
            return [0, 0]

        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        return [polarity, subjectivity]


class VaderFeatureExtractor:
    """Feature extractor for VADER sentiment analysis."""

    def __init__(self):
        """Initialize VADER sentiment analyzer."""
        self.analyzer = SentimentIntensityAnalyzer()

    def extract_features(self, text):
        """
        Extract VADER sentiment features.

        Args:
            text (str): Input text to analyze

        Returns:
            list: List containing [neg, neu, pos, compound] scores
        """
        if not text or pd.isna(text):
            return [0, 0, 0, 0]

        scores = self.analyzer.polarity_scores(text)
        features = [scores["neg"], scores["neu"], scores["pos"], scores["compound"]]

        return features


class EmolexFeatureExtractor:
    """Feature extractor for EmoLex emotion lexicon."""

    def __init__(self, lexicon_path):
        """
        Initialize EmoLex feature extractor.

        Args:
            lexicon_path (str): Path to the EmoLex lexicon file
        """
        self.EMOTIONS = [
            "anger",
            "anticipation",
            "disgust",
            "fear",
            "joy",
            "sadness",
            "surprise",
            "trust",
        ]
        self.SENTIMENTS = ["negative", "positive"]
        self.lexicon = self._load_lexicon(lexicon_path)

    def _load_lexicon(self, lexicon_path):
        """Load and parse the NRC Emotion Lexicon."""
        lexicon = {}

        with open(lexicon_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    continue

                parts = line.strip().split("\t")
                if len(parts) == 3:
                    word, emotion, flag = parts

                    if word not in lexicon:
                        lexicon[word] = {e: 0 for e in self.EMOTIONS + self.SENTIMENTS}

                    if int(flag) == 1:
                        lexicon[word][emotion] = 1

        # print(f"Loaded EmoLex lexicon with {len(lexicon)} words")
        return lexicon

    def extract_features(self, text):
        """
        Extract emotion features using EmoLex.

        Args:
            text (str): Input text to analyze

        Returns:
            numpy.ndarray: Array of emotion features
        """
        if not text or pd.isna(text):
            return np.zeros(2 * len(self.EMOTIONS) + len(self.SENTIMENTS) + 2)

        # Tokenize and lowercase
        tokens = word_tokenize(text.lower())
        total_words = len(tokens)

        if total_words == 0:
            return np.zeros(2 * len(self.EMOTIONS) + len(self.SENTIMENTS) + 2)

        # Initialize counters
        emotion_counts = {emotion: 0 for emotion in self.EMOTIONS}
        sentiment_counts = {sentiment: 0 for sentiment in self.SENTIMENTS}

        # Count emotion words
        for token in tokens:
            if token in self.lexicon:
                for emotion in self.EMOTIONS:
                    emotion_counts[emotion] += self.lexicon[token][emotion]
                for sentiment in self.SENTIMENTS:
                    sentiment_counts[sentiment] += self.lexicon[token][sentiment]

        # Calculate densities
        emotion_densities = {
            emotion: count / total_words for emotion, count in emotion_counts.items()
        }

        # Calculate additional metrics
        emotion_diversity = sum(1 for count in emotion_counts.values() if count > 0)
        dominant_emotion_score = (
            max(emotion_densities.values()) if emotion_densities else 0
        )
        total_emotion_words = sum(emotion_counts.values())
        total_sentiment_words = sum(sentiment_counts.values())
        emotion_sentiment_ratio = (
            total_emotion_words / total_sentiment_words
            if total_sentiment_words > 0
            else 0
        )

        # Construct feature vector
        features = []
        features.extend([emotion_counts[emotion] for emotion in self.EMOTIONS])
        features.extend([emotion_densities[emotion] for emotion in self.EMOTIONS])
        features.extend([sentiment_counts[sentiment] for sentiment in self.SENTIMENTS])
        features.append(emotion_diversity)
        features.append(dominant_emotion_score)
        features.append(emotion_sentiment_ratio)

        return np.array(features, dtype=np.float32)


class FeatureExtractor:
    """
    A comprehensive feature extraction class for text analysis.

    This class provides methods to extract various linguistic and emotional
    features from text, including emotion lexicon features, part-of-speech
    features, sentiment features, and TF-IDF features.

    Attributes:
        vader (SentimentIntensityAnalyzer): VADER sentiment analyzer instance
        EMOTIONS (list): List of emotions tracked by EmoLex
        SENTIMENTS (list): List of sentiment categories
        emolex_lexicon (dict): Loaded EmoLex lexicon
        tfidf_vectorizer (TfidfVectorizer): TF-IDF vectorizer instance
    """

    def __init__(self, feature_config=None, lexicon_path=None):
        """Initialize the FeatureExtractor with necessary components."""
        # Use provided feature_config, or a specific default (all on) if None
        if feature_config is None:
            self.feature_config = {
                "pos": True,
                "textblob": True,
                "vader": True,
                "tfidf": True,
                "emolex": True,
            }
        else:
            self.feature_config = feature_config

        # Initialize components based on feature configuration
        self.vader = (
            SentimentIntensityAnalyzer()
            if self.feature_config.get("vader", True)
            else None
        )
        self.EMOTIONS = [
            "anger",
            "anticipation",
            "disgust",
            "fear",
            "joy",
            "sadness",
            "surprise",
            "trust",
        ]
        self.SENTIMENTS = ["negative", "positive"]
        self.emolex_lexicon = (
            self._load_emolex_lexicon(lexicon_path)
            if self.feature_config.get("emolex", True)
            else None
        )
        self.tfidf_vectorizer = None  # Will be initialized when fit is called

        # Define output columns that will be used for labels
        self.output_columns = ["emotion", "sub_emotion", "intensity"]

        # Initialize feature extractors
        self.pos_extractor = POSFeatureExtractor()
        self.textblob_extractor = TextBlobFeatureExtractor()
        self.vader_extractor = VaderFeatureExtractor()
        self.emolex_extractor = EmolexFeatureExtractor(lexicon_path)

    def _load_emolex_lexicon(self, lexicon_path):
        """
        Load and parse the NRC Emotion Lexicon.

        Args:
            lexicon_path (str): Path to the EmoLex lexicon file

        Returns:
            dict: Dictionary mapping words to their emotion and sentiment scores

        Note:
            The lexicon file should be in the NRC Emotion Lexicon format with
            tab-separated values:
            word    emotion    flag
        """
        lexicon = {}

        with open(lexicon_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    continue

                parts = line.strip().split("\t")
                if len(parts) == 3:
                    word, emotion, flag = parts

                    if word not in lexicon:
                        lexicon[word] = {e: 0 for e in self.EMOTIONS + self.SENTIMENTS}

                    if int(flag) == 1:
                        lexicon[word][emotion] = 1

        # print(f"Loaded EmoLex lexicon with {len(lexicon)} words")
        return lexicon

    def extract_emolex_features(self, text):
        """
        Extract emotion features from text using the EmoLex lexicon.

        Args:
            text (str): Input text to analyze

        Returns:
            numpy.ndarray: Array of emotion features including:
                - Raw emotion counts
                - Emotion densities
                - Sentiment counts
                - Emotion diversity
                - Dominant emotion score
                - Emotion-sentiment ratio
        """
        # Tokenize and lowercase
        tokens = word_tokenize(text.lower())
        total_words = len(tokens)

        if total_words == 0:
            return np.zeros(2 * len(self.EMOTIONS) + len(self.SENTIMENTS) + 2)

        # Initialize counters
        emotion_counts = {emotion: 0 for emotion in self.EMOTIONS}
        sentiment_counts = {sentiment: 0 for sentiment in self.SENTIMENTS}

        # Count emotion words
        for token in tokens:
            if token in self.emolex_lexicon:
                for emotion in self.EMOTIONS:
                    emotion_counts[emotion] += self.emolex_lexicon[token][emotion]
                for sentiment in self.SENTIMENTS:
                    sentiment_counts[sentiment] += self.emolex_lexicon[token][sentiment]

        # Calculate densities
        emotion_densities = {
            emotion: count / total_words for emotion, count in emotion_counts.items()
        }

        # Calculate number of distinct emotions present
        emotion_diversity = sum(1 for count in emotion_counts.values() if count > 0)

        # Find dominant emotion (the one with highest density)
        dominant_emotion_score = (
            max(emotion_densities.values()) if emotion_densities else 0
        )

        # Calculate emotion to sentiment ratio
        total_emotion_words = sum(emotion_counts.values())
        total_sentiment_words = sum(sentiment_counts.values())
        emotion_sentiment_ratio = (
            total_emotion_words / total_sentiment_words
            if total_sentiment_words > 0
            else 0
        )

        # Construct feature vector
        features = []
        features.extend(
            [emotion_counts[emotion] for emotion in self.EMOTIONS]
        )  # Raw counts
        features.extend(
            [emotion_densities[emotion] for emotion in self.EMOTIONS]
        )  # Densities
        features.extend(
            [sentiment_counts[sentiment] for sentiment in self.SENTIMENTS]
        )  # Sentiment counts
        features.append(emotion_diversity)  # Diversity
        features.append(dominant_emotion_score)  # Dominant emotion intensity
        features.append(emotion_sentiment_ratio)  # Emotion-sentiment ratio

        return np.array(features, dtype=np.float32)

    def get_emolex_feature_names(self):
        """
        Get names of the extracted features for interpretability.

        Returns:
            list: List of feature names in the same order as they appear in
            the feature vector
        """
        feature_names = []

        # Emotion counts
        feature_names.extend([f"emolex_{emotion}_count" for emotion in self.EMOTIONS])

        # Emotion densities
        feature_names.extend([f"emolex_{emotion}_density" for emotion in self.EMOTIONS])

        # Sentiment counts
        feature_names.extend(
            [f"emolex_{sentiment}_count" for sentiment in self.SENTIMENTS]
        )

        # Additional metrics
        feature_names.append("emolex_emotion_diversity")
        feature_names.append("emolex_dominant_emotion_score")
        feature_names.append("emolex_emotion_sentiment_ratio")

        return feature_names

    def extract_pos_features(self, text):
        """
        Extract part-of-speech features from text.

        Args:
            text (str): Input text to analyze

        Returns:
            list: List of POS features including normalized counts of:
                - Nouns (NN)
                - Plural nouns (NNS)
                - Verbs (VB)
                - Past tense verbs (VBD)
                - Adjectives (JJ)
                - Adverbs (RB)
                - Personal pronouns (PRP)
                - Prepositions (IN)
                - Determiners (DT)
                - Text length (normalized)
        """
        if not text or pd.isna(text):
            return [0] * 10  # Return zeros for empty text

        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)

        # Count POS tags
        pos_counts = Counter(tag for word, tag in pos_tags)

        # Calculate features (normalized by total tokens)
        total = len(tokens) if tokens else 1
        features = [
            pos_counts.get("NN", 0) / total,  # Nouns
            pos_counts.get("NNS", 0) / total,  # Plural nouns
            pos_counts.get("VB", 0) / total,  # Verbs
            pos_counts.get("VBD", 0) / total,  # Past tense verbs
            pos_counts.get("JJ", 0) / total,  # Adjectives
            pos_counts.get("RB", 0) / total,  # Adverbs
            pos_counts.get("PRP", 0) / total,  # Personal pronouns
            pos_counts.get("IN", 0) / total,  # Prepositions
            pos_counts.get("DT", 0) / total,  # Determiners
            len(tokens) / 30,  # Text length (normalized)
        ]

        return features

    def extract_textblob_sentiment(self, text):
        """
        Extract TextBlob sentiment features.

        Args:
            text (str): Input text to analyze

        Returns:
            list: List containing [polarity, subjectivity] scores
                - polarity: float between -1.0 and 1.0
                - subjectivity: float between 0.0 and 1.0
        """
        if not text or pd.isna(text):
            return [0, 0]

        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        return [polarity, subjectivity]

    def extract_vader_sentiment(self, text):
        """
        Extract VADER sentiment features.

        Args:
            text (str): Input text to analyze

        Returns:
            list: List containing [neg, neu, pos, compound] scores
                - neg: negative sentiment score
                - neu: neutral sentiment score
                - pos: positive sentiment score
                - compound: normalized compound score
        """
        if not text or pd.isna(text):
            return [0, 0, 0, 0]

        scores = self.vader.polarity_scores(text)
        features = [scores["neg"], scores["neu"], scores["pos"], scores["compound"]]

        return features

    def fit_tfidf(self, texts):
        """
        Fit the TF-IDF vectorizer on the provided texts.

        Args:
            texts (list): List of text documents to fit the vectorizer on

        Note:
            This method must be called before using extract_tfidf_features
        """
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100)
        self.tfidf_vectorizer.fit(texts)

    def extract_tfidf_features(self, text):
        """
        Extract TF-IDF features using pre-trained vectorizer.

        Args:
            text (str): Input text to analyze

        Returns:
            numpy.ndarray: Array of TF-IDF features

        Raises:
            ValueError: If fit_tfidf() has not been called yet
        """
        if not text or pd.isna(text):
            return np.zeros(100)

        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf() first.")

        features = self.tfidf_vectorizer.transform([text]).toarray()[0]
        return features

    def extract_all_features(self, text, expected_dim=None):
        """
        Extract all features for a given text based on the feature configuration.
        Pad or truncate features if expected_dim is provided.

        Args:
            text (str): Input text to analyze
            expected_dim (int, optional): The dimension the output features should have.
                                       If None, uses the natural dimension of enabled
                                       features.

        Returns:
            numpy.ndarray: Combined array of all features, padded/truncated to
            expected_dim if provided.
        """
        features_list = []

        if self.feature_config.get("pos", True):
            features_list.extend(self.extract_pos_features(text))

        if self.feature_config.get("textblob", True):
            features_list.extend(self.extract_textblob_sentiment(text))

        if self.feature_config.get("vader", True):
            features_list.extend(self.extract_vader_sentiment(text))

        if self.feature_config.get("tfidf", True):
            # Ensure TF-IDF vectorizer is fitted, especially if called outside
            if self.tfidf_vectorizer is None:
                # Fallback: return zeros for TF-IDF part if not fitted,
                # or fit with dummy if appropriate For prediction, it should
                # have been fit during setup of CustomPredictor. However, if
                # CustomPredictor.predict fits it on the fly, that's different.
                # Let's assume for now if it's None, it means 0 features for
                # this part for a single text.
                num_tfidf_features = (
                    100  # Default expected, should align with get_feature_dim
                )
                if (
                    hasattr(self, "_actual_tfidf_dim")
                    and self._actual_tfidf_dim is not None
                ):
                    num_tfidf_features = self._actual_tfidf_dim
                features_list.extend(np.zeros(num_tfidf_features))
            else:
                features_list.extend(self.extract_tfidf_features(text))

        if self.feature_config.get("emolex", True):
            if self.emolex_lexicon is None:
                # Fallback: Emolex needs lexicon. If none, zero features.
                num_emolex_features = (8 * 2) + 2 + 3
                features_list.extend(np.zeros(num_emolex_features))
            else:
                features_list.extend(self.extract_emolex_features(text))

        actual_features = np.array(features_list, dtype=np.float32)

        if expected_dim is not None:
            current_dim = len(actual_features)
            if current_dim < expected_dim:
                padded_features = np.zeros(expected_dim, dtype=np.float32)
                padded_features[:current_dim] = actual_features
                return padded_features
            elif current_dim > expected_dim:
                return actual_features[:expected_dim]
            # else current_dim == expected_dim, return as is

        return actual_features

    def get_feature_dim(self, expected_dim_from_model=None):
        """
        Calculate the total dimension of all features.
        If expected_dim_from_model is provided, it might influence calculations for
        uninitialized parts (like TF-IDF). However, this function should primarily
        reflect the natural dimension based on config. The padding/truncation should
        ideally happen in extract_all_features.

        Returns:
            int: Total dimension of all enabled features based on current configuration.
        """
        total_dim = 0
        if self.feature_config.get("pos", True):
            total_dim += 10
        if self.feature_config.get("textblob", True):
            total_dim += 2
        if self.feature_config.get("vader", True):
            total_dim += 4
        if self.feature_config.get("emolex", True):
            if self.emolex_lexicon is not None:  # Only add if lexicon loaded
                total_dim += (8 * 2) + 2 + 3  # 21 total EmoLex features
            # else: if emolex is True but lexicon is None, it will produce zeros
            # but occupy space if expected.

        if self.feature_config.get("tfidf", True):
            if self.tfidf_vectorizer is not None and hasattr(
                self.tfidf_vectorizer, "max_features"
            ):
                # Use max_features if vectorizer is initialized, as this is the
                # intended fixed dimension
                self._actual_tfidf_dim = self.tfidf_vectorizer.max_features
                total_dim += self._actual_tfidf_dim
            elif self.tfidf_vectorizer is not None and hasattr(
                self.tfidf_vectorizer, "get_feature_names_out"
            ):
                # Fallback if max_features isn't directly on the instance but
                # vocab is there This case should ideally be avoided by consistent
                # TfidfVectorizer setup
                self._actual_tfidf_dim = len(
                    self.tfidf_vectorizer.get_feature_names_out()
                )
                total_dim += self._actual_tfidf_dim
            else:
                # If TF-IDF is enabled in config but vectorizer not set or lacks
                # max_features info, assume default. This should align with
                # TfidfVectorizer(max_features=100) default in fit_tfidf
                self._actual_tfidf_dim = 100
                total_dim += self._actual_tfidf_dim
        else:
            self._actual_tfidf_dim = None  # Explicitly set to None if tfidf is False

        return total_dim

        #################################
        #        DATA PREPRATION        #
        #################################


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
        emolex_lexicon_path = os.path.join(
            _project_root_dir_dp,
            "models",
            "features",
            "EmoLex",
            "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
        )

        # Initialize feature extractor with configuration and lexicon path
        self.feature_extractor = FeatureExtractor(
            feature_config=feature_config, lexicon_path=emolex_lexicon_path
        )

        # Define output columns that will be used for labels
        self.output_columns = ["emotion", "sub_emotion", "intensity"]

        # The following individual extractors are redundant as
        # FeatureExtractor handles them
        # self.pos_extractor = POSFeatureExtractor()
        # self.textblob_extractor = TextBlobFeatureExtractor()
        # self.vader_extractor = VaderFeatureExtractor()
        # self.emolex_extractor = EmolexFeatureExtractor(
        #   lexicon_path=emolex_lexicon_path)
        # Corrected if it were needed

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
            logger.info(f"  {emotion}: {count} samples ({count/len(train_df):.2%})")

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
        # Create output directory for encoders if it doesn't exist
        os.makedirs("./models/encoders", exist_ok=True)

        # Fit label encoders on training data
        for col in self.output_columns:
            self.label_encoders[col].fit(train_df[col])
            train_df[f"{col}_encoded"] = self.label_encoders[col].transform(
                train_df[col]
            )

            if test_df is not None:
                test_df[f"{col}_encoded"] = self.label_encoders[col].transform(
                    test_df[col]
                )

        # Save label encoders
        self._save_encoders()

        # Split into train and validation sets
        train_indices, val_indices = train_test_split(
            range(len(train_df)),
            test_size=validation_split,
            random_state=42,
            stratify=train_df[
                self.output_columns[0]
            ],  # Stratify by first output column
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
            logger.info("Extracting features for test data...")
            test_features = []
            for text in tqdm(
                test_df["text"], desc="Processing test texts", ncols=120, colour="green"
            ):
                test_features.append(self.feature_extractor.extract_all_features(text))
            test_features = np.array(test_features)

            test_dataset = EmotionDataset(
                texts=test_df["text"].values,
                labels=test_df[
                    [f"{col}_encoded" for col in self.output_columns]
                ].values,
                features=test_features,
                tokenizer=self.tokenizer,
                feature_extractor=self.feature_extractor,
                max_length=self.max_length,
                output_tasks=self.output_columns,
            )
            test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size)

        # Make a copy of the dataframes to avoid modifying the originals
        train_df = train_df.copy()
        if test_df is not None:
            test_df = test_df.copy()

        # Apply data augmentation if requested
        if apply_augmentation:
            train_df = self.apply_data_augmentation(
                train_df,
                balance_strategy=balance_strategy,
                samples_per_class=samples_per_class,
                augmentation_ratio=augmentation_ratio,
            )

        return train_dataloader, val_dataloader, test_dataloader

    def _save_encoders(self):
        """Save label encoders to disk."""
        for col, encoder in self.label_encoders.items():
            # Convert hyphen to underscore in filename
            filename = col.replace("-", "_")
            with open(f"./models/encoders/{filename}_encoder.pkl", "wb") as f:
                pickle.dump(encoder, f)

    def get_num_classes(self):
        """
        Get the number of classes for each output column.

        Returns:
            dict: Dictionary mapping output columns to their number of classes
        """
        return {
            col: len(encoder.classes_) for col, encoder in self.label_encoders.items()
        }


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


#########################################
#        DATA QUALITY ASSESSMENT        #
#########################################


class DataQualityAssessor:
    """
    A class to assess and analyze data quality in emotion classification datasets.

    This class handles:
    - Assessing class imbalance
    - Computing class weights
    - Detecting and analyzing biases
    - Identifying dataset limitations

    Attributes:
        df (pd.DataFrame): The dataset to analyze
        emotion_mapping (dict): Dictionary mapping sub-emotions to standardized emotions
    """

    def __init__(self, df):
        """
        Initialize the DataQualityAssessor.

        Args:
            df (pd.DataFrame): The dataset to analyze
            emotion_mapping (dict): Dictionary mapping sub-emotions to
            standardized emotions
        """
        self.df = df
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

    def assess_class_imbalance(self):
        """
        Assess and visualize class imbalance in the dataset.

        Returns:
            tuple: (emotion_counts, imbalance_ratio)
                - emotion_counts: Series with emotion counts
                - imbalance_ratio: Ratio between majority and minority classes
        """
        # Calculate distribution statistics
        emotion_counts = self.df["emotion"].value_counts()
        total_samples = len(self.df)
        # class_proportions = emotion_counts / total_samples

        # Print imbalance statistics
        print("Class distribution:")
        for emotion, count in emotion_counts.items():
            print(f"{emotion}: {count} samples ({count/total_samples:.2%})")

        # Calculate imbalance metrics
        max_class_size = emotion_counts.max()
        min_class_size = emotion_counts.min()
        imbalance_ratio = max_class_size / min_class_size
        print(f"\nImbalance ratio (majority:minority): {imbalance_ratio:.2f}:1")

        # Visualize with more informative plot
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(
            x=emotion_counts.index, y=emotion_counts.values, palette="viridis"
        )
        plt.title("Emotion Class Distribution", fontsize=14)
        plt.xlabel("Emotion", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(rotation=45)

        # Add count and percentage labels
        for i, count in enumerate(emotion_counts.values):
            percentage = count / total_samples * 100
            ax.text(
                i, count + 5, f"{count}\n({percentage:.1f}%)", ha="center", fontsize=9
            )

        plt.tight_layout()
        plt.show()

        return emotion_counts, imbalance_ratio

    def compute_class_weights(self):
        """
        Compute class weights to address class imbalance.

        Returns:
            torch.Tensor: Tensor of class weights
        """
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(self.df["emotion"]),
            y=self.df["emotion"].tolist(),
        )
        class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print("Class weights:", class_weights_dict)
        return class_weights_tensor

    def assess_bias(self):
        """
        Comprehensive assessment of dataset biases.

        This method analyzes:
        - Text length bias across emotions
        - Lexical bias (word frequency and diversity)
        - Sentiment bias
        - POS distribution bias
        - Cultural/contextual bias indicators

        Returns:
            dict: Dictionary containing comprehensive bias analysis results
        """

        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

        # 1. Text Length Bias Analysis
        ax1 = fig.add_subplot(gs[0, 0])
        self.df["text_length"] = self.df["text"].apply(len)

        # Create violin plot with better scaling
        sns.violinplot(
            data=self.df,
            x="emotion",
            y="text_length",
            palette="viridis",
            cut=1,
            scale="width",
            inner="box",
            ax=ax1,
        )

        # Set y-axis limit to exclude extreme outliers (e.g., 95th percentile)
        # ylim = np.percentile(self.df['text_length'], 95)
        # ax1.set_ylim(0, ylim)

        # Customize plot
        ax1.set_title(
            "Text Length Distribution by Emotion",
            fontsize=14,
            pad=20,
            fontweight="bold",
        )
        ax1.set_xlabel("Emotion", fontsize=12)
        ax1.set_ylabel("Text Length (characters)", fontsize=12)
        ax1.tick_params(axis="x", rotation=45)

        # Add median values with better positioning
        medians = self.df.groupby("emotion")["text_length"].median()
        for i, emotion in enumerate(medians.index):
            ax1.text(
                i,
                0,
                f"Median: {int(medians[emotion])}",
                ha="center",
                va="top",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1),
            )

        # 2. Lexical Bias Analysis
        ax2 = fig.add_subplot(gs[0, 1])
        stop_words = set(stopwords.words("english"))

        # Calculate word frequency and diversity metrics
        emotion_word_stats = {}
        for emotion in self.df["emotion"].unique():
            texts = self.df[self.df["emotion"] == emotion]["text"]
            words = [
                word.lower()
                for text in texts
                for word in nltk.word_tokenize(text)
                if word.isalpha() and word.lower() not in stop_words
            ]

            # Calculate metrics
            total_words = len(words)
            unique_words = len(set(words))
            word_freq = Counter(words)
            top_words = word_freq.most_common(5)

            emotion_word_stats[emotion] = {
                "total_words": total_words,
                "unique_words": unique_words,
                "diversity": unique_words / total_words if total_words > 0 else 0,
                "top_words": top_words,
            }

        # Calculate and plot word diversity with meaningful color gradient
        diversity_values = [stats["diversity"] for stats in emotion_word_stats.values()]
        emotions = list(emotion_word_stats.keys())

        # Create color gradient based on diversity values
        norm = plt.Normalize(min(diversity_values), max(diversity_values))
        colors = plt.cm.RdYlBu(norm(diversity_values))

        bars = ax2.bar(
            emotions, diversity_values, color=colors, edgecolor="black", linewidth=1
        )

        # Customize plot
        ax2.set_title(
            "Vocabulary Diversity by Emotion", fontsize=14, pad=20, fontweight="bold"
        )
        ax2.set_xlabel("Emotion", fontsize=12)
        ax2.set_ylabel("Unique Words / Total Words", fontsize=12)
        ax2.tick_params(axis="x", rotation=45)

        # Add value labels with improved positioning
        # max_height = max(diversity_values)
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1),
            )

        # 3. Sentiment Bias Analysis
        ax3 = fig.add_subplot(gs[1, 0])
        self.df["sentiment"] = self.df["text"].apply(
            lambda x: TextBlob(x).sentiment.polarity
        )

        # Create separate violin and box plots with better visibility
        sns.violinplot(
            data=self.df,
            x="emotion",
            y="sentiment",
            palette="viridis",
            cut=1,
            scale="width",
            inner=None,
            ax=ax3,
        )
        sns.boxplot(
            data=self.df,
            x="emotion",
            y="sentiment",
            color="white",
            width=0.2,
            showfliers=False,
            ax=ax3,
        )

        # Customize plot
        ax3.set_title(
            "Sentiment Distribution by Emotion", fontsize=14, pad=20, fontweight="bold"
        )
        ax3.set_xlabel("Emotion", fontsize=12)
        ax3.set_ylabel("Sentiment Score", fontsize=12)
        ax3.tick_params(axis="x", rotation=45)

        # Add mean values with improved visibility
        means = self.df.groupby("emotion")["sentiment"].mean()
        for i, emotion in enumerate(means.index):
            ax3.text(
                i,
                means[emotion],
                f"Mean: {means[emotion]:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1),
            )

        # 4. POS Distribution Bias
        ax4 = fig.add_subplot(gs[1, 1])

        # Calculate POS ratios for each emotion
        pos_stats = {}
        for emotion in self.df["emotion"].unique():
            texts = self.df[self.df["emotion"] == emotion]["text"]
            all_pos = []
            for text in texts:
                tokens = word_tokenize(text)
                pos_tags = pos_tag(tokens)
                all_pos.extend([tag for word, tag in pos_tags])

            pos_counts = Counter(all_pos)
            total = sum(pos_counts.values())
            pos_ratios = {tag: count / total for tag, count in pos_counts.items()}
            pos_stats[emotion] = pos_ratios

        # Filter for most common POS tags (top 15)
        pos_data = pd.DataFrame(pos_stats).fillna(0)
        top_pos = pos_data.mean(axis=1).nlargest(15).index
        pos_data_filtered = pos_data.loc[top_pos]

        # Create heatmap with better color contrast
        sns.heatmap(
            pos_data_filtered,
            annot=True,
            fmt=".2f",
            cmap="RdYlBu_r",
            ax=ax4,
            cbar_kws={"label": "Ratio"},
        )

        # Customize plot
        ax4.set_title(
            "POS Distribution by Emotion (Top 15 Tags)",
            fontsize=14,
            pad=20,
            fontweight="bold",
        )
        ax4.set_xlabel("Emotion", fontsize=12)
        ax4.set_ylabel("POS Tag", fontsize=12)

        # 5. Cultural/Contextual Bias Indicators
        ax5 = fig.add_subplot(gs[2, :])

        # Analyze potential cultural/contextual indicators
        cultural_indicators = {
            "pronouns": ["i", "you", "he", "she", "they", "we"],
            "modals": ["can", "could", "would", "should", "must"],
            "negation": ["not", "n't", "never", "no"],
            "intensifiers": ["very", "really", "absolutely", "completely"],
        }

        indicator_stats = {}
        for emotion in self.df["emotion"].unique():
            texts = self.df[self.df["emotion"] == emotion]["text"]
            words = [
                word.lower() for text in texts for word in nltk.word_tokenize(text)
            ]
            total_words = len(words)

            stats = {}
            for category, indicators in cultural_indicators.items():
                count = sum(1 for word in words if word in indicators)
                stats[category] = count / total_words if total_words > 0 else 0

            indicator_stats[emotion] = stats

        # Rename categories for better clarity
        category_names = {
            "pronouns": "Personal Pronouns",
            "modals": "Modal Verbs",
            "negation": "Negation Words",
            "intensifiers": "Intensity Modifiers",
        }

        # Update indicator data with new names
        indicator_data = pd.DataFrame(indicator_stats).T
        indicator_data.columns = [category_names[col] for col in indicator_data.columns]

        # Create heatmap with improved color scheme
        sns.heatmap(
            indicator_data,
            annot=True,
            fmt=".2f",
            cmap="RdYlBu",
            ax=ax5,
            cbar_kws={"label": "Usage Frequency"},
        )

        # Customize plot
        ax5.set_title(
            "Cultural and Contextual Language Patterns by Emotion",
            fontsize=14,
            pad=20,
            fontweight="bold",
        )
        ax5.set_xlabel("Language Pattern Category", fontsize=12)
        ax5.set_ylabel("Emotion", fontsize=12)

        # Add main title with subtle background
        fig.suptitle(
            "Comprehensive Dataset Bias Analysis",
            fontsize=16,
            fontweight="bold",
            y=0.95,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=5),
        )

        plt.tight_layout()
        plt.show()

        # Print comprehensive analysis with better formatting
        print("\n" + "=" * 50)
        print("DATASET BIAS ANALYSIS REPORT")
        print("=" * 50)

        # Text length bias
        print("\n1. Text Length Bias:")
        length_stats = self.df.groupby("emotion")["text_length"].agg(
            ["mean", "std", "min", "max"]
        )
        print(length_stats.round(2))

        # Lexical bias
        print("\n2. Lexical Bias:")
        for emotion, stats in emotion_word_stats.items():
            print(f"\n{emotion.upper()}:")
            print(f"  • Total words: {stats['total_words']:,}")
            print(f"  • Unique words: {stats['unique_words']:,}")
            print(f"  • Vocabulary diversity: {stats['diversity']:.2%}")
            print("  • Most common words:")
            for word, count in stats["top_words"]:
                print(f"    - {word}: {count:,}")

        # Sentiment bias
        print("\n3. Sentiment Bias:")
        sentiment_stats = self.df.groupby("emotion")["sentiment"].agg(["mean", "std"])
        print(sentiment_stats.round(3))

        # POS distribution bias
        print("\n4. POS Distribution Bias:")
        for emotion, pos_ratios in pos_stats.items():
            print(f"\n{emotion.upper()}:")
            for pos, ratio in pos_ratios.items():
                print(f"  • {pos}: {ratio:.2%}")

        # Cultural/contextual bias
        print("\n5. Cultural/Contextual Bias Indicators:")
        for emotion, stats in indicator_stats.items():
            print(f"\n{emotion.upper()}:")
            for category, ratio in stats.items():
                print(f"  • {category}: {ratio:.2%}")

        print("\n" + "=" * 50)

        return {
            "length_bias": length_stats,
            "lexical_bias": emotion_word_stats,
            "sentiment_bias": sentiment_stats,
            "pos_bias": pos_stats,
            "cultural_bias": indicator_stats,
        }

    def assess_limitations(self):
        """
        Comprehensive assessment of dataset limitations and biases.

        This method analyzes:
        - Sub-emotion distribution and coverage
        - Intensity patterns across emotions
        - Text length characteristics
        - Vocabulary diversity
        - Cross-emotion relationships

        Returns:
            dict: Dictionary containing comprehensive limitation analysis results
        """
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Sub-emotion Distribution Analysis
        ax1 = fig.add_subplot(gs[0, 0])

        # Get sub-emotion counts
        sub_emotion_counts = self.df["sub_emotion"].value_counts()

        # Create color mapping for main emotions
        main_emotions = self.df["emotion"].unique()
        color_map = dict(
            zip(main_emotions, plt.cm.RdYlBu(np.linspace(0, 1, len(main_emotions))))
        )

        # Group sub-emotions by their main emotion
        sub_emotions_by_main = {}
        for sub_emotion in sub_emotion_counts.index:
            main_emotion = self.df[self.df["sub_emotion"] == sub_emotion][
                "emotion"
            ].iloc[0]
            if main_emotion not in sub_emotions_by_main:
                sub_emotions_by_main[main_emotion] = []
            sub_emotions_by_main[main_emotion].append(
                (sub_emotion, sub_emotion_counts[sub_emotion])
            )

        # Sort sub-emotions within each main emotion by count
        for main_emotion in sub_emotions_by_main:
            sub_emotions_by_main[main_emotion].sort(key=lambda x: x[1], reverse=True)

        # Create grouped bar plot
        # bar_width = 0.8
        x = np.arange(len(sub_emotion_counts))
        current_x = 0

        for main_emotion, sub_emotions in sub_emotions_by_main.items():
            for sub_emotion, count in sub_emotions:
                # bar = ax1.bar(
                #     current_x,
                #     count,
                #     color=color_map[main_emotion],
                #     width=bar_width,
                #     edgecolor="black",
                #     linewidth=0.5,
                # )
                # Add count labels with background
                ax1.text(
                    current_x,
                    count,
                    f"{int(count)}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1),
                )
                current_x += 1

        # Customize plot
        ax1.set_title(
            "Sub-emotion Distribution and Coverage",
            fontsize=14,
            pad=20,
            fontweight="bold",
        )
        ax1.set_xlabel("Sub-emotions", fontsize=12)
        ax1.set_ylabel("Count", fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(
            [sub_emotion for sub_emotion, _ in sub_emotion_counts.items()],
            rotation=45,
            ha="right",
        )

        # Add legend for main emotions with better formatting
        legend_elements = [
            plt.Rectangle(
                (0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.5
            )
            for color in color_map.values()
        ]
        ax1.legend(
            legend_elements,
            color_map.keys(),
            title="Main Emotions",
            loc="upper right",
            title_fontsize=10,
            fontsize=9,
        )

        # 2. Intensity Distribution Analysis
        ax2 = fig.add_subplot(gs[0, 1])

        # Create cross-tabulation of emotion and intensity
        intensity_dist = pd.crosstab(
            self.df["emotion"], self.df["intensity"], normalize="index"
        )

        # Create stacked bar plot with better colors
        intensity_dist.plot(
            kind="bar",
            stacked=True,
            ax=ax2,
            colormap="RdYlBu",
            edgecolor="black",
            linewidth=0.5,
        )

        # Customize plot
        ax2.set_title(
            "Emotion Intensity Distribution", fontsize=14, pad=20, fontweight="bold"
        )
        ax2.set_xlabel("Emotion", fontsize=12)
        ax2.set_ylabel("Proportion", fontsize=12)
        ax2.tick_params(axis="x", rotation=45)
        ax2.legend(title="Intensity Level", bbox_to_anchor=(1.05, 1), loc="upper left")

        # Add percentage labels on bars
        for c in ax2.containers:
            ax2.bar_label(c, fmt="%.1f%%", label_type="center")

        # 3. Text Length Analysis
        ax3 = fig.add_subplot(gs[1, 0])

        # Calculate text length statistics by emotion
        text_length_stats = self.df.groupby("emotion")["text_length"].agg(
            ["mean", "std", "min", "max"]
        )

        # Create violin plot with statistical overlay
        sns.violinplot(
            data=self.df,
            x="emotion",
            y="text_length",
            palette="RdYlBu",
            cut=1,
            inner="box",
            ax=ax3,
        )

        # Add statistical annotations
        for i, (emotion, stats) in enumerate(text_length_stats.iterrows()):
            stats_text = f'Mean: {stats["mean"]:.0f}\nStd: {stats["std"]:.0f}'
            ax3.text(
                i,
                stats["mean"],
                stats_text,
                ha="center",
                va="bottom",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1),
            )

        # Customize plot
        ax3.set_title(
            "Text Length Distribution by Emotion",
            fontsize=14,
            pad=20,
            fontweight="bold",
        )
        ax3.set_xlabel("Emotion", fontsize=12)
        ax3.set_ylabel("Text Length (characters)", fontsize=12)
        ax3.tick_params(axis="x", rotation=45)

        # 4. Vocabulary Overlap Analysis
        ax4 = fig.add_subplot(gs[1, 1])

        # Calculate vocabulary overlap between emotions
        emotion_vocab = {}
        for emotion in self.df["emotion"].unique():
            texts = self.df[self.df["emotion"] == emotion]["text"]
            words = set(
                word.lower()
                for text in texts
                for word in nltk.word_tokenize(text)
                if word.isalpha() and word.lower() not in stopwords.words("english")
            )
            emotion_vocab[emotion] = words

        # Create overlap matrix
        overlap_matrix = np.zeros((len(emotion_vocab), len(emotion_vocab)))
        emotions = list(emotion_vocab.keys())

        for i, e1 in enumerate(emotions):
            for j, e2 in enumerate(emotions):
                if i <= j:  # Only calculate upper triangle
                    overlap = len(emotion_vocab[e1] & emotion_vocab[e2])
                    total = len(emotion_vocab[e1] | emotion_vocab[e2])
                    overlap_matrix[i, j] = overlap_matrix[j, i] = (
                        overlap / total if total > 0 else 0
                    )

        # Create heatmap with better formatting
        sns.heatmap(
            overlap_matrix,
            annot=True,
            fmt=".2f",
            xticklabels=emotions,
            yticklabels=emotions,
            cmap="RdYlBu_r",
            ax=ax4,
            cbar_kws={"label": "Jaccard Similarity"},
        )

        # Customize plot
        ax4.set_title(
            "Vocabulary Overlap Between Emotions",
            fontsize=14,
            pad=20,
            fontweight="bold",
        )
        ax4.set_xlabel("Emotion", fontsize=12)
        ax4.set_ylabel("Emotion", fontsize=12)

        # 5. Dataset Balance Analysis
        ax5 = fig.add_subplot(gs[2, :])

        # Calculate various balance metrics
        balance_metrics = pd.DataFrame(
            {
                "Emotion Count": self.df["emotion"].value_counts(),
                "Sub-emotion Count": self.df.groupby("emotion")[
                    "sub_emotion"
                ].nunique(),
                "Avg Text Length": self.df.groupby("emotion")["text_length"].mean(),
                "Vocab Size": [len(emotion_vocab[e]) for e in emotions],
            }
        ).round(2)

        # Normalize metrics for visualization
        balance_metrics_norm = balance_metrics.div(balance_metrics.max())

        # Create heatmap for balance metrics
        sns.heatmap(
            balance_metrics_norm.T,
            annot=balance_metrics.T,
            fmt=".0f",
            cmap="RdYlBu",
            ax=ax5,
            cbar_kws={"label": "Normalized Score"},
        )

        # Customize plot
        ax5.set_title(
            "Dataset Balance Metrics by Emotion", fontsize=14, pad=20, fontweight="bold"
        )
        ax5.set_xlabel("Emotion", fontsize=12)
        ax5.set_ylabel("Metric", fontsize=12)

        # Add main title with background
        fig.suptitle(
            "Comprehensive Dataset Limitations Analysis",
            fontsize=16,
            fontweight="bold",
            y=0.95,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=5),
        )

        plt.tight_layout()
        plt.show()

        # Print comprehensive analysis
        print("\n" + "=" * 50)
        print("DATASET LIMITATIONS ANALYSIS REPORT")
        print("=" * 50)

        # 1. Sub-emotion Coverage
        print("\n1. Sub-emotion Coverage:")
        for emotion, sub_emotions in sub_emotions_by_main.items():
            print(f"\n{emotion.upper()}:")
            for sub_emotion, count in sub_emotions:
                print(
                    f"  • {sub_emotion}: {count:,} samples ({count/len(self.df):.1%})"
                )

        # 2. Intensity Distribution
        print("\n2. Intensity Distribution:")
        print(intensity_dist.round(3))

        # 3. Text Length Statistics
        print("\n3. Text Length Statistics:")
        print(text_length_stats.round(2))

        # 4. Vocabulary Statistics
        print("\n4. Vocabulary Statistics:")
        for emotion, words in emotion_vocab.items():
            print(f"\n{emotion.upper()}:")
            print(f"  • Vocabulary size: {len(words):,} unique words")
            print(
                f"  • Average words per text: \
                    {len(words)/len(self.df[self.df['emotion'] == emotion]):.1f}"
            )

        # 5. Dataset Balance Metrics
        print("\n5. Dataset Balance Metrics:")
        print(balance_metrics)

        print("\n" + "=" * 50)

        return {
            "sub_emotion_coverage": sub_emotions_by_main,
            "intensity_distribution": intensity_dist,
            "text_length_stats": text_length_stats,
            "vocabulary_stats": emotion_vocab,
            "balance_metrics": balance_metrics,
        }
