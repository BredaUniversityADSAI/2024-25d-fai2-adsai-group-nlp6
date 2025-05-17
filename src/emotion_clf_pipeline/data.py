import os
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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

    def load_test_data(
        self, test_file="./../../Data/group 21_url1.csv", version="modified"
    ):
        """
        Load and preprocess test data from a CSV file.

        Args:
            test_file (str): Path to the test data CSV file
            version (str): Test data version to use - "raw" or "modified" (default)

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

        # Apply modifications if version is "modified"
        if version == "modified":
            # Load the test_gpt.csv file
            test_gpt = pd.read_csv("./../../Data/raw/test_gpt.csv")

            # Merge based on text column
            self.test_df = pd.merge(test_gpt, self.test_df, on="text", how="left")

            # Initialize the sub-emotion and intensity to the gpt values
            self.test_df["gpt_sub_emotion"], self.test_df["gpt_intensity"] = (
                self.test_df["sub_emotion"],
                self.test_df["intensity"],
            )

            # Update the sub-emotion and intensity to neutral if the emotion is neutral
            self.test_df.loc[
                self.test_df["gpt_emotion"] == "neutral", "gpt_sub_emotion"
            ] = "neutral"
            self.test_df.loc[
                self.test_df["gpt_intensity"] == "neutral", "gpt_intensity"
            ] = "neutral"

            # Keep the necessary columns
            self.test_df = self.test_df[
                ["text", "gpt_emotion", "gpt_sub_emotion", "gpt_intensity"]
            ]

            # Rename the columns
            self.test_df.rename(
                columns={
                    "gpt_emotion": "emotion",
                    "gpt_sub_emotion": "sub_emotion",
                    "gpt_intensity": "intensity",
                },
                inplace=True,
            )

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
        # self.output_columns = ['emotion', 'sub_emotion', 'intensity']

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

        print(f"Applying data augmentation with strategy: {balance_strategy}")
        original_class_dist = train_df["emotion"].value_counts()
        print("Original class distribution:")
        for emotion, count in original_class_dist.items():
            print(f"  {emotion}: {count} samples ({count/len(train_df):.2%})")

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
            print("\nAfter emotion balancing, checking sub-emotion distribution:")
            sub_emotion_dist = balanced_df["sub_emotion"].value_counts()
            print(f"Sub-emotion classes: {len(sub_emotion_dist)}")
            print(
                f"Min class size: {sub_emotion_dist.min()}, "
                f"Max class size: {sub_emotion_dist.max()}"
            )

            # If sub-emotion is highly imbalanced, apply additional balancing
            imbalance_ratio = sub_emotion_dist.max() / sub_emotion_dist.min()
            if imbalance_ratio > 5:  # If max/min ratio is greater than 5
                print(
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
        os.makedirs("./results/encoders", exist_ok=True)

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
        print("Fitting TF-IDF vectorizer...")
        self.feature_extractor.fit_tfidf(train_df["text"].values)

        # Extract features for all texts
        print("Extracting features for training data...")
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
            print("Extracting features for test data...")
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
            with open(f"./results/encoders/{filename}_encoder.pkl", "wb") as f:
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
