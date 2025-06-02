import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

# Mock the required modules before importing the target module
sys.modules["torch"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["seaborn"] = MagicMock()
sys.modules["nltk"] = MagicMock()
sys.modules["nltk.sentiment.vader"] = MagicMock()
sys.modules["nltk.tokenize"] = MagicMock()
sys.modules["nltk.corpus"] = MagicMock()
sys.modules["sklearn.feature_extraction.text"] = MagicMock()
sys.modules["sklearn.model_selection"] = MagicMock()
sys.modules["sklearn.preprocessing"] = MagicMock()
sys.modules["textblob"] = MagicMock()
sys.modules["torch.utils.data"] = MagicMock()
sys.modules["tqdm"] = MagicMock()


# Import the class to be tested
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.emotion_clf_pipeline.data import DatasetLoader  # noqa: E402


class TestDatasetLoader(unittest.TestCase):
    """
    Unit tests for the DatasetLoader class.
    """

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.loader = DatasetLoader()

        # Sample training data for testing
        self.sample_train_data = pd.DataFrame(
            {
                "Translation": ["I am happy", "I am angry", "I feel sad"],
                "Emotion": ["joy", "anger", "sadness"],
                "Intensity": ["high", "medium", "low"],
            }
        )

        # Sample test data for testing
        self.sample_test_data = pd.DataFrame(
            {
                "Corrected Sentence": [
                    "I am surprised",
                    "I am disgusted",
                    "I am neutral",
                ],
                "Emotion": ["surprise", "disgust", "neutral"],
                "Intensity": ["high", "medium", "low"],
            }
        )

    def test_init(self):
        """Test the initialization of the DatasetLoader class."""
        # Check if emotion_mapping is correctly initialized
        self.assertIn("joy", self.loader.emotion_mapping)
        self.assertEqual(self.loader.emotion_mapping["joy"], "happiness")
        self.assertEqual(self.loader.emotion_mapping["anger"], "anger")
        self.assertEqual(self.loader.emotion_mapping["neutral"], "neutral")

        # Check if train_df and test_df are initially None
        self.assertIsNone(self.loader.train_df)
        self.assertIsNone(self.loader.test_df)

    @patch("os.listdir")
    @patch("pandas.read_csv")
    @patch("pandas.concat")
    def test_load_training_data(self, mock_concat, mock_read_csv, mock_listdir):
        """Test the load_training_data method."""
        # Setup mock return values
        mock_listdir.return_value = ["file1.csv", "file2.csv"]
        mock_read_csv.return_value = self.sample_train_data

        # Mock concat to return a dataframe with the expected number of rows
        combined_df = pd.DataFrame(
            {
                "Translation": ["I am happy", "I am angry", "I feel sad"] * 2,
                "Emotion": ["joy", "anger", "sadness"] * 2,
                "Intensity": ["high", "medium", "low"] * 2,
            }
        )
        mock_concat.return_value = combined_df

        # Call the method
        result = self.loader.load_training_data(data_dir="dummy_dir")

        # Assertions
        self.assertIsNotNone(result)
        self.assertIn("text", result.columns)
        self.assertIn("sub_emotion", result.columns)
        self.assertIn("intensity", result.columns)
        self.assertIn("emotion", result.columns)

        # Check if emotion mapping was applied correctly
        self.assertEqual(result.loc[0, "emotion"], "happiness")  # joy -> happiness
        self.assertEqual(result.loc[1, "emotion"], "anger")  # anger -> anger
        self.assertEqual(result.loc[2, "emotion"], "sadness")  # sadness -> sadness

        # Verify that the loader's train_df attribute was set
        self.assertIsNotNone(self.loader.train_df)

    @patch("pandas.read_csv")
    def test_load_test_data(self, mock_read_csv):
        """Test the load_test_data method."""
        # Setup mock return value
        mock_read_csv.return_value = self.sample_test_data

        # Call the method
        result = self.loader.load_test_data(test_file="dummy_test_file.csv")

        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        self.assertIn("text", result.columns)
        self.assertIn("sub_emotion", result.columns)
        self.assertIn("intensity", result.columns)
        self.assertIn("emotion", result.columns)

        # Verify emotion mapping was applied correctly
        self.assertEqual(result.loc[0, "emotion"], "surprise")
        self.assertEqual(result.loc[1, "emotion"], "disgust")
        self.assertEqual(result.loc[2, "emotion"], "neutral")

        # Verify that the loader's test_df attribute was set
        self.assertIsNotNone(self.loader.test_df)

        # Verify read_csv was called with correct parameters
        mock_read_csv.assert_called_once_with("dummy_test_file.csv")

    def test_emotion_mapping_completeness(self):
        """Test that emotion mapping contains expected emotions."""
        expected_emotions = [
            "joy",
            "anger",
            "sadness",
            "surprise",
            "disgust",
            "neutral",
            "fear",
        ]

        for emotion in expected_emotions:
            self.assertIn(emotion, self.loader.emotion_mapping)

    def test_emotion_mapping_values(self):
        """Test that emotion mapping values are valid standardized emotions."""
        valid_standard_emotions = [
            "happiness",
            "anger",
            "sadness",
            "surprise",
            "disgust",
            "neutral",
            "fear",
        ]

        for mapped_emotion in self.loader.emotion_mapping.values():
            self.assertIn(mapped_emotion, valid_standard_emotions)


if __name__ == "__main__":
    unittest.main()
