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
sys.modules["sklearn.feature_extraction.text"] = MagicMock()
sys.modules["sklearn.model_selection"] = MagicMock()
sys.modules["sklearn.preprocessing"] = MagicMock()
sys.modules["textblob"] = MagicMock()
sys.modules["torch.utils.data"] = MagicMock()
sys.modules["tqdm"] = MagicMock()


# Import the class to be tested
# Update this path to match your project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.emotion_clf_pipeline.data import DatasetLoader  # noqa: E402


class TestDatasetLoader(unittest.TestCase):
    """
    Unit tests for the DatasetLoader class.
    """

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.loader = DatasetLoader()

        # Sample data for testing
        self.sample_train_data = pd.DataFrame(
            {
                "Translation": ["I am happy", "I am angry", "I feel sad"],
                "Emotion": ["joy", "anger", "sadness"],
                "Intensity": ["high", "medium", "low"],
            }
        )

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

        # Update the test_gpt data structure to match what the method expects
        self.sample_test_gpt_data = pd.DataFrame(
            {
                "text": ["I am surprised", "I am disgusted", "I am neutral"],
                "gpt_emotion": ["surprise", "disgust", "neutral"],
                "gpt_sub_emotion": ["surprise", "disgust", "neutral"],
                "gpt_intensity": ["high", "medium", "low"],
            }
        )

    def test_init(self):
        """Test the initialization of the DatasetLoader class."""
        # Check if emotion_mapping is correctly initialized
        self.assertIn("joy", self.loader.emotion_mapping)
        self.assertEqual(self.loader.emotion_mapping["joy"], "happiness")

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
        # (concat would normally combine the two copies, resulting in 6 rows)
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
        self.assertIn("text", result.columns)
        self.assertIn("sub_emotion", result.columns)
        self.assertIn("intensity", result.columns)
        self.assertIn("emotion", result.columns)

        # Check if emotion mapping was applied correctly
        self.assertEqual(result.loc[0, "emotion"], "happiness")  # joy -> happiness
        self.assertEqual(result.loc[1, "emotion"], "anger")  # anger -> anger
        self.assertEqual(result.loc[2, "emotion"], "sadness")  # sadness -> sadness

    @patch("pandas.read_csv")
    @patch("pandas.merge")
    def test_load_test_data(self, mock_merge, mock_read_csv):
        """Test the load_test_data method for both raw and modified versions."""
        # --------------- Test raw version ---------------
        # Setup mock return values for raw version
        mock_read_csv.return_value = self.sample_test_data

        # Call the method with raw version
        raw_result = self.loader.load_test_data(
            test_file="dummy_test_file.csv", version="raw"
        )

        # Assertions for raw version
        self.assertEqual(len(raw_result), 3)
        self.assertIn("text", raw_result.columns)
        self.assertIn("sub_emotion", raw_result.columns)
        self.assertIn("intensity", raw_result.columns)
        self.assertIn("emotion", raw_result.columns)

        # Verify emotion mapping for raw version
        self.assertEqual(raw_result.loc[0, "emotion"], "surprise")
        self.assertEqual(raw_result.loc[1, "emotion"], "disgust")
        self.assertEqual(raw_result.loc[2, "emotion"], "neutral")

        # Verify merge was not called for raw version
        mock_merge.assert_not_called()

        # Reset mocks for modified version test
        mock_read_csv.reset_mock()
        mock_merge.reset_mock()

        # --------------- Test modified version ---------------
        # Setup mocks for modified version
        mock_read_csv.side_effect = [
            self.sample_test_data,  # First call returns test data
            self.sample_test_gpt_data,  # Second call returns test_gpt data
        ]

        # Mock the merge result for modified version
        merged_df = pd.DataFrame(
            {
                "text": ["I am surprised", "I am disgusted", "I am neutral"],
                "sub_emotion": ["surprise", "disgust", "neutral"],
                "intensity": ["high", "medium", "low"],
                "emotion": ["surprise", "disgust", "neutral"],
                "gpt_emotion": ["surprise", "disgust", "neutral"],
                "gpt_sub_emotion": ["surprise", "disgust", "neutral"],
                "gpt_intensity": ["high", "medium", "low"],
            }
        )
        mock_merge.return_value = merged_df

        # Call the method with modified version
        modified_result = self.loader.load_test_data(
            test_file="dummy_test_file.csv", version="modified"
        )

        # Assertions for modified version
        self.assertIn("text", modified_result.columns)
        self.assertIn("sub_emotion", modified_result.columns)
        self.assertIn("intensity", modified_result.columns)
        self.assertIn("emotion", modified_result.columns)

        # Check if read_csv was called twice for modified version
        self.assertEqual(mock_read_csv.call_count, 2)


if __name__ == "__main__":
    unittest.main()
