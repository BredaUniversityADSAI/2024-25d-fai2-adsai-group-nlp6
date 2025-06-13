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

        self.sample_train_data = pd.DataFrame(
            {
                "start_time": [0, 1, 2],
                "end_time": [5, 6, 7],
                "text": ["I am happy", "I am angry", "I feel sad"],
                "emotion": ["joy", "anger", "sadness"],
                "sub-emotion": ["elated", "furious", "depressed"],
                "intensity": ["high", "medium", "low"],
            }
        )

        self.sample_test_data = pd.DataFrame(
            {
                "start_time": [0, 1, 2, 3],
                "end_time": [5, 6, 7, 8],
                "text": [
                    "I am surprised",
                    "I am disgusted",
                    "I am neutral",
                    "I feel fear",
                ],
                "emotion": ["surprise", "disgust", "neutral", "fear"],
                "sub-emotion": ["amazed", "revolted", "calm", "terrified"],
                "intensity": ["high", "medium", "low", "high"],
            }
        )

        self.dirty_data = pd.DataFrame(
            {
                "start_time": [0, 1, 2, 0, None],
                "end_time": [5, 6, 7, 5, 6],
                "text": ["I am happy", "I am angry", "I feel sad", "I am happy", None],
                "emotion": ["joy", "anger", "sadness", "joy", "fear"],
                "sub-emotion": ["elated", "furious", "depressed", "elated", "scared"],
                "intensity": ["high", "medium", "low", "high", "medium"],
            }
        )

    def test_init(self):
        """Test the initialization of the DatasetLoader class."""
        # Check that train_df and test_df are initially None
        self.assertIsNone(self.loader.train_df)
        self.assertIsNone(self.loader.test_df)

        # Verify that the DatasetLoader instance is created successfully
        self.assertIsInstance(self.loader, DatasetLoader)

    @patch("os.listdir")
    @patch("pandas.read_csv")
    def test_load_training_data_success(self, mock_read_csv, mock_listdir):
        """Test successful loading of training data from multiple CSV files."""
        # Setup mock return values
        mock_listdir.return_value = [
            "train_data-0001.csv",
            "train_data-0002.csv",
            "not_a_csv.txt",
        ]

        file1_data = pd.DataFrame(
            {
                "start_time": [0, 1],
                "end_time": [5, 6],
                "text": ["I am happy", "I am angry"],
                "emotion": ["joy", "anger"],
                "sub-emotion": ["elated", "furious"],
                "intensity": ["high", "medium"],
            }
        )

        file2_data = pd.DataFrame(
            {
                "start_time": [2, 3],
                "end_time": [7, 8],
                "text": ["I feel sad", "I am surprised"],
                "emotion": ["sadness", "surprise"],
                "sub-emotion": ["depressed", "amazed"],
                "intensity": ["low", "high"],
            }
        )

        # Mock read_csv to return different data for each call
        mock_read_csv.side_effect = [file1_data, file2_data]

        # Call the method
        result = self.loader.load_training_data(data_dir="dummy_dir")

        # Assertions
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)

        # Check that the result has the expected columns after processing
        expected_columns = [
            "start_time",
            "end_time",
            "text",
            "emotion",
            "sub_emotion",
            "intensity",
        ]
        for col in expected_columns:
            self.assertIn(col, result.columns)

        # Verify that non-CSV files are skipped (only 2 CSV files should be processed)
        self.assertEqual(mock_read_csv.call_count, 2)

        # Verify that the loader's train_df attribute was set
        self.assertIsNotNone(self.loader.train_df)
        # Should have 4 rows total (2 from each file)
        self.assertEqual(len(self.loader.train_df), 4)

    @patch("os.listdir")
    @patch("pandas.read_csv")
    def test_load_training_data_column_renaming(self, mock_read_csv, mock_listdir):
        """Test that sub-emotion column is renamed to sub_emotion."""
        mock_listdir.return_value = ["train_data-0001.csv"]
        mock_read_csv.return_value = self.sample_train_data

        result = self.loader.load_training_data(data_dir="dummy_dir")

        # Check that sub-emotion was renamed to sub_emotion
        self.assertIn("sub_emotion", result.columns)
        self.assertNotIn("sub-emotion", result.columns)

    @patch("os.listdir")
    @patch("pandas.read_csv")
    def test_load_training_data_cleaning(self, mock_read_csv, mock_listdir):
        """Test that duplicates and null values are properly cleaned."""
        mock_listdir.return_value = ["train_data-0001.csv"]
        mock_read_csv.return_value = self.dirty_data

        result = self.loader.load_training_data(data_dir="dummy_dir")

        # Check that nulls and duplicates are removed
        self.assertEqual(len(result), 3)  # Should have 3 unique, non-null rows
        self.assertFalse(result.isnull().any().any())  # No null values
        self.assertFalse(result.duplicated().any())  # No duplicates

    @patch("os.listdir")
    @patch("pandas.read_csv")
    def test_load_training_data_file_error(self, mock_read_csv, mock_listdir):
        """Test handling of file reading errors."""
        mock_listdir.return_value = ["bad_file.csv", "good_file.csv"]

        # First call raises exception, second call succeeds
        mock_read_csv.side_effect = [
            Exception("File not found"),
            self.sample_train_data,
        ]

        with patch("src.emotion_clf_pipeline.data.logger") as mock_logger:
            result = self.loader.load_training_data(data_dir="dummy_dir")

            # Check that error was logged
            mock_logger.error.assert_called()

            # Check that the good file was still processed
            self.assertIsNotNone(result)
            self.assertEqual(len(result), 3)

    @patch("pandas.read_csv")
    def test_load_test_data_success(self, mock_read_csv):
        """Test successful loading of test data."""
        mock_read_csv.return_value = self.sample_test_data

        result = self.loader.load_test_data(test_file="dummy_test_file.csv")

        # Assertions
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 4)

        # Check expected columns
        expected_columns = [
            "start_time",
            "end_time",
            "text",
            "emotion",
            "sub_emotion",
            "intensity",
        ]
        for col in expected_columns:
            self.assertIn(col, result.columns)

        # Verify that the loader's test_df attribute was set
        self.assertIsNotNone(self.loader.test_df)

        # Verify read_csv was called with correct parameters
        mock_read_csv.assert_called_once_with("dummy_test_file.csv")

    @patch("pandas.read_csv")
    def test_load_test_data_column_renaming(self, mock_read_csv):
        """Test that sub-emotion column is renamed to sub_emotion in test data."""
        mock_read_csv.return_value = self.sample_test_data

        result = self.loader.load_test_data(test_file="dummy_test_file.csv")

        # Check that sub-emotion was renamed to sub_emotion
        self.assertIn("sub_emotion", result.columns)
        self.assertNotIn("sub-emotion", result.columns)

    @patch("pandas.read_csv")
    def test_load_test_data_cleaning(self, mock_read_csv):
        """Test that test data is properly cleaned."""
        mock_read_csv.return_value = self.dirty_data

        result = self.loader.load_test_data(test_file="dummy_test_file.csv")

        # Check that nulls and duplicates are removed
        self.assertEqual(len(result), 3)  # Should have 3 unique, non-null rows
        self.assertFalse(result.isnull().any().any())  # No null values
        self.assertFalse(result.duplicated().any())  # No duplicates

    @patch("pandas.read_csv")
    def test_load_test_data_file_error(self, mock_read_csv):
        """Test handling of test file reading errors."""
        mock_read_csv.side_effect = Exception("File not found")

        with patch("src.emotion_clf_pipeline.data.logger") as mock_logger:
            result = self.loader.load_test_data(test_file="bad_file.csv")

            # Check that error was logged and None was returned
            mock_logger.error.assert_called()
            self.assertIsNone(result)

    @patch("os.listdir")
    def test_load_training_data_empty_directory(self, mock_listdir):
        """Test handling of empty directory."""
        mock_listdir.return_value = []

        result = self.loader.load_training_data(data_dir="empty_dir")

        # Should return empty DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    @patch("os.listdir")
    def test_load_training_data_no_csv_files(self, mock_listdir):
        """Test handling of directory with no CSV files."""
        mock_listdir.return_value = ["file1.txt", "file2.json", "file3.xml"]

        with patch("src.emotion_clf_pipeline.data.logger") as mock_logger:
            result = self.loader.load_training_data(data_dir="no_csv_dir")

            # Check that warnings were logged for non-CSV files
            self.assertEqual(mock_logger.warning.call_count, 3)

            # Should return empty DataFrame
            self.assertIsInstance(result, pd.DataFrame)
            self.assertTrue(result.empty)

    def test_plot_distributions_with_data(self):
        """Test plot_distributions method when both datasets are loaded."""
        # Set up test data
        self.loader.train_df = self.sample_train_data.rename(
            columns={"sub-emotion": "sub_emotion"}
        )
        self.loader.test_df = self.sample_test_data.rename(
            columns={"sub-emotion": "sub_emotion"}
        )

        # Mock matplotlib and seaborn
        with (
            patch("matplotlib.pyplot.subplots") as mock_subplots,
            patch("seaborn.countplot") as mock_countplot,
            patch("matplotlib.pyplot.tight_layout") as mock_tight_layout,  # noqa: F841
            patch("matplotlib.pyplot.show") as mock_show,
        ):

            # Mock the subplots return value
            mock_fig = MagicMock()
            mock_axes = [MagicMock() for _ in range(3)]
            mock_subplots.return_value = (mock_fig, mock_axes)

            # Call the method
            self.loader.plot_distributions()

            # Verify that subplots was called twice (train and test)
            self.assertEqual(mock_subplots.call_count, 2)

            # Verify that countplot was called 6 times (3 columns * 2 datasets)
            self.assertEqual(mock_countplot.call_count, 6)

            # Verify that show was called twice
            self.assertEqual(mock_show.call_count, 2)

    def test_data_integrity_after_loading(self):
        """Test that data maintains integrity after loading and processing."""
        with (
            patch("os.listdir") as mock_listdir,
            patch("pandas.read_csv") as mock_read_csv,
        ):

            mock_listdir.return_value = ["train_data-0001.csv"]
            mock_read_csv.return_value = self.sample_train_data.copy()

            result = self.loader.load_training_data(data_dir="dummy_dir")

            # Check that text data is preserved correctly
            expected_texts = ["I am happy", "I am angry", "I feel sad"]
            actual_texts = result["text"].tolist()
            self.assertEqual(actual_texts, expected_texts)

            # Check that emotions are preserved correctly
            expected_emotions = ["joy", "anger", "sadness"]
            actual_emotions = result["emotion"].tolist()
            self.assertEqual(actual_emotions, expected_emotions)

    def test_index_reset_after_processing(self):
        """Test that DataFrame index is properly reset after processing."""
        with (
            patch("os.listdir") as mock_listdir,
            patch("pandas.read_csv") as mock_read_csv,
        ):

            # Create data with non-sequential index
            data_with_bad_index = self.sample_train_data.copy()
            data_with_bad_index.index = [5, 10, 15]

            mock_listdir.return_value = ["train_data-0001.csv"]
            mock_read_csv.return_value = data_with_bad_index

            result = self.loader.load_training_data(data_dir="dummy_dir")

            # Check that index is reset to sequential
            expected_index = list(range(len(result)))
            actual_index = result.index.tolist()
            self.assertEqual(actual_index, expected_index)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
