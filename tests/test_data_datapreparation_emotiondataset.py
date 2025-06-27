import contextlib
import io
import os
import sys
import unittest
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd

# Mock ALL Azure-related modules f
sys.modules["azure"] = MagicMock()
sys.modules["azure.ai"] = MagicMock()
sys.modules["azure.ai.ml"] = MagicMock()
sys.modules["azure.ai.ml.entities"] = MagicMock()
sys.modules["azure.identity"] = MagicMock()
sys.modules["azure.core"] = MagicMock()
sys.modules["azure.core.exceptions"] = MagicMock()

# Create mock MLClient class and Job class
mock_ml_client = MagicMock()
mock_job = MagicMock()
sys.modules["azure.ai.ml"].MLClient = mock_ml_client
sys.modules["azure.ai.ml.entities"].Data = MagicMock()
sys.modules["azure.identity"].DefaultAzureCredential = MagicMock()

# Mock the entire azure_pipeline
mock_azure_pipeline = MagicMock()
mock_azure_pipeline.register_processed_data_assets_from_paths = MagicMock()
sys.modules["azure_pipeline"] = mock_azure_pipeline
sys.modules["src.emotion_clf_pipeline.azure_pipeline"] = mock_azure_pipeline

# Mock the deep learning and ML libraries
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.optim"] = MagicMock()
sys.modules["torch.utils"] = MagicMock()
sys.modules["torch.utils.data"] = MagicMock()
sys.modules["torch.utils.data.dataset"] = MagicMock()
sys.modules["torch.utils.data.dataloader"] = MagicMock()

# Mock transformers
sys.modules["transformers"] = MagicMock()

# Mock sklearn modules
sys.modules["sklearn"] = MagicMock()
sys.modules["sklearn.feature_extraction"] = MagicMock()
sys.modules["sklearn.feature_extraction.text"] = MagicMock()
sys.modules["sklearn.model_selection"] = MagicMock()
sys.modules["sklearn.preprocessing"] = MagicMock()
sys.modules["sklearn.metrics"] = MagicMock()
sys.modules["sklearn.utils"] = MagicMock()

# Mock matplotlib and its submodules
matplotlib_mock = MagicMock()
sys.modules["matplotlib"] = matplotlib_mock
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["matplotlib.colors"] = MagicMock()
sys.modules["matplotlib.cbook"] = MagicMock()
sys.modules["matplotlib.figure"] = MagicMock()
sys.modules["matplotlib.collections"] = MagicMock()
sys.modules["matplotlib.markers"] = MagicMock()
sys.modules["matplotlib.patches"] = MagicMock()
sys.modules["matplotlib.ticker"] = MagicMock()
sys.modules["matplotlib.dates"] = MagicMock()
sys.modules["matplotlib.axis"] = MagicMock()
sys.modules["matplotlib.scale"] = MagicMock()
sys.modules["matplotlib.transforms"] = MagicMock()

# Mock textblob
sys.modules["textblob"] = MagicMock()

# Mock tqdm
sys.modules["tqdm"] = MagicMock()

# Mock pickle
sys.modules["pickle"] = MagicMock()

# Mock seaborn
sys.modules["seaborn"] = MagicMock()

# Mock scipy and its submodules
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.sparse"] = MagicMock()
sys.modules["scipy.stats"] = MagicMock()
sys.modules["scipy.optimize"] = MagicMock()

# Mock nltk
sys.modules["nltk"] = MagicMock()
sys.modules["nltk.corpus"] = MagicMock()
sys.modules["nltk.tokenize"] = MagicMock()
sys.modules["nltk.stem"] = MagicMock()
sys.modules["nltk.sentiment.vader"] = MagicMock()

# Mock spacy
sys.modules["spacy"] = MagicMock()

# Mock gensim
sys.modules["gensim"] = MagicMock()
sys.modules["gensim.models"] = MagicMock()

# Mock wordcloud
sys.modules["wordcloud"] = MagicMock()

# Mock plotly
sys.modules["plotly"] = MagicMock()
sys.modules["plotly.express"] = MagicMock()
sys.modules["plotly.graph_objects"] = MagicMock()

# Mock tensorflow
sys.modules["tensorflow"] = MagicMock()

# Mock keras
sys.modules["keras"] = MagicMock()

# Mock xgboost
sys.modules["xgboost"] = MagicMock()

# Mock lightgbm
sys.modules["lightgbm"] = MagicMock()

# Mock catboost
sys.modules["catboost"] = MagicMock()

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Create mock torch module with proper tensor functionality
torch_mock = MagicMock()
torch_mock.tensor = MagicMock(
    side_effect=lambda x, dtype=None: MagicMock(flatten=MagicMock(return_value=x))
)
torch_mock.float32 = float
torch_mock.long = int
torch_mock.equal = MagicMock(return_value=True)
sys.modules["torch"] = torch_mock


# Mock the Dataset class from torch.utils.data
class MockDataset:
    def __init__(self, *args, **kwargs):
        pass


sys.modules["torch.utils.data"].Dataset = MockDataset


# Mock supporting classes that will be needed by the imported classes
class MockLabelEncoder:
    def __init__(self):
        self.classes_ = ["joy", "sadness", "anger", "surprise", "fear"]

    def fit(self, y):
        pass

    def transform(self, y):
        return list(range(len(y)))


class MockFeatureExtractor:
    def __init__(self, feature_config=None, lexicon_path=None):
        self.feature_config = feature_config
        self.lexicon_path = lexicon_path

    def fit_tfidf(self, texts):
        pass

    def extract_all_features(self, text):
        return [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
        ]  # Return 5 features to match expected dimensionality


class MockDataLoader:
    def __init__(self, dataset, batch_size=16, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle


class MockTextAugmentor:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def generate_equal_samples(
        self, df, text_column, emotion_column, samples_per_class, random_state
    ):
        return df.copy()

    def balance_dataset(
        self,
        df,
        text_column,
        emotion_column,
        target_count,
        augmentation_ratio,
        random_state,
    ):
        return df.copy()


# Mock train_test_split function that returns numpy arrays instead of lists
def mock_train_test_split(*arrays, test_size=None, random_state=None, stratify=None):
    total_size = len(arrays[0]) if arrays else 10
    split_point = (
        int(total_size * (1 - test_size)) if test_size else int(total_size * 0.8)
    )

    # Return numpy arrays that can be used for indexing
    train_indices = np.array(list(range(split_point)))
    val_indices = np.array(list(range(split_point, total_size)))
    return train_indices, val_indices


# Apply the mocks to their respective modules
sys.modules["sklearn.model_selection"].train_test_split = mock_train_test_split
sys.modules["sklearn.preprocessing"].LabelEncoder = MockLabelEncoder
sys.modules["torch.utils.data"].DataLoader = MockDataLoader

# Mock the augmentation module
augmentation_mock = MagicMock()
augmentation_mock.TextAugmentor = MockTextAugmentor
sys.modules["src.emotion_clf_pipeline.augmentation"] = augmentation_mock

# mock tqdm.tqdm for the progress bars
tqdm_mock = MagicMock()
tqdm_mock.tqdm = MagicMock(
    side_effect=lambda x, **kwargs: x
)  # Just return the iterable
sys.modules["tqdm"] = tqdm_mock
sys.modules["tqdm.tqdm"] = tqdm_mock.tqdm

sys.modules["src.emotion_clf_pipeline.feature_extraction"] = MagicMock()

# Ensure the feature extractor class is available
feature_extraction_mock = MagicMock()
feature_extraction_mock.FeatureExtractor = MockFeatureExtractor
sys.modules["src.emotion_clf_pipeline.feature_extraction"] = feature_extraction_mock


# Suppress any output during import
with (
    contextlib.redirect_stdout(io.StringIO()),
    contextlib.redirect_stderr(io.StringIO()),
):
    try:
        from src.emotion_clf_pipeline.data import (  # noqa: E402
            DataPreparation,
            EmotionDataset,
        )

        print("Successfully imported real classes from src.emotion_clf_pipeline.data")
    except ImportError as e:
        print(f"Import failed with error: {e}")
        print("Checking project structure...")

        expected_path = os.path.join(
            project_root, "src", "emotion_clf_pipeline", "data.py"
        )
        if os.path.exists(expected_path):
            print(f"Found data.py at: {expected_path}")
        else:
            print(f"Could not find data.py at: {expected_path}")
            data_dir = os.path.join(project_root, "src", "emotion_clf_pipeline")
            if os.path.exists(data_dir):
                print(f"Contents of {data_dir}:")
                for item in os.listdir(data_dir):
                    print(f"  - {item}")

        # Re-raise the error so you can see exactly what's missing
        raise ImportError(f"Failed to import required classes: {e}")


class TestEmotionDataset(unittest.TestCase):
    """Test cases for the EmotionDataset class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.tokenizer = MagicMock()
        self.tokenizer.return_value = {
            "input_ids": MagicMock(
                flatten=MagicMock(return_value=[101, 2054, 2003, 102, 0, 0])
            ),
            "attention_mask": MagicMock(
                flatten=MagicMock(return_value=[1, 1, 1, 1, 0, 0])
            ),
        }

        self.texts = ["This is a test", "Another test text"]
        self.features = [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.4, 0.5, 0.6, 0.7, 0.8],
        ]  # 5 features each
        self.labels = [[0, 1, 2], [1, 0, 1]]
        self.output_tasks = ["emotion", "sub_emotion", "intensity"]
        self.max_length = 128

    def test_dataset_initialization(self):
        """Test dataset initialization."""
        dataset = EmotionDataset(
            texts=self.texts,
            tokenizer=self.tokenizer,
            features=self.features,
            labels=self.labels,
            max_length=self.max_length,
            output_tasks=self.output_tasks,
        )

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.texts, self.texts)
        self.assertEqual(dataset.max_length, self.max_length)
        self.assertEqual(dataset.features, self.features)
        self.assertEqual(dataset.labels, self.labels)
        self.assertEqual(dataset.output_tasks, self.output_tasks)

    def test_getitem_with_labels(self):
        """Test __getitem__ method with labels."""
        dataset = EmotionDataset(
            texts=self.texts,
            tokenizer=self.tokenizer,
            features=self.features,
            labels=self.labels,
            max_length=self.max_length,
            output_tasks=self.output_tasks,
        )

        item = dataset[0]

        self.tokenizer.assert_called_with(
            self.texts[0],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        self.assertIn("input_ids", item)
        self.assertIn("attention_mask", item)
        self.assertIn("features", item)
        self.assertIn("emotion_label", item)
        self.assertIn("sub_emotion_label", item)
        self.assertIn("intensity_label", item)

    def test_getitem_without_labels(self):
        """Test __getitem__ method without labels."""
        dataset = EmotionDataset(
            texts=self.texts,
            tokenizer=self.tokenizer,
            features=self.features,
            labels=None,
            max_length=self.max_length,
        )

        item = dataset[0]

        self.assertIn("input_ids", item)
        self.assertIn("attention_mask", item)
        self.assertIn("features", item)
        self.assertNotIn("emotion_label", item)
        self.assertNotIn("sub_emotion_label", item)
        self.assertNotIn("intensity_label", item)

    def test_dataset_length(self):
        """Test dataset length method."""
        dataset = EmotionDataset(
            texts=self.texts,
            tokenizer=self.tokenizer,
            features=self.features,
        )
        self.assertEqual(len(dataset), len(self.texts))


class TestDataPreparation(unittest.TestCase):
    """Test cases for the DataPreparation class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.tokenizer = MagicMock()
        self.output_columns = ["emotion", "sub_emotion", "intensity"]
        self.max_length = 128
        self.batch_size = 16

        self.train_df = pd.DataFrame(
            {
                "text": ["I am happy", "I am sad", "I am angry", "I am surprised"],
                "emotion": ["joy", "sadness", "anger", "surprise"],
                "sub_emotion": ["contentment", "disappointment", "rage", "amazement"],
                "intensity": ["high", "medium", "high", "low"],
            }
        )

        self.test_df = pd.DataFrame(
            {
                "text": ["I am excited", "I am worried"],
                "emotion": ["joy", "fear"],
                "sub_emotion": ["excitement", "anxiety"],
                "intensity": ["medium", "high"],
            }
        )

    def test_initialization(self):
        """Test initialization of DataPreparation."""
        data_prep = DataPreparation(
            output_columns=self.output_columns,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )

        self.assertEqual(data_prep.output_columns, self.output_columns)
        self.assertEqual(data_prep.tokenizer, self.tokenizer)
        self.assertEqual(data_prep.max_length, self.max_length)
        self.assertEqual(data_prep.batch_size, self.batch_size)
        self.assertEqual(len(data_prep.label_encoders), len(self.output_columns))

    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pickle.dump")
    @patch("sys.stdout", new_callable=io.StringIO)  # Suppress print statements
    def test_prepare_data_without_augmentation(
        self, mock_stdout, mock_pickle_dump, mock_file_open, mock_makedirs
    ):
        """Test prepare_data method without data augmentation."""
        data_prep = DataPreparation(
            output_columns=self.output_columns,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )

        with patch.object(
            data_prep.feature_extractor,
            "extract_all_features",
            side_effect=lambda x: [0.1, 0.2, 0.3, 0.4, 0.5],
        ):
            train_dataloader, val_dataloader, test_dataloader = data_prep.prepare_data(
                train_df=self.train_df,
                test_df=self.test_df,
                validation_split=0.2,
                apply_augmentation=False,
            )

            self.assertIsNotNone(train_dataloader)
            self.assertIsNotNone(val_dataloader)
            self.assertIsNotNone(test_dataloader)

    def test_get_num_classes(self):
        """Test get_num_classes method."""
        data_prep = DataPreparation(
            output_columns=self.output_columns,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )

        num_classes = data_prep.get_num_classes()

        self.assertIsInstance(num_classes, dict)
        self.assertEqual(len(num_classes), len(self.output_columns))

        for col in self.output_columns:
            self.assertIn(col, num_classes)
            self.assertIsInstance(num_classes[col], int)

    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pickle.dump")
    def test_save_encoders(self, mock_pickle_dump, mock_file_open, mock_makedirs):
        """Test _save_encoders method."""
        data_prep = DataPreparation(
            output_columns=self.output_columns,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )

        # Call the method
        data_prep._save_encoders()

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
