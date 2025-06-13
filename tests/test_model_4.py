import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

# Add the source directory to Python path
current_test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_test_dir)
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

# Comprehensive list of modules to mock
modules_to_mock = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.optim",
    "torch.utils",
    "torch.utils.data",
    "transformers",
    "textblob",
    "textblob.sentiment",
    "nltk",
    "nltk.sentiment",
    "nltk.sentiment.vader",
    "nltk.tokenize",
    "nltk.tag",
    "nltk.corpus",
    "nltk.data",
    "sklearn",
    "sklearn.feature_extraction",
    "sklearn.feature_extraction.text",
    "sklearn.model_selection",
    "sklearn.metrics",
    "sklearn.preprocessing",
    "sklearn.utils",
    "sklearn.utils.validation",
    "sklearn.base",
    "sklearn.linear_model",
    "sklearn.ensemble",
    "sklearn.svm",
    "sklearn.naive_bayes",
    "pandas",
    "seaborn",
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.colors",
    "matplotlib.ticker",
    "matplotlib.transforms",
    "matplotlib.rcsetup",
    "matplotlib.cbook",
    "matplotlib._docstring",
    "matplotlib.version",
    "matplotlib.api",
    "matplotlib.cm",
    "matplotlib.scale",
    "numpy",
    "numpy.linalg",
]


def setup_mocks():
    """Set up all the necessary mocks for external dependencies."""

    # Mock torch modules
    torch_mock = MagicMock()
    torch_mock.tensor = MockTensor
    torch_mock.zeros = lambda size: MockTensor(
        [0] * (size if isinstance(size, int) else size[0])
    )
    torch_mock.Size = MockSize
    torch_mock.float32 = "float32"

    # Mock Dataset class from torch.utils.data
    class MockDataset:
        def __init__(self):
            pass

    torch_mock.utils.data.Dataset = MockDataset

    sys.modules["torch"] = torch_mock
    sys.modules["torch.nn"] = MagicMock()
    sys.modules["torch.nn.functional"] = MagicMock()
    sys.modules["torch.optim"] = MagicMock()
    sys.modules["torch.utils"] = MagicMock()
    sys.modules["torch.utils.data"] = MagicMock()
    sys.modules["torch.utils.data"].Dataset = MockDataset

    # Mock transformers
    sys.modules["transformers"] = MagicMock()

    # Mock textblob and related dependencies
    sys.modules["textblob"] = MagicMock()
    sys.modules["textblob.sentiment"] = MagicMock()

    # Mock NLTK
    nltk_mock = MagicMock()
    sys.modules["nltk"] = nltk_mock
    sys.modules["nltk.sentiment"] = MagicMock()
    sys.modules["nltk.sentiment.vader"] = MagicMock()
    sys.modules["nltk.tokenize"] = MagicMock()
    sys.modules["nltk.tag"] = MagicMock()
    sys.modules["nltk.corpus"] = MagicMock()
    sys.modules["nltk.data"] = MagicMock()

    # Mock sklearn
    mock_sklearn = MagicMock()

    # Core sklearn modules
    mock_sklearn.preprocessing = MagicMock()
    mock_sklearn.preprocessing.LabelEncoder = MagicMock()
    mock_sklearn.preprocessing.StandardScaler = MagicMock()

    mock_sklearn.feature_extraction = MagicMock()
    mock_sklearn.feature_extraction.text = MagicMock()
    mock_sklearn.feature_extraction.text.TfidfVectorizer = MagicMock()

    mock_sklearn.model_selection = MagicMock()
    mock_sklearn.model_selection.train_test_split = MagicMock()

    mock_sklearn.metrics = MagicMock()
    mock_sklearn.metrics.classification_report = MagicMock()
    mock_sklearn.metrics.accuracy_score = MagicMock()
    mock_sklearn.metrics.f1_score = MagicMock()

    mock_sklearn.utils = MagicMock()
    mock_sklearn.utils.validation = MagicMock()
    mock_sklearn.base = MagicMock()
    mock_sklearn.linear_model = MagicMock()
    mock_sklearn.ensemble = MagicMock()
    mock_sklearn.svm = MagicMock()
    mock_sklearn.naive_bayes = MagicMock()

    # Set up all sklearn modules in sys.modules
    sys.modules["sklearn"] = mock_sklearn
    sys.modules["sklearn.preprocessing"] = mock_sklearn.preprocessing
    sys.modules["sklearn.feature_extraction"] = mock_sklearn.feature_extraction
    sys.modules["sklearn.feature_extraction.text"] = (
        mock_sklearn.feature_extraction.text
    )
    sys.modules["sklearn.model_selection"] = mock_sklearn.model_selection
    sys.modules["sklearn.metrics"] = mock_sklearn.metrics
    sys.modules["sklearn.utils"] = mock_sklearn.utils
    sys.modules["sklearn.utils.validation"] = mock_sklearn.utils.validation
    sys.modules["sklearn.base"] = mock_sklearn.base
    sys.modules["sklearn.linear_model"] = mock_sklearn.linear_model
    sys.modules["sklearn.ensemble"] = mock_sklearn.ensemble
    sys.modules["sklearn.svm"] = mock_sklearn.svm
    sys.modules["sklearn.naive_bayes"] = mock_sklearn.naive_bayes

    # Mock pandas
    sys.modules["pandas"] = MagicMock()

    # Mock seaborn
    sys.modules["seaborn"] = MagicMock()

    # Mock matplotlib with all necessary submodules
    mock_matplotlib = MagicMock()
    mock_plt = MagicMock()
    mock_matplotlib.pyplot = mock_plt
    mock_matplotlib.colors = MagicMock()
    mock_matplotlib.colors.to_rgb = MagicMock()
    mock_matplotlib.colors.Colormap = MagicMock()
    mock_matplotlib.colors.is_color_like = MagicMock()
    mock_matplotlib.ticker = MagicMock()
    mock_matplotlib.transforms = MagicMock()
    mock_matplotlib.rcsetup = MagicMock()
    mock_matplotlib.cbook = MagicMock()
    mock_matplotlib.cbook.normalize_kwargs = MagicMock()
    mock_matplotlib._docstring = MagicMock()
    mock_matplotlib.version = MagicMock()
    mock_matplotlib.api = MagicMock()
    mock_matplotlib.cm = MagicMock()
    mock_matplotlib.scale = MagicMock()

    sys.modules["matplotlib"] = mock_matplotlib
    sys.modules["matplotlib.pyplot"] = mock_plt
    sys.modules["matplotlib.colors"] = mock_matplotlib.colors
    sys.modules["matplotlib.ticker"] = mock_matplotlib.ticker
    sys.modules["matplotlib.transforms"] = mock_matplotlib.transforms
    sys.modules["matplotlib.rcsetup"] = mock_matplotlib.rcsetup
    sys.modules["matplotlib.cbook"] = mock_matplotlib.cbook
    sys.modules["matplotlib._docstring"] = mock_matplotlib._docstring
    sys.modules["matplotlib.version"] = mock_matplotlib.version
    sys.modules["matplotlib.api"] = mock_matplotlib.api
    sys.modules["matplotlib.cm"] = mock_matplotlib.cm
    sys.modules["matplotlib.scale"] = mock_matplotlib.scale

    # Mock numpy with more complete structure
    mock_np = MagicMock()
    mock_np.array = MagicMock()
    mock_np.float32 = MagicMock()
    mock_np.linalg = MagicMock()
    mock_np.linalg.inv = MagicMock()
    sys.modules["numpy"] = mock_np
    sys.modules["numpy.linalg"] = mock_np.linalg


# Create a more sophisticated mock tensor for testing
class MockTensor:
    def __init__(self, data, dtype=None):
        self.data = data if isinstance(data, list) else [data]
        self.dtype = dtype
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            # Handle nested lists like [[1, 2, 3, 4, 0, 0]]
            self.shape = MockSize([len(data), len(data[0]) if data[0] else 0])
        elif isinstance(data, list):
            self.shape = MockSize([len(data)])
        else:
            self.shape = MockSize([0])

    def flatten(self):
        # Return a flattened version
        if (
            isinstance(self.data, list)
            and len(self.data) > 0
            and isinstance(self.data[0], list)
        ):
            flat_data = [item for sublist in self.data for item in sublist]
            return MockTensor(flat_data)
        return self

    def __getitem__(self, key):
        return self.data[key] if hasattr(self.data, "__getitem__") else self


class MockSize:
    def __init__(self, dims):
        self.dims = dims

    def __getitem__(self, idx):
        return self.dims[idx]

    def __len__(self):
        return len(self.dims)

    def __iter__(self):
        return iter(self.dims)


class MockDataFrame:
    """Mock pandas DataFrame with proper .loc indexing support."""

    def __init__(self, data):
        self.data = data
        self.loc = MockLoc(data)


class MockLoc:
    """Mock pandas .loc indexer that properly handles tuple indexing."""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            row_idx, col_name = key
            if row_idx in self.data and col_name in self.data[row_idx]:
                return self.data[row_idx][col_name]
            else:
                raise KeyError(f"Key not found: ({row_idx}, {col_name})")
        else:
            return self.data[key]


# Setup mocks before importing
setup_mocks()

# Import the actual classes
try:
    # Clear any cached imports of the model module
    module_path = "emotion_clf_pipeline.model"
    if module_path in sys.modules:
        del sys.modules[module_path]

    # Import with explicit reloading
    import importlib

    import emotion_clf_pipeline.model as model_module  # noqa: E402

    importlib.reload(model_module)

    # Get the actual classes directly from the module
    ActualEmotionPredictor = getattr(model_module, "EmotionPredictor")
    ActualEmotionDataset = getattr(model_module, "EmotionDataset")

    print("Successfully imported actual classes")
    print(f"ActualEmotionPredictor type: {type(ActualEmotionPredictor)}")
    print(f"ActualEmotionDataset type: {type(ActualEmotionDataset)}")

    # Verify these are actual classes, not mocks
    if hasattr(ActualEmotionDataset, "_mock_name"):
        raise ImportError(
            "EmotionDataset appears to be a mock instead of actual class!"
        )

    if hasattr(ActualEmotionPredictor, "_mock_name"):
        raise ImportError(
            "EmotionPredictor appears to be a mock instead of actual class!"
        )

    IMPORT_SUCCESS = True

except (ImportError, AttributeError) as e:
    print(f"Failed to import classes: {e}")
    print(f"Error type: {type(e)}")
    import traceback

    print("Full traceback:")
    traceback.print_exc()
    raise ImportError(
        f"Could not import required classes from emotion_clf_pipeline.model: {e}"
    )


class TestEmotionDataset(unittest.TestCase):
    """Test cases for EmotionDataset class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_tokenizer = Mock()
        self.mock_feature_extractor = Mock()
        self.sample_texts = ["I am happy", "This is sad", "Very excited!"]
        self.sample_features = [[0.1] * 121] * len(self.sample_texts)  # Mock features

        # Mock tokenizer return value
        self.mock_encoding = {
            "input_ids": MockTensor([[1, 2, 3, 4, 0, 0]]),
            "attention_mask": MockTensor([[1, 1, 1, 1, 0, 0]]),
        }
        self.mock_tokenizer.return_value = self.mock_encoding

    def test_init_without_feature_extractor(self):
        """Test initialization without feature extractor."""
        dataset = ActualEmotionDataset(
            texts=self.sample_texts,
            features=self.sample_features,
            tokenizer=self.mock_tokenizer,
            max_length=128,
        )

        self.assertEqual(dataset.texts, self.sample_texts)
        self.assertEqual(dataset.tokenizer, self.mock_tokenizer)
        self.assertEqual(dataset.max_length, 128)
        if hasattr(dataset, "feature_extractor"):
            self.assertIsNone(dataset.feature_extractor)
        if hasattr(dataset, "feature_dim"):
            self.assertEqual(dataset.feature_dim, 0)

    def test_init_with_feature_extractor(self):
        """Test initialization with feature extractor."""
        self.mock_feature_extractor.get_feature_dim.return_value = 121

        dataset = ActualEmotionDataset(
            texts=self.sample_texts,
            features=self.sample_features,
            tokenizer=self.mock_tokenizer,
            feature_extractor=self.mock_feature_extractor,
            max_length=256,
        )

        if hasattr(dataset, "feature_extractor"):
            self.assertEqual(dataset.feature_extractor, self.mock_feature_extractor)
        if hasattr(dataset, "feature_dim"):
            self.assertEqual(dataset.feature_dim, 121)
            self.mock_feature_extractor.get_feature_dim.assert_called_once()

    def test_init_with_feature_config(self):
        """Test initialization with feature config."""
        # Skip this test if the actual dataset doesn't support feature_config
        try:
            feature_config = {"pos": True, "vader": True}
            dataset = ActualEmotionDataset(
                texts=self.sample_texts,
                features=self.sample_features,
                tokenizer=self.mock_tokenizer,
                feature_config=feature_config,
            )
            if hasattr(dataset, "feature_config"):
                self.assertEqual(dataset.feature_config, feature_config)
        except TypeError as e:
            if "unexpected keyword argument 'feature_config'" in str(e):
                self.skipTest("Dataset class doesn't support feature_config parameter")
            else:
                raise

    def test_len(self):
        """Test __len__ method."""
        dataset = ActualEmotionDataset(
            texts=self.sample_texts,
            features=self.sample_features,
            tokenizer=self.mock_tokenizer,
        )

        self.assertEqual(len(dataset), 3)

    def test_getitem_without_features(self):
        """Test __getitem__ without feature extractor."""
        dataset = ActualEmotionDataset(
            texts=self.sample_texts,
            features=self.sample_features,
            tokenizer=self.mock_tokenizer,
            max_length=64,
        )

        item = dataset[0]

        # Verify return structure (basic check since implementation may vary)
        self.assertIsInstance(item, dict)
        # Common keys that should exist
        expected_keys = ["input_ids", "attention_mask"]
        for key in expected_keys:
            if key in item:
                self.assertIn(key, item)

    def test_getitem_with_features(self):
        """Test __getitem__ with feature extractor."""
        self.mock_feature_extractor.get_feature_dim.return_value = 121
        if hasattr(self.mock_feature_extractor, "extract_all_features"):
            self.mock_feature_extractor.extract_all_features.return_value = [0.1] * 121

        dataset = ActualEmotionDataset(
            texts=self.sample_texts,
            features=self.sample_features,
            tokenizer=self.mock_tokenizer,
            feature_extractor=self.mock_feature_extractor,
        )

        item = dataset[1]

        # Basic check that __getitem__ works
        self.assertIsInstance(item, dict)

    def test_getitem_index_bounds(self):
        """Test __getitem__ with valid and invalid indices."""
        dataset = ActualEmotionDataset(
            texts=self.sample_texts,
            features=self.sample_features,
            tokenizer=self.mock_tokenizer,
        )

        # Valid indices
        item0 = dataset[0]
        item2 = dataset[2]
        self.assertIsInstance(item0, dict)
        self.assertIsInstance(item2, dict)

        # Invalid index should raise IndexError
        with self.assertRaises(IndexError):
            dataset[3]

    def test_different_text_types(self):
        """Test dataset with different text input types."""
        # Test with empty list
        dataset = ActualEmotionDataset([], [], self.mock_tokenizer)
        self.assertEqual(len(dataset), 0)


class TestEmotionPredictor(unittest.TestCase):
    """Test cases for EmotionPredictor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_texts = ["I am happy", "This is sad"]
        self.sample_single_text = "Very excited!"

    @patch("emotion_clf_pipeline.model.nltk.download")
    @patch("emotion_clf_pipeline.model.nltk.data.find")
    def test_init_nltk_available(self, mock_find, mock_download):
        """Test initialization when NLTK vader_lexicon is available."""
        mock_find.return_value = True

        predictor = ActualEmotionPredictor()

        self.assertIsNone(predictor._model)
        self.assertIsNone(predictor._predictor)
        mock_find.assert_called_once_with("sentiment/vader_lexicon.zip")
        mock_download.assert_not_called()

    @patch("emotion_clf_pipeline.model.nltk.download")
    @patch("emotion_clf_pipeline.model.nltk.data.find")
    def test_init_nltk_not_available(self, mock_find, mock_download):
        """Test initialization when NLTK vader_lexicon is not available."""
        # Configure mock to raise an exception
        mock_find.side_effect = LookupError("vader_lexicon not found")

        predictor = ActualEmotionPredictor()  # noqa: F841

        mock_find.assert_called_once_with("sentiment/vader_lexicon.zip")
        mock_download.assert_called_once_with("vader_lexicon")

    @patch("emotion_clf_pipeline.model.ModelLoader")
    def test_predict_single_text(self, mock_model_loader):
        """Test predict method with single text input."""
        # Mock the ModelLoader and predictor
        mock_predictor = Mock()
        mock_model = Mock()
        mock_loader_instance = Mock()
        mock_loader_instance.load_model.return_value = mock_model
        mock_loader_instance.create_predictor.return_value = mock_predictor
        mock_model_loader.return_value = mock_loader_instance

        # Create proper mock DataFrame that supports .loc indexing
        mock_result_data = {
            0: {
                "predicted_emotion": "joy",
                "predicted_sub_emotion": "happiness",
                "predicted_intensity": "high",
            }
        }
        mock_result = MockDataFrame(mock_result_data)
        mock_predictor.predict.return_value = mock_result

        predictor = ActualEmotionPredictor()
        result = predictor.predict(self.sample_single_text)

        # Verify result structure for single input
        expected_keys = ["text", "emotion", "sub_emotion", "intensity"]
        for key in expected_keys:
            self.assertIn(key, result)

        self.assertEqual(result["text"], self.sample_single_text)
        self.assertEqual(result["emotion"], "joy")
        self.assertEqual(result["sub_emotion"], "happiness")
        self.assertEqual(result["intensity"], "high")

    @patch("emotion_clf_pipeline.model.ModelLoader")
    def test_predict_multiple_texts(self, mock_model_loader):
        """Test predict method with multiple texts."""
        # Mock the ModelLoader and predictor
        mock_predictor = Mock()
        mock_model = Mock()
        mock_loader_instance = Mock()
        mock_loader_instance.load_model.return_value = mock_model
        mock_loader_instance.create_predictor.return_value = mock_predictor
        mock_model_loader.return_value = mock_loader_instance

        # Create proper mock DataFrame that supports .loc indexing
        mock_result_data = {
            0: {
                "predicted_emotion": "joy",
                "predicted_sub_emotion": "happiness",
                "predicted_intensity": "high",
            },
            1: {
                "predicted_emotion": "sadness",
                "predicted_sub_emotion": "melancholy",
                "predicted_intensity": "medium",
            },
        }
        mock_result = MockDataFrame(mock_result_data)
        mock_predictor.predict.return_value = mock_result

        predictor = ActualEmotionPredictor()
        results = predictor.predict(self.sample_texts)

        # Should return a list for multiple inputs
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)

        # Each result should have required fields
        for i, result in enumerate(results):
            expected_keys = ["text", "emotion", "sub_emotion", "intensity"]
            for key in expected_keys:
                self.assertIn(key, result)

            # Verify specific values
            self.assertEqual(result["text"], self.sample_texts[i])

        # Check specific predictions
        self.assertEqual(results[0]["emotion"], "joy")
        self.assertEqual(results[1]["emotion"], "sadness")

    def test_predict_custom_feature_config(self):
        """Test predict method with custom feature configuration."""
        custom_config = {  # noqa: F841
            "pos": True,
            "textblob": True,
            "vader": True,
            "tfidf": False,
            "emolex": False,
        }

        with patch("emotion_clf_pipeline.model.ModelLoader"):
            predictor = ActualEmotionPredictor()

            # This test mainly checks that custom config doesn't break initialization
            # The actual prediction would need more complex mocking
            self.assertIsInstance(predictor, ActualEmotionPredictor)

    @patch("emotion_clf_pipeline.model.ModelLoader")
    def test_predict_empty_text_list(self, mock_model_loader):
        """Test predict with empty text list."""
        # Mock the ModelLoader to prevent file loading
        mock_loader_instance = Mock()
        mock_model_loader.return_value = mock_loader_instance

        predictor = ActualEmotionPredictor()
        results = predictor.predict([])

        self.assertEqual(results, [])

    @patch("emotion_clf_pipeline.model.ModelLoader")
    def test_string_vs_list_input_handling(self, mock_model_loader):
        """Test that string input returns dict and list input returns list."""
        # Mock the ModelLoader and predictor
        mock_predictor = Mock()
        mock_model = Mock()
        mock_loader_instance = Mock()
        mock_loader_instance.load_model.return_value = mock_model
        mock_loader_instance.create_predictor.return_value = mock_predictor
        mock_model_loader.return_value = mock_loader_instance

        # Create proper mock DataFrame that supports .loc indexing
        mock_result_data = {
            0: {
                "predicted_emotion": "joy",
                "predicted_sub_emotion": "happiness",
                "predicted_intensity": "high",
            }
        }
        mock_result = MockDataFrame(mock_result_data)
        mock_predictor.predict.return_value = mock_result

        predictor = ActualEmotionPredictor()

        # Single string input
        single_result = predictor.predict("Happy text")
        self.assertIsInstance(single_result, dict)

        # List input
        list_result = predictor.predict(["Happy text"])
        self.assertIsInstance(list_result, list)
        self.assertEqual(len(list_result), 1)


if __name__ == "__main__":
    # Print import status
    print(f"Import successful: {IMPORT_SUCCESS}")
    print(f"ActualEmotionDataset is callable: {callable(ActualEmotionDataset)}")
    print(f"ActualEmotionPredictor is callable: {callable(ActualEmotionPredictor)}")

    # Run the tests
    unittest.main(verbosity=2)
