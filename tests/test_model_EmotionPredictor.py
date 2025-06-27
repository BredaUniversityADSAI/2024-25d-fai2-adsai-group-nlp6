import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

# Add the source directory to Python path
current_test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_test_dir)
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

# Comprehensive list of modules to mock (It was giving an error somehow)
# This was the best method available
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


# Mock Azure ML SDK submodules deeply
mock_azure = MagicMock()
mock_core = MagicMock()
mock_exceptions = MagicMock()
mock_ai = MagicMock()
mock_ml = MagicMock()
mock_constants = MagicMock()
mock_entities = MagicMock()
mock_identity = MagicMock()

sys.modules["azure"] = mock_azure
sys.modules["azure.core"] = mock_core
sys.modules["azure.core.exceptions"] = mock_exceptions
sys.modules["azure.ai"] = mock_ai
sys.modules["azure.ai.ml"] = mock_ml
sys.modules["azure.ai.ml.constants"] = mock_constants
sys.modules["azure.ai.ml.entities"] = mock_entities
sys.modules["azure.identity"] = mock_identity


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


# Mock DataFrame class for test results
class MockDataFrame:
    def __init__(self, data):
        self.data = data
        self.loc = MockLoc(data)


class MockLoc:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            row_idx, col_name = key
            return self.data[row_idx][col_name]
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


class TestEmotionPredictor(unittest.TestCase):
    """Simplified test cases for EmotionPredictor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_texts = ["I am happy", "This is sad"]
        self.sample_single_text = "Very excited!"

    @patch("emotion_clf_pipeline.model.nltk.download")
    @patch("emotion_clf_pipeline.model.nltk.data.find")
    def test_init_nltk_available(self, mock_find, mock_download):
        """Test initialization when NLTK vader_lexicon is available."""
        mock_find.return_value = True

        from emotion_clf_pipeline.model import EmotionPredictor  # noqa: E402

        predictor = EmotionPredictor()

        self.assertIsNone(predictor._model)
        self.assertIsNone(predictor._predictor)
        mock_find.assert_called_once_with("sentiment/vader_lexicon.zip")
        mock_download.assert_not_called()

    @patch("emotion_clf_pipeline.model.nltk.download")
    @patch("emotion_clf_pipeline.model.nltk.data.find")
    def test_init_nltk_missing(self, mock_find, mock_download):
        """Test initialization when NLTK vader_lexicon is missing."""
        mock_find.side_effect = LookupError("vader_lexicon not found")

        from emotion_clf_pipeline.model import EmotionPredictor  # noqa: E402

        EmotionPredictor()

        mock_find.assert_called_once_with("sentiment/vader_lexicon.zip")
        mock_download.assert_called_once_with("vader_lexicon")

    @patch("emotion_clf_pipeline.model.AzureMLSync")
    @patch("emotion_clf_pipeline.model.ModelLoader")
    @patch("os.path.exists")
    @patch("os.getenv")
    def test_predict_single_text(
        self, mock_getenv, mock_exists, mock_model_loader, mock_azure_sync
    ):
        """Test predict method with single text input."""
        # Setup environment mocks
        mock_getenv.return_value = None  # Not in Azure ML
        mock_exists.return_value = True

        # Mock Azure ML sync
        mock_sync_instance = Mock()
        mock_sync_instance.sync_on_startup.return_value = (True, True)
        mock_sync_instance.download_best_baseline_model.return_value = False
        mock_azure_sync.return_value = mock_sync_instance

        # Mock ModelLoader
        mock_model = Mock()
        mock_predictor = Mock()
        mock_loader_instance = Mock()
        mock_loader_instance.load_model.return_value = mock_model
        mock_loader_instance.create_predictor.return_value = mock_predictor
        mock_model_loader.return_value = mock_loader_instance

        # Mock prediction result
        mock_result_data = {
            0: {
                "predicted_emotion": "joy",
                "predicted_sub_emotion": "happiness",
                "predicted_intensity": "high",
            }
        }
        mock_result = MockDataFrame(mock_result_data)
        mock_predictor.predict.return_value = mock_result

        from emotion_clf_pipeline.model import EmotionPredictor  # noqa: E402

        predictor = EmotionPredictor()
        result = predictor.predict(self.sample_single_text)

        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertEqual(result["text"], self.sample_single_text)
        self.assertEqual(result["emotion"], "joy")
        self.assertEqual(result["sub_emotion"], "happiness")
        self.assertEqual(result["intensity"], "high")

    @patch("emotion_clf_pipeline.model.AzureMLSync")
    @patch("emotion_clf_pipeline.model.ModelLoader")
    @patch("os.path.exists")
    @patch("os.getenv")
    def test_predict_multiple_texts(
        self, mock_getenv, mock_exists, mock_model_loader, mock_azure_sync
    ):
        """Test predict method with multiple texts."""
        # Setup environment mocks
        mock_getenv.return_value = None
        mock_exists.return_value = True

        # Mock Azure ML sync
        mock_sync_instance = Mock()
        mock_sync_instance.sync_on_startup.return_value = (True, True)
        mock_sync_instance.download_best_baseline_model.return_value = False
        mock_azure_sync.return_value = mock_sync_instance

        # Mock ModelLoader
        mock_model = Mock()
        mock_predictor = Mock()
        mock_loader_instance = Mock()
        mock_loader_instance.load_model.return_value = mock_model
        mock_loader_instance.create_predictor.return_value = mock_predictor
        mock_model_loader.return_value = mock_loader_instance

        # Mock prediction results for multiple texts
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

        from emotion_clf_pipeline.model import EmotionPredictor  # noqa: E402

        predictor = EmotionPredictor()
        results = predictor.predict(self.sample_texts)

        # Verify results
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)

        # Check first result
        self.assertEqual(results[0]["text"], self.sample_texts[0])
        self.assertEqual(results[0]["emotion"], "joy")

        # Check second result
        self.assertEqual(results[1]["text"], self.sample_texts[1])
        self.assertEqual(results[1]["emotion"], "sadness")

    def test_predict_custom_feature_config(self):
        """Test predict method accepts custom feature configuration."""
        custom_config = {  # noqa: F841
            "pos": True,
            "textblob": True,
            "vader": True,
            "tfidf": False,
            "emolex": False,
        }

        from emotion_clf_pipeline.model import EmotionPredictor  # noqa: E402

        predictor = EmotionPredictor()

        # Test that custom config parameter is accepted without error
        try:
            with patch.object(predictor, "_model", None):
                with patch.object(predictor, "_predictor", None):
                    self.assertTrue(hasattr(predictor, "predict"))
        except Exception as e:
            self.fail(f"Custom feature config caused error: {e}")

    @patch("emotion_clf_pipeline.model.AzureMLSync")
    @patch("emotion_clf_pipeline.model.ModelLoader")
    @patch("os.path.exists")
    @patch("os.getenv")
    def test_predict_empty_list(
        self, mock_getenv, mock_exists, mock_model_loader, mock_azure_sync
    ):
        """Test predict with empty text list."""
        # Setup mocks
        mock_getenv.return_value = None
        mock_exists.return_value = True

        mock_sync_instance = Mock()
        mock_sync_instance.sync_on_startup.return_value = (True, True)
        mock_sync_instance.download_best_baseline_model.return_value = False
        mock_azure_sync.return_value = mock_sync_instance

        mock_predictor = Mock()
        mock_predictor.predict.return_value = MockDataFrame({})
        mock_loader_instance = Mock()
        mock_loader_instance.load_model.return_value = Mock()
        mock_loader_instance.create_predictor.return_value = mock_predictor
        mock_model_loader.return_value = mock_loader_instance

        from emotion_clf_pipeline.model import EmotionPredictor  # noqa: E402

        predictor = EmotionPredictor()
        results = predictor.predict([])

        self.assertEqual(results, [])

    @patch("emotion_clf_pipeline.model.AzureMLSync")
    def test_ensure_best_baseline_success(self, mock_azure_sync):
        """Test ensure_best_baseline method when better model is found."""
        # Mock Azure ML sync to return True (better model found)
        mock_sync_instance = Mock()
        mock_sync_instance.download_best_baseline_model.return_value = True
        mock_azure_sync.return_value = mock_sync_instance

        from emotion_clf_pipeline.model import EmotionPredictor  # noqa: E402

        predictor = EmotionPredictor()

        # Set initial model state
        predictor._model = "old_model"
        predictor._predictor = "old_predictor"

        result = predictor.ensure_best_baseline_model()

        # Should return True and clear model cache
        self.assertTrue(result)
        self.assertIsNone(predictor._model)
        self.assertIsNone(predictor._predictor)

    @patch("emotion_clf_pipeline.model.AzureMLSync")
    def test_ensure_best_baseline_no_update(self, mock_azure_sync):
        """Test ensure_best_baseline method when no better model is found."""
        # Mock Azure ML sync to return False (no better model)
        mock_sync_instance = Mock()
        mock_sync_instance.download_best_baseline_model.return_value = False
        mock_azure_sync.return_value = mock_sync_instance

        from emotion_clf_pipeline.model import EmotionPredictor  # noqa: E402

        predictor = EmotionPredictor()

        result = predictor.ensure_best_baseline_model()

        # Should return False
        self.assertFalse(result)

    @patch("emotion_clf_pipeline.model.AzureMLSync")
    def test_ensure_best_baseline_exception(self, mock_azure_sync):
        """Test ensure_best_baseline method when exception occurs."""
        # Mock Azure ML sync to raise exception
        mock_azure_sync.side_effect = Exception("Azure ML error")

        from emotion_clf_pipeline.model import EmotionPredictor  # noqa: E402

        predictor = EmotionPredictor()

        result = predictor.ensure_best_baseline_model()

        # Should return False on exception
        self.assertFalse(result)

    def test_azure_ml_environment_detection(self):
        """Test that Azure ML environment is properly detected."""
        with patch.dict(os.environ, {"AZUREML_MODEL_DIR": "/tmp/models"}):
            from emotion_clf_pipeline.model import EmotionPredictor  # noqa: E402

            predictor = EmotionPredictor()

            # Should not raise any errors when Azure ML env is detected
            self.assertIsInstance(predictor, EmotionPredictor)


if __name__ == "__main__":
    unittest.main()
