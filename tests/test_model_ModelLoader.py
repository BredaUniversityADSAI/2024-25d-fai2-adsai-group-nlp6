import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, mock_open, patch

# Source directory to Python path
current_test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_test_dir)
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

# Mock dependencies
# Mock PyTorch with proper nested structure
mock_torch = Mock()
mock_torch.device = Mock()
mock_torch.cuda = Mock()
mock_torch.cuda.is_available = Mock(return_value=False)
mock_torch.load = Mock()
mock_torch.nn = Mock()

# Properly mock torch.utils and its submodules
mock_torch_utils = Mock()
mock_torch_utils.data = Mock()
mock_torch_utils.data.DataLoader = Mock()
mock_torch_utils.data.Dataset = Mock()
mock_torch.utils = mock_torch_utils

sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = mock_torch.nn
sys.modules["torch.utils"] = mock_torch_utils
sys.modules["torch.utils.data"] = mock_torch_utils.data

# Mock transformers
mock_transformers = Mock()
mock_deberta_tokenizer = Mock()
mock_deberta_tokenizer.from_pretrained = Mock()
mock_transformers.DebertaV2Tokenizer = mock_deberta_tokenizer
sys.modules["transformers"] = mock_transformers

# Mock NLTK with more complete structure
mock_nltk = Mock()
mock_nltk.corpus = Mock()
mock_nltk.corpus.stopwords = Mock()
mock_nltk.corpus.stopwords.words = Mock(return_value=[])
mock_nltk.tokenize = Mock()
mock_nltk.tokenize.word_tokenize = Mock()
mock_nltk.stem = Mock()
mock_nltk.stem.PorterStemmer = Mock()
mock_nltk.sentiment = Mock()
mock_nltk.sentiment.vader = Mock()
mock_nltk.sentiment.vader.SentimentIntensityAnalyzer = Mock()

# Mock nltk.data.path as an iterable
mock_nltk.data = Mock()
mock_nltk.data.path = ["/fake/nltk/data/path"]  # Make it iterable

sys.modules["nltk"] = mock_nltk
sys.modules["nltk.corpus"] = mock_nltk.corpus
sys.modules["nltk.tokenize"] = mock_nltk.tokenize
sys.modules["nltk.stem"] = mock_nltk.stem
sys.modules["nltk.sentiment"] = mock_nltk.sentiment
sys.modules["nltk.sentiment.vader"] = mock_nltk.sentiment.vader
sys.modules["nltk.data"] = mock_nltk.data

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

# Mock TextBlob with proper structure
mock_textblob = Mock()
mock_textblob_instance = Mock()
mock_textblob_instance.sentiment = Mock()
mock_textblob_instance.sentiment.polarity = 0.0
mock_textblob_instance.sentiment.subjectivity = 0.0
mock_textblob.TextBlob = Mock(return_value=mock_textblob_instance)
sys.modules["textblob"] = mock_textblob

# Mock other potential dependencies
sys.modules["pandas"] = Mock()
sys.modules["numpy"] = Mock()
sys.modules["sklearn"] = Mock()
sys.modules["sklearn.preprocessing"] = Mock()
sys.modules["sklearn.feature_extraction"] = Mock()
sys.modules["sklearn.feature_extraction.text"] = Mock()
sys.modules["sklearn.model_selection"] = Mock()
sys.modules["sklearn.utils"] = Mock()
sys.modules["pickle"] = Mock()
sys.modules["joblib"] = Mock()
sys.modules["matplotlib"] = Mock()
sys.modules["matplotlib.pyplot"] = Mock()
sys.modules["seaborn"] = Mock()

from emotion_clf_pipeline.model import ModelLoader  # noqa: E402


class TestModelLoader(unittest.TestCase):
    """Test cases for the ModelLoader class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.model_name = "microsoft/deberta-v3-xsmall"
        self.feature_dim = 10
        self.num_classes = {"emotion": 6, "sub_emotion": 25, "intensity": 3}
        self.weights_dir = "models/weights"

        # Reset mocks
        mock_torch.load.reset_mock()
        mock_deberta_tokenizer.from_pretrained.reset_mock()

    def test_init_default(self):
        """Test initialization with default parameters."""
        loader = ModelLoader()

        self.assertEqual(loader.model_name, self.model_name)
        self.assertIsNotNone(loader.device)
        self.assertIsNotNone(loader.tokenizer)
        mock_deberta_tokenizer.from_pretrained.assert_called_with(self.model_name)

    def test_init_custom_model(self):
        """Test initialization with custom model name."""
        custom_model = "microsoft/deberta-v3-base"
        loader = ModelLoader(model_name=custom_model)

        self.assertEqual(loader.model_name, custom_model)
        mock_deberta_tokenizer.from_pretrained.assert_called_with(custom_model)

    def test_load_model_basic(self):
        """Test basic model loading without weights."""
        loader = ModelLoader()

        with patch("emotion_clf_pipeline.model.DEBERTAClassifier") as mock_classifier:
            mock_model = Mock()
            mock_model.to = Mock(return_value=mock_model)
            mock_classifier.return_value = mock_model

            model = loader.load_model(self.feature_dim, self.num_classes)

            mock_classifier.assert_called_once()
            mock_model.to.assert_called_with(loader.device)
            self.assertEqual(model, mock_model)

    @patch("builtins.open", new_callable=mock_open, read_data=b"fake_data")
    def test_load_model_with_weights(self, mock_file):
        """Test model loading with weights file."""
        loader = ModelLoader()
        weights_path = "/fake/path/model.pth"

        with patch("emotion_clf_pipeline.model.DEBERTAClassifier") as mock_classifier:
            mock_model = Mock()
            mock_model.to = Mock(return_value=mock_model)
            mock_model.load_state_dict = Mock()
            mock_classifier.return_value = mock_model

            # Mock state dict with bert keys that need correction
            mock_state_dict = {
                "bert.embeddings.weight": Mock(),
                "emotion_classifier.weight": Mock(),
            }
            mock_torch.load.return_value = mock_state_dict

            model = loader.load_model(
                self.feature_dim, self.num_classes, weights_path
            )  # noqa: F841

            mock_file.assert_called_once_with(weights_path, "rb")
            mock_torch.load.assert_called_once()
            mock_model.load_state_dict.assert_called_once()

            # Verify bert keys were corrected to deberta keys
            corrected_state_dict = mock_model.load_state_dict.call_args[0][0]
            self.assertIn("deberta.embeddings.weight", corrected_state_dict)
            self.assertNotIn("bert.embeddings.weight", corrected_state_dict)

    def test_load_model_file_not_found(self):
        """Test handling of missing weight files."""
        loader = ModelLoader()

        with patch("emotion_clf_pipeline.model.DEBERTAClassifier") as mock_classifier:
            mock_model = Mock()
            mock_classifier.return_value = mock_model

            with self.assertRaises(FileNotFoundError):
                loader.load_model(
                    self.feature_dim, self.num_classes, "/nonexistent/path.pth"
                )

    def test_create_predictor(self):
        """Test creating a predictor instance."""
        loader = ModelLoader()

        with patch(
            "emotion_clf_pipeline.model.CustomPredictor"
        ) as mock_predictor_class:
            mock_predictor = Mock()
            mock_predictor_class.return_value = mock_predictor
            mock_model = Mock()

            predictor = loader.create_predictor(mock_model)

            mock_predictor_class.assert_called_once_with(
                model=mock_model,
                tokenizer=loader.tokenizer,
                device=loader.device,
                encoders_dir="models/encoders",
                feature_config=None,
            )
            self.assertEqual(predictor, mock_predictor)

    @patch("os.path.join")
    @patch("torch.load")
    def test_load_baseline_model_success(self, mock_torch_load, mock_path_join):
        """Test successful baseline model loading."""
        loader = ModelLoader()
        loader.model = Mock()  # Add model attribute
        loader.model.load_state_dict = Mock()
        loader.model.to = Mock()
        loader.model.eval = Mock()

        baseline_path = "models/weights/baseline_weights.pt"
        mock_path_join.return_value = baseline_path

        with patch("emotion_clf_pipeline.model.AzureMLSync") as mock_azure:
            mock_manager = Mock()
            mock_azure.return_value = mock_manager

            loader.load_baseline_model(sync_azure=True)

            mock_azure.assert_called_once_with(self.weights_dir)
            mock_manager.auto_sync_on_startup.assert_called_once_with(
                check_for_updates=True
            )
            mock_torch_load.assert_called_once_with(
                baseline_path, map_location=loader.device
            )
            loader.model.to.assert_called_once_with(loader.device)
            loader.model.eval.assert_called_once()

    @patch("os.path.join")
    def test_load_baseline_model_file_not_found(self, mock_path_join):
        """Test baseline model loading with missing file."""
        loader = ModelLoader()
        loader.model = Mock()

        baseline_path = "models/weights/baseline_weights.pt"
        mock_path_join.return_value = baseline_path

        with patch("torch.load", side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError):
                loader.load_baseline_model(sync_azure=False)

    @patch("os.path.join")
    @patch("torch.load")
    def test_load_dynamic_model_success(self, mock_torch_load, mock_path_join):
        """Test successful dynamic model loading."""
        loader = ModelLoader()
        loader.model = Mock()
        loader.model.load_state_dict = Mock()
        loader.model.to = Mock()
        loader.model.eval = Mock()

        dynamic_path = "models/weights/dynamic_weights.pt"
        mock_path_join.return_value = dynamic_path

        with patch("emotion_clf_pipeline.model.AzureMLSync") as mock_azure:
            mock_manager = Mock()
            mock_azure.return_value = mock_manager

            loader.load_dynamic_model(sync_azure=True)

            mock_azure.assert_called_once_with(self.weights_dir)
            mock_manager.auto_sync_on_startup.assert_called_once_with(
                check_for_updates=True
            )
            mock_torch_load.assert_called_once_with(
                dynamic_path, map_location=loader.device
            )
            loader.model.to.assert_called_once_with(loader.device)
            loader.model.eval.assert_called_once()

    @patch("shutil.copy")
    @patch("os.path.join")
    def test_promote_dynamic_to_baseline_local(self, mock_path_join, mock_copy):
        """Test promoting dynamic to baseline locally."""
        loader = ModelLoader()

        dynamic_path = "models/weights/dynamic_weights.pt"
        baseline_path = "models/weights/baseline_weights.pt"
        mock_path_join.side_effect = [dynamic_path, baseline_path]

        with patch("emotion_clf_pipeline.model.AzureMLSync") as mock_azure:
            mock_manager = Mock()
            mock_manager.promote_dynamic_to_baseline.return_value = False
            mock_azure.return_value = mock_manager

            loader.promote_dynamic_to_baseline(sync_azure=True)

            mock_azure.assert_called_once_with(self.weights_dir)
            mock_manager.promote_dynamic_to_baseline.assert_called_once()
            mock_copy.assert_called_once_with(dynamic_path, baseline_path)

    @patch("shutil.copy")
    @patch("os.path.join")
    def test_promote_dynamic_to_baseline_azure_success(self, mock_path_join, mock_copy):
        """Test promoting dynamic to baseline via Azure ML."""
        loader = ModelLoader()

        with patch("emotion_clf_pipeline.model.AzureMLSync") as mock_azure:
            mock_manager = Mock()
            mock_manager.promote_dynamic_to_baseline.return_value = True
            mock_azure.return_value = mock_manager

            loader.promote_dynamic_to_baseline(sync_azure=True)

            mock_azure.assert_called_once_with(self.weights_dir)
            mock_manager.promote_dynamic_to_baseline.assert_called_once()
            # Should not fallback to local copy if Azure succeeds
            mock_copy.assert_not_called()

    @patch("os.path.exists")
    @patch("os.path.dirname")
    @patch("os.path.abspath")
    def test_ensure_best_baseline_model_success(
        self, mock_abspath, mock_dirname, mock_exists
    ):
        """Test ensuring best baseline model downloads better model."""
        loader = ModelLoader()
        loader._model = Mock()
        loader._predictor = Mock()

        # Mock path resolution
        mock_abspath.return_value = "/fake/path/to/file.py"
        mock_dirname.side_effect = ["/fake/path/to", "/fake/path", "/fake"]
        mock_exists.return_value = False  # Not in Docker

        with patch("emotion_clf_pipeline.model.AzureMLSync") as mock_azure:
            mock_manager = Mock()
            mock_manager.download_best_baseline_model.return_value = True
            mock_azure.return_value = mock_manager

            result = loader.ensure_best_baseline_model()

            self.assertTrue(result)
            self.assertIsNone(loader._model)
            self.assertIsNone(loader._predictor)
            mock_manager.download_best_baseline_model.assert_called_once()

    @patch("os.path.exists")
    @patch("os.path.dirname")
    @patch("os.path.abspath")
    def test_ensure_best_baseline_model_no_update(
        self, mock_abspath, mock_dirname, mock_exists
    ):
        """Test ensuring best baseline model when no better model exists."""
        loader = ModelLoader()

        # Mock path resolution
        mock_abspath.return_value = "/fake/path/to/file.py"
        mock_dirname.side_effect = ["/fake/path/to", "/fake/path", "/fake"]
        mock_exists.return_value = False

        with patch("emotion_clf_pipeline.model.AzureMLSync") as mock_azure:
            mock_manager = Mock()
            mock_manager.download_best_baseline_model.return_value = False
            mock_azure.return_value = mock_manager

            result = loader.ensure_best_baseline_model()

            self.assertFalse(result)
            mock_manager.download_best_baseline_model.assert_called_once()

    @patch("os.path.exists")
    @patch("os.path.dirname")
    @patch("os.path.abspath")
    def test_ensure_best_baseline_model_docker_environment(
        self, mock_abspath, mock_dirname, mock_exists
    ):
        """Test ensuring best baseline model in Docker environment."""
        loader = ModelLoader()

        # Mock Docker environment detection
        mock_abspath.return_value = "/file.py"
        mock_dirname.side_effect = ["/", "/", "/"]
        mock_exists.return_value = True  # /app/models exists

        with patch("emotion_clf_pipeline.model.AzureMLSync") as mock_azure:
            mock_manager = Mock()
            mock_manager.download_best_baseline_model.return_value = False
            mock_azure.return_value = mock_manager

            result = loader.ensure_best_baseline_model()

            # Check that AzureMLSync was called (path format may vary by OS)
            mock_azure.assert_called_once()
            call_args = mock_azure.call_args
            weights_dir = (
                call_args[1]["weights_dir"] if call_args[1] else call_args[0][0]
            )
            # Verify the path ends with the expected structure
            # (handles both / and \ separators)
            self.assertTrue(
                weights_dir.endswith(os.path.join("app", "models", "weights"))
            )
            self.assertFalse(result)

    def test_ensure_best_baseline_model_exception_handling(self):
        """Test ensure_best_baseline_model handles exceptions gracefully."""
        loader = ModelLoader()

        with patch("os.path.abspath", side_effect=Exception("Test error")):
            result = loader.ensure_best_baseline_model()

            self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
