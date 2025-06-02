import os
import sys
import unittest
from unittest.mock import Mock, mock_open, patch

# Source directory to Python path
current_test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_test_dir)
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

# Mock ALL external dependencies before any imports
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

# Important: Mock nltk.data.path as an iterable
mock_nltk.data = Mock()
mock_nltk.data.path = ["/fake/nltk/data/path"]  # Make it iterable

sys.modules["nltk"] = mock_nltk
sys.modules["nltk.corpus"] = mock_nltk.corpus
sys.modules["nltk.tokenize"] = mock_nltk.tokenize
sys.modules["nltk.stem"] = mock_nltk.stem
sys.modules["nltk.sentiment"] = mock_nltk.sentiment
sys.modules["nltk.sentiment.vader"] = mock_nltk.sentiment.vader
sys.modules["nltk.data"] = mock_nltk.data

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
        self.default_model_name = "microsoft/deberta-v3-xsmall"
        self.custom_model_name = "microsoft/deberta-v3-base"
        self.feature_dim = 10
        self.num_classes = {"emotion": 6, "sub_emotion": 25, "intensity": 3}

        # Reset mocks before each test
        mock_torch.load.reset_mock()
        mock_deberta_tokenizer.from_pretrained.reset_mock()

    def test_init_default_model(self):
        """Test initialization with default model name."""
        loader = ModelLoader()

        self.assertEqual(loader.model_name, self.default_model_name)
        self.assertIsNotNone(loader.device)
        self.assertIsNotNone(loader.tokenizer)

        # Verify tokenizer was loaded with correct model name
        mock_deberta_tokenizer.from_pretrained.assert_called_with(
            self.default_model_name
        )

    def test_init_custom_model(self):
        """Test initialization with custom model name."""
        loader = ModelLoader(model_name=self.custom_model_name)

        self.assertEqual(loader.model_name, self.custom_model_name)
        self.assertIsNotNone(loader.device)
        self.assertIsNotNone(loader.tokenizer)

    def test_init_with_device(self):
        """Test initialization with specific device."""
        mock_device = Mock()
        loader = ModelLoader(device=mock_device)

        self.assertEqual(loader.device, mock_device)
        self.assertEqual(loader.model_name, self.default_model_name)

    @patch("torch.cuda.is_available")
    def test_device_selection_cuda_available(self, mock_cuda_available):
        """Test device selection when CUDA is available."""
        mock_cuda_available.return_value = True
        mock_torch.cuda.is_available = mock_cuda_available

        loader = ModelLoader()  # noqa: F841

        # Should attempt to use CUDA device
        mock_torch.device.assert_called()

    @patch("torch.cuda.is_available")
    def test_device_selection_cuda_not_available(self, mock_cuda_available):
        """Test device selection when CUDA is not available."""
        mock_cuda_available.return_value = False
        mock_torch.cuda.is_available = mock_cuda_available

        loader = ModelLoader()  # noqa: F841

        # Should use CPU device
        mock_torch.device.assert_called()

    def test_load_model_basic(self):
        """Test basic model loading without weights."""
        loader = ModelLoader()

        # Mock the DEBERTAClassifier class directly in the loader's module
        with patch(
            "emotion_clf_pipeline.model.DEBERTAClassifier"
        ) as mock_classifier_class:
            # Setup mock model instance
            mock_model = Mock()
            mock_model.to = Mock(return_value=mock_model)
            mock_classifier_class.return_value = mock_model

            model = loader.load_model(
                feature_dim=self.feature_dim, num_classes=self.num_classes
            )

            # Verify model class was instantiated with correct parameters
            mock_classifier_class.assert_called_once_with(
                model_name=self.default_model_name,
                feature_dim=self.feature_dim,
                num_classes=self.num_classes,
                hidden_dim=256,
                dropout=0.1,
            )

            # Verify model was moved to device
            mock_model.to.assert_called_with(loader.device)
            self.assertEqual(model, mock_model)

    def test_load_model_with_custom_params(self):
        """Test model loading with custom parameters."""
        loader = ModelLoader()

        with patch(
            "emotion_clf_pipeline.model.DEBERTAClassifier"
        ) as mock_classifier_class:
            mock_model = Mock()
            mock_model.to = Mock(return_value=mock_model)
            mock_classifier_class.return_value = mock_model

            hidden_dim = 512
            dropout = 0.2

            model = loader.load_model(  # noqa: F841
                feature_dim=self.feature_dim,
                num_classes=self.num_classes,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )

            # Verify model was created with custom parameters
            mock_classifier_class.assert_called_once_with(
                model_name=self.default_model_name,
                feature_dim=self.feature_dim,
                num_classes=self.num_classes,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )

    @patch("builtins.open", new_callable=mock_open, read_data=b"fake_model_data")
    def test_load_model_with_weights(self, mock_file):
        """Test model loading with weight file."""
        loader = ModelLoader()

        with patch(
            "emotion_clf_pipeline.model.DEBERTAClassifier"
        ) as mock_classifier_class:
            # Setup mock model instance
            mock_model = Mock()
            mock_model.to = Mock(return_value=mock_model)
            mock_model.load_state_dict = Mock()
            mock_classifier_class.return_value = mock_model

            weights_path = "/fake/path/model.pth"

            # Mock the state dict that would be loaded
            mock_state_dict = {
                "bert.embeddings.weight": Mock(),
                "bert.encoder.layer.0.weight": Mock(),
                "emotion_classifier.0.weight": Mock(),
            }
            mock_torch.load.return_value = mock_state_dict

            model = loader.load_model(  # noqa: F841
                feature_dim=self.feature_dim,
                num_classes=self.num_classes,
                weights_path=weights_path,
            )

            # Verify file was opened
            mock_file.assert_called_once_with(weights_path, "rb")

            # Verify torch.load was called
            mock_torch.load.assert_called_once()

            # Verify model state dict was loaded
            mock_model.load_state_dict.assert_called_once()

    @patch("builtins.open", new_callable=mock_open, read_data=b"fake_model_data")
    def test_load_model_with_bert_key_correction(self, mock_file):
        """Test that bert. keys are corrected to deberta. keys."""
        loader = ModelLoader()

        with patch(
            "emotion_clf_pipeline.model.DEBERTAClassifier"
        ) as mock_classifier_class:
            # Setup mock model instance
            mock_model = Mock()
            mock_model.to = Mock(return_value=mock_model)
            mock_model.load_state_dict = Mock()
            mock_classifier_class.return_value = mock_model

            weights_path = "/fake/path/model.pth"

            # Mock state dict with bert. keys that need correction
            original_state_dict = {
                "bert.embeddings.weight": Mock(),
                "bert.encoder.layer.0.weight": Mock(),
                "emotion_classifier.0.weight": Mock(),
                "sub_emotion_classifier.0.weight": Mock(),
            }
            mock_torch.load.return_value = original_state_dict

            model = loader.load_model(  # noqa: F841
                feature_dim=self.feature_dim,
                num_classes=self.num_classes,
                weights_path=weights_path,
            )

            # Verify model.load_state_dict was called
            mock_model.load_state_dict.assert_called_once()

            # Get the corrected state dict that was passed to load_state_dict
            call_args = mock_model.load_state_dict.call_args[0][0]

            # Verify that bert. keys were corrected to deberta. keys
            self.assertIn("deberta.embeddings.weight", call_args)
            self.assertIn("deberta.encoder.layer.0.weight", call_args)
            self.assertNotIn("bert.embeddings.weight", call_args)
            self.assertNotIn("bert.encoder.layer.0.weight", call_args)
            # Non-bert keys should remain unchanged
            self.assertIn("emotion_classifier.0.weight", call_args)
            self.assertIn("sub_emotion_classifier.0.weight", call_args)

    def test_create_predictor_basic(self):
        """Test creating a basic predictor."""
        loader = ModelLoader()

        with patch(
            "emotion_clf_pipeline.model.CustomPredictor"
        ) as mock_predictor_class:
            mock_predictor = Mock()
            mock_predictor_class.return_value = mock_predictor
            mock_model = Mock()

            predictor = loader.create_predictor(mock_model)

            # Verify CustomPredictor was instantiated with correct parameters
            mock_predictor_class.assert_called_once_with(
                model=mock_model,
                tokenizer=loader.tokenizer,
                device=loader.device,
                encoders_dir="/app/models/encoders",
                feature_config=None,
            )
            self.assertEqual(predictor, mock_predictor)

    def test_create_predictor_with_custom_params(self):
        """Test creating predictor with custom parameters."""
        loader = ModelLoader()

        with patch(
            "emotion_clf_pipeline.model.CustomPredictor"
        ) as mock_predictor_class:
            mock_predictor = Mock()
            mock_predictor_class.return_value = mock_predictor
            mock_model = Mock()
            custom_encoders_dir = "/custom/encoders/path"
            custom_feature_config = {"feature1": "config1"}

            predictor = loader.create_predictor(  # noqa: F841
                model=mock_model,
                encoders_dir=custom_encoders_dir,
                feature_config=custom_feature_config,
            )

            # Verify CustomPredictor was instantiated with custom parameters
            mock_predictor_class.assert_called_once_with(
                model=mock_model,
                tokenizer=loader.tokenizer,
                device=loader.device,
                encoders_dir=custom_encoders_dir,
                feature_config=custom_feature_config,
            )

    def test_full_workflow(self):
        """Test the complete workflow: init -> load_model -> create_predictor."""
        loader = ModelLoader(model_name=self.custom_model_name)

        with (
            patch(
                "emotion_clf_pipeline.model.DEBERTAClassifier"
            ) as mock_classifier_class,
            patch("emotion_clf_pipeline.model.CustomPredictor") as mock_predictor_class,
        ):

            # Setup mocks
            mock_model = Mock()
            mock_model.to = Mock(return_value=mock_model)
            mock_classifier_class.return_value = mock_model

            mock_predictor = Mock()
            mock_predictor_class.return_value = mock_predictor

            # Load model
            model = loader.load_model(
                feature_dim=self.feature_dim,
                num_classes=self.num_classes,
                hidden_dim=512,
                dropout=0.1,
            )

            # Create predictor
            predictor = loader.create_predictor(
                model=model,
                encoders_dir="/custom/path",
                feature_config={"test": "config"},
            )

            # Verify the workflow
            self.assertEqual(loader.model_name, self.custom_model_name)
            self.assertEqual(model, mock_model)
            self.assertEqual(predictor, mock_predictor)

            # Verify model was created correctly
            mock_classifier_class.assert_called_once()
            # Verify predictor was created correctly
            mock_predictor_class.assert_called_once()

    def test_multiple_model_loading(self):
        """Test loading multiple models with the same loader."""
        loader = ModelLoader()

        with patch(
            "emotion_clf_pipeline.model.DEBERTAClassifier"
        ) as mock_classifier_class:
            # Setup mock models
            mock_model1 = Mock()
            mock_model1.to = Mock(return_value=mock_model1)
            mock_model2 = Mock()
            mock_model2.to = Mock(return_value=mock_model2)
            mock_classifier_class.side_effect = [mock_model1, mock_model2]

            # Load first model
            model1 = loader.load_model(
                feature_dim=10,
                num_classes={"emotion": 5, "sub_emotion": 20, "intensity": 3},
            )

            # Load second model with different parameters
            model2 = loader.load_model(
                feature_dim=15,
                num_classes={"emotion": 7, "sub_emotion": 25, "intensity": 4},
                hidden_dim=512,
            )

            # Both models should be loaded successfully
            self.assertEqual(model1, mock_model1)
            self.assertEqual(model2, mock_model2)
            self.assertEqual(mock_classifier_class.call_count, 2)

    def test_load_model_file_not_found(self):
        """Test handling of missing weight files."""
        loader = ModelLoader()

        with patch(
            "emotion_clf_pipeline.model.DEBERTAClassifier"
        ) as mock_classifier_class:
            mock_model = Mock()
            mock_model.to = Mock(return_value=mock_model)
            mock_classifier_class.return_value = mock_model

            with self.assertRaises(FileNotFoundError):
                loader.load_model(
                    feature_dim=self.feature_dim,
                    num_classes=self.num_classes,
                    weights_path="/nonexistent/path.pth",
                )

    def test_tokenizer_consistency(self):
        """Test that tokenizer is consistent across operations."""
        loader = ModelLoader()

        # Get tokenizer reference
        tokenizer1 = loader.tokenizer

        # Tokenizer should be the same object
        self.assertEqual(tokenizer1, loader.tokenizer)

    def test_device_consistency(self):
        """Test that device is consistent across operations."""
        mock_device = Mock()
        loader = ModelLoader(device=mock_device)

        # Device should be consistent
        self.assertEqual(loader.device, mock_device)


if __name__ == "__main__":
    unittest.main()
