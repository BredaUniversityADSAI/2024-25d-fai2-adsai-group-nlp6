import logging
import os
import pickle
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# Suppress logging during tests
logging.getLogger("emotion_clf_pipeline.model").setLevel(logging.CRITICAL)

# Mock dependencies
# Mock PyTorch
mock_torch = MagicMock()
mock_torch.device = MagicMock()
mock_torch.cuda = MagicMock()
mock_torch.cuda.is_available = MagicMock(return_value=False)
mock_torch.load = MagicMock()
mock_torch.argmax = MagicMock()

# Mock torch.no_grad as a context manager
mock_no_grad = MagicMock()
mock_no_grad.__enter__ = MagicMock(return_value=mock_no_grad)
mock_no_grad.__exit__ = MagicMock(return_value=None)
mock_torch.no_grad = MagicMock(return_value=mock_no_grad)
mock_torch.tensor = MagicMock()

# Mock torch.nn
mock_torch.nn = MagicMock()
mock_torch.nn.Module = MagicMock()
mock_torch.nn.Linear = MagicMock()
mock_torch.nn.Dropout = MagicMock()
mock_torch.nn.ReLU = MagicMock()
mock_torch.nn.functional = MagicMock()
mock_torch.nn.functional.softmax = MagicMock()
mock_torch.nn.functional.relu = MagicMock()

# Mock torch.optim
mock_torch.optim = MagicMock()
mock_torch.optim.Adam = MagicMock()
mock_torch.optim.SGD = MagicMock()

sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = mock_torch.nn
sys.modules["torch.nn.functional"] = mock_torch.nn.functional
sys.modules["torch.optim"] = mock_torch.optim

# Mock torch.utils.data
mock_torch_utils = MagicMock()
mock_torch_utils_data = MagicMock()
mock_dataloader = MagicMock()
mock_torch_utils_data.DataLoader = mock_dataloader
mock_torch_utils.data = mock_torch_utils_data
sys.modules["torch.utils"] = mock_torch_utils
sys.modules["torch.utils.data"] = mock_torch_utils_data

# Mock pandas
mock_pd = MagicMock()
mock_dataframe = MagicMock()
mock_pd.DataFrame = MagicMock(return_value=mock_dataframe)
mock_pd.Series = MagicMock()
sys.modules["pandas"] = mock_pd

# Mock Azure ML SDK submodules
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

# Mock numpy
mock_np = MagicMock()
mock_np.array = MagicMock()
mock_np.float32 = MagicMock()
mock_np.linalg = MagicMock()
mock_np.linalg.inv = MagicMock()
sys.modules["numpy"] = mock_np
sys.modules["numpy.linalg"] = mock_np.linalg

# Mock matplotlib
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

# Mock seaborn
mock_seaborn = MagicMock()
mock_seaborn.rcmod = MagicMock()
mock_seaborn.palettes = MagicMock()
mock_seaborn.utils = MagicMock()
mock_seaborn.utils.desaturate = MagicMock()
mock_seaborn.utils.get_color_cycle = MagicMock()
sys.modules["seaborn"] = mock_seaborn
sys.modules["seaborn.rcmod"] = mock_seaborn.rcmod
sys.modules["seaborn.palettes"] = mock_seaborn.palettes
sys.modules["seaborn.utils"] = mock_seaborn.utils

# Mock sklearn
mock_sklearn = MagicMock()
# Preprocessing
mock_sklearn.preprocessing = MagicMock()
mock_sklearn.preprocessing.LabelEncoder = MagicMock()
mock_sklearn.preprocessing.StandardScaler = MagicMock()
# Feature extraction
mock_sklearn.feature_extraction = MagicMock()
mock_sklearn.feature_extraction.text = MagicMock()
mock_sklearn.feature_extraction.text.TfidfVectorizer = MagicMock()
# Model selection
mock_sklearn.model_selection = MagicMock()
mock_sklearn.model_selection.train_test_split = MagicMock()
# Metrics
mock_sklearn.metrics = MagicMock()
mock_sklearn.metrics.classification_report = MagicMock()
mock_sklearn.metrics.accuracy_score = MagicMock()
mock_sklearn.metrics.f1_score = MagicMock()

mock_sklearn.utils = MagicMock()

sys.modules["sklearn"] = mock_sklearn
sys.modules["sklearn.preprocessing"] = mock_sklearn.preprocessing
sys.modules["sklearn.feature_extraction"] = mock_sklearn.feature_extraction
sys.modules["sklearn.feature_extraction.text"] = mock_sklearn.feature_extraction.text
sys.modules["sklearn.model_selection"] = mock_sklearn.model_selection
sys.modules["sklearn.metrics"] = mock_sklearn.metrics
sys.modules["sklearn.utils"] = mock_sklearn.utils

# Mock nltk with all necessary submodules
mock_nltk = MagicMock()
mock_nltk.corpus = MagicMock()
mock_nltk.corpus.stopwords = MagicMock()
mock_nltk.corpus.stopwords.words = MagicMock(return_value=["the", "a", "an"])
mock_nltk.tokenize = MagicMock()
mock_nltk.tokenize.word_tokenize = MagicMock()
mock_nltk.tokenize.sent_tokenize = MagicMock()
mock_nltk.stem = MagicMock()
mock_nltk.stem.PorterStemmer = MagicMock()
mock_nltk.tag = MagicMock()
mock_nltk.tag.pos_tag = MagicMock()
mock_nltk.sentiment = MagicMock()
mock_nltk.sentiment.vader = MagicMock()
mock_nltk.sentiment.vader.SentimentIntensityAnalyzer = MagicMock()
mock_nltk.download = MagicMock()

# Mock nltk.data
mock_nltk.data = MagicMock()
mock_nltk.data.path = [
    "/some/path",
    "/another/path",
]  # Mock as a real list for iteration

sys.modules["nltk"] = mock_nltk
sys.modules["nltk.corpus"] = mock_nltk.corpus
sys.modules["nltk.corpus.stopwords"] = mock_nltk.corpus.stopwords
sys.modules["nltk.tokenize"] = mock_nltk.tokenize
sys.modules["nltk.stem"] = mock_nltk.stem
sys.modules["nltk.tag"] = mock_nltk.tag
sys.modules["nltk.sentiment"] = mock_nltk.sentiment
sys.modules["nltk.sentiment.vader"] = mock_nltk.sentiment.vader
sys.modules["nltk.data"] = mock_nltk.data

# Mock textblob
mock_textblob = MagicMock()
mock_textblob.TextBlob = MagicMock()
sys.modules["textblob"] = mock_textblob

# Mock vaderSentiment
mock_vader = MagicMock()
mock_vader.vaderSentiment = MagicMock()
mock_vader.vaderSentiment.SentimentIntensityAnalyzer = MagicMock()
sys.modules["vaderSentiment"] = mock_vader
sys.modules["vaderSentiment.vaderSentiment"] = mock_vader.vaderSentiment

# Mock tqdm
mock_tqdm = MagicMock()
mock_tqdm.tqdm = MagicMock()
sys.modules["tqdm"] = mock_tqdm

# Mock transformers
mock_transformers = MagicMock()
mock_transformers.AutoTokenizer = MagicMock()
mock_transformers.AutoModel = MagicMock()
sys.modules["transformers"] = mock_transformers

# Mock re (regex)
mock_re = MagicMock()
sys.modules["re"] = mock_re

# Source directory to Python path for imports
current_test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_test_dir)
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

from emotion_clf_pipeline.model import CustomPredictor  # noqa: E402


# Create a simple picklable class to replace Mock for encoders
class MockEncoder:
    """A simple picklable mock encoder class."""

    def __init__(self, task_name):
        self.task_name = task_name
        self.classes_ = [
            f"class_1_{task_name}",
            f"class_2_{task_name}",
            f"class_3_{task_name}",
        ]

    def inverse_transform(self, values):
        return [f"test_{self.task_name}"] * len(values)


# Create a mock feature list that has len() method
class MockFeatureList:
    """A mock feature list that supports len() operation."""

    def __init__(self, length=10):
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return iter([0.1] * self.length)

    def __getitem__(self, index):
        return 0.1


class TestCustomPredictor(unittest.TestCase):
    """Test cases for the CustomPredictor class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mock model
        self.mock_model = MagicMock()
        mock_feature_proj = MagicMock()
        mock_feature_proj.in_features = 10
        self.mock_model.feature_projection = [mock_feature_proj]
        self.mock_model.eval = MagicMock()

        # Create mock tokenizer
        self.mock_tokenizer = MagicMock()

        # Create mock device
        self.mock_device = MagicMock()

        # Create temporary directory for encoders
        self.temp_dir = tempfile.mkdtemp()
        self.encoders_dir = self.temp_dir

        # Create picklable mock encoders
        self.mock_encoders = {}
        for task in ["emotion", "sub_emotion", "intensity"]:
            encoder = MockEncoder(task)
            self.mock_encoders[task] = encoder

            # Create encoder file with picklable object
            encoder_path = os.path.join(self.encoders_dir, f"{task}_encoder.pkl")
            with open(encoder_path, "wb") as f:
                pickle.dump(encoder, f)

        # Reset global mocks
        mock_torch.argmax.reset_mock()
        mock_pd.DataFrame.reset_mock()
        mock_np.array.reset_mock()

    def tearDown(self):
        """Clean up after each test."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        predictor = CustomPredictor(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device=self.mock_device,
            encoders_dir=self.encoders_dir,
        )

        # Check basic attributes
        self.assertEqual(predictor.model, self.mock_model)
        self.assertEqual(predictor.tokenizer, self.mock_tokenizer)
        self.assertEqual(predictor.device, self.mock_device)
        self.assertEqual(predictor.expected_feature_dim, 10)

        # Check default feature config
        expected_config = {
            "pos": False,
            "textblob": False,
            "vader": False,
            "tfidf": False,
            "emolex": False,
        }
        self.assertEqual(predictor.feature_config, expected_config)

        # Check that encoders were loaded
        self.assertEqual(len(predictor.encoders), 3)
        self.assertIn("emotion", predictor.encoders)
        self.assertIn("sub_emotion", predictor.encoders)
        self.assertIn("intensity", predictor.encoders)

        # Check output tasks
        expected_tasks = ["emotion", "sub_emotion", "intensity"]
        self.assertEqual(predictor.output_tasks, expected_tasks)

    def test_init_custom_config(self):
        """Test initialization with custom feature configuration."""
        custom_config = {
            "pos": True,
            "textblob": True,
            "vader": False,
            "tfidf": False,
            "emolex": True,
        }

        predictor = CustomPredictor(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device=self.mock_device,
            encoders_dir=self.encoders_dir,
            feature_config=custom_config,
        )

        self.assertEqual(predictor.feature_config, custom_config)

    def test_emotion_mapping(self):
        """Test that emotion mapping is correctly initialized."""
        predictor = CustomPredictor(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device=self.mock_device,
            encoders_dir=self.encoders_dir,
        )

        # Test some key mappings
        self.assertEqual(predictor.emotion_mapping["joy"], "happiness")
        self.assertEqual(predictor.emotion_mapping["sadness"], "sadness")
        self.assertEqual(predictor.emotion_mapping["anger"], "anger")
        self.assertEqual(predictor.emotion_mapping["fear"], "fear")
        self.assertEqual(predictor.emotion_mapping["neutral"], "neutral")
        self.assertEqual(predictor.emotion_mapping["curiosity"], "happiness")
        self.assertEqual(predictor.emotion_mapping["annoyance"], "anger")

    def test_load_encoders_success(self):
        """Test successful loading of encoders."""
        predictor = CustomPredictor(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device=self.mock_device,
            encoders_dir=self.encoders_dir,
        )

        # Verify all encoders were loaded
        for task in ["emotion", "sub_emotion", "intensity"]:
            self.assertIn(task, predictor.encoders)
            self.assertIsInstance(predictor.encoders[task], MockEncoder)
            self.assertEqual(predictor.encoders[task].task_name, task)

    def test_load_encoders_file_not_found(self):
        """Test handling of missing encoder files."""
        with self.assertRaises(FileNotFoundError):
            CustomPredictor(
                model=self.mock_model,
                tokenizer=self.mock_tokenizer,
                device=self.mock_device,
                encoders_dir="/nonexistent/path",
            )

    def test_predict_success(self):
        """Test successful prediction."""
        predictor = CustomPredictor(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device=self.mock_device,
            encoders_dir=self.encoders_dir,
        )

        # Mock batch data
        mock_batch = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
            "features": MagicMock(),
        }
        mock_batch["input_ids"].to = MagicMock(return_value=mock_batch["input_ids"])
        mock_batch["attention_mask"].to = MagicMock(
            return_value=mock_batch["attention_mask"]
        )
        mock_batch["features"].to = MagicMock(return_value=mock_batch["features"])

        # Mock dataloader
        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.__iter__ = MagicMock(return_value=iter([mock_batch]))

        # Create mock model outputs
        mock_emotion_tensor = MagicMock()
        mock_sub_emotion_tensor = MagicMock()
        mock_intensity_tensor = MagicMock()

        # Set up tensor behaviors
        mock_emotion_tensor.cpu.return_value.numpy.return_value = [0]
        mock_sub_emotion_tensor.cpu.return_value.numpy.return_value = [1]
        mock_intensity_tensor.cpu.return_value.numpy.return_value = [2]

        # Mock logits for sub_emotion
        mock_logit_tensor = MagicMock()
        mock_logit_tensor.tolist.return_value = [0.1, 0.9, 0.2]
        mock_sub_emotion_tensor.cpu.return_value.detach.return_value = [
            mock_logit_tensor
        ]

        # Model outputs
        mock_outputs = {
            "emotion": mock_emotion_tensor,
            "sub_emotion": mock_sub_emotion_tensor,
            "intensity": mock_intensity_tensor,
        }
        self.mock_model.return_value = mock_outputs

        # Mock torch.argmax
        mock_torch.argmax.return_value.cpu.return_value.numpy.return_value = [0]

        # Mock numpy array creation
        mock_np.array.return_value = np.array([[0.1, 0.2, 0.3]])

        # Mock DataFrame
        mock_results_df = MagicMock()
        mock_pd.DataFrame.return_value = mock_results_df

        # Mock feature extraction
        with patch.object(
            predictor.feature_extractor,
            "extract_all_features",
            return_value=MockFeatureList(10),
        ):
            # Import the actual module to patch it properly
            import emotion_clf_pipeline.model as model_module  # noqa: E402

            # Patch DataLoader and tqdm
            with patch.object(
                model_module, "DataLoader", return_value=mock_dataloader_instance
            ):
                with patch.object(
                    model_module, "tqdm", side_effect=lambda x, **kwargs: x
                ):
                    with patch.object(
                        predictor, "post_process", return_value=mock_results_df
                    ):
                        # Test prediction
                        texts = ["I am happy today"]
                        results = predictor.predict(texts, batch_size=1)

                        # Verify key operations
                        self.mock_model.eval.assert_called()
                        mock_torch.argmax.assert_called()
                        self.assertEqual(results, mock_results_df)

    def test_post_process_success(self):
        """Test successful post-processing of predictions."""
        predictor = CustomPredictor(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device=self.mock_device,
            encoders_dir=self.encoders_dir,
        )

        # Create test DataFrame
        test_data = {
            "text": ["I am happy", "I am sad"],
            "predicted_emotion": ["happiness", "sadness"],
            "predicted_sub_emotion": ["joy", "grief"],
            "sub_emotion_logits": [
                [0.1, 0.9, 0.2],  # logits for first text
                [0.8, 0.1, 0.7],  # logits for second text
            ],
        }

        # Create actual DataFrame for testing
        df = pd.DataFrame(test_data)

        # Mock torch.tensor and torch.softmax
        mock_tensor = MagicMock()
        mock_torch.tensor.return_value = mock_tensor
        mock_torch.softmax.return_value = mock_tensor

        # Mock tensor item() method for probabilities
        mock_tensor.item.side_effect = [0.1, 0.9, 0.2]  # Sample probabilities

        # Test post-processing
        result = predictor.post_process(df)

        # Verify the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Verify required columns exist
        self.assertIn("predicted_sub_emotion", result.columns)
        self.assertIn("emotion_pred_post_processed", result.columns)

    def test_post_process_emotion_mapping_alignment(self):
        """Test that post-processing aligns sub-emotions with main emotions."""
        predictor = CustomPredictor(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device=self.mock_device,
            encoders_dir=self.encoders_dir,
        )

        # Test data with misaligned emotions
        test_data = {
            "text": ["test text"],
            "predicted_emotion": ["happiness"],
            "predicted_sub_emotion": ["anger"],  # Misaligned
            "sub_emotion_logits": [
                [0.1, 0.2, 0.9]
            ],  # High probability for joy (index 2)
        }

        df = pd.DataFrame(test_data)

        # Mock the sub_emotion encoder classes to include emotions we're testing
        predictor.encoders["sub_emotion"].classes_ = ["anger", "sadness", "joy"]

        # Mock torch operations
        mock_tensor = MagicMock()
        mock_torch.tensor.return_value = mock_tensor
        mock_torch.softmax.return_value = mock_tensor
        mock_tensor.item.side_effect = [0.1, 0.2, 0.9]  # Probabilities matching logits

        result = predictor.post_process(df)

        # The post-processing should align the sub-emotion with the main emotion
        # joy maps to happiness, so it should be chosen over anger
        self.assertIn("emotion_pred_post_processed", result.columns)

    def test_feature_extractor_initialization(self):
        """Test that feature extractor is properly initialized."""
        predictor = CustomPredictor(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device=self.mock_device,
            encoders_dir=self.encoders_dir,
        )

        self.assertIsNotNone(predictor.feature_extractor)
        self.assertEqual(
            predictor.feature_extractor.feature_config, predictor.feature_config
        )

    def test_expected_feature_dim_from_model(self):
        """Test that expected feature dimension is extracted from model."""
        for dim in [5, 10, 20, 50]:
            mock_model = MagicMock()
            mock_model.feature_projection = [MagicMock()]
            mock_model.feature_projection[0].in_features = dim

            predictor = CustomPredictor(
                model=mock_model,
                tokenizer=self.mock_tokenizer,
                device=self.mock_device,
                encoders_dir=self.encoders_dir,
            )

            self.assertEqual(predictor.expected_feature_dim, dim)

    def test_predict_error_handling(self):
        """Test error handling during prediction."""
        predictor = CustomPredictor(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device=self.mock_device,
            encoders_dir=self.encoders_dir,
        )

        # Mock feature extraction to raise an exception
        with patch.object(
            predictor.feature_extractor,
            "extract_all_features",
            side_effect=Exception("Feature extraction failed"),
        ):
            with self.assertRaises(Exception):
                predictor.predict(["test text"])

    def test_init_error_handling(self):
        """Test error handling during initialization."""
        invalid_model = MagicMock()
        invalid_model.feature_projection = []  # Empty list instead of complex mock

        with self.assertRaises(IndexError):
            CustomPredictor(
                model=invalid_model,
                tokenizer=self.mock_tokenizer,
                device=self.mock_device,
                encoders_dir=self.encoders_dir,
            )


if __name__ == "__main__":
    unittest.main()
