import os
import sys
import unittest
from unittest.mock import MagicMock, Mock


# Mock dependecies
# Mock PyTorch with proper nn.Module base class that supports subscriptable tensors
class MockTensor:
    """Mock tensor that supports indexing operations"""

    def __init__(self, shape=None):
        self.shape = shape or (1, 768)

    def __getitem__(self, key):
        # Return another MockTensor for any indexing operation
        return MockTensor()

    def __len__(self):
        return self.shape[0] if self.shape else 1


class MockModule:
    """Mock base class that acts like nn.Module"""

    def __init__(self):
        self.training = True

    def train(self, mode=True):
        self.training = mode
        return self

    def eval(self):
        self.training = False
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


mock_torch = Mock()
mock_torch.nn = Mock()
mock_torch.nn.Module = MockModule
mock_torch.nn.Linear = Mock()
mock_torch.nn.ReLU = Mock()
mock_torch.nn.Dropout = Mock()
mock_torch.nn.Sequential = Mock()
mock_torch.randn = Mock(return_value=MockTensor())
mock_torch.randint = Mock(return_value=MockTensor())
mock_torch.ones = Mock(return_value=MockTensor())
mock_torch.cat = Mock(return_value=MockTensor())
mock_torch.save = Mock()
mock_torch.load = Mock()
mock_torch.equal = Mock(return_value=True)
mock_torch.allclose = Mock(return_value=True)
mock_torch.softmax = Mock(return_value=MockTensor())
mock_torch.argmax = Mock(return_value=MockTensor())
mock_torch.all = Mock(return_value=True)
mock_torch.no_grad = Mock()
sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = mock_torch.nn
sys.modules["torch.utils"] = Mock()
sys.modules["torch.utils.data"] = Mock()

# Mock numpy with complete structure including __version__
mock_np = MagicMock()
mock_np.__version__ = "1.21.0"  # Add version attribute
mock_np.array = MagicMock()
mock_np.float32 = MagicMock()
mock_np.int64 = MagicMock()
mock_np.float64 = MagicMock()
mock_np.linalg = MagicMock()
mock_np.linalg.inv = MagicMock()
mock_np.random = MagicMock()
mock_np.random.seed = MagicMock()
mock_np.zeros = MagicMock()
mock_np.ones = MagicMock()
mock_np.mean = MagicMock()
mock_np.std = MagicMock()
mock_np.concatenate = MagicMock()
mock_np.stack = MagicMock()
mock_np.squeeze = MagicMock()
mock_np.expand_dims = MagicMock()
mock_np.reshape = MagicMock()
sys.modules["numpy"] = mock_np
sys.modules["numpy.linalg"] = mock_np.linalg
sys.modules["numpy.random"] = mock_np.random

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

# Mock scipy completely
mock_scipy = MagicMock()
mock_scipy.sparse = MagicMock()
mock_scipy.sparse.issparse = MagicMock(return_value=False)
mock_scipy.sparse.csr_matrix = MagicMock()
mock_scipy.sparse.csc_matrix = MagicMock()
mock_scipy.__version__ = "1.7.0"
sys.modules["scipy"] = mock_scipy
sys.modules["scipy.sparse"] = mock_scipy.sparse

# Mock sklearn completely
mock_sklearn = MagicMock()
mock_sklearn.__version__ = "1.0.0"
mock_sklearn.base = MagicMock()
mock_sklearn.base.clone = MagicMock()
mock_sklearn.utils = MagicMock()
mock_sklearn.utils.IS_32BIT = False
mock_sklearn.feature_extraction = MagicMock()
mock_sklearn.feature_extraction.text = MagicMock()
mock_sklearn.feature_extraction.text.TfidfVectorizer = MagicMock()
mock_sklearn.preprocessing = MagicMock()
mock_sklearn.preprocessing.LabelEncoder = MagicMock()
mock_sklearn.model_selection = MagicMock()
mock_sklearn.model_selection.train_test_split = MagicMock()
mock_sklearn.metrics = MagicMock()
mock_sklearn.metrics.accuracy_score = MagicMock()
mock_sklearn.metrics.classification_report = MagicMock()
sys.modules["sklearn"] = mock_sklearn
sys.modules["sklearn.base"] = mock_sklearn.base
sys.modules["sklearn.utils"] = mock_sklearn.utils
sys.modules["sklearn.feature_extraction"] = mock_sklearn.feature_extraction
sys.modules["sklearn.feature_extraction.text"] = mock_sklearn.feature_extraction.text
sys.modules["sklearn.preprocessing"] = mock_sklearn.preprocessing
sys.modules["sklearn.model_selection"] = mock_sklearn.model_selection
sys.modules["sklearn.metrics"] = mock_sklearn.metrics

# Mock NLTK
nltk_mock = MagicMock()
sys.modules["nltk"] = nltk_mock
sys.modules["nltk.sentiment"] = MagicMock()
sys.modules["nltk.sentiment.vader"] = MagicMock()
sys.modules["nltk.tokenize"] = MagicMock()
sys.modules["nltk.tag"] = MagicMock()
sys.modules["nltk.corpus"] = MagicMock()
sys.modules["nltk.data"] = MagicMock()

# Mock other dependencies
mock_transformers = Mock()
mock_transformers.AutoModel = Mock()
mock_transformers.AutoTokenizer = Mock()
sys.modules["transformers"] = mock_transformers
sys.modules["tqdm"] = Mock()

# Mock pandas
mock_pandas = MagicMock()
mock_pandas.DataFrame = MagicMock()
mock_pandas.read_csv = MagicMock()
sys.modules["pandas"] = mock_pandas

# Mock the data module
mock_data_module = Mock()
mock_data_module.EmotionDataset = Mock()
mock_data_module.FeatureExtractor = Mock()
sys.modules["data"] = mock_data_module

# Mock TextBlob with proper structure
mock_textblob = Mock()
mock_textblob_instance = Mock()
mock_textblob_instance.sentiment = Mock()
mock_textblob_instance.sentiment.polarity = 0.0
mock_textblob_instance.sentiment.subjectivity = 0.0
mock_textblob.TextBlob = Mock(return_value=mock_textblob_instance)
sys.modules["textblob"] = mock_textblob

# Mock AutoModel for transformers with subscriptable output
mock_auto_model = Mock()
mock_auto_model.from_pretrained = Mock()
mock_config = Mock()
mock_config.hidden_size = 768
mock_auto_model_instance = Mock()
mock_auto_model_instance.config = mock_config
mock_auto_model.from_pretrained.return_value = mock_auto_model_instance
mock_transformers.AutoModel = mock_auto_model

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
mock_matplotlib.figure = MagicMock()
mock_matplotlib.collections = MagicMock()
mock_matplotlib.markers = MagicMock()
mock_matplotlib.patches = MagicMock()
mock_matplotlib.dates = MagicMock()
mock_matplotlib.axis = MagicMock()

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
sys.modules["matplotlib.figure"] = mock_matplotlib.figure
sys.modules["matplotlib.collections"] = mock_matplotlib.collections
sys.modules["matplotlib.markers"] = mock_matplotlib.markers
sys.modules["matplotlib.patches"] = mock_matplotlib.patches
sys.modules["matplotlib.dates"] = mock_matplotlib.dates
sys.modules["matplotlib.axis"] = mock_matplotlib.axis

# First, add the source directory to Python path
current_test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_test_dir)
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

# Mock the entire emotion_clf_pipeline.data module before importing the model
mock_emotion_data_module = Mock()
mock_emotion_data_module.EmotionDataset = Mock()
mock_emotion_data_module.FeatureExtractor = Mock()
sys.modules["emotion_clf_pipeline.data"] = mock_emotion_data_module

from emotion_clf_pipeline.model import DEBERTAClassifier  # noqa: E402

# Make sure nn.Module is available for the class definition
nn = mock_torch.nn


class TestDEBERTAClassifier(unittest.TestCase):
    """Simplified test cases for the DEBERTAClassifier model."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.model_name = "microsoft/deberta-v3-base"
        self.feature_dim = 10
        self.num_classes = {"emotion": 6, "sub_emotion": 25, "intensity": 3}
        self.hidden_dim = 256
        self.dropout = 0.1

    def test_model_initialization(self):
        """Test that the model initializes correctly with all components."""
        model = DEBERTAClassifier(
            model_name=self.model_name,
            feature_dim=self.feature_dim,
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )

        # Check that all components are initialized
        self.assertIsNotNone(model.deberta)
        self.assertIsNotNone(model.feature_projection)
        self.assertIsNotNone(model.emotion_classifier)
        self.assertIsNotNone(model.sub_emotion_classifier)
        self.assertIsNotNone(model.intensity_classifier)

        # Check that model inherits from nn.Module (MockModule in our case)
        # Since DEBERTAClassifier inherits from torch.nn.Module,
        # and we mocked that to be MockModule
        self.assertTrue(hasattr(model, "train"))
        self.assertTrue(hasattr(model, "eval"))
        self.assertTrue(hasattr(model, "training"))

        # Check stored configuration
        self.assertEqual(model.model_name, self.model_name)
        self.assertEqual(model.num_classes, self.num_classes)
        self.assertEqual(model.hidden_dim, self.hidden_dim)
        self.assertEqual(model.dropout, self.dropout)

    def test_model_forward_pass(self):
        """Test the forward pass returns correct output structure."""
        model = DEBERTAClassifier(
            model_name=self.model_name,
            feature_dim=self.feature_dim,
            num_classes=self.num_classes,
        )

        # Mock the deberta output with subscriptable last_hidden_state
        mock_deberta_output = Mock()
        mock_deberta_output.last_hidden_state = MockTensor(
            shape=(2, 512, 768)
        )  # batch_size, seq_len, hidden_size
        model.deberta = Mock(return_value=mock_deberta_output)

        # Mock the classifiers to return mock tensors
        model.feature_projection = Mock(return_value=MockTensor())
        model.emotion_classifier = Mock(return_value=MockTensor())
        model.sub_emotion_classifier = Mock(return_value=MockTensor())
        model.intensity_classifier = Mock(return_value=MockTensor())

        # Create mock inputs
        input_ids = MockTensor()
        attention_mask = MockTensor()
        features = MockTensor()

        # Test forward pass
        outputs = model.forward(input_ids, attention_mask, features)

        # Check that outputs is a dictionary with correct keys
        self.assertIsInstance(outputs, dict)
        self.assertIn("emotion", outputs)
        self.assertIn("sub_emotion", outputs)
        self.assertIn("intensity", outputs)

        # Verify that deberta was called with correct arguments
        model.deberta.assert_called_once_with(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # Verify that feature projection was called
        model.feature_projection.assert_called_once()

        # Verify that all classifiers were called
        model.emotion_classifier.assert_called_once()
        model.sub_emotion_classifier.assert_called_once()
        model.intensity_classifier.assert_called_once()

    def test_model_with_different_configurations(self):
        """Test model with various configuration parameters."""
        configs = [
            {"hidden_dim": 128, "dropout": 0.2},
            {"hidden_dim": 512, "dropout": 0.0},
            {"feature_dim": 20, "hidden_dim": 64},
        ]

        for config in configs:
            with self.subTest(config=config):
                test_config = {
                    "model_name": self.model_name,
                    "feature_dim": config.get("feature_dim", self.feature_dim),
                    "num_classes": self.num_classes,
                    "hidden_dim": config.get("hidden_dim", self.hidden_dim),
                    "dropout": config.get("dropout", self.dropout),
                }

                model = DEBERTAClassifier(**test_config)

                # Check that model was created successfully
                self.assertIsNotNone(model)
                self.assertEqual(model.hidden_dim, test_config["hidden_dim"])
                self.assertEqual(model.dropout, test_config["dropout"])

    def test_model_training_mode_toggle(self):
        """Test switching between training and evaluation modes."""
        model = DEBERTAClassifier(
            model_name=self.model_name,
            feature_dim=self.feature_dim,
            num_classes=self.num_classes,
        )

        # Test initial training state
        self.assertTrue(model.training)

        # Test switching to eval mode
        model.eval()
        self.assertFalse(model.training)

        # Test switching back to train mode
        model.train()
        self.assertTrue(model.training)

    def test_model_callable_interface(self):
        """Test that model can be called like a function."""
        model = DEBERTAClassifier(
            model_name=self.model_name,
            feature_dim=self.feature_dim,
            num_classes=self.num_classes,
        )

        # Mock the forward method
        model.forward = Mock(
            return_value={
                "emotion": MockTensor(),
                "sub_emotion": MockTensor(),
                "intensity": MockTensor(),
            }
        )

        # Create mock inputs
        input_ids = MockTensor()
        attention_mask = MockTensor()
        features = MockTensor()

        # Test calling model directly (should call forward)
        result = model(input_ids, attention_mask, features)

        # Verify forward was called with correct arguments
        model.forward.assert_called_once_with(input_ids, attention_mask, features)
        self.assertIsInstance(result, dict)

    def test_model_with_different_num_classes(self):
        """Test model handles different numbers of classes correctly."""
        different_num_classes = {"emotion": 8, "sub_emotion": 30, "intensity": 5}

        model = DEBERTAClassifier(
            model_name=self.model_name,
            feature_dim=self.feature_dim,
            num_classes=different_num_classes,
        )

        # Check that the configuration was stored correctly
        self.assertEqual(model.num_classes, different_num_classes)
        self.assertEqual(model.num_classes["emotion"], 8)
        self.assertEqual(model.num_classes["sub_emotion"], 30)
        self.assertEqual(model.num_classes["intensity"], 5)

    def test_deberta_config_access(self):
        """Test that DEBERTA config is accessible after initialization."""
        model = DEBERTAClassifier(
            model_name=self.model_name,
            feature_dim=self.feature_dim,
            num_classes=self.num_classes,
        )

        # Verify that deberta model was initialized and has config
        self.assertIsNotNone(model.deberta)
        # The mock should have been called with from_pretrained
        mock_auto_model.from_pretrained.assert_called_with(self.model_name)


if __name__ == "__main__":
    unittest.main()
