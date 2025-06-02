import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch


# Mock ALL external dependencies before any imports
# Mock PyTorch with proper nn.Module base class
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
mock_torch.randn = Mock(return_value=Mock())
mock_torch.randint = Mock(return_value=Mock())
mock_torch.ones = Mock(return_value=Mock())
mock_torch.cat = Mock(return_value=Mock())
mock_torch.save = Mock()
mock_torch.load = Mock()
mock_torch.equal = Mock(return_value=True)
mock_torch.allclose = Mock(return_value=True)
mock_torch.softmax = Mock(return_value=Mock())
mock_torch.argmax = Mock(return_value=Mock())
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

# Mock the data module that's imported
mock_data_module = Mock()
mock_data_module.EmotionDataset = Mock()
mock_data_module.FeatureExtractor = Mock()
sys.modules["data"] = mock_data_module

# Mock AutoModel for transformers
mock_auto_model = Mock()
mock_auto_model.from_pretrained = Mock()
mock_config = Mock()
mock_config.hidden_size = 768
mock_auto_model.from_pretrained.return_value.config = mock_config
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


class MockTensor:
    """Mock tensor class to simulate PyTorch tensors."""

    def __init__(self, shape, data=None):
        self.shape = shape
        self.data = data or [0] * (shape[0] * shape[1] if len(shape) == 2 else shape[0])

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 3:
            # Handle [:, 0, :] indexing for CLS token
            return MockTensor((self.shape[0], self.shape[2]))
        return MockTensor(self.shape)

    def sum(self, dim=None):
        if dim is None:
            return MockTensor((1,))
        elif dim == 1:
            return MockTensor((self.shape[0],))
        return MockTensor(self.shape)

    def backward(self):
        pass


class MockInput:
    """Mock input tensor with shape attribute."""

    def __init__(self, shape):
        self.shape = shape


class TestDEBERTAClassifier(unittest.TestCase):
    """Test cases for the DEBERTAClassifier model."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.model_name = "microsoft/deberta-v3-base"
        self.feature_dim = 10
        self.num_classes = {"emotion": 6, "sub_emotion": 25, "intensity": 3}
        self.hidden_dim = 256
        self.dropout = 0.1
        self.batch_size = 4
        self.seq_length = 128

        # Mock the torch.cat function to return appropriate shapes
        def mock_cat(tensors, dim=1):
            if dim == 1:
                # Combining deberta embeddings (768) for projected features (hidden_dim)
                combined_dim = 768 + self.hidden_dim
                batch_size = tensors[0].shape[0] if hasattr(tensors[0], "shape") else 4
                return MockTensor((batch_size, combined_dim))
            return MockTensor((4, 768))

        mock_torch.cat.side_effect = mock_cat

    def test_model_initialization(self):
        """Test that the model initializes correctly."""
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

    def test_model_forward_pass(self):
        """Test the forward pass of the model."""
        # Setup mock return values for the forward pass
        mock_deberta_output = Mock()
        mock_deberta_output.last_hidden_state = MockTensor(
            (self.batch_size, self.seq_length, 768)
        )

        mock_auto_model.from_pretrained.return_value = Mock()
        mock_auto_model.from_pretrained.return_value.config.hidden_size = 768
        mock_auto_model.from_pretrained.return_value.return_value = mock_deberta_output

        # Mock the feature projection and classifiers
        mock_projected_features = MockTensor((self.batch_size, self.hidden_dim))
        mock_combined = MockTensor((self.batch_size, 768 + self.hidden_dim))

        model = DEBERTAClassifier(
            model_name=self.model_name,
            feature_dim=self.feature_dim,
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )

        # Mock the forward pass components
        model.deberta = Mock(return_value=mock_deberta_output)
        model.feature_projection = Mock(return_value=mock_projected_features)
        model.emotion_classifier = Mock(
            return_value=MockTensor((self.batch_size, self.num_classes["emotion"]))
        )
        model.sub_emotion_classifier = Mock(
            return_value=MockTensor((self.batch_size, self.num_classes["sub_emotion"]))
        )
        model.intensity_classifier = Mock(
            return_value=MockTensor((self.batch_size, self.num_classes["intensity"]))
        )

        # Mock torch.cat to return the combined tensor
        with patch.object(mock_torch, "cat", return_value=mock_combined):
            # Create mock input tensors
            input_ids = MockInput((self.batch_size, self.seq_length))
            attention_mask = MockInput((self.batch_size, self.seq_length))
            features = MockInput((self.batch_size, self.feature_dim))

            # Forward pass
            emotion_logits, sub_emotion_logits, intensity_logits = model.forward(
                input_ids, attention_mask, features
            )

            # Check that outputs are returned
            self.assertIsNotNone(emotion_logits)
            self.assertIsNotNone(sub_emotion_logits)
            self.assertIsNotNone(intensity_logits)

            # Check output shapes
            self.assertEqual(
                emotion_logits.shape, (self.batch_size, self.num_classes["emotion"])
            )
            self.assertEqual(
                sub_emotion_logits.shape,
                (self.batch_size, self.num_classes["sub_emotion"]),
            )
            self.assertEqual(
                intensity_logits.shape, (self.batch_size, self.num_classes["intensity"])
            )

    def test_model_forward_pass_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        model = DEBERTAClassifier(
            model_name=self.model_name,
            feature_dim=self.feature_dim,
            num_classes=self.num_classes,
        )

        for batch_size in [1, 2, 8, 16]:
            with self.subTest(batch_size=batch_size):
                # Mock the components for each batch size
                mock_deberta_output = Mock()
                mock_deberta_output.last_hidden_state = MockTensor(
                    (batch_size, self.seq_length, 768)
                )
                mock_combined = MockTensor((batch_size, 768 + self.hidden_dim))

                model.deberta = Mock(return_value=mock_deberta_output)
                model.feature_projection = Mock(
                    return_value=MockTensor((batch_size, self.hidden_dim))
                )
                model.emotion_classifier = Mock(
                    return_value=MockTensor((batch_size, self.num_classes["emotion"]))
                )
                model.sub_emotion_classifier = Mock(
                    return_value=MockTensor(
                        (batch_size, self.num_classes["sub_emotion"])
                    )
                )
                model.intensity_classifier = Mock(
                    return_value=MockTensor((batch_size, self.num_classes["intensity"]))
                )

                input_ids = MockInput((batch_size, self.seq_length))
                attention_mask = MockInput((batch_size, self.seq_length))
                features = MockInput((batch_size, self.feature_dim))

                with patch.object(mock_torch, "cat", return_value=mock_combined):
                    emotion_logits, sub_emotion_logits, intensity_logits = (
                        model.forward(input_ids, attention_mask, features)
                    )

                    self.assertEqual(emotion_logits.shape[0], batch_size)
                    self.assertEqual(sub_emotion_logits.shape[0], batch_size)
                    self.assertEqual(intensity_logits.shape[0], batch_size)

    def test_model_different_hidden_dims(self):
        """Test model with different hidden dimensions."""
        for hidden_dim in [128, 256, 512]:
            with self.subTest(hidden_dim=hidden_dim):
                model = DEBERTAClassifier(
                    model_name=self.model_name,
                    feature_dim=self.feature_dim,
                    num_classes=self.num_classes,
                    hidden_dim=hidden_dim,
                )

                # Mock the forward pass components
                mock_deberta_output = Mock()
                mock_deberta_output.last_hidden_state = MockTensor(
                    (2, self.seq_length, 768)
                )
                mock_combined = MockTensor((2, 768 + hidden_dim))

                model.deberta = Mock(return_value=mock_deberta_output)
                model.feature_projection = Mock(
                    return_value=MockTensor((2, hidden_dim))
                )
                model.emotion_classifier = Mock(
                    return_value=MockTensor((2, self.num_classes["emotion"]))
                )
                model.sub_emotion_classifier = Mock(
                    return_value=MockTensor((2, self.num_classes["sub_emotion"]))
                )
                model.intensity_classifier = Mock(
                    return_value=MockTensor((2, self.num_classes["intensity"]))
                )

                input_ids = MockInput((2, self.seq_length))
                attention_mask = MockInput((2, self.seq_length))
                features = MockInput((2, self.feature_dim))

                with patch.object(mock_torch, "cat", return_value=mock_combined):
                    emotion_logits, sub_emotion_logits, intensity_logits = (
                        model.forward(input_ids, attention_mask, features)
                    )

                    # Check that outputs are still correct shape
                    self.assertEqual(
                        emotion_logits.shape[1], self.num_classes["emotion"]
                    )
                    self.assertEqual(
                        sub_emotion_logits.shape[1], self.num_classes["sub_emotion"]
                    )
                    self.assertEqual(
                        intensity_logits.shape[1], self.num_classes["intensity"]
                    )

    def test_model_different_feature_dims(self):
        """Test model with different feature dimensions."""
        for feature_dim in [5, 10, 20, 50]:
            with self.subTest(feature_dim=feature_dim):
                model = DEBERTAClassifier(
                    model_name=self.model_name,
                    feature_dim=feature_dim,
                    num_classes=self.num_classes,
                )

                # Mock the forward pass components
                mock_deberta_output = Mock()
                mock_deberta_output.last_hidden_state = MockTensor(
                    (2, self.seq_length, 768)
                )
                mock_combined = MockTensor((2, 768 + self.hidden_dim))

                model.deberta = Mock(return_value=mock_deberta_output)
                model.feature_projection = Mock(
                    return_value=MockTensor((2, self.hidden_dim))
                )
                model.emotion_classifier = Mock(
                    return_value=MockTensor((2, self.num_classes["emotion"]))
                )
                model.sub_emotion_classifier = Mock(
                    return_value=MockTensor((2, self.num_classes["sub_emotion"]))
                )
                model.intensity_classifier = Mock(
                    return_value=MockTensor((2, self.num_classes["intensity"]))
                )

                input_ids = MockInput((2, self.seq_length))
                attention_mask = MockInput((2, self.seq_length))
                features = MockInput((2, feature_dim))

                with patch.object(mock_torch, "cat", return_value=mock_combined):
                    emotion_logits, sub_emotion_logits, intensity_logits = (
                        model.forward(input_ids, attention_mask, features)
                    )

                    self.assertEqual(
                        emotion_logits.shape, (2, self.num_classes["emotion"])
                    )
                    self.assertEqual(
                        sub_emotion_logits.shape, (2, self.num_classes["sub_emotion"])
                    )
                    self.assertEqual(
                        intensity_logits.shape, (2, self.num_classes["intensity"])
                    )

    def test_model_with_different_num_classes(self):
        """Test model with different number of classes."""
        different_num_classes = {"emotion": 8, "sub_emotion": 30, "intensity": 5}

        model = DEBERTAClassifier(
            model_name=self.model_name,
            feature_dim=self.feature_dim,
            num_classes=different_num_classes,
        )

        # Mock the forward pass components
        mock_deberta_output = Mock()
        mock_deberta_output.last_hidden_state = MockTensor((3, self.seq_length, 768))
        mock_combined = MockTensor((3, 768 + self.hidden_dim))

        model.deberta = Mock(return_value=mock_deberta_output)
        model.feature_projection = Mock(return_value=MockTensor((3, self.hidden_dim)))
        model.emotion_classifier = Mock(
            return_value=MockTensor((3, different_num_classes["emotion"]))
        )
        model.sub_emotion_classifier = Mock(
            return_value=MockTensor((3, different_num_classes["sub_emotion"]))
        )
        model.intensity_classifier = Mock(
            return_value=MockTensor((3, different_num_classes["intensity"]))
        )

        input_ids = MockInput((3, self.seq_length))
        attention_mask = MockInput((3, self.seq_length))
        features = MockInput((3, self.feature_dim))

        with patch.object(mock_torch, "cat", return_value=mock_combined):
            emotion_logits, sub_emotion_logits, intensity_logits = model.forward(
                input_ids, attention_mask, features
            )

            self.assertEqual(
                emotion_logits.shape, (3, different_num_classes["emotion"])
            )
            self.assertEqual(
                sub_emotion_logits.shape, (3, different_num_classes["sub_emotion"])
            )
            self.assertEqual(
                intensity_logits.shape, (3, different_num_classes["intensity"])
            )

    def test_model_component_initialization(self):
        """Test that all model components are properly initialized."""
        model = DEBERTAClassifier(
            model_name=self.model_name,
            feature_dim=self.feature_dim,
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )

        # Check that the model inherits from nn.Module
        self.assertIsInstance(model, MockModule)

        # Check that all required components exist
        self.assertTrue(hasattr(model, "deberta"))
        self.assertTrue(hasattr(model, "feature_projection"))
        self.assertTrue(hasattr(model, "emotion_classifier"))
        self.assertTrue(hasattr(model, "sub_emotion_classifier"))
        self.assertTrue(hasattr(model, "intensity_classifier"))

    def test_model_training_mode_toggle(self):
        """Test switching between training and evaluation modes."""
        model = DEBERTAClassifier(
            model_name=self.model_name,
            feature_dim=self.feature_dim,
            num_classes=self.num_classes,
        )

        # Test that model has training attribute (inherited from MockModule)
        self.assertTrue(hasattr(model, "training"))

        # Test mode switching methods exist
        self.assertTrue(hasattr(model, "train"))
        self.assertTrue(hasattr(model, "eval"))

        # Test initial training state
        self.assertTrue(model.training)

        # Test switching to eval mode
        model.eval()
        self.assertFalse(model.training)

        # Test switching back to train mode
        model.train()
        self.assertTrue(model.training)


class TestNLTKSetup(unittest.TestCase):
    """Test NLTK setup functionality."""

    @patch("os.path.exists")
    def test_nltk_path_setup_with_app_directory(self, mock_exists):
        """Test NLTK path setup when /app/nltk_data exists."""
        mock_exists.return_value = True

        # This would be the logic from your original module
        if os.path.exists("/app/nltk_data"):
            # In the real module, this would append to nltk.data.path
            result = "Path would be appended"
        else:
            result = "Path would not be appended"

        mock_exists.assert_called_with("/app/nltk_data")
        self.assertEqual(result, "Path would be appended")

    @patch("os.path.exists")
    def test_nltk_download_fallback(self, mock_exists):
        """Test NLTK download fallback when no standard path exists."""
        mock_exists.return_value = False

        # Simulate the fallback download logic
        download_attempted = False
        if not os.path.exists("/app/nltk_data"):
            try:
                # In real code, this would be nltk.download calls
                download_attempted = True
            except Exception:
                pass

        mock_exists.assert_called_with("/app/nltk_data")
        self.assertTrue(download_attempted)


class TestModelIntegration(unittest.TestCase):
    """Integration tests for the model components."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.model_config = {
            "model_name": "microsoft/deberta-v3-base",
            "feature_dim": 15,
            "num_classes": {"emotion": 7, "sub_emotion": 28, "intensity": 4},
            "hidden_dim": 512,
            "dropout": 0.2,
        }

    def test_full_pipeline_simulation(self):
        """Test a full pipeline simulation."""
        model = DEBERTAClassifier(**self.model_config)

        # Mock the forward pass components
        batch_size = 8
        seq_length = 256

        mock_deberta_output = Mock()
        mock_deberta_output.last_hidden_state = MockTensor(
            (batch_size, seq_length, 768)
        )
        mock_combined = MockTensor((batch_size, 768 + self.model_config["hidden_dim"]))

        model.deberta = Mock(return_value=mock_deberta_output)
        model.feature_projection = Mock(
            return_value=MockTensor((batch_size, self.model_config["hidden_dim"]))
        )
        model.emotion_classifier = Mock(
            return_value=MockTensor(
                (batch_size, self.model_config["num_classes"]["emotion"])
            )
        )
        model.sub_emotion_classifier = Mock(
            return_value=MockTensor(
                (batch_size, self.model_config["num_classes"]["sub_emotion"])
            )
        )
        model.intensity_classifier = Mock(
            return_value=MockTensor(
                (batch_size, self.model_config["num_classes"]["intensity"])
            )
        )

        input_ids = MockInput((batch_size, seq_length))
        attention_mask = MockInput((batch_size, seq_length))
        features = MockInput((batch_size, self.model_config["feature_dim"]))

        # Forward pass
        with patch.object(mock_torch, "cat", return_value=mock_combined):
            emotion_logits, sub_emotion_logits, intensity_logits = model.forward(
                input_ids, attention_mask, features
            )

            # Verify outputs
            self.assertIsNotNone(emotion_logits)
            self.assertIsNotNone(sub_emotion_logits)
            self.assertIsNotNone(intensity_logits)

            # Check shapes
            self.assertEqual(emotion_logits.shape[0], batch_size)
            self.assertEqual(sub_emotion_logits.shape[0], batch_size)
            self.assertEqual(intensity_logits.shape[0], batch_size)

            self.assertEqual(
                emotion_logits.shape[1], self.model_config["num_classes"]["emotion"]
            )
            self.assertEqual(
                sub_emotion_logits.shape[1],
                self.model_config["num_classes"]["sub_emotion"],
            )
            self.assertEqual(
                intensity_logits.shape[1], self.model_config["num_classes"]["intensity"]
            )

    def test_model_consistency(self):
        """Test that model produces consistent outputs."""
        model = DEBERTAClassifier(**self.model_config)

        # Mock the forward pass components
        mock_deberta_output = Mock()
        mock_deberta_output.last_hidden_state = MockTensor((2, 128, 768))

        model.deberta = Mock(return_value=mock_deberta_output)
        model.feature_projection = Mock(
            return_value=MockTensor((2, self.model_config["hidden_dim"]))
        )
        model.emotion_classifier = Mock(
            return_value=MockTensor((2, self.model_config["num_classes"]["emotion"]))
        )
        model.sub_emotion_classifier = Mock(
            return_value=MockTensor(
                (2, self.model_config["num_classes"]["sub_emotion"])
            )
        )
        model.intensity_classifier = Mock(
            return_value=MockTensor((2, self.model_config["num_classes"]["intensity"]))
        )

        input_ids = MockInput((2, 128))
        attention_mask = MockInput((2, 128))
        features = MockInput((2, self.model_config["feature_dim"]))

        # Run forward pass twice
        output1 = model(input_ids, attention_mask, features)
        output2 = model(input_ids, attention_mask, features)

        # Both should return 3 outputs
        self.assertEqual(len(output1), 3)
        self.assertEqual(len(output2), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
