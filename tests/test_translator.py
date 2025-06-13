import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, mock_open, patch

# Mock the modules
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()

# Configure torch mock
torch_mock = sys.modules["torch"]
torch_mock.cuda.is_available.return_value = False
torch_mock.device.return_value = Mock()
torch_mock.no_grad = MagicMock()

# Configure transformers mock
transformers_mock = sys.modules["transformers"]
tokenizer_mock = Mock()
model_mock = Mock()
transformers_mock.AutoTokenizer.from_pretrained.return_value = tokenizer_mock
transformers_mock.AutoModelForSeq2SeqLM.from_pretrained.return_value = model_mock

# Add the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.emotion_clf_pipeline.translator import HuggingFaceTranslator  # noqa: E402

# Base path
BASE_PATH = "src.emotion_clf_pipeline.translator"


class TestHuggingFaceTranslator(unittest.TestCase):
    """Comprehensive unit tests for HuggingFaceTranslator class"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Reset mocks for each test
        self.tokenizer_mock = Mock()
        self.model_mock = Mock()

        # Configure the mocks
        self.tokenizer_mock.return_value = {
            "input_ids": Mock(),
            "attention_mask": Mock(),
        }
        self.tokenizer_mock.decode.return_value = "Mocked translation"

        # Mock model behavior
        self.model_mock.generate.return_value = [Mock()]
        self.model_mock.to.return_value = self.model_mock
        self.model_mock.eval.return_value = self.model_mock

    def test_init_french(self):
        """Test initialization with French language"""
        with (
            patch(f"{BASE_PATH}.torch.cuda.is_available", return_value=False),
            patch(f"{BASE_PATH}.torch.device") as mock_device,
            patch(f"{BASE_PATH}.AutoTokenizer.from_pretrained") as mock_tokenizer,
            patch(f"{BASE_PATH}.AutoModelForSeq2SeqLM.from_pretrained") as mock_model,
            patch("builtins.print"),
        ):
            mock_device.return_value = Mock()
            mock_tokenizer.return_value = self.tokenizer_mock
            mock_model.return_value = self.model_mock

            # Create translator
            translator = HuggingFaceTranslator("french")

            # Assertions
            self.assertEqual(translator.source_language, "french")
            self.assertEqual(translator.model_name, "helsinki-nlp/opus-mt-fr-en")
            mock_tokenizer.assert_called_with("helsinki-nlp/opus-mt-fr-en")
            mock_model.assert_called_with("helsinki-nlp/opus-mt-fr-en")
            self.model_mock.to.assert_called_once()
            self.model_mock.eval.assert_called_once()

    def test_init_spanish(self):
        """Test initialization with Spanish language"""
        with (
            patch(f"{BASE_PATH}.torch.cuda.is_available", return_value=False),
            patch(f"{BASE_PATH}.torch.device") as mock_device,
            patch(f"{BASE_PATH}.AutoTokenizer.from_pretrained") as mock_tokenizer,
            patch(f"{BASE_PATH}.AutoModelForSeq2SeqLM.from_pretrained") as mock_model,
            patch("builtins.print"),
        ):
            mock_device.return_value = Mock()
            mock_tokenizer.return_value = self.tokenizer_mock
            mock_model.return_value = self.model_mock

            translator = HuggingFaceTranslator("spanish")

            self.assertEqual(translator.source_language, "spanish")
            self.assertEqual(translator.model_name, "helsinki-nlp/opus-mt-es-en")

    def test_init_unsupported_language(self):
        """Test initialization with unsupported language raises ValueError"""
        with self.assertRaises(ValueError) as context:
            HuggingFaceTranslator("german")

        self.assertIn("Unsupported language: german", str(context.exception))

    def test_cuda_availability(self):
        """Test CUDA device selection when available"""
        with (
            patch(f"{BASE_PATH}.torch.cuda.is_available", return_value=True),
            patch(f"{BASE_PATH}.torch.device") as mock_device,
            patch(f"{BASE_PATH}.AutoTokenizer.from_pretrained") as mock_tokenizer,
            patch(f"{BASE_PATH}.AutoModelForSeq2SeqLM.from_pretrained") as mock_model,
            patch("builtins.print"),
        ):
            mock_device.return_value = Mock()
            mock_tokenizer.return_value = self.tokenizer_mock
            mock_model.return_value = self.model_mock

            translator = HuggingFaceTranslator("french")  # noqa: F841

            mock_device.assert_called_with("cuda")

    def test_translate_success(self):
        """Test successful translation"""
        with (
            patch(f"{BASE_PATH}.torch.cuda.is_available", return_value=False),
            patch(f"{BASE_PATH}.torch.device") as mock_device,
            patch(f"{BASE_PATH}.AutoTokenizer.from_pretrained") as mock_tokenizer,
            patch(f"{BASE_PATH}.AutoModelForSeq2SeqLM.from_pretrained") as mock_model,
            patch(f"{BASE_PATH}.torch.no_grad"),
            patch("builtins.print"),
        ):
            mock_device.return_value = Mock()

            # Setup tokenizer mock
            tokenizer_instance = Mock()
            tokenizer_instance.return_value = {
                "input_ids": Mock(),
                "attention_mask": Mock(),
            }
            tokenizer_instance.decode.return_value = "Hello world"
            mock_tokenizer.return_value = tokenizer_instance

            # Setup model mock
            model_instance = Mock()
            model_instance.generate.return_value = [Mock()]
            model_instance.to.return_value = model_instance
            model_instance.eval.return_value = model_instance
            mock_model.return_value = model_instance

            translator = HuggingFaceTranslator("french")
            result = translator.translate("Bonjour le monde")

            self.assertEqual(result, "Hello world")
            tokenizer_instance.assert_called()
            model_instance.generate.assert_called_once()

    def test_translate_empty_text(self):
        """Test translation with empty text"""
        with (
            patch(f"{BASE_PATH}.torch.cuda.is_available", return_value=False),
            patch(f"{BASE_PATH}.torch.device") as mock_device,
            patch(f"{BASE_PATH}.AutoTokenizer.from_pretrained") as mock_tokenizer,
            patch(f"{BASE_PATH}.AutoModelForSeq2SeqLM.from_pretrained") as mock_model,
            patch("builtins.print"),
        ):
            mock_device.return_value = Mock()
            mock_tokenizer.return_value = self.tokenizer_mock
            mock_model.return_value = self.model_mock

            translator = HuggingFaceTranslator("french")

            # Test empty string
            self.assertEqual(translator.translate(""), "")
            # Test whitespace only
            self.assertEqual(translator.translate("   "), "")
            # Test None
            self.assertEqual(translator.translate(None), "")

    def test_translate_exception_handling(self):
        """Test translation exception handling"""
        with (
            patch(f"{BASE_PATH}.torch.cuda.is_available", return_value=False),
            patch(f"{BASE_PATH}.torch.device") as mock_device,
            patch(f"{BASE_PATH}.AutoTokenizer.from_pretrained") as mock_tokenizer,
            patch(f"{BASE_PATH}.AutoModelForSeq2SeqLM.from_pretrained") as mock_model,
            patch("builtins.print"),
        ):
            mock_device.return_value = Mock()

            # Setup tokenizer to raise exception
            tokenizer_instance = Mock()
            tokenizer_instance.side_effect = Exception("Tokenization error")
            mock_tokenizer.return_value = tokenizer_instance
            mock_model.return_value = self.model_mock

            translator = HuggingFaceTranslator("french")

            test_text = "Bonjour"
            result = translator.translate(test_text)

            # Should return original text on error
            self.assertEqual(result, test_text)

    def test_translate_batch_success(self):
        """Test successful batch translation"""
        with (
            patch(f"{BASE_PATH}.torch.cuda.is_available", return_value=False),
            patch(f"{BASE_PATH}.torch.device") as mock_device,
            patch(f"{BASE_PATH}.AutoTokenizer.from_pretrained") as mock_tokenizer,
            patch(f"{BASE_PATH}.AutoModelForSeq2SeqLM.from_pretrained") as mock_model,
            patch(f"{BASE_PATH}.torch.no_grad"),
            patch("builtins.print"),
        ):
            mock_device.return_value = Mock()

            # Setup tokenizer mock
            tokenizer_instance = Mock()
            tokenizer_instance.return_value = {
                "input_ids": Mock(),
                "attention_mask": Mock(),
            }
            tokenizer_instance.decode.side_effect = ["Hello", "World"]
            mock_tokenizer.return_value = tokenizer_instance

            # Setup model mock
            model_instance = Mock()
            model_instance.generate.return_value = [Mock(), Mock()]
            model_instance.to.return_value = model_instance
            model_instance.eval.return_value = model_instance
            mock_model.return_value = model_instance

            translator = HuggingFaceTranslator("french")

            texts = ["Bonjour", "Monde"]
            results = translator.translate_batch(texts)

            self.assertEqual(results, ["Hello", "World"])

    def test_translate_batch_empty_list(self):
        """Test batch translation with empty list"""
        with (
            patch(f"{BASE_PATH}.torch.cuda.is_available", return_value=False),
            patch(f"{BASE_PATH}.torch.device") as mock_device,
            patch(f"{BASE_PATH}.AutoTokenizer.from_pretrained") as mock_tokenizer,
            patch(f"{BASE_PATH}.AutoModelForSeq2SeqLM.from_pretrained") as mock_model,
            patch("builtins.print"),
        ):
            mock_device.return_value = Mock()
            mock_tokenizer.return_value = self.tokenizer_mock
            mock_model.return_value = self.model_mock

            translator = HuggingFaceTranslator("french")

            result = translator.translate_batch([])
            self.assertEqual(result, [])

    def test_translate_batch_fallback(self):
        """Test batch translation fallback to individual translation on error"""
        with (
            patch(f"{BASE_PATH}.torch.cuda.is_available", return_value=False),
            patch(f"{BASE_PATH}.torch.device") as mock_device,
            patch(f"{BASE_PATH}.AutoTokenizer.from_pretrained") as mock_tokenizer,
            patch(f"{BASE_PATH}.AutoModelForSeq2SeqLM.from_pretrained") as mock_model,
            patch(f"{BASE_PATH}.torch.no_grad"),
            patch("builtins.print"),
        ):
            mock_device.return_value = Mock()

            # Setup tokenizer mock - fail first time (batch), succeed individual times
            tokenizer_instance = Mock()
            tokenizer_instance.side_effect = [
                Exception("Batch error"),  # First call fails
                {
                    "input_ids": Mock(),
                    "attention_mask": Mock(),
                },  # Individual calls succeed
                {"input_ids": Mock(), "attention_mask": Mock()},
            ]
            tokenizer_instance.decode.return_value = "Individual translation"
            mock_tokenizer.return_value = tokenizer_instance

            # Setup model mock
            model_instance = Mock()
            model_instance.generate.return_value = [Mock()]
            model_instance.to.return_value = model_instance
            model_instance.eval.return_value = model_instance
            mock_model.return_value = model_instance

            translator = HuggingFaceTranslator("french")

            texts = ["Text1", "Text2"]
            results = translator.translate_batch(texts)

            # Should fallback to individual translations
            self.assertEqual(len(results), 2)

    def test_translate_csv_success(self):
        """Test successful CSV translation"""
        with (
            patch(f"{BASE_PATH}.torch.cuda.is_available", return_value=False),
            patch(f"{BASE_PATH}.torch.device") as mock_device,
            patch(f"{BASE_PATH}.AutoTokenizer.from_pretrained") as mock_tokenizer,
            patch(f"{BASE_PATH}.AutoModelForSeq2SeqLM.from_pretrained") as mock_model,
            patch(f"{BASE_PATH}.torch.no_grad"),
            patch("os.path.exists", return_value=True),
            patch("builtins.open", new_callable=mock_open),
            patch("csv.DictReader") as mock_reader,
            patch("csv.DictWriter") as mock_writer,
            patch("builtins.print"),
        ):
            mock_device.return_value = Mock()

            # Setup tokenizer mock
            tokenizer_instance = Mock()
            tokenizer_instance.return_value = {
                "input_ids": Mock(),
                "attention_mask": Mock(),
            }
            tokenizer_instance.decode.side_effect = ["Hello", "World"]
            mock_tokenizer.return_value = tokenizer_instance

            # Setup model mock
            model_instance = Mock()
            model_instance.generate.return_value = [Mock(), Mock()]
            model_instance.to.return_value = model_instance
            model_instance.eval.return_value = model_instance
            mock_model.return_value = model_instance

            # Mock CSV reader
            mock_reader.return_value = [
                {"text": "Bonjour", "other_column": "value1"},
                {"text": "Monde", "other_column": "value2"},
            ]

            # Mock CSV writer
            mock_writer_instance = Mock()
            mock_writer.return_value = mock_writer_instance

            translator = HuggingFaceTranslator("french")
            result = translator.translate_csv("input.csv", "output.csv", "text")

            self.assertEqual(result, "output.csv")
            mock_writer_instance.writeheader.assert_called_once()
            self.assertEqual(mock_writer_instance.writerow.call_count, 2)

    def test_translate_csv_file_not_found(self):
        """Test CSV translation with non-existent file"""
        with (
            patch(f"{BASE_PATH}.torch.cuda.is_available", return_value=False),
            patch(f"{BASE_PATH}.torch.device") as mock_device,
            patch(f"{BASE_PATH}.AutoTokenizer.from_pretrained") as mock_tokenizer,
            patch(f"{BASE_PATH}.AutoModelForSeq2SeqLM.from_pretrained") as mock_model,
            patch("os.path.exists", return_value=False),
            patch("builtins.print"),
        ):
            mock_device.return_value = Mock()
            mock_tokenizer.return_value = self.tokenizer_mock
            mock_model.return_value = self.model_mock

            translator = HuggingFaceTranslator("french")

            with self.assertRaises(FileNotFoundError):
                translator.translate_csv("nonexistent.csv")

    def test_translate_csv_empty_file(self):
        """Test CSV translation with empty file"""
        with (
            patch(f"{BASE_PATH}.torch.cuda.is_available", return_value=False),
            patch(f"{BASE_PATH}.torch.device") as mock_device,
            patch(f"{BASE_PATH}.AutoTokenizer.from_pretrained") as mock_tokenizer,
            patch(f"{BASE_PATH}.AutoModelForSeq2SeqLM.from_pretrained") as mock_model,
            patch("os.path.exists", return_value=True),
            patch("builtins.open", new_callable=mock_open),
            patch("csv.DictReader") as mock_reader,
            patch("builtins.print"),
        ):
            mock_device.return_value = Mock()
            mock_tokenizer.return_value = self.tokenizer_mock
            mock_model.return_value = self.model_mock

            mock_reader.return_value = []

            translator = HuggingFaceTranslator("french")

            with self.assertRaises(ValueError) as context:
                translator.translate_csv("empty.csv")

            self.assertIn("CSV file is empty", str(context.exception))

    def test_translate_csv_auto_detect_column(self):
        """Test CSV translation with auto-detection of text column"""
        with (
            patch(f"{BASE_PATH}.torch.cuda.is_available", return_value=False),
            patch(f"{BASE_PATH}.torch.device") as mock_device,
            patch(f"{BASE_PATH}.AutoTokenizer.from_pretrained") as mock_tokenizer,
            patch(f"{BASE_PATH}.AutoModelForSeq2SeqLM.from_pretrained") as mock_model,
            patch(f"{BASE_PATH}.torch.no_grad"),
            patch("os.path.exists", return_value=True),
            patch("builtins.open", new_callable=mock_open),
            patch("csv.DictReader") as mock_reader,
            patch("csv.DictWriter") as mock_writer,
            patch("builtins.print"),
        ):
            mock_device.return_value = Mock()

            # Setup tokenizer mock
            tokenizer_instance = Mock()
            tokenizer_instance.return_value = {
                "input_ids": Mock(),
                "attention_mask": Mock(),
            }
            tokenizer_instance.decode.return_value = "Hello"
            mock_tokenizer.return_value = tokenizer_instance

            # Setup model mock
            model_instance = Mock()
            model_instance.generate.return_value = [Mock()]
            model_instance.to.return_value = model_instance
            model_instance.eval.return_value = model_instance
            mock_model.return_value = model_instance

            # Mock CSV with 'Text' column (capitalized)
            mock_reader.return_value = [{"Text": "Bonjour", "other_column": "value1"}]

            mock_writer_instance = Mock()
            mock_writer.return_value = mock_writer_instance

            translator = HuggingFaceTranslator("french")

            # Should auto-detect 'Text' column
            result = translator.translate_csv("input.csv")

            # Verify it completed without error
            self.assertTrue(result.endswith("_translated_french_to_english.csv"))

    def test_model_loading_error(self):
        """Test model loading error handling"""
        with (
            patch(f"{BASE_PATH}.torch.cuda.is_available", return_value=False),
            patch(f"{BASE_PATH}.torch.device", return_value=Mock()),
            patch(f"{BASE_PATH}.AutoTokenizer.from_pretrained") as mock_tokenizer,
            patch("builtins.print"),
        ):
            mock_tokenizer.side_effect = Exception("Model loading failed")

            with self.assertRaises(Exception) as context:
                HuggingFaceTranslator("french")

            self.assertIn("Model loading failed", str(context.exception))

    def test_translate_csv_missing_column(self):
        """Test CSV translation with missing specified column"""
        with (
            patch(f"{BASE_PATH}.torch.cuda.is_available", return_value=False),
            patch(f"{BASE_PATH}.torch.device") as mock_device,
            patch(f"{BASE_PATH}.AutoTokenizer.from_pretrained") as mock_tokenizer,
            patch(f"{BASE_PATH}.AutoModelForSeq2SeqLM.from_pretrained") as mock_model,
            patch("os.path.exists", return_value=True),
            patch("builtins.open", new_callable=mock_open),
            patch("csv.DictReader") as mock_reader,
            patch("builtins.print"),
        ):
            mock_device.return_value = Mock()
            mock_tokenizer.return_value = self.tokenizer_mock
            mock_model.return_value = self.model_mock

            # Mock CSV without the specified column
            mock_reader.return_value = [{"other_column": "value1"}]

            translator = HuggingFaceTranslator("french")

            with self.assertRaises(ValueError) as context:
                translator.translate_csv("input.csv", text_column="missing_column")

            self.assertIn(
                "Column 'missing_column' not found in CSV", str(context.exception)
            )

    def test_translate_csv_no_auto_detect(self):
        """Test CSV translation when auto-detection fails"""
        with (
            patch(f"{BASE_PATH}.torch.cuda.is_available", return_value=False),
            patch(f"{BASE_PATH}.torch.device") as mock_device,
            patch(f"{BASE_PATH}.AutoTokenizer.from_pretrained") as mock_tokenizer,
            patch(f"{BASE_PATH}.AutoModelForSeq2SeqLM.from_pretrained") as mock_model,
            patch("os.path.exists", return_value=True),
            patch("builtins.open", new_callable=mock_open),
            patch("csv.DictReader") as mock_reader,
            patch("builtins.print"),
        ):
            mock_device.return_value = Mock()
            mock_tokenizer.return_value = self.tokenizer_mock
            mock_model.return_value = self.model_mock

            # Mock CSV without any recognizable text columns
            mock_reader.return_value = [
                {"random_column": "value1", "another_column": "value2"}
            ]

            translator = HuggingFaceTranslator("french")

            with self.assertRaises(ValueError) as context:
                translator.translate_csv("input.csv")

            self.assertIn("No text column found", str(context.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
