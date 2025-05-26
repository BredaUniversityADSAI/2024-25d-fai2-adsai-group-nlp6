import json
import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, call, patch

# Mock dependencies
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.utils.data"] = MagicMock()
sys.modules["textblob"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["sklearn"] = MagicMock()
sys.modules["sklearn.feature_extraction"] = MagicMock()
sys.modules["sklearn.feature_extraction.text"] = MagicMock()
sys.modules["sklearn.preprocessing"] = MagicMock()
sys.modules["sklearn.metrics"] = MagicMock()
sys.modules["sklearn.model_selection"] = MagicMock()
mock_nltk = MagicMock()
mock_nltk.download = MagicMock(return_value=True)
mock_nltk.data = MagicMock()
mock_nltk.data.find = MagicMock(return_value=True)
sys.modules["nltk"] = mock_nltk
sys.modules["nltk.corpus"] = MagicMock()
sys.modules["nltk.tokenize"] = MagicMock()
sys.modules["nltk.stem"] = MagicMock()
sys.modules["nltk.sentiment.vader"] = MagicMock()
sys.modules["assemblyai"] = MagicMock()
sys.modules["whisper"] = MagicMock()
sys.modules["yt_dlp"] = MagicMock()
sys.modules["pydub"] = MagicMock()
sys.modules["pytubefix"] = MagicMock()

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, "src")
sys.path.insert(0, src_dir)

# Import the modules under test
cli_module = None
main_function = None

try:
    # Try different import patterns to handle various project structures
    import emotion_clf_pipeline.cli as cli_module  # noqa: E402
    from emotion_clf_pipeline.cli import main as main_function  # noqa: E402
except ImportError:
    try:
        # Try direct import from src
        import cli as cli_module
        from cli import main as main_function
    except ImportError:
        try:
            # Try importing from current directory
            sys.path.insert(0, os.path.join(src_dir, "emotion_clf_pipeline"))
            import cli as cli_module
            from cli import main as main_function
        except ImportError as e:
            print(f"Could not import CLI module: {e}")
            raise


# Test fixtures setup
def get_sample_predictions():
    """Return sample prediction data for testing."""
    return [
        {
            "sentence": "This is a happy sentence.",
            "emotion": "joy",
            "sub_emotion": "happiness",
            "intensity": 0.85,
        },
        {
            "sentence": "This is a sad sentence.",
            "emotion": "sadness",
            "sub_emotion": "melancholy",
            "intensity": 0.72,
        },
    ]


def get_test_url():
    """Return test YouTube URL."""
    return "https://www.youtube.com/watch?v=test123"


# Main test function for the main() CLI function
def test_main_function():
    """Test the main CLI function with various scenarios."""
    sample_predictions = get_sample_predictions()
    test_url = get_test_url()

    # Test 1: Default arguments with successful predictions
    with (
        patch("argparse.ArgumentParser.parse_args") as mock_parse_args,
        patch(
            f"{cli_module.__name__}.process_youtube_url_and_predict", create=True
        ) as mock_predict,
        patch("builtins.print") as mock_print,
    ):
        mock_args = Mock()
        mock_args.url = test_url
        mock_args.filename = "cli_youtube_output"
        mock_args.transcription = "assemblyAI"
        mock_parse_args.return_value = mock_args
        mock_predict.return_value = sample_predictions

        main_function()

        mock_predict.assert_called_once_with(
            youtube_url=test_url,
            output_filename_base="cli_youtube_output",
            transcription_method="assemblyAI",
        )

        expected_calls = [
            call("\n--- Prediction Results ---"),
            call(json.dumps(sample_predictions, indent=4)),
        ]
        mock_print.assert_has_calls(expected_calls)

    # Test 2: Custom arguments
    with (
        patch("argparse.ArgumentParser.parse_args") as mock_parse_args,
        patch(
            f"{cli_module.__name__}.process_youtube_url_and_predict", create=True
        ) as mock_predict,
    ):
        mock_args = Mock()
        mock_args.url = test_url
        mock_args.filename = "custom_output"
        mock_args.transcription = "whisper"
        mock_parse_args.return_value = mock_args
        mock_predict.return_value = sample_predictions

        main_function()

        mock_predict.assert_called_once_with(
            youtube_url=test_url,
            output_filename_base="custom_output",
            transcription_method="whisper",
        )

    # Test 3: Empty predictions
    with (
        patch("argparse.ArgumentParser.parse_args") as mock_parse_args,
        patch(
            f"{cli_module.__name__}.process_youtube_url_and_predict", create=True
        ) as mock_predict,
        patch("builtins.print") as mock_print,
    ):
        mock_args = Mock()
        mock_args.url = test_url
        mock_args.filename = "cli_youtube_output"
        mock_args.transcription = "assemblyAI"
        mock_parse_args.return_value = mock_args
        mock_predict.return_value = []

        main_function()

        expected_calls = [
            call("\n--- Prediction Results ---"),
            call("No predictions were generated."),
        ]
        mock_print.assert_has_calls(expected_calls)

    # Test 4: None predictions
    with (
        patch("argparse.ArgumentParser.parse_args") as mock_parse_args,
        patch(
            f"{cli_module.__name__}.process_youtube_url_and_predict", create=True
        ) as mock_predict,
        patch("builtins.print") as mock_print,
    ):
        mock_args = Mock()
        mock_args.url = test_url
        mock_args.filename = "cli_youtube_output"
        mock_args.transcription = "assemblyAI"
        mock_parse_args.return_value = mock_args
        mock_predict.return_value = None

        main_function()

        expected_calls = [
            call("\n--- Prediction Results ---"),
            call("No predictions were generated."),
        ]
        mock_print.assert_has_calls(expected_calls)

    # Test 5: Exception handling
    with (
        patch("argparse.ArgumentParser.parse_args") as mock_parse_args,
        patch(
            f"{cli_module.__name__}.process_youtube_url_and_predict", create=True
        ) as mock_predict,
        patch("builtins.print") as mock_print,
    ):
        mock_args = Mock()
        mock_args.url = test_url
        mock_args.filename = "cli_youtube_output"
        mock_args.transcription = "assemblyAI"
        mock_parse_args.return_value = mock_args

        test_exception = Exception("Generic test error")
        mock_predict.side_effect = test_exception

        main_function()

        expected_calls = [
            call(
                "An error occurred during the emotion "
                f"prediction pipeline: {test_exception}"
            ),
            call(
                "Please ensure the URL is correct, the video is accessible, "
                "and all configurations are set."
            ),
        ]
        mock_print.assert_has_calls(expected_calls)

    # Test 6: Graceful exit verification
    with (
        patch("argparse.ArgumentParser.parse_args") as mock_parse_args,
        patch(
            f"{cli_module.__name__}.process_youtube_url_and_predict", create=True
        ) as mock_predict,
        patch("builtins.print") as mock_print,
    ):
        mock_args = Mock()
        mock_args.url = test_url
        mock_args.filename = "cli_youtube_output"
        mock_args.transcription = "assemblyAI"
        mock_parse_args.return_value = mock_args

        mock_predict.side_effect = RuntimeError("Critical error")

        try:
            main_function()
            graceful_exit = True
        except Exception:
            graceful_exit = False

        assert graceful_exit, "main() should handle exceptions gracefully"


class TestCLI(unittest.TestCase):
    """Test cases for CLI module functions."""

    def test_main_function_comprehensive(self):
        """Comprehensive test for the main CLI function."""
        test_main_function()


if __name__ == "__main__":
    unittest.main()
