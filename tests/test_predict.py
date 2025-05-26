import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

import pandas as pd

# Comprehensive NLTK mocking - Mock ALL NLTK modules to prevent any imports
mock_nltk = MagicMock()
mock_nltk.download = MagicMock(return_value=True)
mock_nltk.data = MagicMock()
mock_nltk.data.find = MagicMock(return_value=True)
mock_nltk.data.path = []
mock_nltk.word_tokenize = MagicMock(return_value=["test", "tokens"])
mock_nltk.sent_tokenize = MagicMock(return_value=["test sentence"])

# Mock all NLTK submodules
sys.modules["nltk"] = mock_nltk
sys.modules["nltk.corpus"] = MagicMock()
sys.modules["nltk.corpus.stopwords"] = MagicMock()
sys.modules["nltk.tokenize"] = MagicMock()
sys.modules["nltk.tokenize.word_tokenize"] = MagicMock()
sys.modules["nltk.tokenize.sent_tokenize"] = MagicMock()
sys.modules["nltk.stem"] = MagicMock()
sys.modules["nltk.stem.porter"] = MagicMock()
sys.modules["nltk.stem.snowball"] = MagicMock()
sys.modules["nltk.sentiment"] = MagicMock()
sys.modules["nltk.sentiment.vader"] = MagicMock()
sys.modules["nltk.chunk"] = MagicMock()
sys.modules["nltk.tag"] = MagicMock()
sys.modules["nltk.parse"] = MagicMock()
sys.modules["nltk.metrics"] = MagicMock()
sys.modules["nltk.probability"] = MagicMock()
sys.modules["nltk.util"] = MagicMock()

# Mock other ML/NLP dependencies
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.nn.functional"] = MagicMock()
sys.modules["torch.optim"] = MagicMock()
sys.modules["torch.utils"] = MagicMock()
sys.modules["torch.utils.data"] = MagicMock()
sys.modules["textblob"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["sklearn"] = MagicMock()
sys.modules["sklearn.feature_extraction"] = MagicMock()
sys.modules["sklearn.feature_extraction.text"] = MagicMock()
sys.modules["sklearn.preprocessing"] = MagicMock()
sys.modules["sklearn.metrics"] = MagicMock()
sys.modules["sklearn.model_selection"] = MagicMock()
sys.modules["sklearn.linear_model"] = MagicMock()
sys.modules["sklearn.ensemble"] = MagicMock()
sys.modules["sklearn.svm"] = MagicMock()

# Mock audio/video processing dependencies
sys.modules["assemblyai"] = MagicMock()
sys.modules["whisper"] = MagicMock()
sys.modules["yt_dlp"] = MagicMock()
sys.modules["pydub"] = MagicMock()
sys.modules["pydub.AudioSegment"] = MagicMock()
sys.modules["pytubefix"] = MagicMock()

# Add the parent directory to the path to import the predict module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now import the predict module - it should work because dependencies are mocked
from src.emotion_clf_pipeline import predict  # noqa: E402


class TestPredictEmotion(unittest.TestCase):
    """Test predict_emotion function with multiple scenarios"""

    @patch("src.emotion_clf_pipeline.predict.EmotionPredictor")
    @patch("src.emotion_clf_pipeline.predict.time.time")
    @patch("builtins.print")
    def test_predict_emotion_all_scenarios(
        self, mock_print, mock_time, mock_predictor_class
    ):
        """Test all predict_emotion scenarios in one comprehensive test"""
        mock_predictor = Mock()
        mock_predictor_class.return_value = mock_predictor

        # Use a counter to simulate increasing time
        time_counter = [0.0]

        def mock_time_func():
            current = time_counter[0]
            time_counter[0] += 1.5
            return current

        mock_time.side_effect = mock_time_func

        # Test 1: Single text success
        expected_result = {
            "emotion": "happy",
            "sub_emotion": "joy",
            "intensity": "high",
        }
        mock_predictor.predict.return_value = expected_result

        result = predict.predict_emotion("I am very happy today!")
        self.assertEqual(result, expected_result)
        mock_predictor.predict.assert_called_with("I am very happy today!", None, False)

        # Test 2: Multiple texts success
        expected_result = [
            {"emotion": "happy", "sub_emotion": "joy", "intensity": "high"},
            {"emotion": "sad", "sub_emotion": "melancholy", "intensity": "medium"},
        ]
        mock_predictor.predict.return_value = expected_result
        texts = ["I am happy", "I am sad"]

        result = predict.predict_emotion(texts)
        self.assertEqual(result, expected_result)
        mock_predictor.predict.assert_called_with(texts, None, False)

        # Test 3: With feature config
        mock_predictor.predict.return_value = {"emotion": "neutral"}
        feature_config = {"use_tfidf": True, "max_features": 1000}

        result = predict.predict_emotion(
            "Test text", feature_config=feature_config, reload_model=True
        )
        mock_predictor.predict.assert_called_with("Test text", feature_config, True)

        # Test 4: Exception handling
        mock_predictor.predict.side_effect = Exception("Model loading failed")
        result = predict.predict_emotion("Test text")
        self.assertIsNone(result)
        mock_print.assert_called_with(
            "Error in emotion prediction: Model loading failed"
        )

        # Reset side effect for next tests
        mock_predictor.predict.side_effect = None

        # Test 5: Empty text
        mock_predictor.predict.return_value = {
            "emotion": "neutral",
            "sub_emotion": "none",
            "intensity": "low",
        }
        result = predict.predict_emotion("")
        mock_predictor.predict.assert_called_with("", None, False)

        # Test 6: Timing measurement verification
        # The mock should have been called multiple times for timing
        self.assertGreater(mock_time.call_count, 0)


class TestSpeechToText(unittest.TestCase):
    """Test speech_to_text function with multiple scenarios"""

    @patch("src.emotion_clf_pipeline.predict.SpeechToTextTranscriber")
    @patch("src.emotion_clf_pipeline.predict.WhisperTranscriber")
    @patch("src.emotion_clf_pipeline.predict.load_dotenv")
    @patch("src.emotion_clf_pipeline.predict.os.environ.get")
    @patch("src.emotion_clf_pipeline.predict.time.time")
    @patch("builtins.print")
    def test_speech_to_text_all_scenarios(
        self,
        mock_print,
        mock_time,
        mock_env_get,
        mock_load_dotenv,
        mock_whisper_class,
        mock_assembly_class,
    ):
        """Test all speech_to_text scenarios in one comprehensive test"""

        # Use a counter to simulate increasing time
        time_counter = [0.0]

        def mock_time_func():
            current = time_counter[0]
            time_counter[0] += 2.5
            return current

        mock_time.side_effect = mock_time_func

        # Test 1: AssemblyAI success
        mock_env_get.return_value = "test_api_key"
        mock_transcriber = Mock()
        mock_assembly_class.return_value = mock_transcriber

        predict.speech_to_text("assemblyAI", "test_audio.mp3", "test_output.xlsx")
        mock_load_dotenv.assert_called_with(dotenv_path="/app/.env", override=True)
        mock_assembly_class.assert_called_with("test_api_key")
        mock_transcriber.process.assert_called_with(
            "test_audio.mp3", "test_output.xlsx"
        )

        # Test 2: Whisper success
        mock_whisper_transcriber = Mock()
        mock_whisper_class.return_value = mock_whisper_transcriber

        predict.speech_to_text("whisper", "test_audio.wav", "test_output.xlsx")
        mock_whisper_class.assert_called()
        mock_whisper_transcriber.process.assert_called_with(
            "test_audio.wav", "test_output.xlsx"
        )

        # Test 3: AssemblyAI missing API key
        mock_env_get.return_value = None
        predict.speech_to_text("assemblyAI", "test_audio.mp3", "test_output.xlsx")
        expected_error = (
            "Error in speech-to-text: "
            "AssemblyAI API key not found in environment \
                        variables (checked in function)"
        )
        mock_print.assert_called_with(expected_error)

        # Test 4: Unknown method
        predict.speech_to_text("unknown_method", "test_audio.mp3", "test_output.xlsx")
        mock_print.assert_called_with(
            "Error in speech-to-text: Unknown transcription method: unknown_method"
        )

        # Test 5: Whisper exception
        mock_whisper_transcriber.process.side_effect = Exception("Audio file not found")
        predict.speech_to_text("whisper", "nonexistent_audio.mp3", "test_output.xlsx")
        mock_print.assert_called_with("Error in speech-to-text: Audio file not found")

        # Reset side effect
        mock_whisper_transcriber.process.side_effect = None

        # Test 6: Case insensitive methods
        for method in ["WHISPER", "Whisper", "wHiSpEr"]:
            predict.speech_to_text(method, "test_audio.mp3", "test_output.xlsx")
            mock_whisper_transcriber.process.assert_called_with(
                "test_audio.mp3", "test_output.xlsx"
            )

        # Test 7: Timing measurement verification
        self.assertGreater(mock_time.call_count, 0)


class TestProcessYouTubeURL(unittest.TestCase):
    """Test process_youtube_url_and_predict function with multiple scenarios"""

    @patch("src.emotion_clf_pipeline.predict.os.makedirs")
    @patch("src.emotion_clf_pipeline.predict.save_youtube_audio")
    @patch("src.emotion_clf_pipeline.predict.speech_to_text")
    @patch("src.emotion_clf_pipeline.predict.os.path.exists")
    @patch("src.emotion_clf_pipeline.predict.pd.read_excel")
    @patch("src.emotion_clf_pipeline.predict.predict_emotion")
    @patch("src.emotion_clf_pipeline.predict.pd.DataFrame.to_excel")
    @patch("builtins.print")
    def test_process_youtube_url_all_scenarios(
        self,
        mock_print,
        mock_to_excel,
        mock_predict_emotion,
        mock_read_excel,
        mock_path_exists,
        mock_speech_to_text,
        mock_save_youtube_audio,
        mock_makedirs,
    ):
        """Test process_youtube_url_and_predict scenarios in one comprehensive test"""

        # Test 1: Complete pipeline success
        mock_save_youtube_audio.return_value = "/path/to/audio.mp3"
        mock_path_exists.return_value = True
        mock_df = pd.DataFrame(
            {
                "Sentence": ["I am happy", "This is great", "Wonderful day"],
                "Timestamp": [1.0, 2.0, 3.0],
            }
        )
        mock_read_excel.return_value = mock_df
        expected_predictions = [
            {"emotion": "happy", "sub_emotion": "joy", "intensity": "high"},
            {"emotion": "happy", "sub_emotion": "excitement", "intensity": "medium"},
            {"emotion": "happy", "sub_emotion": "contentment", "intensity": "high"},
        ]
        mock_predict_emotion.return_value = expected_predictions

        result = predict.process_youtube_url_and_predict(
            "https://www.youtube.com/watch?v=test123", "test_video", "whisper"
        )
        self.assertEqual(result, expected_predictions)
        mock_save_youtube_audio.assert_called()
        mock_speech_to_text.assert_called()
        mock_predict_emotion.assert_called_with(
            ["I am happy", "This is great", "Wonderful day"]
        )
        mock_to_excel.assert_called()

        # Test 2: Transcription failed
        mock_path_exists.return_value = False
        with self.assertRaises(RuntimeError) as context:
            predict.process_youtube_url_and_predict(
                "https://www.youtube.com/watch?v=test123", "test_video", "assemblyAI"
            )
        self.assertIn("Transcription failed", str(context.exception))
        self.assertIn("AssemblyAI API key is missing", str(context.exception))

        # Reset for next test
        mock_path_exists.return_value = True

        # Test 3: Empty transcript
        mock_df = pd.DataFrame(
            {"Sentence": [None, None, None], "Timestamp": [1.0, 2.0, 3.0]}
        )
        mock_read_excel.return_value = mock_df
        result = predict.process_youtube_url_and_predict(
            "https://www.youtube.com/watch?v=test123", "test_video", "whisper"
        )
        self.assertEqual(result, [])
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        self.assertTrue(
            any("No sentences found in transcript" in call for call in print_calls)
        )

        # Test 4: Prediction returns None
        mock_df = pd.DataFrame({"Sentence": ["Test sentence"], "Timestamp": [1.0]})
        mock_read_excel.return_value = mock_df
        mock_predict_emotion.return_value = None
        result = predict.process_youtube_url_and_predict(
            "https://www.youtube.com/watch?v=test123", "test_video", "whisper"
        )
        self.assertEqual(result, [])
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        self.assertTrue(
            any("Emotion prediction step returned None" in call for call in print_calls)
        )

        # Test 5: Directory creation verification
        self.assertGreater(mock_makedirs.call_count, 0)
        for call in mock_makedirs.call_args_list:
            self.assertTrue(call[1]["exist_ok"])

        # Test 6: Invalid URL format
        mock_save_youtube_audio.side_effect = Exception("Invalid URL format")
        with self.assertRaises(Exception) as context:
            predict.process_youtube_url_and_predict(
                "not-a-valid-url", "test_filename", "whisper"
            )
        self.assertIn("Invalid URL format", str(context.exception))


if __name__ == "__main__":
    unittest.main()
