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

# Comprehensive sklearn mocking - Add ALL missing submodules
sys.modules["sklearn"] = MagicMock()
sys.modules["sklearn.feature_extraction"] = MagicMock()
sys.modules["sklearn.feature_extraction.text"] = MagicMock()
sys.modules["sklearn.preprocessing"] = MagicMock()
sys.modules["sklearn.metrics"] = MagicMock()
sys.modules["sklearn.model_selection"] = MagicMock()
sys.modules["sklearn.linear_model"] = MagicMock()
sys.modules["sklearn.ensemble"] = MagicMock()
sys.modules["sklearn.svm"] = MagicMock()
sys.modules["sklearn.utils"] = MagicMock()  # This was missing!
sys.modules["sklearn.base"] = MagicMock()
sys.modules["sklearn.pipeline"] = MagicMock()
sys.modules["sklearn.compose"] = MagicMock()
sys.modules["sklearn.decomposition"] = MagicMock()
sys.modules["sklearn.cluster"] = MagicMock()
sys.modules["sklearn.naive_bayes"] = MagicMock()
sys.modules["sklearn.tree"] = MagicMock()
sys.modules["sklearn.neighbors"] = MagicMock()
sys.modules["sklearn.neural_network"] = MagicMock()
sys.modules["sklearn.discriminant_analysis"] = MagicMock()
sys.modules["sklearn.gaussian_process"] = MagicMock()
sys.modules["sklearn.cross_decomposition"] = MagicMock()
sys.modules["sklearn.feature_selection"] = MagicMock()
sys.modules["sklearn.semi_supervised"] = MagicMock()
sys.modules["sklearn.isotonic"] = MagicMock()
sys.modules["sklearn.calibration"] = MagicMock()
sys.modules["sklearn.multiclass"] = MagicMock()
sys.modules["sklearn.multioutput"] = MagicMock()
sys.modules["sklearn.dummy"] = MagicMock()
sys.modules["sklearn.datasets"] = MagicMock()
sys.modules["sklearn.inspection"] = MagicMock()
sys.modules["sklearn.experimental"] = MagicMock()

# Mock audio/video processing dependencies
sys.modules["assemblyai"] = MagicMock()
sys.modules["whisper"] = MagicMock()
sys.modules["yt_dlp"] = MagicMock()
sys.modules["pydub"] = MagicMock()
sys.modules["pydub.AudioSegment"] = MagicMock()
sys.modules["pytubefix"] = MagicMock()

# Mock matplotlib and seaborn if they might be imported
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["seaborn"] = MagicMock()

# Mock joblib if it might be imported
sys.modules["joblib"] = MagicMock()

# Mock dotenv
sys.modules["dotenv"] = MagicMock()

# Add the parent directory to the path to import the predict module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.emotion_clf_pipeline import predict  # noqa: E402


class TestPredictEmotion(unittest.TestCase):
    """Test predict_emotion function with multiple scenarios"""

    @patch("src.emotion_clf_pipeline.predict.EmotionPredictor")
    @patch("src.emotion_clf_pipeline.predict.time.time")
    @patch(
        "src.emotion_clf_pipeline.predict.logger"
    )  # Mock the logger instead of print
    def test_predict_emotion_all_scenarios(
        self, mock_logger, mock_time, mock_predictor_class
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

        # Test 4: Exception handling - Reset the mock and setup exception
        mock_predictor.predict.side_effect = Exception("Model loading failed")
        result = predict.predict_emotion("Test text")
        self.assertIsNone(result)

        # Check if logger.error was called with the expected error message
        mock_logger.error.assert_called()
        error_call_args = [call[0][0] for call in mock_logger.error.call_args_list]
        expected_error = "Error in emotion prediction: Model loading failed"
        self.assertTrue(
            any(expected_error in call for call in error_call_args),
            f"Expected error message not found in logger calls: {error_call_args}",
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
    @patch(
        "src.emotion_clf_pipeline.predict.logger"
    )  # Mock the logger instead of print
    def test_speech_to_text_all_scenarios(
        self,
        mock_logger,
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

        # Check if logger.warning was called with expected error
        mock_logger.warning.assert_called()
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        expected_keywords = ["AssemblyAI API key not found"]
        found_warning = any(
            all(keyword in call for keyword in expected_keywords)
            for call in warning_calls
        )
        self.assertTrue(
            found_warning,
            f"Expected warning message not found in logger calls: {warning_calls}",
        )

        # Test 4: Unknown method - Test for ValueError instead of logged error
        with self.assertRaises(ValueError) as context:
            predict.speech_to_text(
                "unknown_method", "test_audio.mp3", "test_output.xlsx"
            )
        self.assertIn(
            "Unknown transcription method: unknown_method", str(context.exception)
        )

        # Test 5: Whisper exception
        mock_whisper_transcriber.process.side_effect = Exception("Audio file not found")
        predict.speech_to_text("whisper", "nonexistent_audio.mp3", "test_output.xlsx")
        error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
        # Check for the actual error message format
        expected_error = "Error during Whisper transcription: Audio file not found"
        self.assertTrue(
            any(expected_error in call for call in error_calls),
            f"Expected audio file error not found: {error_calls}",
        )

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

    def create_mock_dataframe(
        self, sentences, start_times=None, end_times=None, is_empty=False
    ):
        """Helper method to create properly configured mock DataFrame"""
        mock_df = Mock()

        # Configure dropna to return self
        mock_df.dropna.return_value = mock_df
        mock_df.reset_index.return_value = mock_df

        # Configure empty property
        mock_df.empty = is_empty

        # Make columns iterable to fix "Sentence" in df.columns check
        if not is_empty and sentences:
            mock_df.columns = ["Sentence", "Start Time", "End Time"]
        else:
            mock_df.columns = []

        if not is_empty and sentences:
            # Create mock series for column access
            mock_sentence_series = Mock()
            mock_sentence_series.tolist.return_value = sentences

            # Default times if not provided
            if start_times is None:
                start_times = [i * 1.0 for i in range(len(sentences))]
            if end_times is None:
                end_times = [(i + 1) * 1.0 for i in range(len(sentences))]

            mock_start_series = Mock()
            mock_start_series.tolist.return_value = start_times

            mock_end_series = Mock()
            mock_end_series.tolist.return_value = end_times

            # Use a function to handle column access - FIXED: Added self parameter
            def mock_getitem(self_param, key):  # Added self_param parameter
                if key == "Sentence":
                    return mock_sentence_series
                elif key == "Start Time":
                    return mock_start_series
                elif key == "End Time":
                    return mock_end_series
                else:
                    raise KeyError(f"Column {key} not found")

            mock_df.__getitem__ = mock_getitem

        return mock_df

    @patch("src.emotion_clf_pipeline.predict.os.makedirs")
    @patch("src.emotion_clf_pipeline.predict.save_youtube_audio")
    @patch("src.emotion_clf_pipeline.predict.speech_to_text")
    @patch("src.emotion_clf_pipeline.predict.os.path.exists")
    @patch("src.emotion_clf_pipeline.predict.pd.read_excel")
    @patch("src.emotion_clf_pipeline.predict.predict_emotion")
    @patch("src.emotion_clf_pipeline.predict.pd.DataFrame.to_excel")
    @patch("src.emotion_clf_pipeline.predict.logger")
    def test_process_youtube_url_all_scenarios(
        self,
        mock_logger,
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

        sentences = ["I am happy", "This is great", "Wonderful day"]
        start_times = [0.0, 1.5, 3.0]
        end_times = [1.4, 2.9, 4.5]
        mock_df = self.create_mock_dataframe(sentences, start_times, end_times)
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

        # Check that we get the expected structure
        self.assertEqual(len(result), 3)
        for i, item in enumerate(result):
            self.assertEqual(item["sentence"], sentences[i])
            self.assertEqual(item["start_time"], start_times[i])
            self.assertEqual(item["end_time"], end_times[i])
            self.assertEqual(item["emotion"], expected_predictions[i]["emotion"])
            self.assertEqual(
                item["sub_emotion"], expected_predictions[i]["sub_emotion"]
            )
            self.assertEqual(item["intensity"], expected_predictions[i]["intensity"])

        mock_df.dropna.assert_called_with(subset=["Sentence"])

        # Test 2: Transcription failed
        mock_path_exists.return_value = False
        with self.assertRaises(RuntimeError) as context:
            predict.process_youtube_url_and_predict(
                "https://www.youtube.com/watch?v=test123", "test_video", "assemblyAI"
            )
        self.assertIn("Transcription failed", str(context.exception))

        # Reset for next test
        mock_path_exists.return_value = True

        # Test 3: Empty transcript - Mock empty DataFrame
        empty_mock_df = self.create_mock_dataframe([], is_empty=True)
        mock_read_excel.return_value = empty_mock_df

        result = predict.process_youtube_url_and_predict(
            "https://www.youtube.com/watch?v=test123", "test_video", "whisper"
        )
        self.assertEqual(result, [])

        # Test 4: Prediction returns None
        normal_mock_df = self.create_mock_dataframe(["Test sentence"])
        mock_read_excel.return_value = normal_mock_df
        mock_predict_emotion.return_value = None

        result = predict.process_youtube_url_and_predict(
            "https://www.youtube.com/watch?v=test123", "test_video", "whisper"
        )
        # Should still return data structure but with "unknown" values
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["emotion"], "unknown")
        self.assertEqual(result[0]["sub_emotion"], "unknown")
        self.assertEqual(result[0]["intensity"], "unknown")

        # Test 5: Invalid URL format
        mock_save_youtube_audio.side_effect = Exception("Invalid URL format")
        with self.assertRaises(Exception) as context:
            predict.process_youtube_url_and_predict(
                "not-a-valid-url", "test_filename", "whisper"
            )
        self.assertIn("Invalid URL format", str(context.exception))


class TestProcessYouTubeURLReal(unittest.TestCase):
    """Alternative approach: Use real pandas DataFrame for testing"""

    @patch("src.emotion_clf_pipeline.predict.os.makedirs")
    @patch("src.emotion_clf_pipeline.predict.save_youtube_audio")
    @patch("src.emotion_clf_pipeline.predict.speech_to_text")
    @patch("src.emotion_clf_pipeline.predict.os.path.exists")
    @patch("src.emotion_clf_pipeline.predict.pd.DataFrame.to_excel")
    @patch("src.emotion_clf_pipeline.predict.logger")
    def test_with_real_pandas(
        self,
        mock_logger,
        mock_to_excel,
        mock_path_exists,
        mock_speech_to_text,
        mock_save_youtube_audio,
        mock_makedirs,
    ):
        """Test using real pandas operations"""

        # Test 1: Success case
        mock_save_youtube_audio.return_value = "/path/to/audio.mp3"
        mock_path_exists.return_value = True

        # Create real DataFrame with the correct column names that the function expects
        test_df = pd.DataFrame(
            {
                "Sentence": ["I am happy", "This is great", "Wonderful day"],
                "Start Time": [0.0, 1.5, 3.0],
                "End Time": [1.4, 2.9, 4.5],
            }
        )

        expected_predictions = [
            {"emotion": "happy", "sub_emotion": "joy", "intensity": "high"},
            {"emotion": "happy", "sub_emotion": "excitement", "intensity": "medium"},
            {"emotion": "happy", "sub_emotion": "contentment", "intensity": "high"},
        ]

        # Mock both pd.read_excel and predict_emotion within the function call
        with (
            patch(
                "src.emotion_clf_pipeline.predict.pd.read_excel", return_value=test_df
            ) as mock_read_excel,
            patch(
                "src.emotion_clf_pipeline.predict.predict_emotion",
                return_value=expected_predictions,
            ) as mock_predict_emotion,
        ):

            result = predict.process_youtube_url_and_predict(
                "https://www.youtube.com/watch?v=test123", "test_video", "whisper"
            )

            # Debug information
            print(f"Mock read_excel called: {mock_read_excel.called}")
            print(f"Mock predict_emotion called: {mock_predict_emotion.called}")
            print(f"Mock predict_emotion call_args: {mock_predict_emotion.call_args}")
            print(f"Actual result length: {len(result)}")
            print(f"Expected result length: {len(expected_predictions)}")

            # Verify the result structure
            self.assertEqual(len(result), 3)

            # Check the structure of each result item
            for i, item in enumerate(result):
                self.assertEqual(item["sentence"], test_df.iloc[i]["Sentence"])
                self.assertEqual(item["start_time"], test_df.iloc[i]["Start Time"])
                self.assertEqual(item["end_time"], test_df.iloc[i]["End Time"])
                self.assertEqual(item["emotion"], expected_predictions[i]["emotion"])
                self.assertEqual(
                    item["sub_emotion"], expected_predictions[i]["sub_emotion"]
                )
                self.assertEqual(
                    item["intensity"], expected_predictions[i]["intensity"]
                )

            # Verify that read_excel was called
            mock_read_excel.assert_called_once()

            # Verify that predict_emotion was called with the right sentences
            mock_predict_emotion.assert_called_once_with(
                ["I am happy", "This is great", "Wonderful day"]
            )

        # Test 2: Empty transcript with None values
        empty_df = pd.DataFrame(
            {
                "Sentence": [None, None, None],
                "Start Time": [1.0, 2.0, 3.0],
                "End Time": [1.9, 2.9, 3.9],
            }
        )

        with (
            patch(
                "src.emotion_clf_pipeline.predict.pd.read_excel", return_value=empty_df
            ) as mock_read_excel,
            patch(
                "src.emotion_clf_pipeline.predict.predict_emotion"
            ) as mock_predict_emotion,
        ):

            result = predict.process_youtube_url_and_predict(
                "https://www.youtube.com/watch?v=test123", "test_video", "whisper"
            )

            # Should return empty list for all None values
            self.assertEqual(result, [])

            # predict_emotion should not be called when DataFrame is empty after dropna
            mock_predict_emotion.assert_not_called()

        # Test 3: Mixed data with some None values
        mixed_df = pd.DataFrame(
            {
                "Sentence": [
                    "I am happy",
                    None,
                    "This is great",
                    None,
                    "Wonderful day",
                ],
                "Start Time": [1.0, 2.0, 3.0, 4.0, 5.0],
                "End Time": [1.9, 2.9, 3.9, 4.9, 5.9],
            }
        )

        with (
            patch(
                "src.emotion_clf_pipeline.predict.pd.read_excel", return_value=mixed_df
            ) as mock_read_excel,
            patch(
                "src.emotion_clf_pipeline.predict.predict_emotion",
                return_value=expected_predictions,
            ) as mock_predict_emotion,
        ):

            result = predict.process_youtube_url_and_predict(
                "https://www.youtube.com/watch?v=test123", "test_video", "whisper"
            )

            # Should only process the non-None sentences
            mock_predict_emotion.assert_called_once_with(
                ["I am happy", "This is great", "Wonderful day"]
            )
            self.assertEqual(len(result), 3)  # Only non-None sentences

        # Test 4: Transcription failure
        mock_path_exists.return_value = False

        with self.assertRaises(RuntimeError) as context:
            predict.process_youtube_url_and_predict(
                "https://www.youtube.com/watch?v=test123", "test_video", "whisper"
            )

        self.assertIn("Transcription failed", str(context.exception))


if __name__ == "__main__":
    unittest.main()
