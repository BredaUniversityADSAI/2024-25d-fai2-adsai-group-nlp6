import json
import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, mock_open, patch

import pandas as pd

# Add the src folder to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Comprehensive NLTK mocking
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

# Comprehensive sklearn mocking
sys.modules["sklearn"] = MagicMock()
sys.modules["sklearn.feature_extraction"] = MagicMock()
sys.modules["sklearn.feature_extraction.text"] = MagicMock()
sys.modules["sklearn.preprocessing"] = MagicMock()
sys.modules["sklearn.metrics"] = MagicMock()
sys.modules["sklearn.model_selection"] = MagicMock()
sys.modules["sklearn.linear_model"] = MagicMock()
sys.modules["sklearn.ensemble"] = MagicMock()
sys.modules["sklearn.svm"] = MagicMock()
sys.modules["sklearn.utils"] = MagicMock()
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

# Mock matplotlib and seaborn
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["seaborn"] = MagicMock()

# Mock joblib
sys.modules["joblib"] = MagicMock()

# Mock dotenv
sys.modules["dotenv"] = MagicMock()

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

# Minimal mocking just to isolate AzureMLSync
import src.emotion_clf_pipeline.model  # noqa: E402

# Patch the class AFTER import
patcher = patch.object(src.emotion_clf_pipeline.model, "AzureMLSync", MagicMock())
patcher.start()

from src.emotion_clf_pipeline import predict  # noqa: E402
from src.emotion_clf_pipeline.predict import (  # noqa: E402
    extract_audio_transcript,
    extract_transcript,
    predict_emotions_local,
    transcribe_youtube_url,
)


def test_extract_transcript():
    """Test extract_transcript function with all scenarios."""

    # Test 1: Success with manual subtitles
    with (
        patch("yt_dlp.YoutubeDL") as mock_ytdl,
        patch("urllib.request.urlopen") as mock_urlopen,
    ):

        mock_info = {
            "title": "Test Video",
            "subtitles": {"en": [{"url": "http://example.com/subtitles.vtt"}]},
        }

        mock_subtitle_content = """WEBVTT

00:00:01.000 --> 00:00:03.000
<v Speaker>Hello world this is a test

00:00:04.000 --> 00:00:06.000
<v Speaker>This is another line"""

        mock_ytdl_instance = MagicMock()
        mock_ytdl_instance.extract_info.return_value = mock_info
        mock_ytdl.return_value.__enter__.return_value = mock_ytdl_instance

        mock_response = MagicMock()
        mock_response.read.return_value = mock_subtitle_content.encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = extract_transcript("https://youtube.com/watch?v=test")

        assert result["source"] == "subtitles"
        assert result["title"] == "Test Video"
        assert "Hello world this is a test" in result["text"]
        assert "This is another line" in result["text"]

    # Test 2: Success with auto captions
    with (
        patch("yt_dlp.YoutubeDL") as mock_ytdl,
        patch("urllib.request.urlopen") as mock_urlopen,
    ):

        mock_info = {
            "title": "Test Video",
            "subtitles": {},
            "automatic_captions": {
                "en": [{"url": "http://example.com/auto_subtitles.vtt"}]
            },
        }

        mock_ytdl_instance = MagicMock()
        mock_ytdl_instance.extract_info.return_value = mock_info
        mock_ytdl.return_value.__enter__.return_value = mock_ytdl_instance

        mock_response = MagicMock()
        mock_response.read.return_value = b"Test auto caption content"
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = extract_transcript("https://youtube.com/watch?v=test")

        assert result["source"] == "subtitles"
        assert result["title"] == "Test Video"

    # Test 3: Fallback to STT when no subtitles
    with (
        patch("yt_dlp.YoutubeDL") as mock_ytdl,
        patch("your_module.extract_audio_transcript") as mock_extract_audio,
    ):  # Replace with actual module path

        mock_info = {"title": "Test Video", "subtitles": {}, "automatic_captions": {}}

        mock_ytdl_instance = MagicMock()
        mock_ytdl_instance.extract_info.return_value = mock_info
        mock_ytdl.return_value.__enter__.return_value = mock_ytdl_instance

        mock_extract_audio.return_value = {
            "text": "STT extracted text",
            "source": "stt_whisper",
        }

        result = extract_transcript("https://youtube.com/watch?v=test")

        mock_extract_audio.assert_called_once_with("https://youtube.com/watch?v=test")
        assert result["source"] == "stt_whisper"

    # Test 4: Invalid URL handling
    with (
        patch("yt_dlp.YoutubeDL") as mock_ytdl,
        patch("your_module.extract_audio_transcript") as mock_extract_audio,
    ):  # Replace with actual module path

        mock_ytdl_instance = MagicMock()
        mock_ytdl_instance.extract_info.side_effect = Exception("Invalid URL")
        mock_ytdl.return_value.__enter__.return_value = mock_ytdl_instance

        mock_extract_audio.return_value = {"text": "fallback", "source": "stt_whisper"}

        result = extract_transcript("invalid_url")

        mock_extract_audio.assert_called_once_with("invalid_url")

    print("All extract_transcript tests passed!")


def test_extract_audio_transcript():
    """Test extract_audio_transcript function with all scenarios."""

    # Test 1: Success case
    with (
        patch("tempfile.TemporaryDirectory") as mock_temp_dir,
        patch("your_module.save_youtube_audio") as mock_save_audio,
        patch("your_module.WhisperTranscriber") as mock_whisper_class,
    ):  # Replace with actual module path

        mock_temp_dir.return_value.__enter__.return_value = "/tmp/test"
        mock_save_audio.return_value = ("/tmp/test/audio.wav", "Test Video Title")

        mock_transcriber = MagicMock()
        mock_whisper_result = {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "First segment"},
                {"start": 2.0, "end": 4.0, "text": "Second segment"},
            ]
        }
        mock_transcriber.transcribe_audio.return_value = mock_whisper_result

        mock_transcript_data = [
            {"Sentence": "First segment", "Start": 0.0, "End": 2.0},
            {"Sentence": "Second segment", "Start": 2.0, "End": 4.0},
        ]
        mock_transcriber.extract_sentences.return_value = mock_transcript_data
        mock_whisper_class.return_value = mock_transcriber

        result = extract_audio_transcript("https://youtube.com/watch?v=test")

        assert result["source"] == "stt_whisper"
        assert result["title"] == "Test Video Title"
        assert result["text"] == "First segment Second segment"
        assert len(result["sentences"]) == 2
        assert "segments" in result

        mock_save_audio.assert_called_once_with(
            "https://youtube.com/watch?v=test", "/tmp/test"
        )
        mock_transcriber.transcribe_audio.assert_called_once_with("/tmp/test/audio.wav")

    # Test 2: Download failure
    with (
        patch("tempfile.TemporaryDirectory") as mock_temp_dir,
        patch("your_module.save_youtube_audio") as mock_save_audio,
    ):  # Replace with actual module path

        mock_temp_dir.return_value.__enter__.return_value = "/tmp/test"
        mock_save_audio.side_effect = Exception("Download failed")

        try:
            extract_audio_transcript("https://youtube.com/watch?v=test")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Failed to extract transcript using STT" in str(e)

    # Test 3: Whisper transcription failure
    with (
        patch("tempfile.TemporaryDirectory") as mock_temp_dir,
        patch("your_module.save_youtube_audio") as mock_save_audio,
        patch("your_module.WhisperTranscriber") as mock_whisper_class,
    ):  # Replace with actual module path

        mock_temp_dir.return_value.__enter__.return_value = "/tmp/test"
        mock_save_audio.return_value = ("/tmp/test/audio.wav", "Test Video")

        mock_transcriber = MagicMock()
        mock_transcriber.transcribe_audio.side_effect = Exception("Whisper failed")
        mock_whisper_class.return_value = mock_transcriber

        try:
            extract_audio_transcript("https://youtube.com/watch?v=test")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Failed to extract transcript using STT" in str(e)

    print("All extract_audio_transcript tests passed!")


def test_transcribe_youtube_url():
    """Test transcribe_youtube_url function with all scenarios."""

    # Test 1: Force STT option
    with patch(
        "your_module.extract_audio_transcript"
    ) as mock_extract_audio:  # Replace with actual module path
        mock_extract_audio.return_value = {"text": "STT text", "source": "stt_whisper"}

        result = transcribe_youtube_url(
            "https://youtube.com/watch?v=test", use_stt=True
        )

        mock_extract_audio.assert_called_once_with("https://youtube.com/watch?v=test")
        assert result["source"] == "stt_whisper"

    # Test 2: Try subtitles first
    with patch(
        "your_module.extract_transcript"
    ) as mock_extract_transcript:  # Replace with actual module path
        mock_extract_transcript.return_value = {
            "text": "Subtitle text",
            "source": "subtitles",
        }

        result = transcribe_youtube_url(
            "https://youtube.com/watch?v=test", use_stt=False
        )

        mock_extract_transcript.assert_called_once_with(
            "https://youtube.com/watch?v=test"
        )
        assert result["source"] == "subtitles"

    print("All transcribe_youtube_url tests passed!")


class TestPredictEmotion(unittest.TestCase):
    """Test predict_emotion function with multiple scenarios"""

    @patch("src.emotion_clf_pipeline.predict.EmotionPredictor")
    @patch("src.emotion_clf_pipeline.predict.time.time")
    @patch("src.emotion_clf_pipeline.predict.logger")  # Mock the logger
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
    @patch("src.emotion_clf_pipeline.predict.logger")
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

        # Test 3: AssemblyAI failure with automatic fallback to Whisper
        mock_env_get.return_value = "test_api_key"
        mock_assembly_transcriber_fail = Mock()
        mock_assembly_transcriber_fail.process.side_effect = Exception("API Error")
        mock_assembly_class.return_value = mock_assembly_transcriber_fail

        # Reset whisper mock for fallback test
        mock_whisper_transcriber_fallback = Mock()
        mock_whisper_class.return_value = mock_whisper_transcriber_fallback

        predict.speech_to_text("assemblyAI", "test_audio.mp3", "test_output.xlsx")

        # Verify AssemblyAI was attempted first
        mock_assembly_class.assert_called_with("test_api_key")
        mock_assembly_transcriber_fail.process.assert_called_with(
            "test_audio.mp3", "test_output.xlsx"
        )

        # Verify fallback to Whisper occurred
        mock_whisper_class.assert_called()
        mock_whisper_transcriber_fallback.process.assert_called_with(
            "test_audio.mp3", "test_output.xlsx"
        )

        # Verify appropriate log messages
        mock_logger.error.assert_called()
        mock_logger.info.assert_any_call("Falling back to Whisper transcription")
        mock_logger.info.assert_any_call("Whisper transcription successful.")

        # Test 4: AssemblyAI missing API key (should not attempt AssemblyAI)
        mock_env_get.return_value = None
        mock_whisper_transcriber_no_key = Mock()
        mock_whisper_class.return_value = mock_whisper_transcriber_no_key

        predict.speech_to_text("assemblyAI", "test_audio.mp3", "test_output.xlsx")

        # Should skip AssemblyAI and go directly to Whisper
        mock_whisper_class.assert_called()
        mock_whisper_transcriber_no_key.process.assert_called_with(
            "test_audio.mp3", "test_output.xlsx"
        )

        # Test 5: Unknown method - Test for ValueError
        with self.assertRaises(ValueError) as context:
            predict.speech_to_text(
                "unknown_method", "test_audio.mp3", "test_output.xlsx"
            )
        self.assertIn(
            "Unknown transcription method: unknown_method", str(context.exception)
        )

        # Test 6: Whisper exception (no fallback available)
        mock_whisper_transcriber_fail = Mock()
        mock_whisper_transcriber_fail.process.side_effect = Exception(
            "Audio file not found"
        )
        mock_whisper_class.return_value = mock_whisper_transcriber_fail

        predict.speech_to_text("whisper", "nonexistent_audio.mp3", "test_output.xlsx")

        # Verify error was logged
        error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
        expected_error = "Error during Whisper transcription: Audio file not found"
        self.assertTrue(
            any(expected_error in call for call in error_calls),
            f"Expected audio file error not found: {error_calls}",
        )

        # Test 7: Case insensitive methods
        mock_whisper_transcriber_case = Mock()
        mock_whisper_class.return_value = mock_whisper_transcriber_case

        for method in ["WHISPER", "Whisper", "wHiSpEr"]:
            predict.speech_to_text(method, "test_audio.mp3", "test_output.xlsx")
            mock_whisper_transcriber_case.process.assert_called_with(
                "test_audio.mp3", "test_output.xlsx"
            )

        # Test 8: Case insensitive AssemblyAI
        mock_env_get.return_value = "test_api_key"  # Reset API key
        mock_assembly_transcriber_case = Mock()
        mock_assembly_class.return_value = mock_assembly_transcriber_case

        for method in ["ASSEMBLYAI", "AssemblyAI", "assemblyai"]:
            predict.speech_to_text(method, "test_audio.mp3", "test_output.xlsx")
            mock_assembly_class.assert_called_with("test_api_key")
            mock_assembly_transcriber_case.process.assert_called_with(
                "test_audio.mp3", "test_output.xlsx"
            )

        # Test 9: Both AssemblyAI and Whisper fail
        mock_env_get.return_value = "test_api_key"
        mock_assembly_fail = Mock()
        mock_assembly_fail.process.side_effect = Exception("AssemblyAI failed")
        mock_assembly_class.return_value = mock_assembly_fail

        mock_whisper_fail = Mock()
        mock_whisper_fail.process.side_effect = Exception("Whisper failed")
        mock_whisper_class.return_value = mock_whisper_fail

        predict.speech_to_text("assemblyAI", "test_audio.mp3", "test_output.xlsx")

        # Verify both were attempted and both failed
        mock_assembly_fail.process.assert_called_with(
            "test_audio.mp3", "test_output.xlsx"
        )
        mock_whisper_fail.process.assert_called_with(
            "test_audio.mp3", "test_output.xlsx"
        )

        # Verify warning message about complete failure
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        self.assertTrue(
            any("Speech-to-Text failed" in call for call in warning_calls),
            f"Expected failure warning not found: {warning_calls}",
        )

        # Test 10: Timing measurement verification
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

            # Use a function to handle column access - Fixed to accept self parameter
            def mock_getitem(self, key):
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
    @patch(
        "src.emotion_clf_pipeline.predict.save_youtube_video"
    )  # Add video download mock
    @patch("src.emotion_clf_pipeline.predict.speech_to_text")
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
        mock_speech_to_text,
        mock_save_youtube_video,
        mock_save_youtube_audio,
        mock_makedirs,
    ):
        """Test process_youtube_url_and_predict scenarios in one comprehensive test"""

        # Test 1: Complete pipeline success (with video download)
        mock_save_youtube_audio.return_value = (
            "/path/to/audio.mp3",
            "Test Video Title",
        )
        mock_save_youtube_video.return_value = (
            "/path/to/video.mp4",
            "Test Video Title",
        )

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
            "https://www.youtube.com/watch?v=test123", "whisper"
        )

        # Verify both audio and video downloads were attempted
        mock_save_youtube_audio.assert_called_once()
        mock_save_youtube_video.assert_called_once()

        # Check that we get the expected structure
        self.assertEqual(len(result), 3)
        for i, item in enumerate(result):
            self.assertEqual(item["text"], sentences[i])
            self.assertEqual(item["start_time"], start_times[i])
            self.assertEqual(item["end_time"], end_times[i])
            self.assertEqual(item["emotion"], expected_predictions[i]["emotion"])
            self.assertEqual(
                item["sub_emotion"], expected_predictions[i]["sub_emotion"]
            )
            self.assertEqual(item["intensity"], expected_predictions[i]["intensity"])

        mock_df.dropna.assert_called_with(subset=["Sentence"])

        # Test 2: Video download fails but audio succeeds
        mock_save_youtube_video.side_effect = Exception("Video download failed")
        mock_save_youtube_audio.side_effect = None  # Reset side effect
        mock_save_youtube_audio.return_value = (
            "/path/to/audio.mp3",
            "Test Video Title",
        )

        result = predict.process_youtube_url_and_predict(
            "https://www.youtube.com/watch?v=test123", "assemblyAI"
        )

        # Should still succeed with audio only
        self.assertEqual(len(result), 3)
        # Verify warning was logged
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        self.assertTrue(
            any("Video download failed" in call for call in warning_calls),
            f"Expected video download warning not found: {warning_calls}",
        )

        # Test 3: Missing required columns
        mock_save_youtube_video.side_effect = None  # Reset
        mock_df_missing_cols = Mock()
        mock_df_missing_cols.dropna.return_value = mock_df_missing_cols
        mock_df_missing_cols.reset_index.return_value = mock_df_missing_cols
        mock_df_missing_cols.columns = ["Text", "Start", "End"]  # Wrong column names
        mock_read_excel.return_value = mock_df_missing_cols

        result = predict.process_youtube_url_and_predict(
            "https://www.youtube.com/watch?v=test123", "assemblyAI"
        )
        self.assertEqual(result, [])  # Should return empty list

        # Test 4: Empty transcript - Mock empty DataFrame
        empty_mock_df = self.create_mock_dataframe([], is_empty=True)
        mock_read_excel.return_value = empty_mock_df

        result = predict.process_youtube_url_and_predict(
            "https://www.youtube.com/watch?v=test123", "whisper"
        )
        self.assertEqual(result, [])

        # Test 5: Prediction returns None
        normal_mock_df = self.create_mock_dataframe(["Test sentence"])
        mock_read_excel.return_value = normal_mock_df
        mock_predict_emotion.return_value = None

        result = predict.process_youtube_url_and_predict(
            "https://www.youtube.com/watch?v=test123", "whisper"
        )
        # Should still return data structure but with "unknown" values
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["emotion"], "unknown")
        self.assertEqual(result[0]["sub_emotion"], "unknown")
        self.assertEqual(result[0]["intensity"], "unknown")

        # Test 6: Invalid URL format
        mock_save_youtube_audio.side_effect = Exception("Invalid URL format")
        with self.assertRaises(Exception) as context:
            predict.process_youtube_url_and_predict("not-a-valid-url", "whisper")
        self.assertIn("Invalid URL format", str(context.exception))


class TestProcessYouTubeURLReal(unittest.TestCase):
    """Alternative approach: Use real pandas DataFrame for testing"""

    @patch("src.emotion_clf_pipeline.predict.os.makedirs")
    @patch("src.emotion_clf_pipeline.predict.save_youtube_audio")
    @patch("src.emotion_clf_pipeline.predict.save_youtube_video")
    @patch("src.emotion_clf_pipeline.predict.speech_to_text")
    @patch("src.emotion_clf_pipeline.predict.pd.read_excel")
    @patch("src.emotion_clf_pipeline.predict.pd.DataFrame.to_excel")
    @patch("src.emotion_clf_pipeline.predict.logger")
    def test_with_real_pandas(
        self,
        mock_logger,
        mock_to_excel,
        mock_read_excel,
        mock_speech_to_text,
        mock_save_youtube_video,
        mock_save_youtube_audio,
        mock_makedirs,
    ):
        """Test using real pandas operations"""

        # Test 1: Success case
        mock_save_youtube_audio.return_value = (
            "/path/to/audio.mp3",
            "Test Video Title",
        )
        mock_save_youtube_video.return_value = (
            "/path/to/video.mp4",
            "Test Video Title",
        )

        # Create real DataFrame with the correct column names that the function expects
        test_df = pd.DataFrame(
            {
                "Sentence": ["I am happy", "This is great", "Wonderful day"],
                "Start Time": [0.0, 1.5, 3.0],
                "End Time": [1.4, 2.9, 4.5],
            }
        )
        mock_read_excel.return_value = test_df

        expected_predictions = [
            {"emotion": "happy", "sub_emotion": "joy", "intensity": "high"},
            {"emotion": "happy", "sub_emotion": "excitement", "intensity": "medium"},
            {"emotion": "happy", "sub_emotion": "contentment", "intensity": "high"},
        ]

        # Mock predict_emotion within the function call
        with patch(
            "src.emotion_clf_pipeline.predict.predict_emotion",
            return_value=expected_predictions,
        ) as mock_predict_emotion:

            result = predict.process_youtube_url_and_predict(
                "https://www.youtube.com/watch?v=test123", "whisper"
            )

            # Verify the result structure
            self.assertEqual(len(result), 3)

            # Check the structure of each result item
            for i, item in enumerate(result):
                self.assertEqual(item["text"], test_df.iloc[i]["Sentence"])
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
        mock_read_excel.return_value = empty_df

        with patch(
            "src.emotion_clf_pipeline.predict.predict_emotion"
        ) as mock_predict_emotion:

            result = predict.process_youtube_url_and_predict(
                "https://www.youtube.com/watch?v=test123", "whisper"
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
        mock_read_excel.return_value = mixed_df

        with patch(
            "src.emotion_clf_pipeline.predict.predict_emotion",
            return_value=expected_predictions,
        ) as mock_predict_emotion:

            result = predict.process_youtube_url_and_predict(
                "https://www.youtube.com/watch?v=test123", "whisper"
            )

            # Should only process the non-None sentences
            mock_predict_emotion.assert_called_once_with(
                ["I am happy", "This is great", "Wonderful day"]
            )
            self.assertEqual(len(result), 3)  # Only non-None sentences

        # Test 4: Audio download failure
        mock_save_youtube_audio.side_effect = Exception("Download failed")

        with self.assertRaises(Exception) as context:
            predict.process_youtube_url_and_predict(
                "https://www.youtube.com/watch?v=invalid", "whisper"
            )

        self.assertIn("Download failed", str(context.exception))

        # Test 5: Video download fails but continues processing
        mock_save_youtube_audio.side_effect = None  # Reset
        mock_save_youtube_audio.return_value = (
            "/path/to/audio.mp3",
            "Test Video Title",
        )
        mock_save_youtube_video.side_effect = Exception("Video failed")

        # Should still work with just audio
        test_df = pd.DataFrame(
            {
                "Sentence": ["Test sentence"],
                "Start Time": [0.0],
                "End Time": [1.0],
            }
        )
        mock_read_excel.return_value = test_df

        with patch(
            "src.emotion_clf_pipeline.predict.predict_emotion",
            return_value=[
                {"emotion": "neutral", "sub_emotion": "calm", "intensity": "low"}
            ],
        ) as mock_predict_emotion:

            result = predict.process_youtube_url_and_predict(
                "https://www.youtube.com/watch?v=test123", "whisper"
            )

            # Should still succeed
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["text"], "Test sentence")


def test_predict_emotions_local():
    """Test predict_emotions_local function with all scenarios."""

    # Test 1: Successful local emotion prediction
    with (
        patch("your_module.transcribe_youtube_url") as mock_transcribe,
        patch("builtins.open", new_callable=mock_open) as mock_file,
        patch("os.path.exists") as mock_exists,
        patch("torch.load") as mock_torch_load,
        patch("your_module.FeatureExtractor") as mock_feature_extractor_class,
        patch("your_module.DEBERTAClassifier") as mock_deberta_class,
        patch("your_module.process_text_chunks") as mock_process_chunks,
        patch("torch.device"),
        patch("torch.cuda.is_available", return_value=False),
    ):  # Replace with actual module path

        # Mock transcript data
        mock_transcript_data = {
            "text": "This is a test transcript with emotional content.",
            "title": "Test Video",
            "source": "subtitles",
        }
        mock_transcribe.return_value = mock_transcript_data

        # Mock config file content
        mock_config = {
            "model_name": "microsoft/deberta-base",
            "feature_dim": 121,
            "num_classes": 8,
            "hidden_dim": 256,
            "dropout": 0.1,
            "feature_config": {
                "pos": False,
                "textblob": False,
                "vader": False,
                "tfidf": True,
                "emolex": True,
            },
        }
        mock_file.return_value.read.return_value = json.dumps(mock_config)
        mock_exists.return_value = True

        # Mock feature extractor and model
        mock_feature_extractor = MagicMock()
        mock_feature_extractor_class.return_value = mock_feature_extractor
        mock_model = MagicMock()
        mock_deberta_class.return_value = mock_model
        mock_torch_load.return_value = {"model_state": "test"}

        # Mock predictions
        mock_predictions = [
            {"emotion": "joy", "confidence": 0.8, "text_chunk": "test chunk"}
        ]
        mock_process_chunks.return_value = mock_predictions

        result = predict_emotions_local(
            video_url="https://youtube.com/watch?v=test",
            model_path="models/weights/baseline_weights.pt",
            config_path="models/weights/model_config.json",
        )

        # Assertions
        assert "predictions" in result
        assert "metadata" in result
        assert "transcript_data" in result
        assert result["metadata"]["model_type"] == "local"
        assert result["metadata"]["stt_used"] is False
        assert len(result["predictions"]) == 1

        mock_transcribe.assert_called_once_with(
            "https://youtube.com/watch?v=test", use_stt=False
        )

    # Test 2: No transcript extracted
    with patch(
        "your_module.transcribe_youtube_url"
    ) as mock_transcribe:  # Replace with actual module path
        mock_transcribe.return_value = {"text": "", "title": "Test"}

        try:
            predict_emotions_local("https://youtube.com/watch?v=test")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "No transcript text extracted" in str(e)

    # Test 3: Config file loading failure
    with (
        patch("your_module.transcribe_youtube_url") as mock_transcribe,
        patch("builtins.open", new_callable=mock_open) as mock_file,
    ):  # Replace with actual module path

        mock_transcribe.return_value = {
            "text": "Test transcript",
            "title": "Test Video",
        }
        mock_file.side_effect = FileNotFoundError("Config file not found")

        try:
            predict_emotions_local("https://youtube.com/watch?v=test")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass  # Expected

    # Test 4: Model loading failure
    with (
        patch("your_module.transcribe_youtube_url") as mock_transcribe,
        patch("builtins.open", new_callable=mock_open) as mock_file,
        patch("os.path.exists") as mock_exists,
        patch("torch.load") as mock_torch_load,
    ):  # Replace with actual module path

        mock_transcribe.return_value = {
            "text": "Test transcript",
            "title": "Test Video",
        }

        mock_config = {
            "model_name": "microsoft/deberta-base",
            "feature_dim": 121,
            "num_classes": 8,
            "hidden_dim": 256,
            "dropout": 0.1,
        }
        mock_file.return_value.read.return_value = json.dumps(mock_config)
        mock_exists.return_value = False
        mock_torch_load.side_effect = Exception("Model file not found")

        try:
            predict_emotions_local("https://youtube.com/watch?v=test")
            assert False, "Should have raised Exception"
        except Exception:
            pass  # Expected

    # Test 5: With STT enabled
    with (
        patch("your_module.transcribe_youtube_url") as mock_transcribe,
        patch("builtins.open", new_callable=mock_open) as mock_file,
        patch("os.path.exists") as mock_exists,
        patch("torch.load") as mock_torch_load,
        patch("your_module.FeatureExtractor") as mock_feature_extractor_class,
        patch("your_module.DEBERTAClassifier") as mock_deberta_class,
        patch("your_module.process_text_chunks") as mock_process_chunks,
        patch("torch.device"),
        patch("torch.cuda.is_available", return_value=False),
    ):  # Replace with actual module path

        mock_transcript_data = {
            "text": "STT extracted text",
            "title": "Test Video",
            "source": "stt_whisper",
        }
        mock_transcribe.return_value = mock_transcript_data

        mock_config = {
            "model_name": "microsoft/deberta-base",
            "feature_dim": 121,
            "num_classes": 8,
            "hidden_dim": 256,
            "dropout": 0.1,
            "feature_config": {"tfidf": True, "emolex": True},
        }
        mock_file.return_value.read.return_value = json.dumps(mock_config)
        mock_exists.return_value = True

        mock_feature_extractor = MagicMock()
        mock_feature_extractor_class.return_value = mock_feature_extractor
        mock_model = MagicMock()
        mock_deberta_class.return_value = mock_model
        mock_torch_load.return_value = {"test": "weights"}
        mock_process_chunks.return_value = [{"emotion": "sadness"}]

        result = predict_emotions_local(
            video_url="https://youtube.com/watch?v=test", use_stt=True
        )

        mock_transcribe.assert_called_once_with(
            "https://youtube.com/watch?v=test", use_stt=True
        )
        assert result["metadata"]["stt_used"] is True

    # Test 6: TF-IDF fitting with short text
    with (
        patch("your_module.transcribe_youtube_url") as mock_transcribe,
        patch("builtins.open", new_callable=mock_open) as mock_file,
        patch("os.path.exists") as mock_exists,
        patch("torch.load") as mock_torch_load,
        patch("your_module.FeatureExtractor") as mock_feature_extractor_class,
        patch("your_module.DEBERTAClassifier") as mock_deberta_class,
        patch(
            "your_module.process_text_chunks", return_value=[]
        ) as mock_process_chunks,
        patch("torch.device"),
        patch("torch.cuda.is_available", return_value=False),
    ):  # Replace with actual module path

        mock_transcript_data = {"text": "Short text", "title": "Test Video"}
        mock_transcribe.return_value = mock_transcript_data

        mock_config = {
            "model_name": "microsoft/deberta-base",
            "feature_dim": 121,
            "num_classes": 8,
            "hidden_dim": 256,
            "dropout": 0.1,
            "feature_config": {"tfidf": True},
        }
        mock_file.return_value.read.return_value = json.dumps(mock_config)
        mock_exists.return_value = True

        mock_feature_extractor = MagicMock()
        mock_feature_extractor_class.return_value = mock_feature_extractor
        mock_model = MagicMock()
        mock_deberta_class.return_value = mock_model
        mock_torch_load.return_value = {}

        predict_emotions_local("https://youtube.com/watch?v=test")

        # Verify TF-IDF fitting was called
        mock_feature_extractor.fit_tfidf.assert_called_once()

    print("All predict_emotions_local tests passed!")


if __name__ == "__main__":
    unittest.main()
