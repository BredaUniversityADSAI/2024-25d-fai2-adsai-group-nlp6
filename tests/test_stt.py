import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, call, mock_open, patch

# Mock all dependencies
sys.modules["assemblyai"] = MagicMock()
sys.modules["whisper"] = MagicMock()
sys.modules["pytubefix"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["dotenv"] = MagicMock()

# Create mock classes for pytubefix
mock_youtube_class = MagicMock()
sys.modules["pytubefix"].YouTube = mock_youtube_class

# Create mock for assemblyai
aai = sys.modules["assemblyai"]
aai.settings = MagicMock()
aai.Transcriber = MagicMock()
aai.TranscriptionConfig = MagicMock()
aai.TranscriptStatus = MagicMock()
aai.TranscriptStatus.completed = "completed"
aai.TranscriptStatus.error = "error"

# Create mock for whisper
whisper = sys.modules["whisper"]
whisper.load_model = MagicMock()

# Create mock for torch
torch = sys.modules["torch"]
torch.cuda = MagicMock()
torch.cuda.is_available = MagicMock()
torch.cuda.device_count = MagicMock()
torch.cuda.get_device_name = MagicMock()
torch.cuda.current_device = MagicMock()
torch.version = MagicMock()
torch.version.cuda = "11.8"

# Create mock for dotenv
dotenv = sys.modules["dotenv"]
dotenv.load_dotenv = MagicMock()

# Add the source directory to Python path using relative path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
src_path = os.path.join(project_root, "src", "emotion_clf_pipeline")
sys.path.insert(0, src_path)

# Import the classes and functions to test
from stt import (  # noqa: E402
    SpeechToTextTranscriber,
    WhisperTranscriber,
    check_cuda_status,
    sanitize_filename,
    save_youtube_audio,
    save_youtube_video,
)


class TestSanitizeFilename(unittest.TestCase):
    """Test the sanitize_filename function."""

    def test_sanitize_filename_all_scenarios(self):
        """Test all sanitization scenarios in one comprehensive test."""
        # Test normal filename
        self.assertEqual(sanitize_filename("normal_file"), "normal_file")

        # Test empty/None inputs
        self.assertEqual(sanitize_filename(""), "untitled")
        self.assertEqual(sanitize_filename(None), "untitled")
        self.assertEqual(sanitize_filename("   "), "untitled")

        # Test invalid characters removal
        self.assertEqual(sanitize_filename('file<>:"|?*name'), "file_name")
        self.assertEqual(sanitize_filename("file\x00\x1fname"), "file_name")

        # Test multiple underscores
        self.assertEqual(sanitize_filename("file___name"), "file_name")

        # Test leading/trailing dots and spaces
        self.assertEqual(sanitize_filename("  .file.  "), "file")
        self.assertEqual(sanitize_filename("...file..."), "file")

        # Test Windows reserved names
        self.assertEqual(sanitize_filename("CON"), "_CON")
        self.assertEqual(sanitize_filename("PRN.txt"), "_PRN.txt")
        self.assertEqual(sanitize_filename("COM1"), "_COM1")
        self.assertEqual(sanitize_filename("LPT1"), "_LPT1")

        # Test length truncation
        long_name = "a" * 250
        result = sanitize_filename(long_name, max_length=100)
        self.assertEqual(len(result), 100)

        # Test length truncation with extension
        long_name_with_ext = "a" * 250 + ".txt"
        result = sanitize_filename(long_name_with_ext, max_length=100)
        self.assertEqual(len(result), 100)
        self.assertTrue(result.endswith(".txt"))

        # Test edge cases
        self.assertEqual(sanitize_filename("."), "untitled")
        self.assertEqual(sanitize_filename(".."), "untitled")


class TestSpeechToTextTranscriber(unittest.TestCase):
    """Test the SpeechToTextTranscriber class."""

    def setUp(self):
        """Set up test fixtures."""
        self.api_key = "test_api_key"
        self.transcriber = SpeechToTextTranscriber(self.api_key)

    @patch("stt.aai")
    def test_init_and_setup(self, mock_aai):
        """Test initialization and AssemblyAI setup."""
        transcriber = SpeechToTextTranscriber("test_key")
        mock_aai.settings.api_key = "test_key"
        self.assertEqual(transcriber.api_key, "test_key")

    @patch("stt.aai")
    @patch("os.path.exists")
    def test_transcribe_audio_success(self, mock_exists, mock_aai):
        """Test successful audio transcription."""
        mock_exists.return_value = True
        mock_transcript = Mock()
        mock_transcript.status = mock_aai.TranscriptStatus.completed
        mock_transcriber = Mock()
        mock_transcriber.transcribe.return_value = mock_transcript
        mock_aai.Transcriber.return_value = mock_transcriber

        result = self.transcriber.transcribe_audio("test.mp3")

        self.assertEqual(result, mock_transcript)
        mock_transcriber.transcribe.assert_called_once()

    @patch("stt.aai")
    @patch("os.path.exists")
    def test_transcribe_audio_error(self, mock_exists, mock_aai):
        """Test transcription with error status."""
        mock_exists.return_value = True
        mock_transcript = Mock()
        mock_transcript.status = mock_aai.TranscriptStatus.error
        mock_transcript.error = "Test error"
        mock_transcriber = Mock()
        mock_transcriber.transcribe.return_value = mock_transcript
        mock_aai.Transcriber.return_value = mock_transcriber

        with self.assertRaises(Exception) as context:
            self.transcriber.transcribe_audio("test.mp3")

        self.assertIn("Transcription failed", str(context.exception))

    @patch("os.path.exists")
    def test_transcribe_audio_file_not_found(self, mock_exists):
        """Test transcription with non-existent file."""
        mock_exists.return_value = False

        with self.assertRaises(FileNotFoundError):
            self.transcriber.transcribe_audio("nonexistent.mp3")

    @patch("stt.pd.DataFrame")
    def test_save_transcript_csv(self, mock_dataframe):
        """Test saving transcript to CSV."""
        mock_sentence = Mock()
        mock_sentence.text = "Test sentence."
        mock_sentence.start = 1000
        mock_sentence.end = 3000

        mock_transcript = Mock()
        mock_transcript.get_sentences.return_value = [mock_sentence]

        mock_df = Mock()
        mock_dataframe.return_value = mock_df

        self.transcriber.save_transcript(mock_transcript, "output.csv")

        mock_dataframe.assert_called_once()
        mock_df.to_csv.assert_called_once_with("output.csv", index=False)

    @patch("stt.pd.DataFrame")
    def test_save_transcript_xlsx(self, mock_dataframe):
        """Test saving transcript to Excel."""
        mock_sentence = Mock()
        mock_sentence.text = "Test sentence."
        mock_sentence.start = 1000
        mock_sentence.end = 3000

        mock_transcript = Mock()
        mock_transcript.get_sentences.return_value = [mock_sentence]

        mock_df = Mock()
        mock_dataframe.return_value = mock_df

        self.transcriber.save_transcript(mock_transcript, "output.xlsx")

        mock_dataframe.assert_called_once()
        mock_df.to_excel.assert_called_once_with("output.xlsx", index=False)

    def test_save_transcript_unsupported_format(self):
        """Test saving transcript with unsupported format."""
        # Create mock sentence objects with the required attributes
        mock_sentence1 = Mock()
        mock_sentence1.text = "This is a test sentence."
        mock_sentence1.start = 1000
        mock_sentence1.end = 3000

        mock_sentence2 = Mock()
        mock_sentence2.text = "This is another sentence."
        mock_sentence2.start = 3000
        mock_sentence2.end = 5000

        mock_transcript = Mock()
        mock_transcript.get_sentences.return_value = [mock_sentence1, mock_sentence2]

        with self.assertRaises(ValueError):
            self.transcriber.save_transcript(mock_transcript, "output.txt")

    def test_format_timestamp(self):
        """Test timestamp formatting."""
        self.assertEqual(self.transcriber._format_timestamp(0), "00:00:00")
        self.assertEqual(self.transcriber._format_timestamp(65), "00:01:05")
        self.assertEqual(self.transcriber._format_timestamp(3661), "01:01:01")

    @patch.object(SpeechToTextTranscriber, "transcribe_audio")
    @patch.object(SpeechToTextTranscriber, "save_transcript")
    @patch("stt.aai")
    def test_process_success(self, mock_aai, mock_save, mock_transcribe):
        """Test successful processing."""
        mock_config = Mock()
        mock_aai.TranscriptionConfig.return_value = mock_config
        mock_transcript = Mock()
        mock_transcribe.return_value = mock_transcript

        self.transcriber.process("input.mp3", "output.xlsx")

        mock_transcribe.assert_called_once_with("input.mp3", mock_config)
        mock_save.assert_called_once_with(mock_transcript, "output.xlsx")


class TestWhisperTranscriber(unittest.TestCase):
    """Test the WhisperTranscriber class."""

    @patch("stt.torch")
    @patch("stt.whisper")
    @patch("stt.check_cuda_status")
    def setUp(self, mock_check_cuda, mock_whisper, mock_torch):
        """Set up test fixtures."""
        mock_torch.cuda.is_available.return_value = False
        mock_check_cuda.return_value = False
        mock_model = Mock()
        mock_whisper.load_model.return_value = mock_model
        mock_model.to.return_value = mock_model

        self.transcriber = WhisperTranscriber("base")

    @patch("stt.torch")
    @patch("stt.whisper")
    @patch("stt.check_cuda_status")
    def test_init_with_cuda(self, mock_check_cuda, mock_whisper, mock_torch):
        """Test initialization with CUDA available."""
        mock_torch.cuda.is_available.return_value = True
        mock_check_cuda.return_value = True
        mock_model = Mock()
        mock_whisper.load_model.return_value = mock_model
        mock_model.to.return_value = mock_model

        transcriber = WhisperTranscriber("base")

        self.assertEqual(transcriber.device, "cuda")

    @patch("stt.torch")
    @patch("stt.whisper")
    @patch("stt.check_cuda_status")
    def test_init_force_cpu(self, mock_check_cuda, mock_whisper, mock_torch):
        """Test initialization with forced CPU."""
        mock_torch.cuda.is_available.return_value = True
        mock_check_cuda.return_value = True
        mock_model = Mock()
        mock_whisper.load_model.return_value = mock_model
        mock_model.to.return_value = mock_model

        transcriber = WhisperTranscriber("base", force_cpu=True)

        self.assertEqual(transcriber.device, "cpu")

    @patch("os.path.exists")
    @patch("os.path.abspath")
    def test_transcribe_audio_success(self, mock_abspath, mock_exists):
        """Test successful audio transcription."""
        mock_exists.return_value = True
        mock_abspath.return_value = "/abs/path/test.mp3"

        mock_result = {
            "segments": [{"text": "Test sentence", "start": 0.0, "end": 2.0}]
        }
        self.transcriber.model.transcribe.return_value = mock_result

        result = self.transcriber.transcribe_audio("test.mp3")

        self.assertEqual(result, mock_result)
        self.transcriber.model.transcribe.assert_called_once_with(
            "/abs/path/test.mp3", language=None, word_timestamps=True, verbose=False
        )

    @patch("os.path.exists")
    def test_transcribe_audio_file_not_found(self, mock_exists):
        """Test transcription with non-existent file."""
        mock_exists.return_value = False

        with self.assertRaises(FileNotFoundError):
            self.transcriber.transcribe_audio("nonexistent.mp3")

    def test_format_timestamp_static(self):
        """Test static timestamp formatting method."""
        self.assertEqual(WhisperTranscriber.format_timestamp(0), "00:00:00")
        self.assertEqual(WhisperTranscriber.format_timestamp(65), "00:01:05")
        self.assertEqual(WhisperTranscriber.format_timestamp(3661), "01:01:01")

    def test_extract_sentences(self):
        """Test sentence extraction from Whisper result."""
        mock_result = {
            "segments": [
                {"text": "  First sentence  ", "start": 0.0, "end": 2.5},
                {"text": "Second sentence", "start": 2.5, "end": 5.0},
                {"text": "   ", "start": 5.0, "end": 6.0},  # Empty text
            ]
        }

        result = self.transcriber.extract_sentences(mock_result)

        self.assertEqual(len(result), 2)  # Empty text should be filtered out
        self.assertEqual(result[0]["Sentence"], "First sentence")
        self.assertEqual(result[0]["Start Time"], "00:00:00")
        self.assertEqual(result[0]["End Time"], "00:00:02")
        self.assertEqual(result[1]["Sentence"], "Second sentence")

    @patch("stt.pd.DataFrame")
    def test_save_transcript_static_csv(self, mock_dataframe):
        """Test static save_transcript method for CSV."""
        transcript_data = [
            {"Sentence": "Test", "Start Time": "00:00:00", "End Time": "00:00:02"}
        ]

        mock_df = Mock()
        mock_dataframe.return_value = mock_df

        WhisperTranscriber.save_transcript(transcript_data, "output.csv")

        mock_dataframe.assert_called_once_with(transcript_data)
        mock_df.to_csv.assert_called_once_with("output.csv", index=False)

    @patch("stt.pd.DataFrame")
    def test_save_transcript_static_xlsx(self, mock_dataframe):
        """Test static save_transcript method for Excel."""
        transcript_data = [
            {"Sentence": "Test", "Start Time": "00:00:00", "End Time": "00:00:02"}
        ]

        mock_df = Mock()
        mock_dataframe.return_value = mock_df

        WhisperTranscriber.save_transcript(transcript_data, "output.xlsx")

        mock_dataframe.assert_called_once_with(transcript_data)
        mock_df.to_excel.assert_called_once_with("output.xlsx", index=False)

    def test_save_transcript_static_unsupported(self):
        """Test static save_transcript with unsupported format."""
        transcript_data = [
            {"Sentence": "Test", "Start Time": "00:00:00", "End Time": "00:00:02"}
        ]

        with self.assertRaises(ValueError):
            WhisperTranscriber.save_transcript(transcript_data, "output.txt")

    @patch.object(WhisperTranscriber, "transcribe_audio")
    @patch.object(WhisperTranscriber, "extract_sentences")
    @patch.object(WhisperTranscriber, "save_transcript")
    @patch("os.path.abspath")
    @patch("os.makedirs")
    def test_process_success(
        self, mock_makedirs, mock_abspath, mock_save, mock_extract, mock_transcribe
    ):
        """Test successful processing."""
        mock_abspath.side_effect = lambda x: f"/abs/{x}"
        mock_result = {"segments": []}
        mock_transcript_data = []
        mock_transcribe.return_value = mock_result
        mock_extract.return_value = mock_transcript_data

        self.transcriber.process("input.mp3", "output.xlsx")

        mock_transcribe.assert_called_once_with("/abs/input.mp3", None)
        mock_extract.assert_called_once_with(mock_result)
        mock_save.assert_called_once_with(mock_transcript_data, "/abs/output.xlsx")


class TestCheckCudaStatus(unittest.TestCase):
    """Test the check_cuda_status function."""

    @patch("stt.torch")
    def test_check_cuda_status_available(self, mock_torch):
        """Test CUDA status check when CUDA is available."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_name.return_value = "Test GPU"
        mock_torch.cuda.current_device.return_value = 0
        mock_torch.version.cuda = "11.8"

        result = check_cuda_status()

        self.assertTrue(result)
        mock_torch.cuda.is_available.assert_called_once()
        mock_torch.cuda.device_count.assert_called_once()

    @patch("stt.torch")
    def test_check_cuda_status_not_available(self, mock_torch):
        """Test CUDA status check when CUDA is not available."""
        mock_torch.cuda.is_available.return_value = False

        result = check_cuda_status()

        self.assertFalse(result)
        mock_torch.cuda.is_available.assert_called_once()

    @patch("stt.torch")
    def test_check_cuda_status_exception(self, mock_torch):
        """Test CUDA status check with exception."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.side_effect = Exception("CUDA error")

        result = check_cuda_status()

        self.assertFalse(result)


class TestYouTubeAudioDownload(unittest.TestCase):
    """Test the save_youtube_audio function."""

    @patch("stt.YouTube")
    @patch("stt.sanitize_filename")
    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("os.rename")
    def test_save_youtube_audio_success(
        self, mock_rename, mock_makedirs, mock_exists, mock_sanitize, mock_youtube
    ):
        """Test successful YouTube audio download."""
        mock_exists.side_effect = lambda path: False
        mock_sanitize.return_value = "test_video"

        mock_yt = Mock()
        mock_yt.title = "Test Video"
        mock_youtube.return_value = mock_yt

        mock_stream = Mock()
        mock_stream.download.return_value = "/dest/temp_file.webm"
        mock_yt.streams.filter.return_value.first.return_value = mock_stream

        result_path, title = save_youtube_audio(
            "https://youtube.com/watch?v=test", "/dest"
        )

        expected_path = os.path.normpath("/dest/test_video.mp3")
        actual_path = os.path.normpath(result_path)

        self.assertEqual(actual_path, expected_path)
        self.assertEqual(title, "test_video")
        mock_youtube.assert_called_once_with(
            "https://youtube.com/watch?v=test", use_po_token=False
        )
        mock_stream.download.assert_called_once_with(output_path="/dest")

    @patch("stt.YouTube")
    @patch("stt.sanitize_filename")
    @patch("os.path.exists")
    def test_save_youtube_audio_file_exists(
        self, mock_exists, mock_sanitize, mock_youtube
    ):
        """Test when audio file already exists."""
        mock_sanitize.return_value = "test_video"
        mock_exists.return_value = True

        mock_yt = Mock()
        mock_yt.title = "Test Video"
        mock_youtube.return_value = mock_yt

        result_path, title = save_youtube_audio(
            "https://youtube.com/watch?v=test", "/dest"
        )

        expected_path = os.path.normpath("/dest/test_video.mp3")
        actual_path = os.path.normpath(result_path)

        self.assertEqual(actual_path, expected_path)
        self.assertEqual(title, "test_video")


class TestYouTubeVideoDownload(unittest.TestCase):
    """Test the save_youtube_video function."""

    @patch("stt.YouTube")
    @patch("stt.sanitize_filename")
    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("os.rename")
    def test_save_youtube_video_progressive(
        self, mock_rename, mock_makedirs, mock_exists, mock_sanitize, mock_youtube
    ):
        """Test successful YouTube video download with progressive stream."""
        mock_exists.side_effect = lambda path: False
        mock_sanitize.return_value = "test_video"

        mock_yt = Mock()
        mock_yt.title = "Test Video"
        mock_youtube.return_value = mock_yt

        mock_progressive_stream = Mock()
        mock_progressive_stream.download.return_value = "/dest/temp_file.mp4"
        mock_progressive_stream.resolution = "720p"

        mock_progressive_filter = Mock()
        ordered = mock_progressive_filter.order_by.return_value
        desc_ordered = ordered.desc.return_value
        first_result = desc_ordered.first
        first_result.return_value = mock_progressive_stream

        mock_yt.streams.filter.return_value = mock_progressive_filter

        result_path, title = save_youtube_video(
            "https://youtube.com/watch?v=test", "/dest"
        )

        expected_path = os.path.normpath("/dest/test_video.mp4")
        actual_path = os.path.normpath(result_path)

        self.assertEqual(actual_path, expected_path)
        self.assertEqual(title, "test_video")
        mock_progressive_stream.download.assert_called_once_with(output_path="/dest")

    @patch("stt.YouTube")
    @patch("stt.sanitize_filename")
    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("os.rename")
    def test_save_youtube_video_adaptive_fallback(
        self, mock_rename, mock_makedirs, mock_exists, mock_sanitize, mock_youtube
    ):
        """Test YouTube video download with adaptive stream fallback."""
        mock_exists.side_effect = lambda path: False
        mock_sanitize.return_value = "test_video"

        mock_yt = Mock()
        mock_yt.title = "Test Video"
        mock_youtube.return_value = mock_yt

        # Mock progressive stream not available
        mock_progressive_filter = Mock()
        progressive_ordered = mock_progressive_filter.order_by.return_value
        progressive_desc_ordered = progressive_ordered.desc.return_value
        progressive_first_result = progressive_desc_ordered.first
        progressive_first_result.return_value = None

        # Mock adaptive stream available
        mock_adaptive_stream = Mock()
        mock_adaptive_stream.download.return_value = "/dest/temp_file.mp4"
        mock_adaptive_stream.resolution = "1080p"
        mock_adaptive_filter = Mock()
        adaptive_ordered = mock_adaptive_filter.order_by.return_value
        adaptive_desc_ordered = adaptive_ordered.desc.return_value
        adaptive_first_result = adaptive_desc_ordered.first
        adaptive_first_result.return_value = mock_adaptive_stream

        # Set up filter call sequence
        mock_yt.streams.filter.side_effect = [
            mock_progressive_filter,
            mock_adaptive_filter,
        ]

        result_path, title = save_youtube_video(
            "https://youtube.com/watch?v=test", "/dest"
        )

        expected_path = os.path.normpath("/dest/test_video.mp4")
        actual_path = os.path.normpath(result_path)

        self.assertEqual(actual_path, expected_path)
        self.assertEqual(title, "test_video")
        mock_adaptive_stream.download.assert_called_once_with(output_path="/dest")

    @patch("stt.YouTube")
    @patch("stt.sanitize_filename")
    @patch("os.path.exists")
    def test_save_youtube_video_file_exists(
        self, mock_exists, mock_sanitize, mock_youtube
    ):
        """Test when video file already exists."""
        mock_sanitize.return_value = "test_video"
        mock_exists.return_value = True

        mock_yt = Mock()
        mock_yt.title = "Test Video"
        mock_youtube.return_value = mock_yt

        result_path, title = save_youtube_video(
            "https://youtube.com/watch?v=test", "/dest"
        )

        expected_path = os.path.normpath("/dest/test_video.mp4")
        actual_path = os.path.normpath(result_path)

        self.assertEqual(actual_path, expected_path)
        self.assertEqual(title, "test_video")

    @patch("stt.YouTube")
    @patch("stt.sanitize_filename")
    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_save_youtube_video_no_streams(
        self, mock_makedirs, mock_exists, mock_sanitize, mock_youtube
    ):
        """Test when no suitable video streams are found."""
        mock_exists.side_effect = lambda path: False
        mock_sanitize.return_value = "test_video"

        mock_yt = Mock()
        mock_yt.title = "Test Video"
        mock_youtube.return_value = mock_yt

        # Mock no streams available
        mock_filter = Mock()
        mock_filter.order_by.return_value.desc.return_value.first.return_value = None
        mock_yt.streams.filter.return_value = mock_filter

        with self.assertRaises(Exception) as context:
            save_youtube_video("https://youtube.com/watch?v=test", "/dest")

        self.assertIn("No suitable video streams found", str(context.exception))


if __name__ == "__main__":
    unittest.main()
