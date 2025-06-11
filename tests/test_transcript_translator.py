import io
import os
import sys
import unittest
from unittest.mock import MagicMock, mock_open, patch

# Mock dependecies
sys.modules["whisper"] = MagicMock()
sys.modules["pytubefix"] = MagicMock()
sys.modules["assemblyai"] = MagicMock()
sys.modules["translator"] = MagicMock()

# Create mock classes for pytubefix
mock_youtube_class = MagicMock()
sys.modules["pytubefix"].YouTube = mock_youtube_class

# Create mock for assemblyai
aai = sys.modules["assemblyai"]

# To access a different directory
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "src", "emotion_clf_pipeline")
)

# Now import the module
from transcript_translator import Transcript  # noqa: E402


class TestTranscript(unittest.TestCase):
    def setUp(self):
        # Capture output to avoid printing during tests
        self.held, sys.stdout = sys.stdout, io.StringIO()

        # Mock all user inputs for initialization
        with patch("builtins.input") as mock_input:
            mock_input.side_effect = [
                "youtube",  # input_type
                "whisper",  # transcription service
                "english",  # language
            ]
            self.transcript = Transcript()

    def tearDown(self):
        # Restore output
        sys.stdout = self.held

    def test_get_input_type_valid_inputs(self):
        """Test get_input_type with valid inputs"""
        valid_inputs = ["youtube", "video", "csv", "YOUTUBE", "Video", "CSV"]
        expected_outputs = ["youtube", "video", "csv", "youtube", "video", "csv"]

        for input_val, expected in zip(valid_inputs, expected_outputs):
            with patch("builtins.input", return_value=input_val):
                result = self.transcript.get_input_type()
                self.assertEqual(result, expected)

    def test_get_input_type_invalid_then_valid(self):
        """Test get_input_type with invalid input followed by valid input"""
        with patch("builtins.input", side_effect=["invalid", "bad", "youtube"]):
            result = self.transcript.get_input_type()
            self.assertEqual(result, "youtube")

    @patch("builtins.input")
    def test_init_youtube_whisper_english(self, mock_input):
        """Test initialization with YouTube, Whisper, English"""
        mock_input.side_effect = ["youtube", "whisper", "english"]

        transcript = Transcript()

        self.assertEqual(transcript.input_type, "youtube")
        self.assertEqual(transcript.choice, "whisper")
        self.assertEqual(transcript.language, "english")
        self.assertFalse(transcript.translate_to_english)

    @patch("builtins.input")
    def test_init_video_assembly_french(self, mock_input):
        """Test initialization with Video, AssemblyAI, French"""
        mock_input.side_effect = ["video", "assembly", "french"]

        transcript = Transcript()

        self.assertEqual(transcript.input_type, "video")
        self.assertEqual(transcript.choice, "assembly")
        self.assertEqual(transcript.language, "french")
        self.assertTrue(transcript.translate_to_english)

    @patch("builtins.input")
    def test_init_csv_spanish(self, mock_input):
        """Test initialization with CSV, Spanish"""
        mock_input.side_effect = ["csv", "spanish"]

        transcript = Transcript()

        self.assertEqual(transcript.input_type, "csv")
        self.assertIsNone(transcript.choice)  # No transcription service for CSV
        self.assertEqual(transcript.language, "spanish")
        self.assertTrue(transcript.translate_to_english)

    @patch("builtins.input")
    def test_verify_youtube_link_success(self, mock_input):
        """Test successful YouTube link verification"""
        mock_input.side_effect = [
            "https://www.youtube.com/watch?v=test",  # URL input
            "yes",  # Confirm video is correct
        ]

        # Create a fresh mock YouTube object for this test
        mock_yt = MagicMock()
        mock_yt.title = "Test Video"
        mock_yt.length = 180  # 3 minutes

        # Use the existing mock_youtube_class that's already set up
        mock_youtube_class.reset_mock()
        mock_youtube_class.return_value = mock_yt
        mock_youtube_class.side_effect = None  # Ensure no exception is raised

        result = self.transcript.verify_youtube_link()

        # Verify the YouTube constructor was called with correct URL
        mock_youtube_class.assert_called_with("https://www.youtube.com/watch?v=test")

        # Verify we got the expected result
        self.assertEqual(result, mock_yt)

    @patch("builtins.input")
    def test_verify_youtube_link_user_rejects(self, mock_input):
        """Test YouTube link verification when user rejects the video"""
        mock_input.side_effect = [
            "https://www.youtube.com/watch?v=test1",  # URL input
            "no",  # Reject video
            "no",  # Don't try another URL
        ]

        # Mock YouTube object
        mock_yt = MagicMock()
        mock_yt.title = "Wrong Video"
        mock_yt.length = 120
        mock_youtube_class.return_value = mock_yt

        result = self.transcript.verify_youtube_link()

        self.assertIsNone(result)

    @patch("builtins.input")
    def test_verify_youtube_link_exception_handling(self, mock_input):
        """Test YouTube link verification with connection error"""
        mock_input.side_effect = [
            "https://www.youtube.com/watch?v=invalid",  # URL input
            "no",  # Don't try another URL
        ]

        # Mock YouTube to raise exception
        mock_youtube_class.side_effect = Exception("Connection error")

        result = self.transcript.verify_youtube_link()

        self.assertIsNone(result)

    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("os.rename")
    def test_download_youtube_audio_with_yt_object(
        self, mock_rename, mock_makedirs, mock_exists
    ):
        """Test download_youtube_audio with provided YouTube object"""

        # Mock os.path.exists to return appropriate values
        def exists_side_effect(path):
            # Directory doesn't exist initially
            if path == os.path.join("data", "transcript"):
                return False
            # Downloaded file exists (before rename)
            elif "Test Video.mp4" in path:
                return True
            # Renamed file exists (after rename)
            elif "Test Video.mp3" in path:
                return True
            return False

        mock_exists.side_effect = exists_side_effect

        # Create mock YouTube object
        mock_yt = MagicMock()
        mock_yt.title = "Test Video"
        mock_streams = MagicMock()
        mock_yt.streams = mock_streams
        mock_filter = MagicMock()
        mock_streams.filter.return_value = mock_filter
        mock_video = MagicMock()
        mock_filter.first.return_value = mock_video

        # Mock the download to return a path with .mp4 extension
        download_path = os.path.join("data", "transcript", "Test Video.mp4")
        mock_video.download.return_value = download_path

        result = self.transcript.download_youtube_audio(mock_yt)

        # Verify the correct methods were called
        mock_streams.filter.assert_called_with(only_audio=True)
        mock_filter.first.assert_called()
        mock_video.download.assert_called_with(
            output_path=os.path.join("data", "transcript")
        )

        # Check that rename was called and result is the renamed file
        expected_new_file = os.path.join("data", "transcript", "Test Video.mp3")
        mock_rename.assert_called_with(download_path, expected_new_file)
        self.assertEqual(result, expected_new_file)

    @patch("builtins.input")
    @patch("os.path.exists")
    def test_get_video_file_path_success(self, mock_exists, mock_input):
        """Test successful video file path input"""
        mock_input.return_value = "/path/to/video.mp4"
        mock_exists.return_value = True

        result = self.transcript.get_video_file_path()

        self.assertEqual(result, "/path/to/video.mp4")

    @patch("builtins.input")
    @patch("os.path.exists")
    def test_get_video_file_path_file_not_found(self, mock_exists, mock_input):
        """Test video file path input when file doesn't exist"""
        mock_input.side_effect = ["/nonexistent/video.mp4", "no"]
        mock_exists.return_value = False

        result = self.transcript.get_video_file_path()

        self.assertIsNone(result)

    @patch("builtins.input")
    @patch("os.path.exists")
    def test_get_csv_file_path_success(self, mock_exists, mock_input):
        """Test successful CSV file path input"""
        mock_input.return_value = "/path/to/data.csv"
        mock_exists.return_value = True

        result = self.transcript.get_csv_file_path()

        self.assertEqual(result, "/path/to/data.csv")

    @patch("subprocess.run")
    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_convert_video_to_audio_success(self, mock_makedirs, mock_exists, mock_run):
        """Test successful video to audio conversion"""
        mock_exists.return_value = False
        mock_run.return_value.returncode = 0

        result = self.transcript.convert_video_to_audio("/path/to/video.mp4")

        expected_path = os.path.join("data", "transcript", "video.mp3")
        self.assertEqual(result, expected_path)
        mock_makedirs.assert_called_once()

    @patch("subprocess.run")
    def test_convert_video_to_audio_ffmpeg_error(self, mock_run):
        """Test video to audio conversion with ffmpeg error"""
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "ffmpeg error"

        result = self.transcript.convert_video_to_audio("/path/to/video.mp4")

        self.assertIsNone(result)

    def test_process_csv_translation_no_translation_needed(self):
        """Test CSV processing when no translation is needed"""
        self.transcript.translate_to_english = False

        sys.stdout = io.StringIO()
        self.transcript.process_csv_translation("/path/to/file.csv")

        output = sys.stdout.getvalue()
        self.assertIn("No translation needed", output)

    @patch("translator.HuggingFaceTranslator")
    def test_process_csv_translation_with_translation(self, mock_translator_class):
        """Test CSV processing with translation"""
        self.transcript.translate_to_english = True
        self.transcript.language = "french"

        # Mock translator
        mock_translator = MagicMock()
        mock_translator_class.return_value = mock_translator
        mock_translator.translate_csv.return_value = "/path/to/translated.csv"

        sys.stdout = io.StringIO()
        self.transcript.process_csv_translation("/path/to/file.csv")

        mock_translator_class.assert_called_with("french")
        mock_translator.translate_csv.assert_called_with("/path/to/file.csv")

    def test_seconds_to_hms(self):
        """Test seconds to HH:MM:SS conversion"""
        self.assertEqual(self.transcript.seconds_to_hms(3661), "1:01:01")
        self.assertEqual(self.transcript.seconds_to_hms(0), "0:00:00")
        self.assertEqual(self.transcript.seconds_to_hms(120), "0:02:00")
        self.assertEqual(self.transcript.seconds_to_hms(45), "0:00:45")

    def test_get_whisper_language_code(self):
        """Test Whisper language code mapping"""
        self.transcript.language = "english"
        self.assertEqual(self.transcript.get_whisper_language_code(), "en")

        self.transcript.language = "french"
        self.assertEqual(self.transcript.get_whisper_language_code(), "fr")

        self.transcript.language = "spanish"
        self.assertEqual(self.transcript.get_whisper_language_code(), "es")

    def test_get_assemblyai_language_code(self):
        """Test AssemblyAI language code mapping"""
        self.transcript.language = "english"
        self.assertEqual(self.transcript.get_assemblyai_language_code(), "en")

        self.transcript.language = "french"
        self.assertEqual(self.transcript.get_assemblyai_language_code(), "fr")

        self.transcript.language = "spanish"
        self.assertEqual(self.transcript.get_assemblyai_language_code(), "es")

    @patch("subprocess.run")
    def test_check_ffmpeg_availability_success(self, mock_run):
        """Test ffmpeg availability check - success"""
        mock_run.return_value.returncode = 0

        result = self.transcript.check_ffmpeg_availability()

        self.assertTrue(result)

    @patch("subprocess.run")
    def test_check_ffmpeg_availability_failure(self, mock_run):
        """Test ffmpeg availability check - failure"""
        mock_run.side_effect = FileNotFoundError()

        result = self.transcript.check_ffmpeg_availability()

        self.assertFalse(result)

    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("os.remove")
    @patch("os.path.getsize")
    @patch("builtins.open", new_callable=mock_open)
    @patch("csv.writer")
    @patch("whisper.load_model")
    def test_whisper_model_english_no_translation(
        self,
        mock_load_model,
        mock_csv_writer,
        mock_file,
        mock_getsize,
        mock_remove,
        mock_makedirs,
        mock_exists,
    ):
        """Test Whisper model with English language (no translation)"""
        # Setup
        self.transcript.language = "english"
        self.transcript.translate_to_english = False
        self.transcript.input_type = "youtube"

        mock_exists.side_effect = lambda path: "test_audio.mp3" in path
        mock_getsize.return_value = 1024 * 1024  # 1MB file size
        mock_load_model.return_value.transcribe.return_value = {
            "segments": [
                {"start": 0, "end": 5, "text": " Hello world"},
                {"start": 5, "end": 10, "text": " This is a test"},
            ]
        }

        mock_writer = MagicMock()
        mock_csv_writer.return_value = mock_writer

        with patch.object(
            self.transcript, "check_ffmpeg_availability", return_value=True
        ):
            sys.stdout = io.StringIO()
            self.transcript.whisper_model("test_audio.mp3")

        # Verify CSV header for English
        mock_writer.writerow.assert_any_call(
            ["Start (HH:MM:SS)", "End (HH:MM:SS)", "Sentence"]
        )
        mock_writer.writerow.assert_any_call(["0:00:00", "0:00:05", "Hello world"])

    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("os.remove")
    @patch("os.path.getsize")
    @patch("builtins.open", new_callable=mock_open)
    @patch("csv.writer")
    @patch("whisper.load_model")
    @patch("translator.HuggingFaceTranslator")
    def test_whisper_model_french_with_translation(
        self,
        mock_translator_class,
        mock_load_model,
        mock_csv_writer,
        mock_file,
        mock_getsize,
        mock_remove,
        mock_makedirs,
        mock_exists,
    ):
        """Test Whisper model with French language (with translation)"""
        # Setup
        self.transcript.language = "french"
        self.transcript.translate_to_english = True
        self.transcript.input_type = "youtube"

        mock_exists.side_effect = lambda path: "test_audio.mp3" in path
        mock_getsize.return_value = 1024 * 1024  # 1MB file size
        mock_load_model.return_value.transcribe.return_value = {
            "segments": [{"start": 0, "end": 5, "text": " Bonjour monde"}]
        }

        # Mock translator
        mock_translator = MagicMock()
        mock_translator_class.return_value = mock_translator
        mock_translator.translate.return_value = "Hello world"

        mock_writer = MagicMock()
        mock_csv_writer.return_value = mock_writer

        with patch.object(
            self.transcript, "check_ffmpeg_availability", return_value=True
        ):
            sys.stdout = io.StringIO()
            self.transcript.whisper_model("test_audio.mp3")

        # Verify translation was called
        mock_translator.translate.assert_called_with("Bonjour monde")

        # Verify CSV includes both original and translated text
        mock_writer.writerow.assert_any_call(
            ["Start (HH:MM:SS)", "End (HH:MM:SS)", "Sentence", "Text_french"]
        )

    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("os.remove")
    @patch("builtins.open", new_callable=mock_open)
    @patch("csv.writer")
    def test_transcribe_audio_with_assemblyai_english(
        self, mock_csv_writer, mock_file, mock_remove, mock_makedirs, mock_exists
    ):
        """Test AssemblyAI transcription with English"""
        # Setup
        self.transcript.language = "english"
        self.transcript.translate_to_english = False
        self.transcript.input_type = "youtube"

        mock_exists.return_value = True

        # Mock AssemblyAI
        mock_transcriber = MagicMock()
        aai.Transcriber.return_value = mock_transcriber
        mock_transcript = MagicMock()
        mock_transcriber.transcribe.return_value = mock_transcript

        mock_sentences = [
            MagicMock(start=0, end=5000, text="Hello world"),
            MagicMock(start=5000, end=10000, text="This is a test"),
        ]
        mock_transcript.get_sentences.return_value = mock_sentences

        mock_writer = MagicMock()
        mock_csv_writer.return_value = mock_writer

        sys.stdout = io.StringIO()
        self.transcript.transcribe_audio_with_assemblyai("test_audio.mp3")

        # Verify correct header for English
        mock_writer.writerow.assert_any_call(
            ["Sentence Number", "Text", "Start Time", "End Time"]
        )

    @patch.object(Transcript, "verify_youtube_link")
    @patch.object(Transcript, "download_youtube_audio")
    @patch.object(Transcript, "whisper_model")
    def test_process_youtube_whisper_success(
        self, mock_whisper, mock_download, mock_verify
    ):
        """Test full process with YouTube and Whisper - success path"""
        # Setup
        self.transcript.input_type = "youtube"
        self.transcript.choice = "whisper"

        mock_yt = MagicMock()
        mock_verify.return_value = mock_yt
        mock_download.return_value = "test_audio.mp3"

        sys.stdout = io.StringIO()
        self.transcript.process()

        # Verify the flow
        mock_verify.assert_called_once()
        mock_download.assert_called_once_with(mock_yt)
        mock_whisper.assert_called_once_with("test_audio.mp3")

    @patch.object(Transcript, "get_video_file_path")
    @patch.object(Transcript, "convert_video_to_audio")
    @patch.object(Transcript, "transcribe_audio_with_assemblyai")
    def test_process_video_assembly_success(
        self, mock_assembly, mock_convert, mock_get_path
    ):
        """Test full process with Video and AssemblyAI - success path"""
        # Setup
        self.transcript.input_type = "video"
        self.transcript.choice = "assembly"

        mock_get_path.return_value = "/path/to/video.mp4"
        mock_convert.return_value = "converted_audio.mp3"

        sys.stdout = io.StringIO()
        self.transcript.process()

        # Verify the flow
        mock_get_path.assert_called_once()
        mock_convert.assert_called_once_with("/path/to/video.mp4")
        mock_assembly.assert_called_once_with("converted_audio.mp3")

    @patch.object(Transcript, "get_csv_file_path")
    @patch.object(Transcript, "process_csv_translation")
    def test_process_csv_success(self, mock_process_csv, mock_get_path):
        """Test full process with CSV - success path"""
        # Setup
        self.transcript.input_type = "csv"

        mock_get_path.return_value = "/path/to/data.csv"

        sys.stdout = io.StringIO()
        self.transcript.process()

        # Verify the flow
        mock_get_path.assert_called_once()
        mock_process_csv.assert_called_once_with("/path/to/data.csv")

    @patch.object(Transcript, "verify_youtube_link")
    @patch.object(Transcript, "initialize_settings")
    def test_process_youtube_verification_failure_retry(self, mock_init, mock_verify):
        """Test process with YouTube verification failure and retry"""
        self.transcript.input_type = "youtube"

        # First call fails, second call succeeds
        mock_verify.side_effect = [None, MagicMock()]

        # Mock initialize_settings to change input_type to break the loop
        def side_effect():
            self.transcript.input_type = "csv"  # Change to break loop

        mock_init.side_effect = side_effect

        # Mock CSV processing to avoid further issues
        with (
            patch.object(
                self.transcript, "get_csv_file_path", return_value="/test.csv"
            ),
            patch.object(self.transcript, "process_csv_translation"),
        ):

            sys.stdout = io.StringIO()
            self.transcript.process()

        # Verify initialize_settings was called when verification failed
        mock_init.assert_called_once()


if __name__ == "__main__":
    unittest.main()
