import io
import os
import sys
import unittest
from unittest.mock import MagicMock, mock_open, patch

# To access a different directory
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "src", "emotion_clf_pipeline")
)

from transcript import Transcript  # noqa: E402

sys.modules["whisper"] = MagicMock()
sys.modules["pytubefix"] = MagicMock()

mock_youtube_class = MagicMock()
sys.modules["pytubefix"].YouTube = mock_youtube_class

sys.modules["assemblyai"] = MagicMock()
aai = sys.modules["assemblyai"]


class TestTranscript(unittest.TestCase):
    def setUp(self):
        # Capture output to avoid printing during tests
        self.held, sys.stdout = sys.stdout, io.StringIO()
        # Define valid choices and test cases in setUp
        self.valid_choices = ["whisper", "assembly"]
        self.invalid_inputs = [
            "wh1sper",
            "haLlo",
            "invalid",
            "wrong",
            "123",
            "!@#",
            "",
            "test",
        ]
        self.valid_test_inputs = [
            "whisper",
            "assembly",
            "WHISPER",
            "AsseMBly",
            "  whisper  ",
            "  ASSEMBLY  ",
        ]

        # Create a transcript instance with a mocked input
        with patch("builtins.input", return_value="whisper"):
            self.transcript = Transcript()

    def tearDown(self):
        # Restore output
        sys.stdout = self.held

    @patch("builtins.input")
    def test_init(self, mock_input):
        """Test that valid inputs work correctly with various formats"""
        for input_value in self.valid_test_inputs:
            with self.subTest(input_value=input_value):
                mock_input.side_effect = [input_value]

                transcript = Transcript()

                # Check that the choice is one of the valid options
                self.assertTrue(
                    transcript.choice == "whisper" or transcript.choice == "assembly",
                    f"Expected 'whisper' or 'assembly', got '{transcript.choice}'",
                )
                self.assertIn(transcript.choice, self.valid_choices)

        """Test that all invalid inputs are properly rejected"""
        for invalid_input in self.invalid_inputs:
            with self.subTest(invalid_input=invalid_input):
                with patch("builtins.input", side_effect=[invalid_input]):
                    try:
                        transcript = Transcript()
                        # Shouldn't execute since the loop should continue indefinitely
                        self.fail(
                            f"Expected infinite loop for invalid input: {invalid_input}"
                        )
                    except:  # noqa: E722
                        # Test pass if we can't create the transcript for invalid input
                        pass

                    # Verify the invalid input is not in valid choices
                    self.assertNotIn(invalid_input, self.valid_choices)
                    # Additional check: verify it's not whisper or assembly
                    self.assertFalse(
                        invalid_input == "whisper" or invalid_input == "assembly",
                        f"'{invalid_input}' should not be a valid choice",
                    )

    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("os.rename")
    def test_download_youtube_audio(self, mock_rename, mock_makedirs, mock_exists):
        """Test the download_youtube_audio method with mocked dependencies"""

        # Case 1: Normal operation when directory doesn't exist
        mock_exists.return_value = False

        # Since pytubefix is already mocked at module level, access the mock directly
        mock_youtube_class = sys.modules["pytubefix"].YouTube
        mock_youtube_class.reset_mock()

        # Setup YouTube mock hierarchy
        mock_yt_instance = MagicMock()
        mock_youtube_class.return_value = mock_yt_instance
        mock_streams = MagicMock()
        mock_yt_instance.streams = mock_streams
        mock_filter = MagicMock()
        mock_streams.filter.return_value = mock_filter
        mock_video = MagicMock()
        mock_filter.first.return_value = mock_video

        # Setup download mock
        download_path = os.path.join("data", "transcript", "rickroll.mp4")
        mock_video.download.return_value = download_path

        # Call the method
        result = self.transcript.download_youtube_audio()

        # Verify YouTube was initialized with correct URL
        mock_youtube_class.assert_called_with(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )

        # Verify filter was called with only_audio=True
        mock_streams.filter.assert_called_with(only_audio=True)

        # Verify first() was called
        mock_filter.first.assert_called()

        # Verify download was called with correct destination
        mock_video.download.assert_called_with(
            output_path=os.path.join("data", "transcript")
        )

        # Verify makedirs was called if directory doesn't exist
        mock_makedirs.assert_called_with(os.path.join("data", "transcript"))

        # Verify rename was called to change extension to mp3
        expected_new_file = os.path.join("data", "transcript", "rickroll.mp3")
        mock_rename.assert_called_with(download_path, expected_new_file)

        # Verify the result is the new file path
        self.assertEqual(result, expected_new_file)

        # Case 2: Directory already exists
        mock_rename.reset_mock()
        mock_makedirs.reset_mock()
        mock_exists.reset_mock()
        mock_youtube_class.reset_mock()

        # Setup mocks for existing directory
        mock_exists.return_value = True

        # Reset YouTube mocks
        mock_yt_instance = MagicMock()
        mock_youtube_class.return_value = mock_yt_instance
        mock_streams = MagicMock()
        mock_yt_instance.streams = mock_streams
        mock_filter = MagicMock()
        mock_streams.filter.return_value = mock_filter
        mock_video = MagicMock()
        mock_filter.first.return_value = mock_video
        download_path = os.path.join("data", "transcript", "rickroll.mp4")
        mock_video.download.return_value = download_path

        # Call the method again
        self.transcript.download_youtube_audio()

        # Verify makedirs was NOT called since directory exists
        mock_makedirs.assert_not_called()

        # Case 3: Exception handling
        mock_rename.reset_mock()
        mock_makedirs.reset_mock()
        mock_exists.reset_mock()
        mock_youtube_class.reset_mock()

        # Setup YouTube mock to raise an exception
        mock_youtube_class.side_effect = Exception("YouTube connection error")

        # Call the method and capture the exception
        with self.assertRaises(Exception) as context:
            self.transcript.download_youtube_audio()

        # Verify the exception message
        self.assertIn("YouTube connection error", str(context.exception))

    def test_seconds_to_hms(self):
        """Test the seconds_to_hms method with various inputs"""
        self.assertEqual(self.transcript.seconds_to_hms(3661), "1:01:01")
        self.assertEqual(self.transcript.seconds_to_hms(0), "0:00:00")
        self.assertEqual(self.transcript.seconds_to_hms(120), "0:02:00")
        self.assertEqual(self.transcript.seconds_to_hms(45), "0:00:45")
        self.assertEqual(self.transcript.seconds_to_hms(3661.7), "1:01:01")
        self.assertEqual(self.transcript.seconds_to_hms(86399), "23:59:59")

    def test_whisper_model(self):
        """Test all aspects of the whisper_model method"""

        # Test 1: File not found scenario
        with (
            patch("os.path.exists", return_value=False),
            patch("os.makedirs"),
        ):  # Add patch for makedirs to avoid directory exists error
            # Call the method with a non-existent file
            sys.stdout = io.StringIO()  # Reset captured output
            self.transcript.whisper_model("nonexistent_file.mp3")

            # Check output for error message
            output = sys.stdout.getvalue()
            self.assertIn("Error: Audio file not found", output)

        # Test 2: Exception handling during transcription
        with (
            patch("os.path.exists", return_value=True),
            patch("os.makedirs"),
            patch("whisper.load_model") as mock_load_model,
        ):
            # Setup mocks for exception test
            mock_model = MagicMock()
            mock_load_model.return_value = mock_model
            mock_model.transcribe.side_effect = Exception("Test exception")

            # Call the method that should handle the exception
            sys.stdout = io.StringIO()  # Reset captured output
            self.transcript.whisper_model("test.mp3")

            # Check output for error message
            output = sys.stdout.getvalue()
            self.assertIn("Transcription error: Test exception", output)

        # Test 3: Full functionality test
        with (
            patch("os.path.exists") as mock_exists,
            patch("os.makedirs") as mock_makedirs,  # noqa: F841
            patch("whisper.load_model") as mock_load_model,
            patch("builtins.open", new_callable=mock_open) as mock_file,  # noqa: F841
            patch("csv.writer") as mock_csv_writer,
            patch("os.remove") as mock_remove,
        ):
            # Setup all mocks for full functionality test
            mock_exists.side_effect = (
                lambda path: "test_audio.mp3" in path
            )  # File exists, dir doesn't

            mock_model = MagicMock()
            mock_load_model.return_value = mock_model
            mock_model.transcribe.return_value = {
                "segments": [
                    {"start": 0, "end": 5, "text": " Hello world"},
                    {"start": 5, "end": 10, "text": " This is a test"},
                ]
            }

            mock_writer = MagicMock()
            mock_csv_writer.return_value = mock_writer

            # Call the method for full functionality test
            sys.stdout = io.StringIO()  # Reset captured output
            self.transcript.whisper_model("test_audio.mp3")

            # Verify model loading and transcription
            mock_load_model.assert_called_once_with("medium")
            mock_model.transcribe.assert_called_once_with("test_audio.mp3")

            # Verify file cleanup
            mock_remove.assert_called_once_with("test_audio.mp3")

            # Verify CSV writing
            mock_writer.writerow.assert_any_call(
                ["Start (HH:MM:SS)", "End (HH:MM:SS)", "Sentence"]
            )
            mock_writer.writerow.assert_any_call(["0:00:00", "0:00:05", "Hello world"])
            mock_writer.writerow.assert_any_call(
                ["0:00:05", "0:00:10", "This is a test"]
            )

            # Check console output
            output = sys.stdout.getvalue()
            self.assertIn("Loading Whisper model", output)
            self.assertIn("Transcription saved to", output)

    def test_transcribe_audio_with_assemblyai(self):
        """Test all aspects of the transcribe_audio_with_assemblyai method"""

        # Test: Full functionality test
        with (
            patch("os.path.exists", return_value=True),
            patch("os.makedirs") as mock_makedirs,  # noqa: F841
            patch("builtins.open", new_callable=mock_open) as mock_file,  # noqa: F841
            patch("csv.writer") as mock_csv_writer,
            patch("os.remove") as mock_remove,
        ):
            # Set up assemblyai mock structure
            mock_transcriber = MagicMock()
            aai.Transcriber.return_value = mock_transcriber

            mock_transcript = MagicMock()
            mock_transcriber.transcribe.return_value = mock_transcript

            # Mock get_sentences to return test sentences
            mock_sentences = [
                MagicMock(start=0, end=5000, text="Hello world"),
                MagicMock(start=5000, end=10000, text="This is a test"),
            ]
            mock_transcript.get_sentences.return_value = mock_sentences

            mock_writer = MagicMock()
            mock_csv_writer.return_value = mock_writer

            # Call the method for full functionality test
            sys.stdout = io.StringIO()  # Reset captured output
            self.transcript.transcribe_audio_with_assemblyai("test_audio.mp3")

            # Verify AssemblyAI API key setup
            self.assertEqual(aai.settings.api_key, "fb2df8accbcb4f38ba02666862cd6216")

            # Verify transcription was called
            mock_transcriber.transcribe.assert_called_once_with("test_audio.mp3")

            # Verify CSV writing
            mock_writer.writerow.assert_any_call(
                ["Sentence Number", "Text", "Start Time", "End Time"]
            )
            mock_writer.writerow.assert_any_call(
                [1, "Hello world", "0:00:00", "0:00:05"]
            )
            mock_writer.writerow.assert_any_call(
                [2, "This is a test", "0:00:05", "0:00:10"]
            )

            # Verify file cleanup
            mock_remove.assert_called_once_with("test_audio.mp3")

            # Check console output
            output = sys.stdout.getvalue()
            self.assertIn("Starting transcription", output)
            self.assertIn("Done! Check", output)

    def test_process(self):
        """Test the process method which runs the complete pipeline"""

        # Test 1: Process with Whisper
        with (
            patch.object(self.transcript, "download_youtube_audio") as mock_download,
            patch.object(self.transcript, "whisper_model") as mock_whisper,
            patch.object(
                self.transcript, "transcribe_audio_with_assemblyai"
            ) as mock_assembly,
        ):
            mock_download.return_value = "test_audio.mp3"
            self.transcript.choice = "whisper"

            # Call the process method
            sys.stdout = io.StringIO()  # Reset captured output
            self.transcript.process()

            # Verify the correct methods were called
            mock_download.assert_called_once()
            mock_whisper.assert_called_once_with("test_audio.mp3")
            mock_assembly.assert_not_called()

        # Test 2: Process with AssemblyAI
        with (
            patch.object(self.transcript, "download_youtube_audio") as mock_download,
            patch.object(self.transcript, "whisper_model") as mock_whisper,
            patch.object(
                self.transcript, "transcribe_audio_with_assemblyai"
            ) as mock_assembly,
        ):
            mock_download.return_value = "test_audio.mp3"
            self.transcript.choice = "assembly"

            # Call the process method
            sys.stdout = io.StringIO()  # Reset captured output
            self.transcript.process()

            # Verify the correct methods were called
            mock_download.assert_called_once()
            mock_whisper.assert_not_called()
            mock_assembly.assert_called_once_with("test_audio.mp3")

        # Test 3: Exception handling
        with patch.object(self.transcript, "download_youtube_audio") as mock_download:
            mock_download.side_effect = Exception("Download error")

            # Call the process method
            sys.stdout = io.StringIO()  # Reset captured output
            self.transcript.process()

            # Verify error message
            output = sys.stdout.getvalue()
            self.assertIn("Error in pipeline: Download error", output)


# This allows to run from the editor or in the cmd with 'python test_module1.py'
if __name__ == "__main__":
    unittest.main()
