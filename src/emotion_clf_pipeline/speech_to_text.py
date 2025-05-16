
# Import the libraries
import sys
from pytubefix import YouTube
import os
import assemblyai as aai
import pandas as pd
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class SpeechToTextTranscriber:
    def __init__(self, api_key: str):
        """Initialize the transcriber with an API key.
        
        Args:
            api_key: AssemblyAI API key
        """
        self.api_key = api_key
        self._setup_assemblyai()
        
    def _setup_assemblyai(self) -> None:
        """Initialize AssemblyAI with the API key."""
        aai.settings.api_key = self.api_key
        
    def transcribe_audio(self, file_path: str, config: Optional[aai.TranscriptionConfig] = None) -> aai.Transcript:
        """
        Transcribe the audio file using AssemblyAI.
        
        Args:
            file_path: Path to the audio file
            config: Optional transcription configuration
            
        Returns:
            AssemblyAI transcript object
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        transcriber = aai.Transcriber()
        
        try:
            transcript = transcriber.transcribe(file_path, config)
            
            if transcript.status == aai.TranscriptStatus.error:
                raise Exception(f"Transcription failed: {transcript.error}")
                
            return transcript
        
        except Exception as e:
            raise Exception(f"Transcription error: {str(e)}")

    def save_transcript(self, transcript: aai.Transcript, output_file: str) -> None:
        """
        Save the transcript to a file (CSV/Excel) with sentences and timestamps.
        
        Args:
            transcript: AssemblyAI transcript object
            output_file: Path to save the output file
        """
        # Store sentences with their timestamps
        transcript_data = []
        
        try:
            # Use the sentences endpoint to get properly separated sentences
            for sentence in transcript.get_sentences():
                # Convert timestamps from milliseconds to seconds
                start_time = sentence.start / 1000 if sentence.start is not None else 0
                end_time = sentence.end / 1000 if sentence.end is not None else 0
                
                # Format timestamps as HH:MM:SS
                start_formatted = self._format_timestamp(start_time)
                end_formatted = self._format_timestamp(end_time)
                
                transcript_data.append({
                    'Sentence': sentence.text,
                    'Start Time': start_formatted,
                    'End Time': end_formatted
                })
        except:
            # Fallback: Split the full text into sentences (less accurate)
            sentences = [{'Sentence': s.strip(), 
                         'Start Time': '', 
                         'End Time': ''} 
                        for s in transcript.text.split('.') if s.strip()]
            transcript_data = sentences
        
        # Create DataFrame with sentences and timestamps
        df = pd.DataFrame(transcript_data)
        
        # Save based on file extension
        file_ext = output_file.lower().split('.')[-1]
        if file_ext == 'csv':
            df.to_csv(output_file, index=False)
        elif file_ext in ['xlsx', 'xls']:
            df.to_excel(output_file, index=False)
        else:
            raise ValueError("Unsupported output file format. Use .csv or .xlsx")

    def _format_timestamp(self, seconds: float) -> str:
        """
        Format time in seconds to HH:MM:SS format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string in HH:MM:SS format
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def process(self, audio_file: str, output_file: str = "transcribed_data_assemblyAI.xlsx") -> None:
        """
        Process an audio file and save the transcript.
        
        Args:
            audio_file: Path to the input audio file
            output_file: Path for the output transcript file
        """
        try:
            # Configure transcription
            config = aai.TranscriptionConfig(
                punctuate=True,  # Enable punctuation
                format_text=True  # Enable text formatting
            )
            
            # Perform transcription
            print(f"Transcribing {audio_file}...")
            transcript = self.transcribe_audio(audio_file, config)
            
            # Save results
            print(f"Saving transcript to {output_file}...")
            self.save_transcript(transcript, output_file)
            
            print("Transcription completed successfully!")
            
        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)
            raise



"""
Speech to Text Transcription using OpenAI's Whisper Model

This script takes an MP3 file as input and generates a structured output file containing
transcribed sentences with timestamps. It uses the Whisper model for high-accuracy transcription.

Requirements:
    - whisper package (install with: pip install -U openai-whisper)
    - pandas package (install with: pip install pandas)
    - ffmpeg (install with: apt-get install ffmpeg or brew install ffmpeg)
"""

import whisper
import pandas as pd
import sys
import os
import subprocess
from typing import Optional, List, Dict
import torch
import numpy as np


def check_cuda_status():
    """
    Check and print detailed CUDA status information.
    This function helps diagnose CUDA-related issues.
    
    Returns:
        bool: True if CUDA is available and properly configured
    """
    print("\n===== CUDA Status Check =====")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        try:
            device_count = torch.cuda.device_count()
            print(f"CUDA Device Count: {device_count}")
            
            for i in range(device_count):
                print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
                
            print(f"Current CUDA Device: {torch.cuda.current_device()}")
            print(f"CUDA Version: {torch.version.cuda}")
            
            # Test a simple CUDA operation to confirm functionality
            test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
            result = test_tensor * 2
            print("CUDA operation test successful!")
            
            return True
        except Exception as e:
            print(f"CUDA Error: {str(e)}")
            return False
    else:
        print("CUDA is not available. Possible reasons:")
        print("1. NVIDIA GPU drivers are not installed or outdated")
        print("2. CUDA toolkit is not installed or not in PATH")
        print("3. PyTorch was installed without CUDA support")
        print("\nRecommended solutions:")
        print("1. Verify NVIDIA GPU drivers are installed")
        print("2. Install CUDA toolkit (compatible with your PyTorch version)")
        print("3. Reinstall PyTorch with CUDA support")
        return False


class WhisperTranscriber:
    
    def __init__(self, model_size: str = "base", force_cpu: bool = False):
        """
        Initialize the WhisperTranscriber with a specific model size.
        
        Args:
            model_size: Size of the model to use ("tiny", "base", "small", "medium", "large")
            force_cpu: If True, CPU will be used even if CUDA is available
        """
        self.model_size = model_size
        self.force_cpu = force_cpu
        self.device = self._get_device()
        print(f"Using device: {self.device}")
        self.model = self._load_model()
    
    def _get_device(self) -> str:
        """
        Determine which device to use for model inference.
        
        Returns:
            str: "cuda" if CUDA is available and not forced to use CPU, otherwise "cpu"
        """
        if self.force_cpu:
            print("Force CPU mode enabled, using CPU even if CUDA is available")
            return "cpu"
        
        if torch.cuda.is_available():
            # Check if CUDA is working properly
            cuda_ok = check_cuda_status()
            if cuda_ok:
                return "cuda"
            else:
                print("CUDA is available but encountering issues. Falling back to CPU.")
                return "cpu"
        else:
            print("CUDA is not available. Using CPU (this will be slower).")
            return "cpu"

    def _load_model(self) -> whisper.Whisper:
        """
        Load the Whisper model.
        
        Returns:
            Loaded Whisper model
        """
        try:
            print(f"Loading {self.model_size} model on {self.device}...")
            model = whisper.load_model(self.model_size).to(self.device)
            print(f"Model loaded successfully on {self.device}")
            return model
        except Exception as e:
            print(f"Error loading Whisper model: {str(e)}")
            if self.device == "cuda" and "CUDA" in str(e):
                print("Attempting to fall back to CPU...")
                self.device = "cpu"
                model = whisper.load_model(self.model_size).to(self.device)
                print("Model loaded successfully on CPU")
                return model
            else:
                raise Exception(f"Error loading Whisper model: {str(e)}")

    def transcribe_audio(self, file_path: str, language: Optional[str] = None) -> Dict:
        """
        Transcribe the audio file using Whisper.
        
        Args:
            file_path: Path to the audio file
            language: Optional language code (e.g., "en" for English)
            
        Returns:
            Dictionary containing transcription results
        """
        # Convert to absolute path
        file_path = os.path.abspath(file_path)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        try:
            print(f"Using absolute file path: {file_path}")
            # Transcribe with word-level timestamps
            result = self.model.transcribe(file_path, language=language, word_timestamps=True, verbose=False)
            return result
        
        except Exception as e:
            if "ffmpeg" in str(e).lower():
                raise Exception(f"FFMPEG error during transcription. Please make sure FFMPEG is correctly installed: {str(e)}")
            else:
                raise Exception(f"Transcription error: {str(e)}")

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """
        Format time in seconds to HH:MM:SS format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string in HH:MM:SS format
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def extract_sentences(self, result: Dict) -> List[Dict]:
        """
        Extract sentences with timestamps from Whisper transcription result.
        
        Args:
            result: Whisper transcription result
            
        Returns:
            List of dictionaries containing sentences and their timestamps
        """
        transcript_data = []
        
        for segment in result['segments']:
            # Get the text and timestamps
            text = segment['text'].strip()
            start_time = segment['start']
            end_time = segment['end']
            
            # Format timestamps
            start_formatted = self.format_timestamp(start_time)
            end_formatted = self.format_timestamp(end_time)
            
            # Add to transcript data if there's text
            if text:
                transcript_data.append({
                    'Sentence': text,
                    'Start Time': start_formatted,
                    'End Time': end_formatted
                })
        
        return transcript_data

    @staticmethod
    def save_transcript(transcript_data: List[Dict], output_file: str) -> None:
        """
        Save the transcript to a file (CSV/Excel).
        
        Args:
            transcript_data: List of dictionaries containing transcription data
            output_file: Path to save the output file
        """
        # Create DataFrame
        df = pd.DataFrame(transcript_data)
        
        # Save based on file extension
        file_ext = output_file.lower().split('.')[-1]
        if file_ext == 'csv':
            df.to_csv(output_file, index=False)
        elif file_ext in ['xlsx', 'xls']:
            df.to_excel(output_file, index=False)
        else:
            raise ValueError("Unsupported output file format. Use .csv or .xlsx")

    def process(self, audio_file: str, 
               output_file: str = "transcribed_data_whisper.xlsx",
               language: Optional[str] = None) -> None:
        """
        Process the audio file and generate a transcript.
        
        Args:
            audio_file: Path to the input audio file
            output_file: Path for the output transcript file
            language: Optional language code for transcription
        """
        try:
            # Ensure we have absolute file paths
            audio_file = os.path.abspath(audio_file)
            output_file = os.path.abspath(output_file)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Perform transcription
            print(f"Transcribing {audio_file}...")
            result = self.transcribe_audio(audio_file, language)
            
            # Extract sentences with timestamps
            transcript_data = self.extract_sentences(result)
            
            # Save results
            print(f"Saving transcript to {output_file}...")
            self.save_transcript(transcript_data, output_file)
            
            print("Transcription completed successfully!")
            
        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)
            sys.exit(1)


def save_youtube_audio(url, destination, return_path, filename=None):
    """
    Download a YouTube video and save its audio as an MP3 file.
    
    Args:
        url (str): The YouTube video URL
        destination (str): The destination folder for the audio file
        return_path (bool): If True, returns the path to the saved file
        filename (str, optional): Custom filename for the saved audio file (without extension)
        
    Returns:
        str or None: Path to the saved file if return_path is True, otherwise None
    """

    # Remove the file if it already exists
    if filename:
        clean_filename = os.path.splitext(filename)[0]
        existing_file = os.path.join(destination, f"{clean_filename}.mp3")
        if os.path.exists(existing_file):
            os.remove(existing_file)

    # url input from youtube
    yt = YouTube(url)

    # extract only audio
    video = yt.streams.filter(only_audio=True).first()

    # ensure destination directory exists
    if not os.path.exists(destination):
        os.makedirs(destination)

    # download the file
    out_file = video.download(output_path=destination)

    # save the file with custom filename if provided
    base, ext = os.path.splitext(out_file)
    if filename:
        # Remove any extension from the provided filename and ensure it's safe
        clean_filename = os.path.splitext(filename)[0]
        new_file = os.path.join(destination, f"{clean_filename}.mp3")
    else:
        new_file = base + '.mp3'
    
    os.rename(out_file, new_file)
    
    if return_path:
        return new_file
