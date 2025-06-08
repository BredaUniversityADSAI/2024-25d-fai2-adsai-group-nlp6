#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A complete end-to-end pipeline for extracting and analyzing emotional content
from YouTube videos. This module orchestrates the entire workflow from audio
extraction to emotion classification, providing a streamlined interface for
sentiment analysis research and applications.

The pipeline supports multiple transcription services with automatic fallback
mechanisms to ensure robustness in production environments.

Usage:
    python predict.py "https://youtube.com/watch?v=..." --transcription_method whisper
"""

# Standard library imports for core functionality
import argparse
import logging
import os
import time
import warnings

# Third-party imports for data processing and external services
import pandas as pd
from dotenv import load_dotenv
from pytubefix import YouTube

# Import domain-specific modules for pipeline components
from .model import EmotionPredictor
from .stt import SpeechToTextTranscriber, WhisperTranscriber, save_youtube_audio

# Silence non-critical warnings to improve user experience
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Configure structured logging for better debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Establish project directory structure for consistent file operations
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))  # Project root
DATA_DIR = os.path.join(BASE_DIR, "data")


def time_str_to_seconds(time_str):
    """
    Convert time strings to seconds for numerical operations.

    Handles multiple time formats commonly found in transcription outputs:
    - HH:MM:SS or HH:MM:SS.mmm (hours, minutes, seconds with optional milliseconds)
    - MM:SS or MM:SS.mmm (minutes, seconds with optional milliseconds)
    - Numeric values (already in seconds)

    This conversion is essential for temporal analysis and synchronization
    between audio timestamps and emotion predictions.

    Args:
        time_str: Time in string format or numeric value

    Returns:
        float: Time converted to seconds, or 0.0 if parsing fails
          Note:
        Returns 0.0 for invalid inputs rather than raising exceptions
        to maintain pipeline robustness during batch processing.
    """

    # Return early if time_str is already numeric
    if isinstance(time_str, (int, float)):
        return float(time_str)

    # Error handling
    try:

        # Split time string into components
        parts = str(time_str).split(":")

        # Handle HH:MM:SS or MM:SS formats with optional milliseconds
        if len(parts) == 3:
            h, m, s_parts = parts
            s_and_ms = s_parts.split(".")
            s = float(s_and_ms[0])
            ms = float(s_and_ms[1]) / 1000.0 if len(s_and_ms) > 1 else 0.0
            return float(h) * 3600 + float(m) * 60 + s + ms

        # Handle MM:SS format with optional milliseconds
        elif len(parts) == 2:
            m, s_parts = parts
            s_and_ms = s_parts.split(".")
            s = float(s_and_ms[0])
            ms = float(s_and_ms[1]) / 1000.0 if len(s_and_ms) > 1 else 0.0
            return float(m) * 60 + s + ms

        # Handle numeric values directly
        else:
            return float(time_str)

    # Catch parsing errors and log warnings
    except ValueError:
        logger.warning(f"Could not parse time string: {time_str}. Returning 0.0")
        return 0.0


def get_video_title(youtube_url: str) -> str:
    """
    Extract video title from YouTube URL for meaningful file naming.

    Video titles serve as natural identifiers for organizing processing
    results and enable easy correlation between source content and
    analysis outputs.

    Args:
        youtube_url: Valid YouTube video URL

    Returns:
        str: Video title or "Unknown Title" if extraction fails

    Note:
        Gracefully handles network errors and invalid URLs to prevent
        pipeline interruption during batch processing scenarios.
    """
    # Initialize YouTube object and fetch video title
    try:
        yt = YouTube(youtube_url)
        return yt.title

    # Handle exceptions during title extraction
    except Exception as e:
        logger.error(f"Error fetching YouTube video title for {youtube_url}: {e}")
        return "Unknown Title"


def predict_emotion(texts, feature_config=None, reload_model=False):
    """
    Apply emotion classification to text using trained models.

    This function serves as the core intelligence of the pipeline,
    transforming raw text into structured emotional insights. It supports
    both single text analysis and batch processing for efficiency.

    Args:
        texts: Single text string or list of texts for analysis
        feature_config: Optional configuration for feature extraction methods
        reload_model: Force model reinitialization (useful for memory management)

    Returns:
        dict or list: Emotion predictions with confidence scores.
                     Single dict for one text, list of dicts for multiple texts.
                     Returns None if prediction fails.

    Performance:
        Logs processing latency for performance monitoring and optimization.
    """

    # Initialize emotion predictor with optional model reload
    _emotion_predictor = EmotionPredictor()

    # Start timer
    start = time.time()

    # Error handling
    try:

        # Predict emotion for single text or batch
        output = _emotion_predictor.predict(texts, feature_config, reload_model)

        # End timer
        end = time.time()
        logger.info(f"Latency (Emotion Classification): {end - start:.2f} seconds")

        return output

    # Catch exceptions during prediction
    except Exception as e:
        logger.error(f"Error in emotion prediction: {str(e)}")
        return None


def speech_to_text(transcription_method, audio_file, output_file):
    """
    Convert audio to text using configurable transcription services.

    Implements a robust transcription strategy with automatic fallback:
    - Primary: AssemblyAI (cloud-based, high accuracy)
    - Fallback: Whisper (local processing, privacy-preserving)

    This dual-service approach ensures pipeline reliability even when
    external services are unavailable or API limits are reached.

    Args:
        transcription_method: "assemblyAI" or "whisper" for primary service
        audio_file: Path to input audio file
        output_file: Path where transcript will be saved

    Raises:
        ValueError: If transcription_method is not recognized

    Note:
        AssemblyAI failures trigger automatic Whisper fallback.
        All transcription attempts are logged for debugging purposes.
    """

    # Start timer
    start = time.time()

    # Initialize variables
    transcription_successful = False
    method_used = transcription_method.lower()

    # If "AssemblyAI" is specified
    if method_used == "assemblyai":

        # Error handling
        try:

            # Load AssemblyAI API key
            load_dotenv(dotenv_path="/app/.env", override=True)
            api_key = os.environ.get("ASSEMBLYAI_API_KEY")

            # Transcribe using AssemblyAI
            transcriber = SpeechToTextTranscriber(api_key)
            transcriber.process(audio_file, output_file)
            transcription_successful = True
            logger.info("AssemblyAI transcription successful.")

        # Catch exceptions during AssemblyAI transcription
        except Exception as e_assembly:
            logger.error(f"Error: AssemblyAI transcription: {str(e_assembly)}.")

        # Automatic fallback mechanism for reliability
        if not transcription_successful:
            method_used = "whisper"
            logger.info("Falling back to Whisper transcription")

    # If "Whisper" is specified or fallback is triggered
    if method_used == "whisper":

        # Error handling
        try:

            # Transcribe using Whisper
            transcriber = WhisperTranscriber()
            transcriber.process(audio_file, output_file)
            transcription_successful = True
            logger.info("Whisper transcription successful.")

        # Catch exceptions during Whisper transcription
        except Exception as e_whisper:
            logger.error(f"Error during Whisper transcription: {str(e_whisper)}")

    # If transcription method is not recognized
    if method_used not in ["assemblyai", "whisper"]:
        logger.error(f"Unknown transcription method: {transcription_method}")
        raise ValueError(f"Unknown transcription method: {transcription_method}")

    # End timer
    end = time.time()

    # Show latency metrics if transcription was successful
    if transcription_successful:
        logger.info(f"Latency (Speech-to-Text with {method_used}): {end - start:.2f} s")
    else:
        logger.warning(
            f"Speech-to-Text failed for '{transcription_method}' "
            f"(and fallback if applicable) after {end - start:.2f} s."
        )


def process_youtube_url_and_predict(
    youtube_url: str, transcription_method: str
) -> list[dict]:
    """
    Execute the complete emotion analysis pipeline for a YouTube video.
      This is the main orchestration function that coordinates all pipeline stages:
    1. Audio extraction from YouTube (with title metadata)
    2. Speech-to-text transcription (with fallback mechanisms)
    3. Emotion classification (with temporal alignment)
    4. Results persistence (structured Excel output)

    The function maintains data lineage throughout the process, ensuring
    that timestamps from transcription are preserved and aligned with
    emotion predictions for temporal analysis capabilities.

    Args:
        youtube_url: Valid YouTube video URL for processing
        transcription_method: "assemblyAI" or "whisper" for speech recognition

    Returns:
        list[dict]: Structured emotion analysis results where each dictionary
                   contains temporal and emotional metadata:
                   - start_time/end_time: Temporal boundaries of the segment
                   - text: Transcribed speech content
                   - emotion/sub_emotion: Classified emotional states
                   - intensity: Emotional intensity measurement

    Returns empty list if essential processing steps fail.

    Note:
        Creates necessary output directories automatically.
        All intermediate and final results are persisted to disk
        for reproducibility and further analysis.
    """
    # Initialize directories
    youtube_audio_dir = os.path.join(BASE_DIR, "results", "audio")
    transcripts_dir = os.path.join(BASE_DIR, "results", "transcript")
    results_dir = os.path.join(BASE_DIR, "results", "predictions")

    # Make sure output directories exist
    os.makedirs(youtube_audio_dir, exist_ok=True)
    os.makedirs(transcripts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # STAGE 1: Audio Extraction with Metadata
    logger.info("*" * 50)
    logger.info("Step 1 - Downloading YouTube audio...")

    actual_audio_path, title = save_youtube_audio(
        url=youtube_url,
        destination=youtube_audio_dir,
    )
    logger.info(f"Audio file saved at: {actual_audio_path}")
    logger.info("YouTube audio downloaded successfully!")

    # STAGE 2: Speech Recognition with Fallback Strategy
    logger.info("*" * 50)
    logger.info("Step 2 - Transcribing audio...")
    transcript_output_file = os.path.join(
        transcripts_dir,
        f"{title}.xlsx",
    )
    speech_to_text(transcription_method, actual_audio_path, transcript_output_file)

    # Load and validate transcript structure
    df = pd.read_excel(transcript_output_file)
    df = df.dropna(subset=["Sentence"])  # Remove empty sentences
    df = df.reset_index(drop=True)  # Clean up row indices

    # Initialize results container
    sentences_data = []

    # Validate that transcript has required columns for processing
    if (
        "Sentence" in df.columns
        and "Start Time" in df.columns
        and "End Time" in df.columns
    ):
        # Extract structured data from transcript
        sentences = df["Sentence"].tolist()
        start_times_str = df["Start Time"].tolist()
        end_times_str = df["End Time"].tolist()

        logger.info(
            f"Transcription completed successfully! "
            f"Found {len(sentences)} sentences."
        )

        # STAGE 3: Emotion Analysis with Temporal Alignment
        logger.info("*" * 50)
        logger.info("Step 3 - Classifying emotions, sub-emotions, and intensity...")

        emotion_predictions = []
        if sentences:
            # Process all sentences in batch for efficiency
            emotion_predictions = predict_emotion(sentences)
        else:
            logger.info("Info: No sentences found for emotion classification.")

        # Combine temporal and emotional data for comprehensive analysis
        for i, sentence_text in enumerate(sentences):
            # Initialize data structure with temporal boundaries
            pred_data = {
                "start_time": start_times_str[i],
                "end_time": end_times_str[i],
                "text": sentence_text,
                "emotion": "unknown",
                "sub_emotion": "unknown",
                "intensity": "unknown",
            }
            # Merge emotion predictions if available
            if (
                emotion_predictions
                and i < len(emotion_predictions)
                and emotion_predictions[i]
            ):
                pred_data["emotion"] = emotion_predictions[i].get("emotion", "unknown")
                pred_data["sub_emotion"] = emotion_predictions[i].get(
                    "sub_emotion", "unknown"
                )
                # Ensure intensity is string for consistent output format
                pred_data["intensity"] = str(
                    emotion_predictions[i].get("intensity", "unknown")
                )
            sentences_data.append(pred_data)
    else:
        logger.warning(
            "One or more required columns (Sentence, Start Time, End Time) "
            "are missing from the transcript Excel file."
        )
        # Return empty list if essential columns are missing
        return []    # STAGE 4: Results Persistence for Reproducibility
    results_df = pd.DataFrame(sentences_data)
    results_file = os.path.join(
        results_dir, f"{title}.xlsx"
    )
    # Save structured results with temporal and emotional metadata
    results_df.to_excel(results_file, index=False)
    logger.info(f"Emotion predictions saved to {results_file}")
    logger.info("*" * 50)

    return sentences_data


if __name__ == "__main__":
    # Configure command-line interface for pipeline execution
    parser = argparse.ArgumentParser(
        description="Emotion Classification Pipeline - "
                    "Extract and analyze emotions from YouTube videos",
        epilog="Example: python predict.py 'https://youtube.com/watch?v=...' "
               "--transcription_method whisper"
    )
    parser.add_argument(
        "youtube_url",
        type=str,
        help="YouTube video URL for emotion analysis",
    )
    parser.add_argument(
        "--transcription_method",
        type=str,
        default="assemblyAI",
        choices=["assemblyAI", "whisper"],
        help="Speech-to-text service: 'assemblyAI' (cloud) or 'whisper' (local)",
    )

    args = parser.parse_args()

    # Execute the complete pipeline workflow
    predictions = process_youtube_url_and_predict(
        youtube_url=args.youtube_url,
        transcription_method=args.transcription_method,
    )

    # Provide execution summary and user feedback
    if predictions:
        logger.info("Pipeline completed successfully. Sample predictions:")
        # Display first 5 results as preview
        for i, pred in enumerate(predictions[:5]):
            logger.info(f"  Segment {i+1}: {pred}")
        logger.info(f"Total segments processed: {len(predictions)}")
    else:
        logger.warning("Pipeline completed but no predictions were generated.")
