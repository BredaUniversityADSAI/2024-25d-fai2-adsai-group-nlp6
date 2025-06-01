#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emotion classification pipeline
-----------------------------------
This script implements a complete NLP pipeline that:
1. Downloads audio from YouTube
2. Transcribes audio to text
3. Classifies emotions in text
4. Saves results to file
"""

# Import the libraries
import argparse
import logging
import os
import time
import warnings

import pandas as pd
from dotenv import load_dotenv
from pytubefix import YouTube  # Added for fetching video title

# Import the local modules using relative imports
# from .data import DataPreparation
from .model import EmotionPredictor
from .stt import SpeechToTextTranscriber, WhisperTranscriber, save_youtube_audio

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Get paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))  # Project root
DATA_DIR = os.path.join(BASE_DIR, "data")


def time_str_to_seconds(time_str):
    """Converts a time string (HH:MM:SS or HH:MM:SS.mmm) to seconds."""
    if isinstance(time_str, (int, float)):
        return float(time_str)  # Already in seconds or compatible format
    try:
        parts = str(time_str).split(":")
        if len(parts) == 3:
            h, m, s_parts = parts
            s_and_ms = s_parts.split(".")
            s = float(s_and_ms[0])
            ms = float(s_and_ms[1]) / 1000.0 if len(s_and_ms) > 1 else 0.0
            return float(h) * 3600 + float(m) * 60 + s + ms
        elif len(parts) == 2:  # MM:SS or MM:SS.mmm
            m, s_parts = parts
            s_and_ms = s_parts.split(".")
            s = float(s_and_ms[0])
            ms = float(s_and_ms[1]) / 1000.0 if len(s_and_ms) > 1 else 0.0
            return float(m) * 60 + s + ms
        else:  # Assume it might be seconds already as a string
            return float(time_str)
    except ValueError:
        logger.warning(f"Could not parse time string: {time_str}. Returning 0.0")
        return 0.0  # Or raise an error, or handle differently


def get_video_title(youtube_url: str) -> str:
    """
    Fetches the title of a YouTube video.

    Args:
        youtube_url (str): The URL of the YouTube video.

    Returns:
        str: The title of the video, or "Unknown Title" if an error occurs.
    """
    try:
        yt = YouTube(youtube_url)
        return yt.title
    except Exception as e:
        logger.error(f"Error fetching YouTube video title for {youtube_url}: {e}")
        return "Unknown Title"


def predict_emotion(texts, feature_config=None, reload_model=False):
    """
    Predict emotions for the given text(s).

    Args:
        texts (str or list): Text or list of texts to analyze
        feature_config (dict, optional): Configuration for features to use in prediction
        reload_model (bool): Force reload the model even if cached

    Returns:
        dict or list: Dictionary with emotion predictions for a single text or
                    list of dictionaries for multiple texts
    """
    _emotion_predictor = EmotionPredictor()
    start = time.time()
    try:
        output = _emotion_predictor.predict(texts, feature_config, reload_model)
        end = time.time()
        logger.info(f"Latency (Emotion Classification): {end - start:.2f} seconds")
        return output
    except Exception as e:
        logger.error(f"Error in emotion prediction: {str(e)}")
        return None


def speech_to_text(transcription_method, audio_file, output_file):
    """
    Perform speech-to-text transcription.
    If AssemblyAI is chosen and fails, it will fall back to Whisper.

    Args:
        transcription_method (str): The method to use for transcription
                                  ("assemblyAI" or "whisper")
        audio_file (str): Path to the input audio file
        output_file (str): Path for the output transcript file
    """
    start = time.time()
    transcription_successful = False
    method_used = transcription_method.lower()

    if method_used == "assemblyai":
        logger.info("Attempting transcription with AssemblyAI...")
        try:
            load_dotenv(dotenv_path="/app/.env", override=True)
            api_key = os.environ.get("ASSEMBLYAI_API_KEY")
            if not api_key:
                logger.warning(
                    "AssemblyAI API key not found. Attempting fallback to Whisper."
                )
            else:
                transcriber = SpeechToTextTranscriber(api_key)
                transcriber.process(audio_file, output_file)
                transcription_successful = True
                logger.info("AssemblyAI transcription successful.")
        except Exception as e_assembly:
            logger.error(
                f"Error during AssemblyAI transcription: {str(e_assembly)}. "
                f"Attempting fallback to Whisper."
            )

        if not transcription_successful:
            logger.info("Falling back to Whisper transcription...")
            method_used = "whisper_fallback"  # For logging purposes
            try:
                whisper_transcriber = WhisperTranscriber()
                # Using the same output_file, relying on overwrite.
                # Alt: output_file.replace("assemblyAI", "whisper_fallback")
                whisper_transcriber.process(audio_file, output_file)
                transcription_successful = True
                logger.info("Whisper fallback transcription successful.")
            except Exception as e_whisper_fallback:
                logger.error(
                    f"Error during Whisper fallback: {str(e_whisper_fallback)}"
                )

    elif method_used == "whisper":
        logger.info("Attempting transcription with Whisper...")
        try:
            transcriber = WhisperTranscriber()
            transcriber.process(audio_file, output_file)
            transcription_successful = True
            logger.info("Whisper transcription successful.")
        except Exception as e_whisper:
            logger.error(f"Error during Whisper transcription: {str(e_whisper)}")

    else:
        logger.error(f"Unknown transcription method: {transcription_method}")
        raise ValueError(f"Unknown transcription method: {transcription_method}")

    end = time.time()
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
    Processes a YouTube URL to download audio, transcribe, and predict emotions.

    Args:
        youtube_url (str): The URL of the YouTube video.
        transcription_method (str): "assemblyAI" or "whisper".

    Returns:
        list[dict]: A list of prediction dictionaries, one for each sentence.
                    Each dictionary contains 'sentence', 'start_time', 'end_time',
                    'emotion', 'sub_emotion', 'intensity'.
    """
    logger.info(f"Starting emotion prediction pipeline for URL: {youtube_url}")
    logger.info(f"Using transcription method: {transcription_method}")

    # --- Ensure directories exist (moved from main block for reusability) ---
    youtube_audio_dir = os.path.join(BASE_DIR, "results", "audio")
    transcripts_dir = os.path.join(BASE_DIR, "results", "transcript")
    results_dir = os.path.join(BASE_DIR, "results", "predictions")

    os.makedirs(youtube_audio_dir, exist_ok=True)
    os.makedirs(transcripts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ----------------------------------------------------------------------

    # STEP 1 - DOWNLOAD YOUTUBE AUDIO
    logger.info("*" * 50)
    logger.info("Step 1 - Downloading YouTube audio...")

    actual_audio_path, title = save_youtube_audio(
        url=youtube_url,
        destination=youtube_audio_dir,
    )
    logger.info(f"Audio file saved at: {actual_audio_path}")
    logger.info("YouTube audio downloaded successfully!")

    # Step 2 - SPEECH TO TEXT TRANSCRIPTION
    logger.info("*" * 50)
    logger.info("Step 2 - Transcribing audio...")
    transcript_output_file = os.path.join(
        transcripts_dir,
        f"{title}.xlsx",
    )
    speech_to_text(transcription_method, actual_audio_path, transcript_output_file)

    # Load the transcript
    df = pd.read_excel(transcript_output_file)
    df = df.dropna(subset=["Sentence"])
    df = df.reset_index(drop=True)

    sentences_data = []  # Initialize with an empty list

    if (
        "Sentence" in df.columns
        and "Start Time" in df.columns
        and "End Time" in df.columns
    ):
        sentences = df["Sentence"].tolist()
        start_times_str = df["Start Time"].tolist()
        end_times_str = df["End Time"].tolist()

        logger.info(
            f"Transcription completed successfully! "
            f"Found {len(sentences)} sentences."
        )

        # STEP 3 - EMOTION CLASSIFICATION
        logger.info("*" * 50)
        logger.info("Step 3 - Classifying emotions, sub-emotions, and intensity...")

        emotion_predictions = []
        if sentences:
            emotion_predictions = predict_emotion(sentences)  # List of dicts
        else:
            logger.info("Info: No sentences found for emotion classification.")

        # Combine transcript data with emotion predictions
        for i, sentence_text in enumerate(sentences):
            pred_data = {
                "start_time": start_times_str[i],  # time_str_to_seconds(start_times_str[i]),
                "end_time": end_times_str[i],  # time_str_to_seconds(end_times_str[i]),
                "text": sentence_text,
                "emotion": "unknown",
                "sub_emotion": "unknown",
                "intensity": "unknown",
            }
            if (
                emotion_predictions
                and i < len(emotion_predictions)
                and emotion_predictions[i]
            ):
                pred_data["emotion"] = emotion_predictions[i].get("emotion", "unknown")
                pred_data["sub_emotion"] = emotion_predictions[i].get(
                    "sub_emotion", "unknown"
                )
                pred_data["intensity"] = str(
                    emotion_predictions[i].get("intensity", "unknown")
                )  # Ensure string
            sentences_data.append(pred_data)
    else:
        logger.warning(
            "One or more required columns (Sentence, Start Time, End Time) "
            "are missing from the transcript Excel file."
        )
        # Return empty list if essential columns are missing to prevent further errors
        return []

    # Save results to Excel
    results_df = pd.DataFrame(sentences_data)
    results_file = os.path.join(
        results_dir, f"{title}.xlsx"
    )
    results_df.to_excel(results_file, index=False)
    logger.info(f"Emotion predictions saved to {results_file}")
    logger.info("*" * 50)

    return sentences_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion Classification Pipeline")
    parser.add_argument(
        "youtube_url",
        type=str,
        help=(
            "YouTube URL to process (e.g., "
            "'https://www.youtube.com/watch?v=dQw4w9WgXcQ')"
        ),
    )
    parser.add_argument(
        "--transcription_method",
        type=str,
        default="assemblyAI",
        choices=["assemblyAI", "whisper"],
        help="Transcription method to use ('assemblyAI' or 'whisper')",
    )

    args = parser.parse_args()

    # Example usage of the full pipeline
    predictions = process_youtube_url_and_predict(
        youtube_url=args.youtube_url,
        transcription_method=args.transcription_method,
    )

    if predictions:
        logger.info("Pipeline completed. Final predictions:")
        for i, pred in enumerate(predictions[:5]):  # Print first 5 predictions
            logger.info(f"  Sentence {i+1}: {pred}")
    else:
        logger.warning("Pipeline completed but no predictions were generated.")
