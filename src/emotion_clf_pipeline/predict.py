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
import os
import time
import warnings

import pandas as pd
from dotenv import load_dotenv

# Import the local modules using relative imports
# from .data import DataPreparation
from .model import EmotionPredictor
from .stt import SpeechToTextTranscriber, WhisperTranscriber, save_youtube_audio

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Get paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# BASE_DIR should be the project root, not src/
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_DIR = os.path.join(BASE_DIR, "data")


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
        print(f"Latency (Emotion Classification): {end - start:.2f} seconds")
        return output
    except Exception as e:
        print(f"Error in emotion prediction: {str(e)}")
        return None


def speech_to_text(transcription_method, audio_file, output_file):
    """
    Perform speech-to-text transcription.

    Args:
        transcription_method (str): The method to use for transcription
                                  ("assemblyAI" or "whisper")
        audio_file (str): Path to the input audio file
        output_file (str): Path for the output transcript file
    """
    start = time.time()
    try:
        if transcription_method.lower() == "assemblyai":
            # Explicitly load .env just before it's needed as a robust measure
            load_dotenv(dotenv_path="/app/.env", override=True)
            api_key = os.environ.get("ASSEMBLYAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "AssemblyAI API key not found in environment \
                        variables (checked in function)"
                )
            transcriber = SpeechToTextTranscriber(api_key)
            transcriber.process(audio_file, output_file)
        elif transcription_method.lower() == "whisper":
            transcriber = WhisperTranscriber()
            transcriber.process(audio_file, output_file)
        else:
            raise ValueError(f"Unknown transcription method: {transcription_method}")

        end = time.time()
        print(f"Latency (Speech-to-Text): {end - start:.2f} seconds")
    except Exception as e:
        print(f"Error in speech-to-text: {str(e)}")


def process_youtube_url_and_predict(
    youtube_url: str, output_filename_base: str, transcription_method: str
) -> list[dict]:
    """
    Processes a YouTube URL to download audio, transcribe, and predict emotions.

    Args:
        youtube_url (str): The URL of the YouTube video.
        output_filename_base (str): Base name for output files (audio, transcript).
        transcription_method (str): "assemblyAI" or "whisper".

    Returns:
        list[dict]: A list of prediction dictionaries, one for each sentence.
                    Each dictionary contains 'emotion', 'sub_emotion', 'intensity'.
    """
    print(f"Starting emotion prediction pipeline for URL: {youtube_url}")
    print(
        f"Using transcription method: {transcription_method}, \
          Filename base: {output_filename_base}"
    )

    # --- Ensure directories exist (moved from main block for reusability) ---
    youtube_audio_dir = os.path.join(BASE_DIR, "data", "youtube_audio")
    transcripts_dir = os.path.join(BASE_DIR, "data", "transcripts")
    results_dir = os.path.join(BASE_DIR, "data", "results")

    os.makedirs(youtube_audio_dir, exist_ok=True)
    os.makedirs(transcripts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    # ----------------------------------------------------------------------

    # STEP 1 - DOWNLOAD YOUTUBE AUDIO
    print("Step 1 - Downloading YouTube audio...")
    # audio_file_path = os.path.join(youtube_audio_dir, f"{output_filename_base}.mp3")
    # Assuming save_youtube_audio is available (it is in data.py, ensure
    # it's imported in predict.py if not already) from data import
    # save_youtube_audio # This import might be needed at the top of predict.py
    actual_audio_path = save_youtube_audio(
        url=youtube_url,
        destination=youtube_audio_dir,
        return_path=True,
        filename=output_filename_base,
    )
    print(f"Audio file saved at: {actual_audio_path}")
    print("YouTube audio downloaded successfully!")

    # Step 2 - SPEECH TO TEXT TRANSCRIPTION
    print("Step 2 - Transcribing audio...")
    transcript_output_file = os.path.join(
        transcripts_dir,
        f"transcribed_data_{output_filename_base}_{transcription_method}.xlsx",
    )
    speech_to_text(transcription_method, actual_audio_path, transcript_output_file)

    # Check if transcription was successful by verifying file existence
    if not os.path.exists(transcript_output_file):
        # Construct a more informative error message if possible,
        # e.g., by capturing stdout/stderr from speech_to_text or
        # checking API key presence. For now, a generic message:
        error_message = (
            f"Transcription failed: Output file {transcript_output_file} not found. "
        )
        if transcription_method.lower() == "assemblyai" and not os.environ.get(
            "ASSEMBLYAI_API_KEY"
        ):
            error_message += "AssemblyAI API key is missing. Please set the \
                ASSEMBLYAI_API_KEY environment variable."
        else:
            error_message += "Please check logs for transcription errors."
        raise RuntimeError(error_message)

    df = pd.read_excel(transcript_output_file)
    df = df.dropna(subset=["Sentence"])
    df = df.reset_index(drop=True)
    sentences = df["Sentence"].tolist()
    print(f"Transcription completed successfully! Found {len(sentences)} sentences.")

    # STEP 3 - EMOTION CLASSIFICATION
    print("Step 3 - Classifying emotions, sub-emotions, and intensity...")

    raw_predictions = []  # Default to empty list
    if sentences:  # Only call predict_emotion if there are sentences
        # predict_emotion is expected to return a list of dicts,
        # or None if an error occurs
        raw_predictions = predict_emotion(sentences)
    else:
        print("Info: No sentences found in transcript for emotion classification.")

    # Ensure raw_predictions is a list, even if predict_emotion returned None
    if raw_predictions is None:
        print(
            "Warning: Emotion prediction step returned None. Emotion data will \
                be marked as N/A."
        )
        raw_predictions = []  # Treat None as an empty list for consistent handling

    # Prepare lists for DataFrame columns, defaulting to pd.NA (requires pandas import)
    # Ensure 'pd' is available; pandas is imported as 'pd' at the top of the file.
    num_rows_in_df = len(df)
    emotions_column_data = [pd.NA] * num_rows_in_df
    sub_emotions_column_data = [pd.NA] * num_rows_in_df
    intensities_column_data = [pd.NA] * num_rows_in_df

    # Process predictions if there were sentences and the DataFrame has rows
    if sentences and num_rows_in_df > 0:
        if raw_predictions and len(raw_predictions) == num_rows_in_df:
            # Predictions exist and their count matches the number of rows in
            # the DataFrame
            for i, pred_dict in enumerate(raw_predictions):
                if isinstance(pred_dict, dict):
                    emotions_column_data[i] = pred_dict.get("emotion", pd.NA)
                    sub_emotions_column_data[i] = pred_dict.get("sub_emotion", pd.NA)
                    # Ensure intensity is stored as a string, consistent with API
                    # response model
                    intensities_column_data[i] = str(pred_dict.get("intensity", pd.NA))
                # else: data for this row remains pd.NA as initialized
            print(
                "Emotions, sub-emotions, and intensity data processed from predictions."
            )
        elif raw_predictions:  # Predictions exist but length mismatches
            print(
                f"Warning: Mismatch between number of predictions \
                    ({len(raw_predictions)}) and DataFrame rows ({num_rows_in_df}). \
                    Emotion data will be N/A."
            )
            # Data columns remain as pd.NA (already initialized)
        else:  # No predictions returned (raw_predictions is empty), but
            # there were sentences
            print(
                "Warning: Emotion prediction yielded no results for the provided \
                    sentences. Emotion data will be N/A."
            )
            # Data columns remain as pd.NA (already initialized)
    elif num_rows_in_df == 0:
        print("Info: Transcript DataFrame is empty. Emotion columns will be empty.")
        # emotions_column_data etc. are already [], which is correct for an
        # empty DataFrame.
    # else: No sentences to begin with, num_rows_in_df might be >0 if they were
    # all NaN and dropped. Handled by initialization if num_rows_in_df > 0, or
    # if num_rows_in_df == 0.

    df["Emotion"] = emotions_column_data
    df["Sub Emotion"] = sub_emotions_column_data
    df["Intensity"] = intensities_column_data

    print("Emotion data columns added/updated in DataFrame.")

    # STEP 4 - SAVE RESULTS (full results including text)
    print("Step 4 - Saving detailed results...")
    results_output_file = os.path.join(
        results_dir, f"results_{output_filename_base}_{transcription_method}.xlsx"
    )
    df.to_excel(results_output_file, index=False)
    print(f"Detailed results saved at: {results_output_file}")
    print("Pipeline completed successfully!")

    return raw_predictions  # Return the list of prediction dictionaries


# Start the pipeline
if __name__ == "__main__":
    # Setup command-line arguments
    parser = argparse.ArgumentParser(
        description="Emotion Classification Pipeline for YouTube URLs"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="https://www.youtube.com/watch?v=ZDsfeIyjZUM",
        help="YouTube video URL to download audio from",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="youtube_video",
        help="Base filename for the downloaded audio and other outputs \
            (without extension)",
    )
    parser.add_argument(
        "--transcription",
        type=str,
        choices=["assemblyAI", "whisper"],
        default="assemblyAI",
        help="Method for speech-to-text transcription",
    )
    args = parser.parse_args()

    # Call the main processing function
    pipeline_predictions = process_youtube_url_and_predict(
        youtube_url=args.url,
        output_filename_base=args.filename,
        transcription_method=args.transcription,
    )

    print("\n--- Final Predictions from Pipeline (summary) ---")
    for i, pred in enumerate(pipeline_predictions):
        print(f"Sentence {i+1}: {pred}")
