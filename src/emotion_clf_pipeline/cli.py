"""
Command-Line Interface for the Emotion Classification Pipeline.

This script provides a CLI to predict emotions from the content of a given YouTube URL.
It utilizes the `process_youtube_url_and_predict` function from the `predict` module
to perform the analysis and outputs the results in JSON format.

Usage:
    python -m src.emotion_clf_pipeline.cli <YOUTUBE_URL>
"""
import argparse
import json
from typing import Any, Dict, List

# Use relative import for sibling module predict.py
try:
    from .predict import process_youtube_url_and_predict
except ImportError:
    # Fallback for scenarios where the script might be run directly
    # and the relative import fails (e.g., certain IDE configurations or older
    # Python versions)
    from predict import process_youtube_url_and_predict


def main():
    """
    Parses command-line arguments, performs emotion prediction, and prints the result.

    The script expects a single argument: the URL of a YouTube video.
    It then calls the emotion prediction pipeline and prints the structured
    JSON output containing the emotion, sub-emotion, and intensity.
    Handles potential errors during the prediction process and prints an error message.
    """
    parser = argparse.ArgumentParser(
        description="Analyzes a YouTube video's transcript for emotion content."
        "Outputs predicted emotions for transcribed sentences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Define Command-Line Arguments ---
    parser.add_argument(
        "url",
        type=str,
        help="The YouTube URL from which to extract and analyze content for emotion.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="cli_youtube_output",
        help="Base filename for downloaded audio (without extension).",
    )
    parser.add_argument(
        "--transcription",
        type=str,
        choices=["assemblyAI", "whisper"],
        default="assemblyAI",
        help="Method for speech-to-text transcription.",
    )

    # --- Parse Arguments ---
    args = parser.parse_args()

    # --- Perform Prediction ---
    try:
        list_of_predictions: List[Dict[str, Any]] = process_youtube_url_and_predict(
            youtube_url=args.url,
            output_filename_base=args.filename,
            transcription_method=args.transcription,
        )
    except Exception as e:
        # Provide a user-friendly error message
        print(f"An error occurred during the emotion prediction pipeline: {e}")
        print(
            "Please ensure the URL is correct, the video is accessible, "
            "and all configurations are set."
        )
        return  # Exit gracefully

    # --- Output Result ---
    print("\n--- Prediction Results ---")
    if list_of_predictions:
        # Print the list of prediction dictionaries as a JSON formatted string
        print(json.dumps(list_of_predictions, indent=4))
    else:
        print("No predictions were generated.")


if __name__ == "__main__":
    main()
