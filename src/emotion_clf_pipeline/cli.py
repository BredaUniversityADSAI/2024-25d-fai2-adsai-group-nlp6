import argparse
import json
from typing import Dict, Any

# Assuming predict.py is in the same directory or accessible via PYTHONPATH
# Use relative import for sibling module
try:
    from .predict import predict_emotion
except ImportError:
    # Fallback for running the script directly in some environments
    from predict import predict_emotion


def main():
    """
    Main function to parse arguments and run emotion prediction from the CLI.
    """
    parser = argparse.ArgumentParser(
        description="Predict emotion from text using the command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )

    # --- Define Command-Line Arguments ---
    parser.add_argument(
        "text",
        type=str,
        help="The input text to analyze for emotion."
    )

    # --- Parse Arguments ---
    args = parser.parse_args()

    # --- Perform Prediction ---
    try:
        prediction_result: Dict[str, Any] = predict_emotion(args.text)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return # Exit gracefully on error

    # --- Output Result ---
    # Print the result as a JSON formatted string for clarity
    print(json.dumps(prediction_result, indent=4))


if __name__ == "__main__":
    main()
