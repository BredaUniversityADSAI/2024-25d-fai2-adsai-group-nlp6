from typing import Any, Dict


def predict_emotion(text: str) -> Dict[str, Any]:
    """
    Dummy function to predict emotion from text.

    Args:
        text: The input text string.

    Returns:
        A dictionary containing predicted emotion, sub_emotion, and intensity.
    """
    # In a real scenario, this function would load a trained model
    # and perform inference.
    print(
        f"Received text for prediction: {text[:50]}..."
    )  # Log received text (truncated)

    # Dummy prediction logic
    prediction = {"emotion": "neutral", "sub_emotion": "calm", "intensity": 0.5}

    # Simulate different outputs based on simple keywords (optional example)
    if "happy" in text.lower():
        prediction = {"emotion": "happiness", "sub_emotion": "joy", "intensity": 0.8}
    elif "sad" in text.lower():
        prediction = {"emotion": "sadness", "sub_emotion": "grief", "intensity": 0.7}
    elif "angry" in text.lower():
        prediction = {"emotion": "anger", "sub_emotion": "rage", "intensity": 0.9}

    print(f"Prediction result: {prediction}")  # Log prediction
    return prediction
