"""
Module for the Emotion Classification API.

This module defines a FastAPI application that provides an endpoint for
predicting emotions from text input. It uses a pre-trained emotion
classification pipeline to process the text and return emotion predictions.
"""
from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel

# Assuming predict.py is in the same directory or accessible via PYTHONPATH
from .predict import process_youtube_url_and_predict

# Initialize FastAPI app
app = FastAPI(
    title="Emotion Classification API",
    description="""API for predicting emotion from text using the
    emotion classification pipeline.
    Accepts a URL to an article, processes the content, and returns
    the predicted emotion, sub-emotion, and intensity for the first transcribed
    segment.""",
    version="0.1.0",
)

# --- Pydantic Models ---


class PredictionRequest(BaseModel):
    """
    Request model for the emotion prediction endpoint.

    Attributes:
        url: The URL of the article to be analyzed for emotion.
    """

    url: str
    # Optional: Add parameters for filename and transcription method
    # if desired for API control
    # output_filename_base: str = "api_youtube_output"
    # transcription_method: str = "assemblyAI"


class PredictionResponse(BaseModel):
    """
    Response model for the emotion prediction endpoint.

    Attributes:
        emotion: The primary emotion predicted from the text.
        sub_emotion: A more specific sub-category of the predicted emotion.
        intensity: The predicted intensity of the emotion
        (e.g., "mild", "moderate", "intense").
    """

    emotion: str
    sub_emotion: str
    intensity: str


# --- API Endpoints ---


@app.post("/predict", response_model=PredictionResponse)
def handle_prediction(request: PredictionRequest) -> PredictionResponse:
    """
    Predicts the emotion from the content of an article URL.
    Returns the prediction for the first transcribed sentence/segment.
    """
    list_of_predictions: List[Dict[str, Any]] = process_youtube_url_and_predict(
        youtube_url=request.url,
        output_filename_base="api_output",
        transcription_method="assemblyAI",
    )

    if not list_of_predictions:
        return PredictionResponse(
            emotion="unknown", sub_emotion="unknown", intensity="unknown"
        )

    first_prediction = list_of_predictions[0]

    response = PredictionResponse(
        emotion=first_prediction.get("emotion", "unknown"),
        sub_emotion=first_prediction.get("sub_emotion", "unknown"),
        intensity=str(first_prediction.get("intensity", "unknown")),
    )
    return response


# --- Root Endpoint ---


@app.get("/")
def read_root() -> Dict[str, str]:
    """
    Provides a welcome message for the API root.

    Returns:
        A dictionary containing a welcome message.
    """
    return {
        "message": "Welcome to the Emotion Classification API. "
        "Use the POST /predict endpoint to analyze article emotions."
    }


# --- Running the API (Example using uvicorn) ---
# To run this API locally:
# 1. Ensure FastAPI and Uvicorn are installed: poetry add fastapi uvicorn
# 2. From the project root directory, run:
#    uvicorn src.emotion_clf_pipeline.api:app --reload
