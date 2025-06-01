"""
Module for the Emotion Classification API.

This module defines a FastAPI application that provides an endpoint for
predicting emotions from text input. It uses a pre-trained emotion
classification pipeline to process the text and return emotion predictions.
"""
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Assuming predict.py is in the same directory or accessible via PYTHONPATH
from .predict import get_video_title, process_youtube_url_and_predict

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

# CORS configuration
origins = [
    "http://localhost:3000",  # Allow frontend origin
    "http://localhost:3121",  # Allow new frontend origin
    # You can add other origins if needed, e.g., deployed frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


class TranscriptItem(BaseModel):
    """
    Model for a single item in the transcript.
    """

    sentence: str
    start_time: float  # Assuming time is in seconds
    end_time: float  # Assuming time is in seconds
    emotion: str
    sub_emotion: str
    intensity: str


class PredictionResponse(BaseModel):
    """
    Response model for the emotion prediction endpoint.

    Attributes:
        videoId: A unique identifier for the video.
        title: The title of the YouTube video.
        transcript: A list of transcript items with emotion analysis.
    """

    videoId: str
    title: str
    transcript: List[TranscriptItem]


# --- API Endpoints ---


@app.post("/predict", response_model=PredictionResponse)
def handle_prediction(request: PredictionRequest) -> PredictionResponse:
    """
    Predicts the emotion from the content of an article URL.
    Returns the prediction for the first transcribed sentence/segment.
    """
    # In a real scenario, you might generate a more robust videoId
    video_id = str(hash(request.url))

    # Attempt to get the video title
    try:
        video_title = get_video_title(request.url)
    except Exception as e:
        print(f"Could not fetch video title: {e}")
        video_title = "Unknown Title"

    list_of_predictions: List[Dict[str, Any]] = process_youtube_url_and_predict(
        youtube_url=request.url,
        # Use video_id for unique output filenames
        # output_filename_base=f"api_output_{video_id}",
        transcription_method="assemblyAI",  # Or make this configurable
    )

    if not list_of_predictions:
        # Return an empty or error-indicating response if no predictions
        return PredictionResponse(videoId=video_id, title=video_title, transcript=[])

    # Transform the prediction dictionaries into TranscriptItem models
    transcript_items = [
        TranscriptItem(
            sentence=pred.get("sentence", "N/A"),
            start_time=float(pred.get("start_time", 0.0)),
            end_time=float(pred.get("end_time", 0.0)),
            emotion=pred.get("emotion", "unknown"),
            sub_emotion=pred.get("sub_emotion", "unknown"),
            intensity=str(pred.get("intensity", "unknown")),
        )
        for pred in list_of_predictions
    ]

    response = PredictionResponse(
        videoId=video_id,
        title=video_title,
        transcript=transcript_items,
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
