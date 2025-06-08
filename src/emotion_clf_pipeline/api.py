"""
Emotion Classification API.

A RESTful API service that analyzes YouTube video content for emotional sentiment.
The service transcribes video audio, processes the text through an emotion
classification pipeline, and returns structured emotion predictions with timestamps.

Key Features:
    - YouTube video transcription using AssemblyAI
    - Multi-dimensional emotion analysis (emotion, sub-emotion, intensity)
    - Time-stamped transcript segmentation
    - CORS-enabled for web frontend integration

Dependencies:
    - FastAPI for REST API framework
    - Pydantic for data validation
    - Custom emotion classification pipeline
"""
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .predict import get_video_title, process_youtube_url_and_predict


# Application constants
API_TITLE = "Emotion Classification API"
API_VERSION = "0.1.0"
DEFAULT_VIDEO_TITLE = "Unknown Title"
DEFAULT_TRANSCRIPTION_METHOD = "assemblyAI"

# Default values for missing prediction data
DEFAULT_SENTENCE = "N/A"
DEFAULT_TIME = "00:00:00"
DEFAULT_EMOTION = "unknown"
DEFAULT_INTENSITY = "unknown"


# FastAPI application configuration
app = FastAPI(
    title=API_TITLE,
    description="""Analyzes YouTube videos for emotional content by transcribing
    audio and applying emotion classification. Returns detailed emotion analysis
    with timestamps for each transcript segment.""",
    version=API_VERSION,
)

# CORS middleware configuration for cross-origin requests
origins = [
    "http://localhost:3000",  # Development frontend
    "http://localhost:3121",  # Alternative development port
    "http://194.171.191.226:3121",  # Production frontend
    "*",  # DEVELOPMENT ONLY - allows all origins for testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # DEVELOPMENT: allows all origins
    allow_credentials=False,  # Required when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---


class PredictionRequest(BaseModel):
    """
    Request payload for emotion prediction endpoint.

    This model validates and structures the input data required for analyzing
    YouTube video content. The URL must point to a valid YouTube video that
    can be transcribed and analyzed.

    Attributes:
        url (str): Valid YouTube video URL for emotion analysis.
                  Must be accessible and contain audio content.

    Example:
        {
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        }
    """

    url: str


class TranscriptItem(BaseModel):
    """
    Represents a single analyzed segment from video transcript.

    Each transcript item corresponds to a time-bounded portion of the video
    with associated emotion analysis results. All timing information uses
    HH:MM:SS format for consistency.

    Attributes:
        sentence (str): Transcribed text content for this time segment.
        start_time (str): Segment start time in HH:MM:SS format.
        end_time (str): Segment end time in HH:MM:SS format.
        emotion (str): Primary emotion category detected in text.
        sub_emotion (str): More specific emotion subcategory.
        intensity (str): Emotional intensity level (e.g., low, medium, high).

    Example:
        {
            "sentence": "I'm really excited about this project!",
            "start_time": "00:01:23",
            "end_time": "00:01:27",
            "emotion": "joy",
            "sub_emotion": "excitement",
            "intensity": "high"
        }
    """

    sentence: str
    start_time: str
    end_time: str
    emotion: str
    sub_emotion: str
    intensity: str


class PredictionResponse(BaseModel):
    """
    Complete emotion analysis response for a YouTube video.

    Contains the video metadata and a complete transcript with emotion
    analysis for each time-segmented portion of the content.

    Attributes:
        videoId (str): Unique identifier generated from the video URL hash.
                      Used for tracking and caching purposes.
        title (str): YouTube video title. Falls back to "Unknown Title"
                    if retrieval fails.
        transcript (List[TranscriptItem]): Chronologically ordered list of
                                         transcript segments with emotion data.
                                         Empty list if processing fails.

    Example:
        {
            "videoId": "1234567890",
            "title": "Sample Video Title",
            "transcript": [
                {
                    "sentence": "Hello everyone!",
                    "start_time": "00:00:05",
                    "end_time": "00:00:07",
                    "emotion": "joy",
                    "sub_emotion": "greeting",
                    "intensity": "medium"
                }
            ]
        }
    """

    videoId: str
    title: str
    transcript: List[TranscriptItem]


# --- API Endpoints ---


@app.post("/predict", response_model=PredictionResponse)
def handle_prediction(request: PredictionRequest) -> PredictionResponse:
    """
    Analyze YouTube video content for emotional sentiment.

    Processes a YouTube video by:
    1. Extracting and transcribing audio content using AssemblyAI
    2. Segmenting transcript into time-bounded chunks
    3. Running emotion classification on each segment
    4. Returning structured results with timestamps

    Args:
        request (PredictionRequest): Contains YouTube URL for analysis.
                                   URL must be valid and accessible.

    Returns:
        PredictionResponse: Complete analysis results including:
            - Video metadata (ID, title)
            - Time-segmented transcript with emotion data
            - Empty transcript list if processing fails

    Raises:
        HTTPException: Implicitly raised by FastAPI for invalid requests.

    Processing Notes:
        - Video ID generated from URL hash for uniqueness
        - Graceful degradation: continues processing if title fetch fails
        - Uses AssemblyAI transcription service
        - Returns empty results rather than errors for robustness

    Example:
        POST /predict
        {
            "url": "https://www.youtube.com/watch?v=example"
        }

        Response:
        {
            "videoId": "12345",
            "title": "Example Video",
            "transcript": [...]
        }
    """    # Generate unique identifier from URL for tracking and caching
    video_id = str(hash(request.url))

    # Fetch video metadata with graceful error handling
    try:
        video_title = get_video_title(request.url)
    except Exception as e:
        print(f"Could not fetch video title: {e}")
        video_title = DEFAULT_VIDEO_TITLE

    # Process video through emotion classification pipeline
    list_of_predictions: List[Dict[str, Any]] = (
        process_youtube_url_and_predict(
            youtube_url=request.url,
            transcription_method=DEFAULT_TRANSCRIPTION_METHOD,
        )
    )

    # Handle empty results gracefully - return structured empty response
    if not list_of_predictions:
        return PredictionResponse(
            videoId=video_id,
            title=video_title,
            transcript=[]
        )    # Transform raw prediction data into structured transcript items
    transcript_items = [
        TranscriptItem(
            sentence=pred.get("text", pred.get("sentence", DEFAULT_SENTENCE)),
            start_time=str(pred.get("start_time", DEFAULT_TIME)),
            end_time=str(pred.get("end_time", DEFAULT_TIME)),
            emotion=pred.get("emotion", DEFAULT_EMOTION),
            sub_emotion=pred.get("sub_emotion", DEFAULT_EMOTION),
            intensity=str(pred.get("intensity", DEFAULT_INTENSITY)),
        )
        for pred in list_of_predictions
    ]

    return PredictionResponse(
        videoId=video_id,
        title=video_title,
        transcript=transcript_items,
    )


# --- Health Check Endpoint ---


@app.get("/")
def read_root() -> Dict[str, str]:
    """
    API health check and information endpoint.

    Provides basic API information and confirms service availability.
    Used for health checks, service discovery, and API documentation.

    Returns:
        Dict[str, str]: Service information with:
            - message: Welcome message with usage instructions
            - status: Implicit "healthy" status by successful response

    Example:
        GET /        Response:
        {
            "message": "Welcome to the Emotion Classification API..."
        }
    """
    return {
        "message": (
            "Welcome to the Emotion Classification API. "
            "Use POST /predict to analyze YouTube videos for emotional content."
        )
    }
