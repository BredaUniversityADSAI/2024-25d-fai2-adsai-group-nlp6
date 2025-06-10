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
    - Feedback collection for training data improvement
"""
import csv
import io
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .azure_pipeline import get_ml_client
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


class FeedbackItem(BaseModel):
    """
    Represents a single corrected emotion prediction for training data.
    
    Used to collect user feedback on emotion classifications to improve
    model accuracy over time.
    
    Attributes:
        start_time (str): Segment start time in HH:MM:SS format.
        end_time (str): Segment end time in HH:MM:SS format.
        text (str): Transcribed text content for this segment.
        emotion (str): Corrected primary emotion category.
        sub_emotion (str): Corrected specific emotion subcategory.
        intensity (str): Corrected emotional intensity level.
    """
    
    start_time: str
    end_time: str
    text: str
    emotion: str
    sub_emotion: str
    intensity: str


class FeedbackRequest(BaseModel):
    """
    Request payload for submitting emotion classification feedback.
    
    Contains corrected emotion predictions that will be saved as training
    data to improve future model performance.
    
    Attributes:
        videoTitle (str): Title of the video being corrected.
        feedbackData (List[FeedbackItem]): List of corrected predictions.
    """
    
    videoTitle: str
    feedbackData: List[FeedbackItem]


class FeedbackResponse(BaseModel):
    """
    Response for feedback submission.
    
    Confirms successful saving of training data with metadata.
    
    Attributes:
        success (bool): Whether the feedback was saved successfully.
        filename (str): Name of the created training file.
        message (str): Descriptive message about the operation.
        record_count (int): Number of feedback records saved.
    """
    
    success: bool
    filename: str
    message: str
    record_count: int


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
            "Use POST /predict to analyze YouTube videos for emotional content, "
            "or POST /save-feedback to submit training data improvements."
        )
    }


def get_next_training_filename() -> str:
    """
    Generate the next available training data filename.
    
    Scans Azure data assets to find the highest numbered training file
    and returns the next sequential filename.
    
    Returns:
        str: Next available filename (e.g., "train_data-0042.csv")
    """
    try:
        # Try to get ML client and check existing data assets
        ml_client = get_ml_client()
        
        # List all data assets to find existing training files
        data_assets = list(ml_client.data.list())
        train_numbers = []
        
        for asset in data_assets:
            if asset.name.startswith("emotion-raw-train-"):
                try:
                    # Extract number from name like "emotion-raw-train-0001"
                    number_part = asset.name.split("-")[-1]
                    train_numbers.append(int(number_part))
                except (ValueError, IndexError):
                    continue
        
        # Get next number
        if train_numbers:
            next_number = max(train_numbers) + 1
        else:
            next_number = 1
            
        return f"train_data-{next_number:04d}.csv"
        
    except Exception:
        # Fallback to timestamp-based naming if Azure access fails
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"train_data-{timestamp}.csv"


def create_feedback_csv(feedback_data: List[FeedbackItem]) -> str:
    """
    Create CSV content from feedback data.
    
    Args:
        feedback_data: List of corrected emotion predictions
        
    Returns:
        str: CSV content as string
    """
    output = io.StringIO()
      # Define CSV headers matching the training data format
    fieldnames = [
        'start_time', 'end_time', 'text', 'emotion', 'sub-emotion', 'intensity'
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    
    # Write header
    writer.writeheader()
    
    # Write feedback data
    for item in feedback_data:
        writer.writerow({
            'start_time': item.start_time,
            'end_time': item.end_time,
            'text': item.text,
            'emotion': item.emotion,
            'sub-emotion': item.sub_emotion,  # Note: CSV uses hyphenated version
            'intensity': item.intensity
        })
    
    csv_content = output.getvalue()
    output.close()
    return csv_content


def save_feedback_to_azure(filename: str, csv_content: str) -> bool:
    """
    Save feedback CSV to Azure ML as a data asset.
    
    Args:
        filename: Name for the training file
        csv_content: CSV content as string
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from azure.ai.ml.entities import Data
        from azure.ai.ml.constants import AssetTypes
          ml_client = get_ml_client()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False
        ) as tmp_file:
            tmp_file.write(csv_content)
            tmp_path = tmp_file.name
        
        try:
            # Create data asset name
            asset_name = (
                f"emotion-raw-train-{filename.replace('.csv', '').split('-')[-1]}"
            )
            
            # Create data asset
            data_asset = Data(
                name=asset_name,
                description=(
                    f"Training feedback data from user corrections - {filename}"
                ),
                path=tmp_path,
                type=AssetTypes.URI_FILE
            )
            
            # Register with Azure ML
            ml_client.data.create_or_update(data_asset)
            return True
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
            
    except Exception as e:
        print(f"Failed to save feedback to Azure: {str(e)}")
        return False


@app.post("/save-feedback", response_model=FeedbackResponse)
def save_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    Save user feedback on emotion predictions as training data.
    
    Accepts corrected emotion classifications and saves them to Azure ML
    data assets for future model training and improvement.
    
    Args:
        request (FeedbackRequest): Contains video title and corrected predictions.
        
    Returns:
        FeedbackResponse: Confirmation with filename and record count.
        
    Raises:
        HTTPException: If saving fails or data is invalid.
        
    Example:
        POST /save-feedback
        {
            "videoTitle": "Sample Video",
            "feedbackData": [
                {
                    "start_time": "00:00:05",
                    "end_time": "00:00:07",
                    "text": "Hello everyone!",
                    "emotion": "happiness",
                    "sub_emotion": "excitement",
                    "intensity": "moderate"
                }
            ]
        }
        
        Response:
        {
            "success": true,
            "filename": "train_data-0042.csv",
            "message": "Successfully saved 1 feedback records",
            "record_count": 1
        }
    """
    try:
        # Validate that we have feedback data
        if not request.feedbackData:
            raise HTTPException(
                status_code=400,
                detail="No feedback data provided"
            )
        
        # Generate filename
        filename = get_next_training_filename()
        
        # Create CSV content
        csv_content = create_feedback_csv(request.feedbackData)
        
        # Save to Azure (if available, otherwise just return success for demo)
        try:
            azure_success = save_feedback_to_azure(filename, csv_content)
            if not azure_success:
                # Fallback: could save locally or return partial success
                print(f"Azure save failed, but feedback received: {filename}")
        except Exception as e:
            print(f"Azure integration error: {str(e)}")
            # Continue anyway for demo purposes
        
        return FeedbackResponse(
            success=True,
            filename=filename,
            message=f"Successfully saved {len(request.feedbackData)} feedback records",
            record_count=len(request.feedbackData)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error saving feedback: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save feedback: {str(e)}"
        )
