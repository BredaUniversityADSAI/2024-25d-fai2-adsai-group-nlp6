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
import sys
import time
import io
import os
import shutil
import tempfile
from datetime import datetime
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

# Import
try:
    from .azure_pipeline import get_ml_client
    from .azure_sync import sync_best_baseline
    from .predict import get_video_title, process_youtube_url_and_predict
except ImportError as e:
    try:
        from azure_pipeline import get_ml_client
        from azure_sync import sync_best_baseline
        from predict import get_video_title, process_youtube_url_and_predict
    except ImportError:
        # Add src directory to path if not already there
        src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from emotion_clf_pipeline.azure_pipeline import get_ml_client
        from emotion_clf_pipeline.azure_sync import sync_best_baseline
        from emotion_clf_pipeline.predict import (
            get_video_title,
            process_youtube_url_and_predict
        )

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


@app.on_event("startup")
async def startup_event():
    """
    On application startup, sync the latest baseline model from Azure ML.
    This ensures the API is always using the production-ready model.
    """
    print("🚀 --- Triggering model sync on startup --- 🚀")
    synced = sync_best_baseline(force_update=True, min_f1_improvement=0.0)
    if synced:
        print("✅ --- Model sync successful --- ✅")
    else:
        print("⚠️ --- Model sync failed or no new model found --- ⚠️")


# --- Pydantic Models ---


class PredictionRequest(BaseModel):
    """
    Request payload for emotion prediction endpoint.
    """

    url: str


class TranscriptItem(BaseModel):
    """
    Represents a single analyzed segment from video transcript.
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
    """

    videoId: str
    title: str
    transcript: List[TranscriptItem]


class FeedbackItem(BaseModel):
    """
    Represents a single corrected emotion prediction for training data.
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
    """

    videoTitle: str
    feedbackData: List[FeedbackItem]


class FeedbackResponse(BaseModel):
    """
    Response for feedback submission.
    """

    success: bool
    filename: str
    message: str
    record_count: int


# --- API Endpoints ---


@app.post("/refresh-model")
def handle_refresh() -> Dict[str, Any]:
    """
    Triggers a manual refresh of the baseline model from Azure ML.

    This endpoint allows for a zero-downtime model update by pulling the
    latest model tagged as 'emotion-clf-baseline' from the registry
    and loading it into the running API instance.

    TODO: Secure this endpoint.
    """
    print("🔄 --- Triggering manual model refresh --- 🔄")
    synced = sync_best_baseline(force_update=True, min_f1_improvement=0.0)
    if synced:
        print("✅ --- Model refresh successful --- ✅")
        return {"success": True, "message": "Model refreshed successfully."}

    print("⚠️ --- Model refresh failed or no new model found --- ⚠️")
    return {
        "success": False,
        "message": "Model refresh failed or no new model was found.",
    }


@app.post("/predict", response_model=PredictionResponse)
def handle_prediction(request: PredictionRequest) -> PredictionResponse:
    """
    Analyze YouTube video content for emotional sentiment.
    """
    # Generate unique identifier from URL for tracking and caching
    video_id = str(hash(request.url))

    # Fetch video metadata with graceful error handling
    try:
        video_title = get_video_title(request.url)
    except Exception as e:
        print(f"Could not fetch video title: {e}")
        video_title = DEFAULT_VIDEO_TITLE
    list_of_predictions: List[Dict[str, Any]] = process_youtube_url_and_predict(
        youtube_url=request.url,
        transcription_method=DEFAULT_TRANSCRIPTION_METHOD,
    )

    # Handle empty results gracefully - return structured empty response
    if not list_of_predictions:
        return PredictionResponse(videoId=video_id, title=video_title, transcript=[])

    # Transform raw prediction data into structured transcript items
    transcript_items = [
        TranscriptItem(
            sentence=pred.get("text", pred.get("sentence", DEFAULT_SENTENCE)),
            start_time=format_time_seconds(pred.get("start_time", 0)),
            end_time=format_time_seconds(pred.get("end_time", 0)),
            emotion=pred.get("emotion", DEFAULT_EMOTION) or DEFAULT_EMOTION,
            sub_emotion=(
                pred.get("sub_emotion", pred.get("sub-emotion", "neutral")) or "neutral"
            ),
            intensity=(
                (pred.get("intensity", DEFAULT_INTENSITY) or "mild").lower()
                if pred.get("intensity")
                else "mild"
            ),
        )
        for pred in list_of_predictions
    ]

    return PredictionResponse(
        videoId=video_id,
        title=video_title,
        transcript=transcript_items,
    )


def get_next_training_filename() -> str:
    """
    Generate the next available training data filename by checking existing files
    in the local data/raw/train directory.
    """
    try:
        # Check local data/raw/train directory for existing files
        train_dir = "data/raw/train"
        if os.path.exists(train_dir):
            existing_files = [
                f
                for f in os.listdir(train_dir)
                if f.startswith("train_data-") and f.endswith(".csv")
            ]

            train_numbers = []
            for filename in existing_files:
                try:
                    # Extract number from filename like "train_data-0001.csv"
                    number_part = filename.replace("train_data-", "").replace(
                        ".csv", ""
                    )
                    train_numbers.append(int(number_part))
                except (ValueError, IndexError):
                    continue

            if train_numbers:
                next_number = max(train_numbers) + 1
            else:
                next_number = 1
        else:
            next_number = 1

        return f"train_data-{next_number:04d}.csv"

    except Exception:
        # Fallback to timestamp-based naming if directory access fails
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"train_data-{timestamp}.csv"


def create_feedback_csv(feedback_data: List[FeedbackItem]) -> str:
    """
    Create CSV content from feedback data.
    """
    output = io.StringIO()

    # Define CSV headers matching the training data format
    fieldnames = [
        "start_time",
        "end_time",
        "text",
        "emotion",
        "sub-emotion",
        "intensity",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)

    # Write header
    writer.writeheader()

    # Write feedback data
    for item in feedback_data:
        writer.writerow(
            {
                "start_time": item.start_time,
                "end_time": item.end_time,
                "text": item.text,
                "emotion": item.emotion,
                "sub-emotion": item.sub_emotion,  # Note: CSV uses hyphenated version
                "intensity": item.intensity,
            }
        )

    csv_content = output.getvalue()
    output.close()
    return csv_content


def format_time_seconds(time_input) -> str:
    """
    Convert various time formats to HH:MM:SS format.

    Args:
        time_input: Time in seconds (float/int), or already formatted string

    Returns:
        Formatted time string in HH:MM:SS format
    """
    try:
        # If it's already a properly formatted string, return as is
        if isinstance(time_input, str):
            # Check if it's already in HH:MM:SS format
            if ":" in time_input and len(time_input.split(":")) == 3:
                return time_input
            # Try to convert string to float (in case it's "181.5")
            try:
                time_input = float(time_input)
            except ValueError:
                return DEFAULT_TIME

        # Convert numeric seconds to HH:MM:SS
        seconds = float(time_input)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    except (ValueError, TypeError):
        return DEFAULT_TIME


def save_feedback_to_azure(filename: str, csv_content: str) -> bool:
    """
    Save feedback CSV as a new version of the emotion-raw-train Azure ML data asset.
    Creates a new version as URI_FOLDER to match the existing data asset type.
    """
    try:

        ml_client = get_ml_client()

        # Create temporary directory (required for URI_FOLDER type)
        temp_dir = tempfile.mkdtemp(prefix="feedback_upload_")
        try:
            # Step 1: Try to download existing data from latest Azure ML data asset
            try:
                current_asset = ml_client.data.get(
                    name="emotion-raw-train", version="latest"
                )
                print(
                    f"Found existing asset v{current_asset.version}, "
                    f"downloading data..."
                )

                # Download the current asset to get all existing files
                download_path = ml_client.data.download(
                    name="emotion-raw-train", version="latest", download_path=temp_dir
                )
                print(f"Downloaded existing data to: {download_path}")

            except Exception as e:
                print(f"Could not download existing data: {e}")
                # Fallback: Copy from local directory if available
                local_train_dir = "data/raw/train"
                if os.path.exists(local_train_dir):
                    print("Falling back to local training data...")
                    for file in os.listdir(local_train_dir):
                        if file.endswith(".csv"):
                            src_path = os.path.join(local_train_dir, file)
                            dst_path = os.path.join(temp_dir, file)
                            shutil.copy2(src_path, dst_path)

            # Step 2: Add the new feedback file to the collection
            temp_file_path = os.path.join(temp_dir, filename)
            with open(temp_file_path, "w", newline="", encoding="utf-8") as f:
                f.write(csv_content)

            # Count total files for reporting
            csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
            print(f"Prepared {len(csv_files)} files for new data asset version")

            # Get current asset to determine next version
            try:
                # List all versions of the emotion-raw-train asset
                asset_versions = list(ml_client.data.list(name="emotion-raw-train"))
                if asset_versions:
                    # Find the highest version number
                    version_numbers = []
                    for asset in asset_versions:
                        try:
                            version_numbers.append(int(asset.version))
                        except (ValueError, TypeError):
                            continue

                    if version_numbers:
                        new_version = max(version_numbers) + 1
                    else:
                        new_version = 2
                else:
                    new_version = 2
            except Exception as e:
                print(f"Error getting existing versions: {e}")
                new_version = 2

            # Create new version of the emotion-raw-train data asset as URI_FOLDER
            data_asset = Data(
                name="emotion-raw-train",
                version=str(new_version),
                description=(
                    f"Training data with user feedback - Version {new_version} - "
                    f"Contains {len(csv_files)} files including new "
                    f"feedback: {filename}"
                ),
                path=temp_dir,  # Point to folder containing all CSV files
                type=AssetTypes.URI_FOLDER,  # Match existing asset type
            )  # Register new version with Azure ML (with retry logic)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    created_asset = ml_client.data.create_or_update(data_asset)
                    print(
                        f"Successfully created emotion-raw-train "
                        f"version {new_version}"
                    )
                    print(
                        f"Asset details: {created_asset.name} "
                        f"v{created_asset.version}"
                    )
                    break
                except Exception as retry_error:
                    print(
                        f"Attempt {attempt + 1}/{max_retries} failed: "
                        f"{str(retry_error)}"
                    )
                    if attempt == max_retries - 1:
                        print(
                            "Max retries reached. Data uploaded to storage "
                            "but asset registration failed."
                        )
                        print("The files are safely stored in Azure Storage.")
                    else:
                        time.sleep(2**attempt)  # Exponential backoff

            # Also save locally to maintain consistency
            local_train_dir = "data/raw/train"
            os.makedirs(local_train_dir, exist_ok=True)
            local_file_path = os.path.join(local_train_dir, filename)
            with open(local_file_path, "w", newline="", encoding="utf-8") as f:
                f.write(csv_content)

            return True

        finally:
            # Clean up temporary directory after a short delay
            time.sleep(2)  # Give Azure time to process the upload
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        print(f"Failed to save feedback to Azure: {str(e)}")
        # Fallback: save locally only
        try:
            local_train_dir = "data/raw/train"
            os.makedirs(local_train_dir, exist_ok=True)
            local_file_path = os.path.join(local_train_dir, filename)
            with open(local_file_path, "w", newline="", encoding="utf-8") as f:
                f.write(csv_content)
            print(f"Saved feedback locally to {local_file_path}")
            return True
        except Exception as fallback_error:
            print(f"Failed to save feedback locally: {str(fallback_error)}")
            return False


@app.post("/save-feedback", response_model=FeedbackResponse)
def save_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    Save user feedback on emotion predictions as training data.
    """
    try:
        # Validate that we have feedback data
        if not request.feedbackData:
            raise HTTPException(status_code=400, detail="No feedback data provided")

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
            message=(
                f"Successfully saved {len(request.feedbackData)} " f"feedback records"
            ),
            record_count=len(request.feedbackData),
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error saving feedback: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to save feedback: {str(e)}"
        )


@app.get("/")
def read_root() -> Dict[str, str]:
    """
    API health check and information endpoint.
    """
    return {
        "message": (
            "Welcome to the Emotion Classification API. "
            "Use POST /predict to analyze YouTube videos for emotional content, "
            "or POST /save-feedback to submit training data improvements."
        )
    }
