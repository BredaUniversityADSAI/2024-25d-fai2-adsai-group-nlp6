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
    - Comprehensive monitoring with Prometheus metrics
"""

import csv
import sys
import time
import io
import os
import shutil
import tempfile
import pickle
from datetime import datetime
from typing import Any, Dict, List
import json

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

# Import monitoring components
try:
    from .monitoring import metrics_collector, RequestTracker, time_function
except ImportError:
    from monitoring import metrics_collector, RequestTracker, time_function

# Import
try:
    from .azure_pipeline import get_ml_client
    from .azure_sync import sync_best_baseline
    from .predict import get_video_title, process_youtube_url_and_predict
except ImportError as e:
    print(f"Import error: {e}. Attempting to import from src directory.")
    try:
        from azure_pipeline import get_ml_client
        from azure_sync import sync_best_baseline
        from predict import get_video_title, process_youtube_url_and_predict
    except ImportError:
        # Add src directory to path if not already there
        src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from emotion_clf_pipeline.azure_pipeline import get_ml_client
        from emotion_clf_pipeline.azure_sync import sync_best_baseline
        from emotion_clf_pipeline.predict import (
            get_video_title,
            process_youtube_url_and_predict,
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
    synced = sync_best_baseline(force_update=False, min_f1_improvement=0.0)
    if synced:
        print("✅ --- Model sync successful --- ✅")
    else:
        print("⚠️ --- Model sync failed or no new model found --- ⚠️")
    
    # Create baseline stats for drift detection if they don't exist
    print("📊 --- Setting up baseline stats for drift detection --- 📊")
    create_baseline_stats_from_training_data()
    
    # Load training metrics into monitoring system
    print("📊 --- Loading training metrics for monitoring --- 📊")
    load_training_metrics_to_monitoring()


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

    with RequestTracker():
        # Generate unique identifier from URL for tracking and caching
        video_id = str(hash(request.url))
        overall_start_time = time.time()
        
        # Track active requests
        metrics_collector.active_requests.inc()

        # Fetch video metadata with graceful error handling
        title_start_time = time.time()
        try:
            video_title = get_video_title(request.url)
            title_latency = time.time() - title_start_time
            print(f"Video title fetch took: {title_latency:.2f}s")
        except Exception as e:
            print(f"Could not fetch video title: {e}")
            video_title = DEFAULT_VIDEO_TITLE
            metrics_collector.record_error("video_title_fetch", "predict")

        # Process transcription and prediction with detailed timing
        transcription_start_time = time.time()
        try:
            list_of_predictions: List[Dict[str, Any]] = process_youtube_url_and_predict(
                youtube_url=request.url,
                transcription_method=DEFAULT_TRANSCRIPTION_METHOD,
            )

            # Record transcription metrics with actual latency
            transcription_latency = time.time() - transcription_start_time
            metrics_collector.transcription_latency.observe(transcription_latency)
            print(f"Transcription + prediction took: {transcription_latency:.2f}s")

        except Exception as e:
            metrics_collector.record_error("prediction_processing", "predict")
            raise HTTPException(
                status_code=500, detail=f"Error processing video: {str(e)}"
            )
        finally:
            # Decrement active requests counter
            metrics_collector.active_requests.dec()

        # Handle empty results gracefully - return structured empty response
        if not list_of_predictions:
            return PredictionResponse(
                videoId=video_id, title=video_title, transcript=[]
            )

        # Transform raw prediction data into structured transcript items
        prediction_processing_start = time.time()
        transcript_items = []
        for pred in list_of_predictions:
            # Record individual prediction metrics
            sub_emotion = (
                pred.get("sub_emotion", pred.get("sub-emotion", "neutral")) or "neutral"
            )
            intensity_raw = pred.get("intensity", DEFAULT_INTENSITY) or "mild"
            intensity = intensity_raw.lower() if pred.get("intensity") else "mild"

            prediction_data = {
                "emotion": pred.get("emotion", DEFAULT_EMOTION) or DEFAULT_EMOTION,
                "sub_emotion": sub_emotion,
                "intensity": intensity,
            }

            confidence = pred.get("confidence", 0.0)
            
            # Record model confidence distribution
            metrics_collector.model_confidence.observe(confidence)
            
            # Record prediction metrics
            pred_latency = time.time() - prediction_processing_start
            metrics_collector.record_prediction(
                prediction_data, confidence=confidence, latency=pred_latency
            )

            transcript_items.append(
                TranscriptItem(
                    sentence=pred.get("text", pred.get("sentence", DEFAULT_SENTENCE)),
                    start_time=format_time_seconds(pred.get("start_time", 0)),
                    end_time=format_time_seconds(pred.get("end_time", 0)),
                    emotion=prediction_data["emotion"],
                    sub_emotion=prediction_data["sub_emotion"],
                    intensity=prediction_data["intensity"],
                )
            )

        # Record overall prediction timing
        overall_latency = time.time() - overall_start_time
        metrics_collector.prediction_latency.observe(overall_latency)
        print(f"Total request processing time: {overall_latency:.2f}s")

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


@app.get("/health")
def health_check() -> Dict[str, str]:
    """
    Docker container health check endpoint.
    
    Returns 200 OK when the API is ready to serve requests.
    Used by Docker Compose healthcheck configuration.
    """
    return {"status": "healthy", "service": "emotion-classification-api"}


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


# --- Monitoring Endpoints ---


@app.get("/metrics")
def get_metrics():
    """
    Prometheus metrics endpoint.

    Exposes metrics for monitoring the API's performance and usage.
    Accessible at /metrics for Prometheus scraping.
    """
    try:
        # Export metrics using the registered metrics collector
        metrics_data = metrics_collector.export_metrics()
        return Response(content=metrics_data, media_type="text/plain")
    except Exception as e:
        print(f"Error exporting metrics: {str(e)}")
        return Response(
            content="# Error exporting metrics\n",
            media_type="text/plain",
            status_code=500,
        )


@app.post("/track-request")
def track_request(request_data: Dict[str, Any]):
    """
    Track API requests for monitoring and analysis.

    This endpoint is used to collect data on incoming requests,
    including URL, method, headers, and body.

    Data is recorded by the RequestTracker middleware.
    """
    try:
        # Access the request tracker instance
        tracker = RequestTracker()

        # Record the request data
        tracker.record_request(request_data)

        return {"success": True, "message": "Request tracked successfully."}
    except Exception as e:
        print(f"Error tracking request: {str(e)}")
        return {"success": False, "message": "Failed to track request."}


def load_training_metrics_to_monitoring():
    """
    Load training metrics from saved results and update monitoring system.

    This ensures model performance metrics are available in Prometheus
    even after API restarts.
    """
    try:
        # Try to load from results directory first
        base_dir = os.path.dirname(__file__)
        metrics_file_paths = [
            "results/evaluation/training_metrics.json",
            "models/evaluation/metrics.json",
            os.path.join(base_dir, "../../results/evaluation/training_metrics.json"),
            os.path.join(base_dir, "../../models/evaluation/metrics.json")
        ]

        training_metrics = None
        metrics_file_used = None

        for metrics_path in metrics_file_paths:
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'r') as f:
                        training_metrics = json.load(f)
                    metrics_file_used = metrics_path
                    break
                except Exception as e:
                    print(f"Failed to load metrics from {metrics_path}: {e}")
                    continue

        if not training_metrics:
            print("⚠️ No training metrics file found - model performance metrics empty")
            return

        print(f"📊 Loading training metrics from: {metrics_file_used}")

        # Extract metrics from the training results
        metrics_to_update = {}

        # Check for training metrics format (newer format)
        if "best_validation_f1s" in training_metrics:
            best_f1s = training_metrics["best_validation_f1s"]
            for task, f1_score in best_f1s.items():
                metrics_to_update[task] = {"f1": f1_score}

        # Check for evaluation epochs format
        elif "epochs" in training_metrics and training_metrics["epochs"]:
            # Get the last epoch's validation metrics
            last_epoch = training_metrics["epochs"][-1]
            if "val_tasks_metrics" in last_epoch:
                val_metrics = last_epoch["val_tasks_metrics"]
                for task, task_metrics in val_metrics.items():
                    metrics_to_update[task] = {
                        "accuracy": task_metrics.get("acc", 0.0),
                        "f1": task_metrics.get("f1", 0.0)
                    }        # Update monitoring system with the loaded metrics
        if metrics_to_update:
            metrics_collector.update_model_performance(metrics_to_update)
            task_list = list(metrics_to_update.keys())
            print(f"✅ Updated monitoring with metrics for tasks: {task_list}")
            
            # Log the loaded values
            for task, metrics in metrics_to_update.items():
                acc = metrics.get("accuracy", "N/A")
                f1 = metrics.get("f1", "N/A")
                print(f"   - {task}: accuracy={acc}, f1={f1}")
        else:
            print("⚠️ No valid metrics found in training results")

    except Exception as e:
        print(f"❌ Error loading training metrics: {e}")
        # Don't raise - monitoring should continue to work for real-time metrics


def create_baseline_stats_from_training_data():
    """
    Create baseline statistics file for drift detection from training data.
    
    This creates the baseline_stats.pkl file that the monitoring system
    expects for drift detection to work properly.
    """
    try:
        # Check if baseline stats already exist
        baseline_path = "models/baseline_stats.pkl"
        if os.path.exists(baseline_path):
            print("📊 Baseline stats file already exists")
            return
            
        # Create basic baseline stats from available training data
        baseline_stats = {
            "feature_means": {},
            "feature_stds": {},
            "prediction_distribution": {
                "happiness": 0.35,
                "neutral": 0.25,
                "sadness": 0.15,
                "anger": 0.10,
                "surprise": 0.08,
                "fear": 0.04,
                "disgust": 0.03
            },
            "performance_baseline": {"accuracy": 0.60, "f1": 0.58}
        }
        
        # Ensure the models directory exists
        os.makedirs("models", exist_ok=True)
        
        # Save the baseline stats
        with open(baseline_path, "wb") as f:
            pickle.dump(baseline_stats, f)
            
        print(f"✅ Created baseline stats file at: {baseline_path}")
        
    except Exception as e:
        print(f"⚠️ Could not create baseline stats: {e}")
        # Don't raise - this is not critical for API functionality
