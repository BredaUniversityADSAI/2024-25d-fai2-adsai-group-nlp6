from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

# Assuming predict.py is in the same directory or accessible via PYTHONPATH
from .predict import predict_emotion


# Initialize FastAPI app
app = FastAPI(
    title="Emotion Classification API",
    description=(
        "API for predicting emotion from text using the emotion classification pipeline."
    ),
    version="0.1.0",
)


# --- Pydantic Models ---

class PredictionRequest(BaseModel):
    """Request model for text input."""
    text: str


class PredictionResponse(BaseModel):
    """Response model for prediction output."""
    emotion: str
    sub_emotion: str
    intensity: float


# --- API Endpoints ---

@app.post("/predict", response_model=PredictionResponse)
def handle_prediction(request: PredictionRequest) -> PredictionResponse:
    """
    Accepts text input via POST request and returns emotion predictions.
    
    Args:
        request: The request body containing the text.
        
    Returns:
        The prediction response containing emotion, sub_emotion, and intensity.
    """
    prediction_result: Dict[str, Any] = predict_emotion(request.text)
    
    # Ensure the result matches the response model structure
    # Handles potential discrepancies between predict_emotion output and PredictionResponse
    response = PredictionResponse(
        emotion=prediction_result.get("emotion", "unknown"),
        sub_emotion=prediction_result.get("sub_emotion", "unknown"),
        intensity=float(prediction_result.get("intensity", 0.0))
    )
    
    return response


# --- Root Endpoint (Optional) ---

@app.get("/")
def read_root():
    """Provides a simple welcome message for the API root."""
    return {
        "message": "Welcome to the Emotion Classification API! Use the /predict endpoint to analyze text."
    }


# --- Running the API (Example using uvicorn) ---
# To run this API locally:
# 1. Ensure FastAPI and Uvicorn are installed: poetry add fastapi uvicorn
# 2. Run from the project root directory:
#    uvicorn src.emotion_clf_pipeline.api:app --reloa