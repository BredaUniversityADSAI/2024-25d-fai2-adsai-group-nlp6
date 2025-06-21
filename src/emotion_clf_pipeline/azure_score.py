"""
Azure ML Endpoint Scoring Script
Loads model from Azure ML model registry and handles inference requests.
"""

import os
import json
import torch
from azureml.core.model import Model
from transformers import AutoTokenizer
from src.emotion_clf_pipeline.model import DEBERTAClassifier

model = None
config = None


def init():
    """Initialize the model and configuration."""
    global model, config

    # Get model path from Azure ML
    model_path = Model.get_model_path(model_name="emotion-clf-baseline")
    config_path = os.path.join(model_path, "model_config.json")

    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load model
    model = DEBERTAClassifier(
        model_name=config["model_name"],
        feature_dim=config["feature_dim"],
        num_classes=config["num_classes"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    )

    weights_path = os.path.join(model_path, "baseline_weights.pt")
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()


def run(raw_data):
    """Run inference on the input data."""
    from torch import tensor
    from src.emotion_clf_pipeline.features import FeatureExtractor

    # Parse input data
    data = json.loads(raw_data)
    text = data["text"]

    # Prepare tokenizer and features
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    features = FeatureExtractor.extract(text)
    # Tokenize input
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=128
    )

    # Run inference
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            features=tensor(features).unsqueeze(0),
        )

    # Return predictions as JSON
    return json.dumps({k: v.tolist() for k, v in outputs.items()})
