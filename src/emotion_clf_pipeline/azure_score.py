"""
Azure ML Endpoint Scoring Script
Loads model from Azure ML model registry and handles inference requests.
"""

import os
import sys
import json
import logging
import torch
import nltk
from transformers import AutoTokenizer
from typing import Optional

# HACK: Temporarily add the application's root to the path for module resolution
# in the Azure ML container. This is a workaround for environment limitations
# and should be replaced by proper package installation in the deployment environment.
sys.path.append("/var/azureml-app")
from emotion_clf_pipeline.model import DEBERTAClassifier  # noqa: E402
from emotion_clf_pipeline.features import FeatureExtractor  # noqa: E402

# Global variables
model = None
config = None
tokenizer = None
feature_extractor = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _find_file(name: str, path: str) -> Optional[str]:
    """Recursively find a file in a directory."""
    for root, _, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
    return None


def _download_nltk_resource(resource_name, resource_path):
    """Safely download a single NLTK resource."""
    try:
        nltk.data.find(resource_path)
        logger.info(f"✓ NLTK resource '{resource_name}' already available.")
    except LookupError:
        try:
            logger.info(f"⬇ Downloading NLTK resource '{resource_name}'...")
            nltk.download(resource_name, quiet=True)
            logger.info(f"✅ Successfully downloaded '{resource_name}'.")
        except Exception as e:
            logger.warning(f"⚠ Failed to download NLTK resource '{resource_name}': {e}")


def download_nltk_resources():
    """
    Download all required NLTK resources for the Azure ML environment.
    """
    logger.info("📦 Setting up NLTK resources...")
    required_resources = [
        ("punkt", "tokenizers/punkt"),
        ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
        ("vader_lexicon", "sentiment/vader_lexicon.zip/vader_lexicon/vader_lexicon.xml"),
        ("stopwords", "corpora/stopwords"),
    ]
    for name, path in required_resources:
        _download_nltk_resource(name, path)


def _load_config(model_path):
    """Load model configuration from JSON file."""
    config_path = _find_file("model_config.json", model_path)
    if not config_path:
        logger.error(f"Could not find 'model_config.json' in path: {model_path}")
        # Log directory contents for debugging
        for root, dirs, files in os.walk(model_path):
            logger.error(f"Contents of {root}: Dirs={dirs}, Files={files}")
        raise FileNotFoundError(f"model_config.json not found in {model_path} or its subdirectories.")

    with open(config_path, "r") as f:
        config = json.load(f)
    logger.info(f"⚙ Loaded model config from {config_path}")
    return config, os.path.dirname(config_path)


def _initialize_tokenizer(model_name):
    """Initialize the tokenizer for the specified model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"🔤 Initialized tokenizer for {model_name}")
    return tokenizer


def _initialize_feature_extractor(model_path):
    """Initialize the feature extractor with fallback logic."""
    emolex_path = _find_file("emolex_lexicon.txt", model_path)
    if not emolex_path:
        logger.warning(
            f"⚠ EmoLex lexicon not found in {model_path}. "
            "Feature extractor will use default settings without EmoLex."
        )
        emolex_path = None

    try:
        feature_extractor = FeatureExtractor(lexicon_path=emolex_path)
        logger.info("🎯 Initialized feature extractor.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize feature extractor: {e}, using fallback.")
        feature_extractor = FeatureExtractor(
            feature_config={
                "pos": True,
                "textblob": True,
                "vader": True,
                "tfidf": False,  # Disable for safety
                "emolex": False,
            },
            lexicon_path=None,
        )
        logger.info("🎯 Initialized feature extractor with fallback configuration.")
    return feature_extractor


def init():
    """Initialize the model, tokenizer, and feature extractor."""
    global model, config, tokenizer, feature_extractor

    logger.info("🚀 Initializing Azure ML scoring service...")

    download_nltk_resources()

    model_root_path = os.getenv("AZUREML_MODEL_DIR")
    if not model_root_path or not os.path.isdir(model_root_path):
        raise EnvironmentError(
            "AZUREML_MODEL_DIR environment variable not set or invalid."
        )
    logger.info(f"📁 Loading model from root path: {model_root_path}")

    config, model_content_path = _load_config(model_root_path)
    tokenizer = _initialize_tokenizer(config["model_name"])
    feature_extractor = _initialize_feature_extractor(model_content_path)

    model = DEBERTAClassifier(
        model_name=config["model_name"],
        feature_dim=config["feature_dim"],
        num_classes=config["num_classes"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    )

    weights_path = os.path.join(model_content_path, "baseline_weights.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    logger.info("✅ Azure ML scoring service initialization complete!")


def run(raw_data):
    """Run inference on the input data."""
    global model, config, tokenizer, feature_extractor

    try:
        data = json.loads(raw_data)
        text = data.get("text", "")

        if not text:
            logger.warning("⚠ Empty text received for inference")
            return json.dumps({"error": "Empty text provided"})

        logger.info(f"🔍 Processing text: {text[:50]}...")

        try:
            expected_dim = config.get("feature_dim")
            features = feature_extractor.extract_all_features(text, expected_dim)
            logger.info(f"📊 Extracted {len(features)} features")
        except Exception as e:
            logger.error(f"❌ Feature extraction failed: {e}", exc_info=True)
            return json.dumps({"error": f"Feature extraction failed: {str(e)}"})

        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=128
        )

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                features=torch.tensor(features).unsqueeze(0),
            )

        result = {
            key: value.tolist() if isinstance(value, torch.Tensor) else value
            for key, value in outputs.items()
        }

        logger.info("✅ Inference completed successfully")
        return json.dumps(result)

    except json.JSONDecodeError:
        logger.error("❌ Invalid JSON input received.", exc_info=True)
        return json.dumps({"error": "Invalid JSON format"})
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}", exc_info=True)
        return json.dumps({"error": f"Inference failed: {str(e)}"})
