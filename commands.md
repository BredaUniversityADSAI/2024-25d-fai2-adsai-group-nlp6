# Emotion Classification Pipeline - CLI Commands Reference

This document provides a comprehensive reference for all CLI commands available in the emotion classification pipeline.

## Table of Contents
- [Available Commands](#available-commands)
- [Prediction Commands](#prediction-commands)
- [Training Commands](#training-commands)
- [Azure ML Endpoint Commands](#azure-ml-endpoint-commands)
- [Data Commands](#data-commands)
- [Environment Setup](#environment-setup)
- [Common Use Cases](#common-use-cases)

---

## Available Commands

The CLI supports the following main commands:
```bash
python -m emotion_clf_pipeline.cli {data,preprocess,train,predict,endpoint}
```

- **`data`** - 📊 Data preprocessing commands (not yet implemented)
- **`preprocess`** - 🔄 Preprocess raw data for training
- **`train`** - 🚀 Train the emotion classification model
- **`predict`** - 🎭 Predict emotions from video/audio
- **`endpoint`** - 🚀 Azure ML endpoint management

---

## Prediction Commands

### Basic Local Prediction
```bash
# Predict emotions from a YouTube video using local model
python -m emotion_clf_pipeline.cli predict "https://www.youtube.com/watch?v=VIDEO_ID"

# Example with specific video
python -m emotion_clf_pipeline.cli predict "https://www.youtube.com/watch?v=JXulxHKxED4&ab_channel=Datastream"
```

### Local Prediction with Options
```bash
# Use speech-to-text for audio extraction
python -m emotion_clf_pipeline.cli predict "https://www.youtube.com/watch?v=VIDEO_ID" --use-stt

# Specify custom chunk size for text processing
python -m emotion_clf_pipeline.cli predict "https://www.youtube.com/watch?v=VIDEO_ID" --chunk-size 300

# Use custom model and config paths
python -m emotion_clf_pipeline.cli predict "https://www.youtube.com/watch?v=VIDEO_ID" --model-path "path/to/model.pt" --config-path "path/to/config.json"

# Save output to file
python -m emotion_clf_pipeline.cli predict "https://www.youtube.com/watch?v=VIDEO_ID" --output "results/predictions.json"
```

### Azure ML Prediction
```bash
# Predict using Azure ML endpoint (auto-detects configuration from .env)
python -m emotion_clf_pipeline.cli predict "https://www.youtube.com/watch?v=VIDEO_ID" --use-azure

# Use specific Azure endpoint (overrides .env configuration)
python -m emotion_clf_pipeline.cli predict "https://www.youtube.com/watch?v=VIDEO_ID" --use-azure --azure-endpoint "https://your-endpoint.region.inference.ml.azure.com/score" --azure-api-key "your-api-key"

# Use NGROK tunnel for Azure endpoint
python -m emotion_clf_pipeline.cli predict "https://www.youtube.com/watch?v=VIDEO_ID" --use-azure --use-ngrok --server-ip "226"

# Azure with STT and output file
python -m emotion_clf_pipeline.cli predict "https://www.youtube.com/watch?v=VIDEO_ID" --use-azure --use-stt --output "results/predictions.json"
```

---

## Preprocess Commands

### Data Preprocessing
```bash
# Basic preprocessing with default paths
python -m emotion_clf_pipeline.cli preprocess

# Preprocess with custom paths
python -m emotion_clf_pipeline.cli preprocess --raw-train-path "data/raw/train" --raw-test-path "data/raw/test/test_data-0001.csv"

# Preprocess with verbose output
python -m emotion_clf_pipeline.cli preprocess --verbose --raw-train-path "data/raw/train" --raw-test-path "data/raw/test/test_data-0001.csv"

# Preprocess with custom output directory
python -m emotion_clf_pipeline.cli preprocess --output-dir "data/custom_processed" --encoders-dir "models/custom_encoders"

# Preprocess with Azure ML
python -m emotion_clf_pipeline.cli preprocess --azure --verbose
```

### Available Preprocess Options
- `--raw-train-path`: Path to raw training data (directory or CSV file)
- `--raw-test-path`: Path to raw test data CSV file  
- `--output-dir`: Output directory for processed data (default: "data/processed")
- `--encoders-dir`: Directory to save label encoders (default: "models/encoders")
- `--model-name-tokenizer`: HuggingFace model name for tokenizer
- `--max-length`: Maximum sequence length for tokenization
- `--output-tasks`: Comma-separated list of output tasks
- `--verbose`: Enable verbose logging
- `--mode`: Execution mode (local/azure)
- `--azure`: Use Azure ML for preprocessing

---

## Training Commands

### Train Model
```bash
# Train model with default configuration
python -m emotion_clf_pipeline.cli train

# Train with custom configuration file
python -m emotion_clf_pipeline.cli train --config "path/to/config.json"

# Train with specific hyperparameters
python -m emotion_clf_pipeline.cli train --epochs 10 --batch-size 16 --learning-rate 2e-5

# Train with custom output directory
python -m emotion_clf_pipeline.cli train --output-dir "models/custom"
```

---

## Azure ML Endpoint Commands

### Deploy to Azure ML
```bash
# Deploy complete pipeline to Azure ML Kubernetes endpoint
python -m emotion_clf_pipeline.cli endpoint deploy

# Force update existing deployment
python -m emotion_clf_pipeline.cli endpoint deploy --force-update

# Deploy with JSON output
python -m emotion_clf_pipeline.cli endpoint deploy --json
```

### Test Azure Endpoint
```bash
# Test deployed endpoint with sample data
python -m emotion_clf_pipeline.cli endpoint test

# Test with JSON output
python -m emotion_clf_pipeline.cli endpoint test --json
```

### Get Endpoint Information
```bash
# Get endpoint details including scoring URI and keys
python -m emotion_clf_pipeline.cli endpoint details

# Get details in JSON format
python -m emotion_clf_pipeline.cli endpoint details --json
```

### Clean Up Azure Resources
```bash
# Delete endpoint and all associated deployments
python -m emotion_clf_pipeline.cli endpoint cleanup
```



---

## Data Commands

**⚠️ Note**: Data preprocessing commands are not yet fully implemented in the current CLI version.

```bash
# Check data command status
python -m emotion_clf_pipeline.cli data
# Currently shows: "📊 Data preprocessing commands not yet implemented in this version."
```

### Alternative Data Processing
For data preprocessing, you can use the internal modules directly:

```python
# Using Python directly for data preprocessing
from emotion_clf_pipeline.data import DatasetLoader, DataPreparation

# Load and preprocess data
loader = DatasetLoader()
preprocessor = DataPreparation()

# Process your data files
train_data = loader.load_from_path("data/raw/train/")
test_data = loader.load_from_csv("data/raw/test/test_data-0001.csv")
```

---

## Environment Setup

### Initial Setup
```bash
# Install package in development mode
pip install -e .

# Install with all dependencies
pip install -e ".[dev,azure,monitoring]"

# Set up environment file
cp .env.example .env
# Edit .env with your configuration
```

### Azure Configuration
```bash
# Set up Azure ML configuration
python -m emotion_clf_pipeline.cli setup-azure

# Test Azure connection
python -m emotion_clf_pipeline.cli test-azure-connection

# Generate Azure service principal credentials
python -m emotion_clf_pipeline.cli generate-azure-credentials
```

### Docker Setup
```bash
# Build Docker image
docker-compose build

# Run full stack
docker-compose up

# Run in production mode
docker-compose -f docker-compose.yml -f docker-compose.build.yml up
```

---

## Environment Variables

### Required Environment Variables
```bash
# Azure ML Configuration
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_RESOURCE_GROUP="your-resource-group"
export AZURE_WORKSPACE_NAME="your-workspace-name"
export AZURE_TENANT_ID="your-tenant-id"
export AZURE_CLIENT_ID="your-client-id"
export AZURE_CLIENT_SECRET="your-client-secret"

# Azure ML Endpoint Configuration
export AZURE_ENDPOINT_URL="https://your-endpoint.region.inference.ml.azure.com/score"
export AZURE_API_KEY="your-api-key"

# Optional: NGROK Configuration
export USE_NGROK="true"
export NGROK_SERVER_IP="your-server-ip"

# Optional: Model Configuration
export MODEL_PATH="models/weights/baseline_weights.pt"
export CONFIG_PATH="models/weights/model_config.json"
```

---

## Common Use Cases

### 1. Quick Local Prediction
```bash
# Analyze emotions in a YouTube video
python -m emotion_clf_pipeline.cli predict "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

### 2. Azure Endpoint Prediction
```bash
# Use Azure ML endpoint for prediction (loads config from .env)
python -m emotion_clf_pipeline.cli predict "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --use-azure
```

### 3. Deploy and Test Azure Endpoint
```bash
# Deploy to Azure
python -m emotion_clf_pipeline.cli endpoint deploy --force-update

# Test deployment
python -m emotion_clf_pipeline.cli endpoint test

# Get endpoint details
python -m emotion_clf_pipeline.cli endpoint details --json
```

### 4. Train New Model
```bash
# Train model with custom parameters
python -m emotion_clf_pipeline.cli train --epochs 15 --learning-rate 1e-5 --batch-size 16

# Deploy updated model to Azure
python -m emotion_clf_pipeline.cli endpoint deploy --force-update
```

---

## Troubleshooting

### Getting Help
```bash
# Get general help
python -m emotion_clf_pipeline.cli --help

# Get help for specific commands
python -m emotion_clf_pipeline.cli predict --help
python -m emotion_clf_pipeline.cli train --help
python -m emotion_clf_pipeline.cli endpoint --help
python -m emotion_clf_pipeline.cli endpoint deploy --help
```

### Common Issues

**Issue: Command not found error**
```bash
# ❌ This will fail:
python -m emotion_clf_pipeline.cli preprocess

# ✅ Use this instead:
python -m emotion_clf_pipeline.cli data
# (Note: data preprocessing is not yet implemented)
```

**Issue: Azure authentication**
- Ensure your `.env` file contains valid Azure credentials
- Check that your Azure subscription and workspace details are correct

**Issue: Model loading errors**
- Verify model files exist in `models/weights/` directory
- Check that model configuration matches the actual trained model

---

## Output Formats

### JSON Output
Most commands support `--json` flag for machine-readable output:
```bash
python -m emotion_clf_pipeline.cli predict "URL" --json > results.json
python -m emotion_clf_pipeline.cli endpoint details --json > endpoint_info.json
```

### File Output
Many commands support `--output` flag:
```bash
python -m emotion_clf_pipeline.cli predict "URL" --output "predictions.xlsx"
python -m emotion_clf_pipeline.cli evaluate --output "evaluation_report.json"
```

---

## Performance Tips

1. **Use appropriate chunk sizes**: `--chunk-size 200` for balanced performance
2. **Enable GPU when available**: Models automatically use CUDA if available
3. **Use Azure for large batches**: Azure endpoint can handle multiple concurrent requests
4. **Cache models locally**: Local models are cached after first load
5. **Use JSON output for automation**: `--json` flag provides structured output

---

For more detailed information about specific commands, use:
```bash
python -m emotion_clf_pipeline.cli COMMAND --help
``` 