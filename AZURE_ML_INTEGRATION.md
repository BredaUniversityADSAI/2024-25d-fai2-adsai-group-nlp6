# Azure ML Integration for Emotion Classification Pipeline

## Summary of Changes

This document summarizes the comprehensive Azure ML integration implemented to ensure metrics and images appear properly in the Azure ML job overview.

## Problem Addressed

The original issue was that metrics and evaluation plots were not appearing in Azure ML's job overview when running the training pipeline. This was caused by:

1. **Incomplete Azure ML Integration**: The code relied primarily on MLflow logging without proper Azure ML native logging
2. **Missing Image Logging**: Evaluation plots were not logged as images that Azure ML could visualize
3. **Inadequate Artifact Management**: Files weren't properly uploaded to Azure ML's artifact storage
4. **No Dual Logging Strategy**: No fallback between MLflow and Azure ML native logging

## Solution Implemented

### 1. AzureMLLogger Class

**Location**: `src/emotion_clf_pipeline/train.py` (lines 67-461)

A comprehensive logging class that provides:

- **Dual Logging Strategy**: Logs to both MLflow and Azure ML native APIs
- **Environment Detection**: Automatically detects Azure ML vs local environments
- **Image Logging**: Proper Azure ML image logging for visualization
- **Artifact Management**: File upload to Azure ML outputs directory
- **Error Handling**: Graceful fallbacks when Azure ML is unavailable

Key Methods:
```python
def log_metric(self, key: str, value: float, step: int = None)
def log_param(self, key: str, value)
def log_artifact(self, local_path: str, artifact_path: str = None)
def log_image(self, image_path: str, name: str = None)
def log_table(self, name: str, data: dict)
def create_evaluation_plots(...)
def log_evaluation_artifacts(...)
```

### 2. Enhanced Training Loop

**Location**: `src/emotion_clf_pipeline/train.py` - `train_and_evaluate` method (lines 978-1179)

The training loop now includes:

- **Comprehensive Metric Logging**: Per-epoch and final metrics logged to Azure ML
- **Step-based Tracking**: Metrics logged with step information for trend visualization
- **Image Generation**: Automatic creation of evaluation plots (confusion matrices, performance charts)
- **Artifact Persistence**: All plots and metrics files uploaded as Azure ML artifacts

### 3. Evaluation Visualization

**Location**: `src/emotion_clf_pipeline/train.py` - `create_evaluation_plots` method (lines 306-426)

Generates comprehensive visualizations:

1. **Confusion Matrices**: Per-task confusion matrices with heatmap visualization
2. **Performance Metrics Charts**: Bar charts showing accuracy, F1, precision, recall
3. **Overall Comparison**: Multi-task performance comparison plots

All plots are:
- Saved as high-resolution PNG files
- Logged as Azure ML images for in-UI visualization
- Uploaded as artifacts for persistence

### 4. Azure ML Native Integration

The implementation properly uses Azure ML's native logging APIs:

```python
# Metric logging with step tracking
self.azure_run.log(f"{key}_step", value, step=step)
self.azure_run.log(key, value)

# Image logging for visualization
self.azure_run.log_image(display_name, path=image_path)

# File uploads to outputs directory
self.azure_run.upload_file(artifact_path, azure_path)

# Table logging for structured data
self.azure_run.log_table(name, data)
```

## Key Features

### 1. Environment-Aware Operation

The logger automatically detects the environment and adapts:

- **Azure ML Environment**: Uses both Azure ML native APIs and MLflow
- **Local Environment**: Falls back to MLflow-only logging
- **No Azure ML SDK**: Graceful degradation with warnings

### 2. Comprehensive Error Handling

Every logging operation includes try/catch blocks with meaningful warnings, ensuring training continues even if logging fails.

### 3. Dual Logging Strategy

Metrics are logged to both systems to ensure compatibility:
- **Azure ML**: For job overview visualization and native Azure ML features
- **MLflow**: For backward compatibility and local development

### 4. Proper Artifact Management

Files are managed according to Azure ML conventions:
- **Outputs Directory**: All artifacts saved to `outputs/` for Azure ML persistence
- **Structured Paths**: Organized artifact paths (`images/`, `evaluation/`, `metrics/`)
- **File Copying**: Local files copied to appropriate Azure ML locations

## Usage Examples

### Basic Metric Logging

```python
# Initialize logger
azure_logger = AzureMLLogger()

# Start logging session
azure_logger.start_logging(run_name="emotion-classification-training")

# Log parameters
azure_logger.log_param("learning_rate", 2e-5)
azure_logger.log_param("batch_size", 16)

# Log metrics with steps (for per-epoch tracking)
for epoch in range(epochs):
    azure_logger.log_metric("train_loss", loss_value, step=epoch+1)
    azure_logger.log_metric("val_accuracy", accuracy, step=epoch+1)

# Log final metrics
azure_logger.log_metric("final_f1_score", f1_score)

# End session
azure_logger.end_logging()
```

### Evaluation Plot Creation

```python
# Create evaluation plots
azure_logger.create_evaluation_plots(
    test_preds, test_labels, test_metrics, 
    evaluation_dir, output_tasks
)

# Log all generated artifacts
azure_logger.log_evaluation_artifacts(evaluation_dir)
```

## Expected Azure ML Job Overview

After implementing these changes, the Azure ML job overview should display:

### Metrics Tab
- **Training Progress**: Per-epoch loss and accuracy curves
- **Validation Metrics**: F1, precision, recall trends over epochs
- **Final Performance**: Summary metrics for all tasks

### Images Tab
- **Confusion Matrices**: Interactive heatmaps for each classification task
- **Performance Charts**: Bar charts comparing metrics across tasks
- **Overall Comparison**: Multi-task performance visualization

### Outputs and Logs
- **Evaluation Directory**: Complete evaluation results and plots
- **Metrics Files**: JSON files with detailed training metrics
- **Model Artifacts**: Trained model weights and configuration

## Testing

A comprehensive test script has been created at `test_azure_logging.py` to validate:

1. **Environment Detection**: Proper Azure ML environment identification
2. **Logging Functionality**: Metric, parameter, and artifact logging
3. **Image Creation**: Evaluation plot generation and logging
4. **Error Handling**: Graceful fallbacks when Azure ML unavailable

Run the test with:
```bash
python test_azure_logging.py
```

## Configuration Requirements

### Environment Variables (for Azure ML)
```bash
AZUREML_RUN_ID=<run_id>
AZUREML_SERVICE_ENDPOINT=<endpoint>
AZUREML_RUN_TOKEN=<token>
AZUREML_ARM_SUBSCRIPTION=<subscription_id>
AZUREML_ARM_RESOURCEGROUP=<resource_group>
```

### Required Packages
```bash
pip install azureml-core mlflow matplotlib seaborn scikit-learn
```

## Troubleshooting

### Common Issues

1. **No Metrics in Azure ML UI**
   - Check Azure ML SDK is installed
   - Verify environment variables are set
   - Check Azure ML run context is active

2. **Images Not Appearing**
   - Ensure matplotlib/seaborn are installed
   - Check image files are created successfully
   - Verify Azure ML image logging calls

3. **Artifacts Missing**
   - Check outputs directory permissions
   - Verify file copying operations succeed
   - Check Azure ML upload calls

### Debug Information

The logger provides detailed debug information:
```python
logger.info(f"Azure ML Logger initialized - Azure ML: {self.is_azure_ml}, "
            f"MLflow: {self.mlflow_active}")
```

## Backward Compatibility

The implementation maintains full backward compatibility:
- **Existing MLflow code**: Still works unchanged
- **Local Development**: No Azure ML required for local training
- **Legacy Metrics**: All existing metric calculations preserved

## Performance Impact

The Azure ML integration has minimal performance impact:
- **Async Operations**: Logging operations don't block training
- **Error Isolation**: Logging failures don't affect model training
- **Efficient File Operations**: Files copied only when needed

This comprehensive integration ensures that your emotion classification pipeline provides rich visualization and monitoring capabilities in Azure ML while maintaining robust operation in all environments.
