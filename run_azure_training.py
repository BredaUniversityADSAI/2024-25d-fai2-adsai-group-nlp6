#!/usr/bin/env python3
"""
Script to run the actual Azure ML training pipeline.
This will submit a real training job to Azure ML using the registered data assets.
"""

import os
import sys
import logging
from datetime import datetime

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from emotion_clf_pipeline.azure_pipeline import submit_training_pipeline
from emotion_clf_pipeline.utils import get_azure_ml_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run Azure ML training pipeline with registered data assets."""
    
    try:
        print("=== Running Azure ML Training Pipeline ===")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        # Get Azure ML client
        print("\n1. Connecting to Azure ML workspace...")
        ml_client = get_azure_ml_client()
        print(f"✓ Connected to workspace: {ml_client.workspace_name}")
        
        # Submit training pipeline
        print("\n2. Submitting training pipeline...")
        print("   - Using registered data assets:")
        print("     * train_data: azureml:emotion-processed-train:1")
        print("     * test_data: azureml:emotion-processed-test:1")
        print("   - Model: microsoft/deberta-v3-xsmall")
        print("   - Training parameters: batch_size=16, lr=2e-05, epochs=1")
        
        # Submit the job
        job = submit_training_pipeline(
            ml_client=ml_client,
            model_name="microsoft/deberta-v3-xsmall",
            batch_size=16,
            learning_rate=2e-05,
            epochs=1,
            train_data_asset="azureml:emotion-processed-train:1",
            test_data_asset="azureml:emotion-processed-test:1"
        )
        
        print(f"\n✓ Training job submitted successfully!")
        print(f"  Job Name: {job.name}")
        print(f"  Job ID: {job.id}")
        print(f"  Status: {job.status}")
        
        # Get job URL
        job_url = f"https://ml.azure.com/runs/{job.name}?wsid=/subscriptions/{ml_client.subscription_id}/resourcegroups/{ml_client.resource_group_name}/workspaces/{ml_client.workspace_name}"
        print(f"  Job URL: {job_url}")
        
        print(f"\n=== Training Pipeline Submitted Successfully ===")
        print("The training job is now running in Azure ML.")
        print("You can monitor its progress in the Azure ML Studio.")
        print(f"Job will train for 1 epoch with the registered processed datasets.")
        
        return job
        
    except Exception as e:
        logger.error(f"Failed to submit training pipeline: {str(e)}")
        logger.exception("Full error details:")
        sys.exit(1)

if __name__ == "__main__":
    job = main()
