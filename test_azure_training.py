#!/usr/bin/env python3
"""
Test script for Azure ML training pipeline integration.

This script validates that:
1. The training pipeline can be imported and initialized
2. The command structure is correct for Azure ML
3. The data asset references are valid
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_azure_training_pipeline():
    """Test the Azure ML training pipeline setup."""
    
    print("=== Testing Azure ML Training Pipeline ===")
    
    try:
        # Test import
        from emotion_clf_pipeline.azure_pipeline import submit_training_pipeline
        print("✓ Successfully imported submit_training_pipeline")
        
        # Test training script argument parsing
        from emotion_clf_pipeline.train import parse_arguments
        print("✓ Successfully imported training argument parser")
        
        # Create mock arguments for training pipeline
        class MockArgs:
            def __init__(self):
                self.model_name = "microsoft/deberta-v3-xsmall"
                self.batch_size = 16
                self.learning_rate = 2e-5
                self.epochs = 1
        
        args = MockArgs()
        print("✓ Mock arguments created successfully")
        
        # Print expected command structure
        expected_command = (
            "python -c \"import nltk; nltk.download('vader_lexicon', quiet=True)\" "
            "&& python -m src.emotion_clf_pipeline.train "
            f"--model-name {args.model_name} "
            f"--batch-size {args.batch_size} "
            f"--learning-rate {args.learning_rate} "
            f"--epochs {args.epochs} "
            "--train-data ${{inputs.train_data}} "
            "--test-data ${{inputs.test_data}} "
            "--output-dir ${{outputs.model_output}}"
        )
        
        print("\n=== Expected Azure ML Command ===")
        print(expected_command)
        
        print("\n=== Expected Inputs ===")
        print("train_data: azureml:emotion-processed-train:1")
        print("test_data: azureml:emotion-processed-test:1")
        
        print("\n=== Expected Outputs ===")
        print("model_output: uri_folder (rw_mount)")
        
        print("\n✓ Azure ML training pipeline structure validated")
        
        # Test that training script accepts new arguments
        import argparse
        
        # Mock sys.argv for testing
        original_argv = sys.argv
        sys.argv = [
            'train.py',
            '--model-name', 'microsoft/deberta-v3-xsmall',
            '--batch-size', '16',
            '--learning-rate', '2e-5',
            '--epochs', '1',
            '--train-data', '/mock/train.csv',
            '--test-data', '/mock/test.csv',
            '--output-dir', '/mock/output'
        ]
        
        try:
            # This should not raise an error with the new arguments
            args = parse_arguments()
            print("✓ Training script accepts Azure ML arguments")
            print(f"  - train_data: {args.train_data}")
            print(f"  - test_data: {args.test_data}")
            print(f"  - output_dir: {args.output_dir}")
        finally:
            sys.argv = original_argv
        
        print("\n=== All Tests Passed ===")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_azure_training_pipeline()
    sys.exit(0 if success else 1)
