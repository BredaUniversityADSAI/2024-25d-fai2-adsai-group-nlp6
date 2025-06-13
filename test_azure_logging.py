#!/usr/bin/env python3
"""
Test script for Azure ML logging functionality.

This script validates that the AzureMLLogger class properly handles:
- Metric logging to both MLflow and Azure ML
- Artifact logging including images and files
- Error handling when Azure ML is not available
- Proper cleanup and session management
"""

import os
import sys
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def create_test_plot(output_path):
    """Create a simple test plot for artifact logging."""
    plt.figure(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('Test Plot for Azure ML Logging')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Test plot created: {output_path}")

def test_azure_ml_logger():
    """Test the AzureMLLogger functionality."""
    print("=== Testing Azure ML Logger ===")
    
    try:
        from emotion_clf_pipeline.train import AzureMLLogger
        
        # Initialize logger
        print("1. Initializing AzureMLLogger...")
        logger = AzureMLLogger()
        print(f"   ✓ Logger initialized")
        print(f"   - Azure ML detected: {logger.is_azure_ml}")
        print(f"   - MLflow active: {logger.mlflow_active}")
        
        # Start logging session
        print("\n2. Starting logging session...")
        logger.start_logging(run_name="test-azure-ml-logging")
        print("   ✓ Logging session started")
        
        # Test parameter logging
        print("\n3. Testing parameter logging...")
        test_params = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "model_name": "microsoft/deberta-v3-xsmall",
            "epochs": 5
        }
        
        for param_name, param_value in test_params.items():
            logger.log_param(param_name, param_value)
            print(f"   ✓ Logged parameter: {param_name} = {param_value}")
        
        # Test metric logging
        print("\n4. Testing metric logging...")
        test_metrics = [
            ("train_loss", 0.8, 1),
            ("train_loss", 0.6, 2),
            ("train_loss", 0.4, 3),
            ("val_accuracy", 0.75, 1),
            ("val_accuracy", 0.82, 2),
            ("val_accuracy", 0.89, 3),
            ("final_f1_score", 0.85, None)
        ]
        
        for metric_name, metric_value, step in test_metrics:
            logger.log_metric(metric_name, metric_value, step=step)
            step_str = f"step {step}" if step is not None else "final"
            print(f"   ✓ Logged metric: {metric_name} = {metric_value} ({step_str})")
        
        # Test artifact logging
        print("\n5. Testing artifact logging...")
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_plot_path = os.path.join(temp_dir, "test_evaluation_plot.png")
            test_config_path = os.path.join(temp_dir, "test_config.json")
            
            # Create test plot
            create_test_plot(test_plot_path)
            
            # Create test config file
            import json
            test_config = {
                "model_config": {
                    "architecture": "deberta",
                    "hidden_size": 768,
                    "num_layers": 12
                },
                "training_config": {
                    "optimizer": "AdamW",
                    "scheduler": "linear_warmup"
                }
            }
            
            with open(test_config_path, 'w') as f:
                json.dump(test_config, f, indent=2)
            print(f"   ✓ Test config created: {test_config_path}")
            
            # Log artifacts
            logger.log_image(test_plot_path, name="Test Evaluation Plot")
            print("   ✓ Logged image artifact")
            
            logger.log_artifact(test_config_path, "config/test_config.json")
            print("   ✓ Logged config artifact")
        
        # Test evaluation plots creation
        print("\n6. Testing evaluation plot creation...")
        with tempfile.TemporaryDirectory() as eval_dir:
            # Create mock test data
            output_tasks = ["emotion", "sub_emotion"]
            test_preds = {
                "emotion": [0, 1, 2, 1, 0, 2, 1],
                "sub_emotion": [0, 1, 2, 0, 1, 2, 1]
            }
            test_labels = {
                "emotion": [0, 1, 2, 1, 0, 1, 1],  # Some misclassifications
                "sub_emotion": [0, 1, 2, 0, 1, 1, 1]
            }
            test_metrics = {
                "emotion": {
                    "accuracy": 0.857,
                    "f1_score": 0.832,
                    "precision": 0.845,
                    "recall": 0.820
                },
                "sub_emotion": {
                    "accuracy": 0.714,
                    "f1_score": 0.698,
                    "precision": 0.705,
                    "recall": 0.691
                }
            }
            
            # Create evaluation plots
            logger.create_evaluation_plots(
                test_preds, test_labels, test_metrics, eval_dir, output_tasks
            )
            
            # Check if plots were created
            plot_files = [f for f in os.listdir(eval_dir) if f.endswith('.png')]
            print(f"   ✓ Created {len(plot_files)} evaluation plots:")
            for plot_file in plot_files:
                print(f"     - {plot_file}")
            
            # Log evaluation artifacts
            logger.log_evaluation_artifacts(eval_dir)
            print("   ✓ Logged evaluation artifacts")
        
        # Test table logging (Azure ML specific)
        print("\n7. Testing table logging...")
        if logger.azure_run:
            sample_table = {
                "epoch": [1, 2, 3, 4, 5],
                "train_loss": [0.8, 0.6, 0.4, 0.3, 0.2],
                "val_accuracy": [0.75, 0.82, 0.89, 0.91, 0.93]
            }
            logger.log_table("training_progress", sample_table)
            print("   ✓ Logged table data")
        else:
            print("   - Table logging skipped (Azure ML not available)")
        
        # End logging session
        print("\n8. Ending logging session...")
        logger.end_logging()
        print("   ✓ Logging session ended")
        
        # Complete run (for Azure ML)
        if logger.azure_run:
            logger.complete_run()
            print("   ✓ Azure ML run completed")
        
        print("\n=== Azure ML Logger Test PASSED ===")
        print(f"Environment: {'Azure ML' if logger.is_azure_ml else 'Local'}")
        print(f"MLflow: {'Active' if logger.mlflow_active else 'Inactive'}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Azure ML Logger Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_detection():
    """Test environment detection logic."""
    print("\n=== Testing Environment Detection ===")
    
    try:
        from emotion_clf_pipeline.train import AzureMLLogger
        
        # Test current environment
        logger = AzureMLLogger()
        
        print("Current Environment Analysis:")
        print(f"  - Azure ML detected: {logger.is_azure_ml}")
        print(f"  - MLflow active: {logger.mlflow_active}")
        
        # Check Azure ML environment variables
        azure_vars = [
            'AZUREML_RUN_ID',
            'AZUREML_SERVICE_ENDPOINT', 
            'AZUREML_RUN_TOKEN',
            'AZUREML_ARM_SUBSCRIPTION',
            'AZUREML_ARM_RESOURCEGROUP'
        ]
        
        print("\nAzure ML Environment Variables:")
        for var in azure_vars:
            value = os.getenv(var)
            status = "✓ Set" if value else "✗ Not set"
            print(f"  - {var}: {status}")
        
        # Check Azure ML SDK availability
        try:
            from azureml.core import Run
            print("\nAzure ML SDK: ✓ Available")
            
            try:
                run = Run.get_context()
                if hasattr(run, 'experiment'):
                    print(f"  - Active run context: ✓ {run.id}")
                else:
                    print("  - Run context: ✗ Offline mode")
            except Exception as e:
                print(f"  - Run context error: {e}")
                
        except ImportError:
            print("\nAzure ML SDK: ✗ Not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment detection test failed: {e}")
        return False

def main():
    """Main test function."""
    print("Azure ML Logger Validation Test")
    print("=" * 50)
    
    # Test environment detection
    env_test_passed = test_environment_detection()
    
    # Test Azure ML logger functionality  
    logger_test_passed = test_azure_ml_logger()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY:")
    print(f"  Environment Detection: {'✓ PASSED' if env_test_passed else '❌ FAILED'}")
    print(f"  Azure ML Logger: {'✓ PASSED' if logger_test_passed else '❌ FAILED'}")
    
    overall_success = env_test_passed and logger_test_passed
    print(f"\nOverall Result: {'✓ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if overall_success:
        print("\n🎉 Azure ML logging functionality is ready!")
        print("You can now run training with proper Azure ML integration.")
    else:
        print("\n⚠️  Some issues detected. Check the errors above.")
        print("Training will still work but Azure ML features may be limited.")
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    exit(main())
