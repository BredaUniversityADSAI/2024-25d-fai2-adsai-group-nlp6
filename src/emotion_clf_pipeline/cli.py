"""
Enhanced Command-Line Interface for the Emotion Classification Pipeline.

This script provides a unified CLI for both local and Azure ML execution of:
- Data preprocessing pipeline
- Model training pipeline
- Prediction from YouTube URLs
- Pipeline status monitoring

Supports seamless switching between local and Azure ML execution modes
while maintaining backward compatibility with existing usage patterns.
"""

import argparse
import logging
import os
import sys
import time

# A simple retry decorator
def retry(tries=3, delay=5, backoff=2):
    """
    A simple retry decorator for functions that might fail due to transient issues.
    
    Args:
        tries (int): The maximum number of attempts.
        delay (int): The initial delay between retries in seconds.
        backoff (int): The factor by which the delay should increase for each retry.
    """
    def deco_retry(f):
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    # Check for specific, recoverable network errors
                    if "ConnectionResetError" in str(e) or "Connection aborted" in str(e) or "10054" in str(e):
                        msg = f"Retrying in {mdelay} seconds due to network error: {e}"
                        logger.warning(msg)
                        time.sleep(mdelay)
                        mtries -= 1
                        mdelay *= backoff
                    else:
                        # If it's not a network error, fail fast
                        raise
            # Final attempt
            return f(*args, **kwargs)
        return f_retry
    return deco_retry


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def add_preprocess_args(parser):
    """Add preprocessing-specific arguments."""
    parser.add_argument(
        "--raw-train-path",
        type=str,
        default="data/raw/train",
        help="Path to raw training data (directory or CSV file)"
    )

    parser.add_argument(
        "--raw-test-path",
        type=str,
        default="data/raw/test/test_data-0001.csv",
        help="Path to raw test data CSV file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for processed data"
    )

    parser.add_argument(
        "--encoders-dir",
        type=str,
        default="models/encoders",
        help="Directory to save label encoders"
    )

    parser.add_argument(
        "--model-name-tokenizer",
        type=str,
        default="microsoft/deberta-v3-xsmall",
        help="HuggingFace model name for tokenizer"
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length for tokenization"
    )

    parser.add_argument(
        "--output-tasks",
        type=str,
        default="emotion,sub-emotion,intensity",
        help="Comma-separated list of output tasks"
    )

    parser.add_argument(
        "--register-data-assets",
        action="store_true",
        default=True,
        help="Register processed data as Azure ML data assets after completion"
    )

    parser.add_argument(
        "--no-register-data-assets",
        action="store_false",
        dest="register_data_assets",
        help="Skip registering processed data as Azure ML data assets"
    )


def add_train_args(parser):
    """Add training-specific arguments."""
    parser.add_argument(
        "--processed-train-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed training data"
    )

    parser.add_argument(
        "--processed-test-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed test data"
    )

    parser.add_argument(
        "--encoders-dir",
        type=str,
        default="models/encoders",
        help="Directory containing label encoders"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/deberta-v3-xsmall",
        help="HuggingFace transformer model name"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training and evaluation"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for optimizer"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/weights",
        help="Output directory for trained model weights"
    )

    parser.add_argument(
        "--metrics-file",
        type=str,
        default="models/evaluation/metrics.json",
        help="Output file for training metrics"
    )

    parser.add_argument(
        "--output-tasks",
        type=str,
        default="emotion,sub-emotion,intensity",
        help="Comma-separated list of output tasks"
    )


def add_pipeline_args(parser):
    """Add arguments for the complete training pipeline."""
    # --- Arguments from add_preprocess_args ---
    parser.add_argument(
        "--raw-train-path",
        type=str,
        default="data/raw/train",
        help="Path to raw training data (directory or CSV file) for the pipeline."
    )
    parser.add_argument(
        "--raw-test-path",
        type=str,
        default="data/raw/test/test_data-0001.csv",
        help="Path to raw test data CSV file for the pipeline."
    )
    parser.add_argument(
        "--model-name-tokenizer",
        type=str,
        default="microsoft/deberta-v3-xsmall",
        help="HuggingFace model name for the tokenizer."
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length for tokenization."
    )
    # parser.add_argument(
    #     "--output-tasks",
    #     type=str,
    #     default="emotion,sub_emotion,intensity",
    #     help="Comma-separated list of output tasks."
    # )
    parser.add_argument(
        "--register-data-assets",
        action="store_true",
        help="Register processed data as Azure ML data assets."
    )
    
    # --- Arguments from add_train_args ---
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/deberta-v3-xsmall",
        help="HuggingFace transformer model name."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training and evaluation."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for optimizer."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="models/evaluation/metrics.json",
        help="Output file for training metrics."
    )
    
    # --- Shared/Conflicting Arguments (handled once) ---
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/weights",
        help="Output directory for final trained model weights."
    )
    parser.add_argument(
        "--encoders-dir",
        type=str,
        default="models/encoders",
        help="Directory for label encoders (output from preprocess, input to train)."
    )
    parser.add_argument(
        "--output-tasks",
        type=str,
        default="emotion,sub-emotion,intensity",
        help="Comma-separated list of output tasks for the pipeline."
    )
    
    # --- Pipeline-specific arguments ---
    parser.add_argument(
        "--pipeline-name",
        type=str,
        default="emotion_clf_pipeline",
        help="Base name for the Azure ML pipeline"
    )
    
    parser.add_argument(
        "--registration-f1-threshold",
        type=float,
        default=0.10,
        help="Minimum F1 score for model registration in Azure ML"
    )


def add_evaluate_register_args(parser):
    """Add evaluate and register specific arguments."""
    parser.add_argument(
        "--model-input-dir",
        type=str,
        required=True,
        help="Directory containing the trained model"
    )
    
    parser.add_argument(
        "--processed-test-dir", 
        type=str,
        required=True,
        help="Directory containing processed test data"
    )
    
    parser.add_argument(
        "--train-path",
        type=str,
        required=True,
        help="Path to training data file"
    )
    
    parser.add_argument(
        "--encoders-dir",
        type=str,
        required=True,
        help="Directory containing label encoders"
    )
    
    parser.add_argument(
        "--final-eval-output-dir",
        type=str,
        default="results/evaluation",
        help="Output directory for final evaluation results"
    )
    
    parser.add_argument(
        "--registration-f1-threshold",
        type=float,
        default=0.10,
        help="Minimum F1 score for model registration in Azure ML"
    )


def add_predict_args(parser):
    """Add prediction-specific arguments."""
    parser.add_argument(
        "url",
        type=str,
        help="YouTube URL to analyze"
    )
    
    parser.add_argument(
        "--transcription-method",
        type=str,
        choices=["whisper", "assemblyai"],
        default="whisper",
        help="Transcription method to use"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file path for results (JSON format)"
    )


def run_preprocess_local(args):
    """Run data preprocessing locally."""
    logger.info("Running data preprocessing locally...")

    try:
        # Import here to avoid circular imports
        from .data import DatasetLoader, DataPreparation
        from transformers import AutoTokenizer
        import pandas as pd

        # Parse output tasks
        output_tasks = [task.strip() for task in args.output_tasks.split(',')]

        # Load raw data
        dataset_loader = DatasetLoader()

        if os.path.isdir(args.raw_train_path):
            train_df = dataset_loader.load_training_data(args.raw_train_path)
        else:
            train_df = pd.read_csv(args.raw_train_path)
            # Handle column name consistency - convert hyphen to underscore
            if "sub-emotion" in train_df.columns:
                train_df = train_df.rename(columns={"sub-emotion": "sub_emotion"})

        test_df = dataset_loader.load_test_data(args.raw_test_path)

        logger.info(
            f"Loaded {len(train_df)} training samples and {len(test_df)} test samples"
        )

        # Clean data by removing rows with NaN in critical columns
        critical_columns = ['text', 'emotion', 'sub-emotion', 'intensity']
        # Only check columns that exist in the dataframes
        train_critical = [col for col in critical_columns if col in train_df.columns]
        test_critical = [col for col in critical_columns if col in test_df.columns]

        initial_train_len = len(train_df)
        initial_test_len = len(test_df)

        train_df = train_df.dropna(subset=train_critical)
        test_df = test_df.dropna(subset=test_critical)

        # Rename sub-emotion column to sub_emotion for consistency with training code
        if 'sub-emotion' in train_df.columns:
            train_df = train_df.rename(columns={'sub-emotion': 'sub_emotion'})
        if 'sub-emotion' in test_df.columns:
            test_df = test_df.rename(columns={'sub-emotion': 'sub_emotion'})

        # Update output_tasks to use underscore instead of hyphen
        output_tasks = [task.replace('sub-emotion', 'sub_emotion')
                        for task in output_tasks]

        train_removed = initial_train_len - len(train_df)
        test_removed = initial_test_len - len(test_df)
        logger.info(f"After cleaning: {len(train_df)} training samples "
                    f"({train_removed} removed)")
        logger.info(f"After cleaning: {len(test_df)} test samples "
                    f"({test_removed} removed)")

        # Apply intensity mapping
        intensity_mapping = {
            "mild": "mild", "neutral": "mild", "moderate": "moderate",
            "intense": "strong", "overwhelming": "strong"
        }
        train_df["intensity"] = (train_df["intensity"]
                                 .map(intensity_mapping).fillna("mild"))
        test_df["intensity"] = (test_df["intensity"]
                                .map(intensity_mapping).fillna("mild"))

        # Initialize tokenizer and data preparation
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_tokenizer)

        feature_config = {
            "pos": False, "textblob": False, "vader": False,
            "tfidf": True, "emolex": True
        }

        data_prep = DataPreparation(
            output_columns=output_tasks,
            tokenizer=tokenizer,
            max_length=args.max_length,
            batch_size=16,  # Not used in preprocessing
            feature_config=feature_config,
            encoders_save_dir=args.encoders_dir
        )

        # Create output directories
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.encoders_dir, exist_ok=True)

        # Prepare data (this will save encoders and process data)
        train_dataloader, val_dataloader, test_dataloader = data_prep.prepare_data(
            train_df=train_df,
            test_df=test_df,
            validation_split=0.1
        )

        # Save processed data
        train_output_path = os.path.join(args.output_dir, "train.csv")
        test_output_path = os.path.join(args.output_dir, "test.csv")

        data_prep.train_df_processed.to_csv(train_output_path, index=False)
        data_prep.test_df_processed.to_csv(test_output_path, index=False)

        logger.info("Preprocessing completed successfully!")
        logger.info(f"Processed data saved to: {args.output_dir}")
        logger.info(f"Encoders saved to: {args.encoders_dir}")

    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise


def run_preprocess_azure(args):
    """Run data preprocessing on Azure ML."""
    logger.info("Submitting data preprocessing pipeline to Azure ML...")

    try:
        from .azure_pipeline import (
            submit_preprocess_pipeline,
            register_processed_data_assets
        )

        # Submit Azure ML pipeline
        job = submit_preprocess_pipeline(args)
        logger.info(f"Azure ML preprocessing job submitted: {job.name}")
        logger.info(f"Monitor at: {job.studio_url}")

        # Check if we should register data assets after completion
        if getattr(args, 'register_data_assets', True):
            logger.info("Waiting for job completion to register data assets...")
            register_processed_data_assets(job)

    except ImportError:
        logger.error("Azure ML dependencies not available. Please install azure-ai-ml")
        raise
    except Exception as e:
        logger.error(f"Azure ML preprocessing submission failed: {str(e)}")
        raise


def run_train_local(args):
    """Run model training locally."""
    logger.info("Running model training locally...")

    try:
        import subprocess
        import sys

        # Build command to run train.py as a module
        # Only pass arguments that train.py actually supports
        cmd = [
            sys.executable, "-m", "src.emotion_clf_pipeline.train",
            "--model-name", args.model_name,
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate),
            "--epochs", str(args.epochs)
        ]

        logger.info(f"Running command: {' '.join(cmd)}")

        # Run the training script
        result = subprocess.run(cmd, check=True, capture_output=False)

        if result.returncode == 0:
            logger.info("Local training completed successfully!")
        else:
            raise subprocess.CalledProcessError(result.returncode, cmd)

    except subprocess.CalledProcessError as e:
        logger.error(f"Training process failed with return code {e.returncode}")
        raise
    except Exception as e:
        logger.error(f"Local training failed: {str(e)}")
        raise


def run_train_azure(args):
    """Run model training on Azure ML."""
    logger.info("Submitting model training pipeline to Azure ML...")

    try:
        from .azure_pipeline import submit_training_pipeline

        # Submit Azure ML pipeline
        job = submit_training_pipeline(args)
        logger.info(f"Azure ML training job submitted: {job.name}")
        logger.info(f"Monitor at: {job.studio_url}")

    except ImportError:
        logger.error("Azure ML dependencies not available. Please install azure-ai-ml")
        raise
    except Exception as e:
        logger.error(f"Azure ML training submission failed: {str(e)}")
        raise


def run_predict(args):
    """Run prediction on YouTube URL."""
    logger.info(f"Analyzing YouTube video: {args.url}")

    try:
        from .predict import process_youtube_url_and_predict

        # Process the YouTube URL
        results = process_youtube_url_and_predict(
            youtube_url=args.url,
            transcription_method=args.transcription_method
        )

        logger.info(f"Prediction completed! Found {len(results)} transcript segments.")

        # Save results if output file specified
        if args.output_file:
            import json
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {args.output_file}")
        else:
            # Print first few results
            for i, result in enumerate(results[:3]):
                logger.info(f"Segment {i+1}: {result.get('sentence', '')[:100]}...")
                logger.info(f"  Emotion: {result.get('emotion', 'N/A')}")
                logger.info(f"  Intensity: {result.get('intensity', 'N/A')}")

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise


def run_pipeline_local(args):
    """Run the complete pipeline locally (preprocess + train)."""
    logger.info("Starting complete local pipeline: preprocess + train")
    
    try:
        # Step 1: Run preprocessing
        logger.info("=" * 60)
        logger.info("STEP 1: DATA PREPROCESSING")
        logger.info("=" * 60)
        run_preprocess_local(args)
        
        # Step 2: Run training
        logger.info("=" * 60)
        logger.info("STEP 2: MODEL TRAINING")
        logger.info("=" * 60)
        run_train_local(args)
        
        logger.info("=" * 60)
        logger.info("COMPLETE PIPELINE FINISHED SUCCESSFULLY")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


@retry(tries=3, delay=10, backoff=2)
def run_pipeline_azure(args):
    """Run the complete pipeline on Azure ML."""
    logger.info("🚀 Submitting complete pipeline to Azure ML...")
    from . import azure_pipeline
        
    try:
        job = azure_pipeline.submit_complete_pipeline(args)
        logger.info(f"✅ Pipeline submitted successfully. Job ID: {job.name}")
        logger.info(f"➡️  Monitor job at: {job.studio_url}")

    except Exception as e:
        logger.error(f"❌ Pipeline submission failed: {str(e)}", exc_info=True)
        sys.exit(1)


def cmd_preprocess(args):
    """Handle preprocess command."""
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine execution mode
    mode = "azure" if args.azure else args.mode

    if mode == "local":
        run_preprocess_local(args)
    elif mode == "azure":
        run_preprocess_azure(args)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def cmd_train(args):
    """Handle train command."""
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine execution mode
    mode = "azure" if args.azure else args.mode

    if mode == "local":
        run_train_local(args)
    elif mode == "azure":
        run_train_azure(args)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def cmd_evaluate_register(args):
    """Handle evaluate and register command."""
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Running evaluation and registration...")
    
    try:
        from .evaluate import evaluate_and_register
        
        # The new evaluation script needs a direct path to the test CSV
        # and a path for the status output file. Let's construct them.
        args.processed_test_path = os.path.join(args.processed_test_dir, "test.csv")
        os.makedirs(args.final_eval_output_dir, exist_ok=True)
        args.registration_status_output_file = os.path.join(args.final_eval_output_dir, "registration_status.json")

        # Assume a default batch size if not present in args
        if not hasattr(args, 'batch_size'):
            args.batch_size = 16
        
        evaluate_and_register(args)
        
        logger.info("Evaluation and registration completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation and registration failed: {str(e)}", exc_info=True)
        sys.exit(1)


def cmd_predict(args):
    """Handle predict command."""
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Prediction always runs locally (uses local models)
    run_predict(args)


def cmd_status(args):
    """Check the status of an Azure ML job."""
    logger.info(f"Checking status of job: {args.job_id}")
    from . import azure_pipeline
    status = azure_pipeline.get_pipeline_status(args.job_id)
    logger.info(f"Job status: {status}")


def cmd_pipeline(args):
    """Execute the appropriate pipeline based on mode."""
    if args.azure:
        run_pipeline_azure(args)
    else:
        run_pipeline_local(args)


def main():
    """Main function to parse arguments and execute commands."""
    # Main parser
    parser = argparse.ArgumentParser(
        description="Emotion Classification Pipeline - Local and Azure ML Execution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Parent parser for global arguments that all sub-commands will share
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--mode",
        choices=["local", "azure"],
        default="local",
        help="Execution mode: local (current machine) or azure (Azure ML)"
    )
    parent_parser.add_argument(
        "--azure",
        action="store_true",
        help="Shorthand for --mode azure"
    )
    parent_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Preprocess command
    parser_preprocess = subparsers.add_parser(
        "preprocess",
        parents=[parent_parser],
        help="Run data preprocessing pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_preprocess_args(parser_preprocess)
    parser_preprocess.set_defaults(func=cmd_preprocess)

    # Train command
    parser_train = subparsers.add_parser(
        "train",
        parents=[parent_parser],
        help="Run model training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser_train)
    parser_train.set_defaults(func=cmd_train)

    # Evaluate and Register command
    parser_evaluate_register = subparsers.add_parser(
        "evaluate_register",
        parents=[parent_parser],
        help="Evaluate model and register if meets threshold",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_evaluate_register_args(parser_evaluate_register)
    parser_evaluate_register.set_defaults(func=cmd_evaluate_register)

    # Predict command
    parser_predict = subparsers.add_parser(
        "predict",
        parents=[parent_parser],
        help="Predict emotions from YouTube URL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_predict_args(parser_predict)
    parser_predict.set_defaults(func=cmd_predict)

    # Pipeline command
    parser_pipeline = subparsers.add_parser(
        "train-pipeline",
        parents=[parent_parser],
        help="Run the complete training pipeline (preprocess, train, evaluate, register)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_pipeline_args(parser_pipeline)
    parser_pipeline.set_defaults(func=cmd_pipeline)

    # Status command
    parser_status = subparsers.add_parser(
        "status",
        parents=[parent_parser],
        help="Check Azure ML pipeline status",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_status.add_argument(
        "--job-id",
        type=str,
        help="Azure ML job ID to check status"
    )
    parser_status.set_defaults(func=cmd_status)

    # Parse arguments
    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.command:
        parser.print_help()
        return

    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Command failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
