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

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def create_base_parser():
    """Create base argument parser with common options."""
    parser = argparse.ArgumentParser(
        description="Emotion Classification Pipeline - Local and Azure ML Execution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Execution mode
    parser.add_argument(
        "--mode",
        choices=["local", "azure"],
        default="local",
        help="Execution mode: local (current machine) or azure (Azure ML)"
    )

    # Alternative Azure flag for convenience
    parser.add_argument(
        "--azure",
        action="store_true",
        help="Shorthand for --mode azure"
    )

    # Verbose logging
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser


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


def add_predict_args(parser):
    """Add prediction-specific arguments."""
    parser.add_argument(
        "url",
        type=str,
        help="YouTube URL to analyze for emotions"
    )

    parser.add_argument(
        "--transcription-method",
        choices=["assemblyai", "whisper"],
        default="assemblyai",
        help="Speech-to-text service to use"
    )

    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file to save prediction results (optional)"
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


def cmd_predict(args):
    """Handle predict command."""
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Prediction always runs locally (uses local models)
    run_predict(args)


def cmd_status(args):
    """Handle pipeline status command."""
    logger.info("Checking pipeline status...")

    try:
        from .azure_pipeline import get_pipeline_status

        if args.job_id:
            status = get_pipeline_status(args.job_id)
            logger.info(f"Job {args.job_id} status: {status}")
        else:
            logger.info("Please provide --job-id to check specific job status")

    except ImportError:
        logger.error("Azure ML dependencies not available")
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Emotion Classification Pipeline CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Preprocess command
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Run data preprocessing pipeline",
        parents=[create_base_parser()],
        conflict_handler='resolve'
    )
    add_preprocess_args(preprocess_parser)
    preprocess_parser.set_defaults(func=cmd_preprocess)

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Run model training pipeline",
        parents=[create_base_parser()],
        conflict_handler='resolve'
    )
    add_train_args(train_parser)
    train_parser.set_defaults(func=cmd_train)

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict emotions from YouTube URL",
        parents=[create_base_parser()],
        conflict_handler='resolve'
    )
    add_predict_args(predict_parser)
    predict_parser.set_defaults(func=cmd_predict)

    # Status command
    status_parser = subparsers.add_parser(
        "status",
        help="Check Azure ML pipeline status",
        parents=[create_base_parser()],
        conflict_handler='resolve'
    )
    status_parser.add_argument(
        "--job-id",
        type=str,
        help="Azure ML job ID to check status"
    )
    status_parser.set_defaults(func=cmd_status)

    # Parse arguments
    args = parser.parse_args()

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
