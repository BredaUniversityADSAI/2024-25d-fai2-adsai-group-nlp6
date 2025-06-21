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
from transformers import AutoTokenizer
import pandas as pd
import subprocess
import json
import traceback

try:
    from . import azure_pipeline
    from .azure_pipeline import (
        submit_preprocess_pipeline,
        register_processed_data_assets,
        submit_training_pipeline,
    )
    from .predict import process_youtube_url_and_predict
    from .data import DatasetLoader, DataPreparation
    from .azure_endpoint import AzureEndpointManager
except ImportError:
    import azure_pipeline
    from azure_pipeline import (
        submit_preprocess_pipeline,
        register_processed_data_assets,
        submit_training_pipeline,
    )
    from predict import process_youtube_url_and_predict
    from data import DatasetLoader, DataPreparation
    from azure_endpoint import AzureEndpointManager


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
                    if (
                        "ConnectionResetError" in str(e)
                        or "Connection aborted" in str(e)
                        or "10054" in str(e)
                    ):
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
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def add_preprocess_args(parser):
    """Add preprocessing-specific arguments."""
    parser.add_argument(
        "--raw-train-path",
        type=str,
        default="data/raw/train",
        help="Path to raw training data (directory or CSV file)",
    )

    parser.add_argument(
        "--raw-test-path",
        type=str,
        default="data/raw/test/test_data-0001.csv",
        help="Path to raw test data CSV file",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for processed data",
    )

    parser.add_argument(
        "--encoders-dir",
        type=str,
        default="models/encoders",
        help="Directory to save label encoders",
    )

    parser.add_argument(
        "--model-name-tokenizer",
        type=str,
        default="microsoft/deberta-v3-xsmall",
        help="HuggingFace model name for tokenizer",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length for tokenization",
    )

    parser.add_argument(
        "--output-tasks",
        type=str,
        default="emotion,sub-emotion,intensity",
        help="Comma-separated list of output tasks",
    )

    parser.add_argument(
        "--register-data-assets",
        action="store_true",
        default=True,
        help="Register processed data as Azure ML data assets after completion",
    )

    parser.add_argument(
        "--no-register-data-assets",
        action="store_false",
        dest="register_data_assets",
        help="Skip registering processed data as Azure ML data assets",
    )


def add_train_args(parser):
    """Add training-specific arguments."""
    parser.add_argument(
        "--processed-train-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed training data",
    )

    parser.add_argument(
        "--processed-test-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed test data",
    )

    parser.add_argument(
        "--encoders-dir",
        type=str,
        default="models/encoders",
        help="Directory containing label encoders",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/deberta-v3-xsmall",
        help="HuggingFace transformer model name",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training and evaluation",
    )

    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate for optimizer"
    )

    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/weights",
        help="Output directory for trained model weights",
    )

    parser.add_argument(
        "--metrics-file",
        type=str,
        default="models/evaluation/metrics.json",
        help="Output file for training metrics",
    )

    parser.add_argument(
        "--output-tasks",
        type=str,
        default="emotion,sub-emotion,intensity",
        help="Comma-separated list of output tasks",
    )


def add_pipeline_args(parser):
    """Add arguments for the complete training pipeline."""
    # --- Arguments from add_preprocess_args ---
    parser.add_argument(
        "--raw-train-path",
        type=str,
        default="data/raw/train",
        help="Path to raw training data (directory or CSV file) for the pipeline.",
    )
    parser.add_argument(
        "--raw-test-path",
        type=str,
        default="data/raw/test/test_data-0001.csv",
        help="Path to raw test data CSV file for the pipeline.",
    )
    parser.add_argument(
        "--model-name-tokenizer",
        type=str,
        default="microsoft/deberta-v3-xsmall",
        help="HuggingFace model name for the tokenizer.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length for tokenization.",
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
        default=True,
        help="Register processed data as Azure ML data assets.",
    )

    parser.add_argument(
        "--no-register-data-assets",
        action="store_false",
        dest="register_data_assets",
        help="Skip registering processed data as Azure ML data assets",
    )

    # --- Arguments from add_train_args ---
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/deberta-v3-xsmall",
        help="HuggingFace transformer model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate for optimizer."
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs."
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="models/evaluation/metrics.json",
        help="Output file for training metrics.",
    )

    # --- Shared/Conflicting Arguments (handled once) ---
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/weights",
        help="Output directory for final trained model weights.",
    )
    parser.add_argument(
        "--encoders-dir",
        type=str,
        default="models/encoders",
        help="Directory for label encoders (output from preprocess, input to train).",
    )
    parser.add_argument(
        "--output-tasks",
        type=str,
        default="emotion,sub-emotion,intensity",
        help="Comma-separated list of output tasks for the pipeline.",
    )

    # --- Pipeline-specific arguments ---
    parser.add_argument(
        "--pipeline-name",
        type=str,
        default="deberta-full-pipeline",
        help="Base name for the Azure ML pipeline",
    )

    parser.add_argument(
        "--registration-f1-threshold",
        type=float,
        default=0.10,
        help="Minimum F1 score for model registration in Azure ML",
    )


def add_schedule_pipeline_args(parser):
    """Add pipeline-specific arguments for scheduling (avoiding conflicts)."""
    parser.add_argument(
        "--pipeline-name",
        type=str,
        default="scheduled-deberta-training-pipeline",
        help="Name of the Azure ML pipeline",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/processed",
        help="Path to processed training data directory",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="./models",
        help="Path to output trained models",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="scheduled-deberta-training-experiment",
        help="Name of the Azure ML experiment",
    )

    parser.add_argument(
        "--compute-target",
        type=str,
        default="cpu-cluster",
        help="Azure ML compute target name",
    )


def add_predict_args(parser):
    """Add prediction-specific arguments."""
    parser.add_argument("url", type=str, help="YouTube URL to analyze")

    parser.add_argument(
        "--transcription-method",
        type=str,
        choices=["whisper", "assemblyai"],
        default="whisper",
        help="Transcription method to use",
    )

    parser.add_argument(
        "--output-file", type=str, help="Output file path for results (JSON format)"
    )


def run_preprocess_local(args):
    """Run data preprocessing locally."""
    logger.info("Running data preprocessing locally...")

    try:

        # Parse output tasks
        output_tasks = [task.strip() for task in args.output_tasks.split(",")]

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
        critical_columns = ["text", "emotion", "sub-emotion", "intensity"]
        # Only check columns that exist in the dataframes
        train_critical = [col for col in critical_columns if col in train_df.columns]
        test_critical = [col for col in critical_columns if col in test_df.columns]

        initial_train_len = len(train_df)
        initial_test_len = len(test_df)

        train_df = train_df.dropna(subset=train_critical)
        test_df = test_df.dropna(subset=test_critical)

        # Rename sub-emotion column to sub_emotion for consistency with training code
        if "sub-emotion" in train_df.columns:
            train_df = train_df.rename(columns={"sub-emotion": "sub_emotion"})
        if "sub-emotion" in test_df.columns:
            test_df = test_df.rename(columns={"sub-emotion": "sub_emotion"})

        # Update output_tasks to use underscore instead of hyphen
        output_tasks = [
            task.replace("sub-emotion", "sub_emotion") for task in output_tasks
        ]

        train_removed = initial_train_len - len(train_df)
        test_removed = initial_test_len - len(test_df)
        logger.info(
            f"After cleaning: {len(train_df)} training samples "
            f"({train_removed} removed)"
        )
        logger.info(
            f"After cleaning: {len(test_df)} test samples " f"({test_removed} removed)"
        )

        # Apply intensity mapping
        intensity_mapping = {
            "mild": "mild",
            "neutral": "mild",
            "moderate": "moderate",
            "intense": "strong",
            "overwhelming": "strong",
        }
        train_df["intensity"] = (
            train_df["intensity"].map(intensity_mapping).fillna("mild")
        )
        test_df["intensity"] = (
            test_df["intensity"].map(intensity_mapping).fillna("mild")
        )

        # Initialize tokenizer and data preparation
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_tokenizer)

        feature_config = {
            "pos": False,
            "textblob": False,
            "vader": False,
            "tfidf": True,
            "emolex": True,
        }

        data_prep = DataPreparation(
            output_columns=output_tasks,
            tokenizer=tokenizer,
            max_length=args.max_length,
            batch_size=16,  # Not used in preprocessing
            feature_config=feature_config,
            encoders_save_dir=args.encoders_dir,
        )

        # Create output directories
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.encoders_dir, exist_ok=True)

        # Prepare data (this will save encoders and process data)
        train_dataloader, val_dataloader, test_dataloader = data_prep.prepare_data(
            train_df=train_df, test_df=test_df, validation_split=0.1
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
        # Submit Azure ML pipeline
        job = submit_preprocess_pipeline(args)
        logger.info(f"Azure ML preprocessing job submitted: {job.name}")
        logger.info(f"Monitor at: {job.studio_url}")

        # Check if we should register data assets after completion
        if getattr(args, "register_data_assets", True):
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
        # Build command to run train.py as a module
        # Only pass arguments that train.py actually supports
        cmd = [
            sys.executable,
            "-m",
            "src.emotion_clf_pipeline.train",
            "--model-name",
            args.model_name,
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--epochs",
            str(args.epochs),
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
        # Process the YouTube URL
        results = process_youtube_url_and_predict(
            youtube_url=args.url, transcription_method=args.transcription_method
        )

        logger.info(f"Prediction completed! Found {len(results)} transcript segments.")

        # Save results if output file specified
        if args.output_file:
            with open(args.output_file, "w") as f:
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


def cmd_predict(args):
    """Handle predict command."""
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Prediction always runs locally (uses local models)
    run_predict(args)


def cmd_status(args):
    """Check the status of an Azure ML job."""
    logger.info(f"Checking status of job: {args.job_id}")
    status = azure_pipeline.get_pipeline_status(args.job_id)
    logger.info(f"Job status: {status}")


def cmd_pipeline(args):
    """Execute the appropriate pipeline based on mode."""
    if args.azure:
        run_pipeline_azure(args)
    else:
        run_pipeline_local(args)


def add_schedule_create_args(parser):
    """Add arguments for creating pipeline schedules."""
    parser.add_argument(
        "--schedule-name", type=str, required=True, help="Name for the schedule"
    )

    parser.add_argument(
        "--cron",
        type=str,
        help="Cron expression (e.g., '0 0 * * *' for daily at midnight)",
    )

    parser.add_argument(
        "--daily",
        action="store_true",
        help="Create daily schedule (use with --hour and --minute)",
    )

    parser.add_argument(
        "--weekly",
        type=int,
        metavar="DAY",
        help="Create weekly schedule on specified day (0=Sunday, 1=Monday, etc.)",
    )

    parser.add_argument(
        "--monthly",
        type=int,
        metavar="DAY",
        help="Create monthly schedule on specified day (1-31)",
    )

    parser.add_argument(
        "--hour", type=int, default=0, help="Hour of day (0-23, default: 0)"
    )

    parser.add_argument(
        "--minute", type=int, default=0, help="Minute of hour (0-59, default: 0)"
    )

    parser.add_argument(
        "--timezone",
        type=str,
        default="UTC",
        help="Timezone for the schedule (default: UTC)",
    )

    parser.add_argument("--description", type=str, help="Description for the schedule")

    parser.add_argument(
        "--enabled",
        action="store_true",
        default=False,
        help="Enable the schedule immediately (default: disabled)",
    )

    # Add pipeline configuration arguments
    add_schedule_pipeline_args(parser)


def cmd_schedule_create(args):
    """Handle schedule create command."""
    try:
        # Determine schedule type and create accordingly
        if args.cron:
            schedule_id = azure_pipeline.create_pipeline_schedule(
                pipeline_name=args.pipeline_name,
                schedule_name=args.schedule_name,
                cron_expression=args.cron,
                timezone=args.timezone,
                description=args.description,
                enabled=args.enabled,
                args=args,
            )
        elif args.daily:
            schedule_id = azure_pipeline.create_daily_schedule(
                pipeline_name=args.pipeline_name,
                hour=args.hour,
                minute=args.minute,
                timezone=args.timezone,
                enabled=args.enabled,
            )
        elif args.weekly is not None:
            schedule_id = azure_pipeline.create_weekly_schedule(
                pipeline_name=args.pipeline_name,
                day_of_week=args.weekly,
                hour=args.hour,
                minute=args.minute,
                timezone=args.timezone,
                enabled=args.enabled,
            )
        elif args.monthly is not None:
            schedule_id = azure_pipeline.create_monthly_schedule(
                pipeline_name=args.pipeline_name,
                day_of_month=args.monthly,
                hour=args.hour,
                minute=args.minute,
                timezone=args.timezone,
                enabled=args.enabled,
            )
        else:
            logger.error(
                "Please specify one of: --cron, --daily, --weekly, or --monthly"
            )
            return

        if schedule_id:
            logger.info(f"✅ Successfully created schedule: {schedule_id}")
        else:
            logger.error("❌ Failed to create schedule")

    except Exception as e:
        logger.error(f"❌ Schedule creation failed: {e}")


def cmd_schedule_list(args):
    """Handle schedule list command."""
    try:
        azure_pipeline.print_schedule_summary()

    except Exception as e:
        logger.error(f"❌ Failed to list schedules: {e}")


def cmd_schedule_details(args):
    """Handle schedule details command."""
    try:
        details = azure_pipeline.get_schedule_details(args.schedule_name)

        if details:
            print(f"📅 Schedule Details: {args.schedule_name}")
            print("=" * 50)
            print(f"Enabled: {'🟢 Yes' if details.get('enabled') else '🔴 No'}")
            print(f"Description: {details.get('description', 'N/A')}")
            print(f"Trigger Type: {details.get('trigger_type', 'Unknown')}")

            if details.get("cron_expression"):
                print(f"Cron Expression: {details['cron_expression']}")
                print(f"Timezone: {details.get('timezone', 'UTC')}")
            elif details.get("frequency"):
                print(
                    f"Frequency: Every {details.get('interval', 1)} \
                    {details.get('frequency')}"
                )

            if details.get("created_time"):
                print(f"Created: {details['created_time']}")
            if details.get("last_modified"):
                print(f"Modified: {details['last_modified']}")

            if details.get("create_job"):
                job_info = details["create_job"]
                print(f"Pipeline: {job_info.get('name', 'N/A')}")
                print(f"Experiment: {job_info.get('experiment', 'N/A')}")
                if job_info.get("compute"):
                    print(f"Compute: {job_info['compute']}")

            if details.get("tags"):
                print("Tags:")
                for key, value in details["tags"].items():
                    print(f"  {key}: {value}")
        else:
            logger.error(f"❌ Schedule '{args.schedule_name}' not found")

    except Exception as e:
        logger.error(f"❌ Failed to get schedule details: {e}")


def cmd_schedule_enable(args):
    """Handle schedule enable command."""
    try:
        if azure_pipeline.enable_schedule(args.schedule_name):
            logger.info(f"✅ Schedule '{args.schedule_name}' enabled successfully")
        else:
            logger.error(f"❌ Failed to enable schedule '{args.schedule_name}'")

    except Exception as e:
        logger.error(f"❌ Failed to enable schedule: {e}")


def cmd_schedule_disable(args):
    """Handle schedule disable command."""
    try:
        if azure_pipeline.disable_schedule(args.schedule_name):
            logger.info(f"✅ Schedule '{args.schedule_name}' disabled successfully")
        else:
            logger.error(f"❌ Failed to disable schedule '{args.schedule_name}'")

    except Exception as e:
        logger.error(f"❌ Failed to disable schedule: {e}")


def cmd_schedule_delete(args):
    """Handle schedule delete command."""
    try:
        # Confirm deletion unless --confirm is used
        if not args.confirm:
            response = input(
                f"Are you sure you want to delete schedule \
                '{args.schedule_name}'? (y/N): "
            )
            if response.lower() not in ["y", "yes"]:
                logger.info("❌ Deletion cancelled")
                return

        if azure_pipeline.delete_schedule(args.schedule_name):
            logger.info(f"✅ Schedule '{args.schedule_name}' deleted successfully")
        else:
            logger.error(f"❌ Failed to delete schedule '{args.schedule_name}'")

    except Exception as e:
        logger.error(f"❌ Failed to delete schedule: {e}")


def cmd_schedule_setup_defaults(args):
    """Handle setup default schedules command."""
    try:
        logger.info(f"🕐 Setting up default schedules for '{args.pipeline_name}'...")
        results = azure_pipeline.setup_default_schedules(args.pipeline_name)

        successful = [k for k, v in results.items() if v is not None]
        failed = [k for k, v in results.items() if v is None]

        if successful:
            logger.info(f"✅ Created {len(successful)} default schedules:")
            for schedule_type in successful:
                logger.info(f"   - {schedule_type}: {results[schedule_type]}")

        if failed:
            logger.warning(
                f"❌ Failed to create {len(failed)} schedules: \
                {', '.join(failed)}"
            )

        logger.info(
            "💡 All schedules are created in disabled state. Use \
            'schedule enable' to activate them."
        )

    except Exception as e:
        logger.error(f"❌ Failed to setup default schedules: {e}")


def get_azure_config():
    """Get Azure configuration from environment variables."""
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP")
    workspace_name = os.getenv("AZURE_WORKSPACE_NAME")

    if not all([subscription_id, resource_group, workspace_name]):
        raise ValueError(
            "Missing required Azure environment variables: "
            "AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE_NAME"
        )

    return subscription_id, resource_group, workspace_name


def cmd_endpoint_deploy(args):
    """Deploy model to Azure ML Kubernetes endpoint using blue-green strategy."""
    if AzureEndpointManager is None:
        logger.error(
            "❌ Azure endpoint functionality not available. "
            "Please check azure_endpoint.py imports."
        )
        sys.exit(1)

    try:
        subscription_id, resource_group, workspace_name = get_azure_config()

        manager = AzureEndpointManager(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            endpoint_name=args.endpoint_name,
        )

        logger.info(f"🚀 Starting blue-green deployment to {args.endpoint_name}")

        manager.blue_green_deploy(
            model_name=args.model_name,
            model_version=args.model_version,
            environment=args.environment,
            code_path=args.code_path,
            scoring_script=args.scoring_script,
            instance_type=getattr(args, "instance_type", "defaultinstancetype"),
        )

        logger.info("✅ Blue-green deployment completed successfully")

    except Exception as e:
        logger.error(f"❌ Endpoint deployment failed: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


def cmd_endpoint_status(args):
    """Check status and traffic distribution of the endpoint."""
    try:
        subscription_id, resource_group, workspace_name = get_azure_config()

        manager = AzureEndpointManager(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            endpoint_name=args.endpoint_name,
        )

        endpoint = manager.ml_client.online_endpoints.get(args.endpoint_name)
        active_deployment = manager.get_active_deployment()

        logger.info(f"📊 Endpoint Status: {args.endpoint_name}")
        logger.info(f"   Active Deployment: {active_deployment}")
        logger.info(f"   Traffic Distribution: {endpoint.traffic}")
        logger.info(f"   Auth Mode: {endpoint.auth_mode}")

    except Exception as e:
        logger.error(f"❌ Failed to get endpoint status: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


def cmd_endpoint_switch_traffic(args):
    """Switch traffic between blue and green deployments."""
    try:
        subscription_id, resource_group, workspace_name = get_azure_config()

        manager = AzureEndpointManager(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            endpoint_name=args.endpoint_name,
        )

        logger.info(
            f"🔄 Switching traffic: blue={args.blue_weight}%, "
            f"green={args.green_weight}%"
        )

        manager.update_traffic(
            blue_weight=args.blue_weight, green_weight=args.green_weight
        )

        logger.info("✅ Traffic switch completed successfully")

    except Exception as e:
        logger.error(f"❌ Traffic switch failed: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


def cmd_endpoint_rollback(args):
    """Rollback to the previous deployment by switching traffic."""
    try:
        subscription_id, resource_group, workspace_name = get_azure_config()

        manager = AzureEndpointManager(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            endpoint_name=args.endpoint_name,
        )

        active = manager.get_active_deployment()
        if not active:
            logger.error("❌ No active deployment found")
            sys.exit(1)

        # Switch to the other deployment
        new_active = "green" if active == "blue" else "blue"

        logger.info(f"🔄 Rolling back from {active} to {new_active}")

        traffic_config = {new_active: 100, active: 0}
        manager.update_traffic(**traffic_config)

        logger.info(f"✅ Rollback completed. Active deployment: {new_active}")

    except Exception as e:
        logger.error(f"❌ Rollback failed: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


def add_endpoint_deploy_args(parser):
    """Add arguments for endpoint deployment command."""
    parser.add_argument(
        "--endpoint-name",
        type=str,
        required=True,
        help="Name of the Azure ML Kubernetes endpoint",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="emotion-clf-baseline",
        help="Name of the registered model in Azure ML",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="latest",
        help="Version of the model to deploy",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="emotion-clf-pipeline-env:latest",
        help="Azure ML environment for the deployment",
    )
    parser.add_argument(
        "--instance-type",
        type=str,
        default="defaultinstancetype",
        help="Instance type for Kubernetes deployment (K8s-compatible)",
    )
    parser.add_argument(
        "--code-path",
        type=str,
        default="src/emotion_clf_pipeline",
        help="Path to the code directory",
    )
    parser.add_argument(
        "--scoring-script",
        type=str,
        default="azure_score.py",
        help="Name of the scoring script",
    )


def add_endpoint_status_args(parser):
    """Add arguments for endpoint status command."""
    parser.add_argument(
        "--endpoint-name",
        type=str,
        required=True,
        help="Name of the Azure ML Kubernetes endpoint",
    )


def add_endpoint_traffic_args(parser):
    """Add arguments for traffic switching command."""
    parser.add_argument(
        "--endpoint-name",
        type=str,
        required=True,
        help="Name of the Azure ML Kubernetes endpoint",
    )
    parser.add_argument(
        "--blue-weight",
        type=int,
        default=100,
        help="Percentage of traffic for blue deployment (0-100)",
    )
    parser.add_argument(
        "--green-weight",
        type=int,
        default=0,
        help="Percentage of traffic for green deployment (0-100)",
    )


def add_endpoint_rollback_args(parser):
    """Add arguments for rollback command."""
    parser.add_argument(
        "--endpoint-name",
        type=str,
        required=True,
        help="Name of the Azure ML Kubernetes endpoint",
    )


def main():
    """Main function to parse arguments and execute commands."""
    # Main parser
    parser = argparse.ArgumentParser(
        description="Emotion Classification Pipeline - Local and Azure ML Execution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Parent parser for global arguments that all sub-commands will share
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--mode",
        choices=["local", "azure"],
        default="local",
        help="Execution mode: local (current machine) or azure (Azure ML)",
    )
    parent_parser.add_argument(
        "--azure", action="store_true", help="Shorthand for --mode azure"
    )
    parent_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Preprocess command
    parser_preprocess = subparsers.add_parser(
        "preprocess",
        parents=[parent_parser],
        help="Run data preprocessing pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_preprocess_args(parser_preprocess)
    parser_preprocess.set_defaults(func=cmd_preprocess)

    # Train command
    parser_train = subparsers.add_parser(
        "train",
        parents=[parent_parser],
        help="Run model training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_train_args(parser_train)
    parser_train.set_defaults(func=cmd_train)

    # Predict command
    parser_predict = subparsers.add_parser(
        "predict",
        parents=[parent_parser],
        help="Predict emotions from YouTube URL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_predict_args(parser_predict)
    parser_predict.set_defaults(func=cmd_predict)

    # Pipeline command
    parser_pipeline = subparsers.add_parser(
        "train-pipeline",
        parents=[parent_parser],
        help=(
            "Run the complete training pipeline " "(preprocess, train with evaluation)"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_pipeline_args(parser_pipeline)
    parser_pipeline.set_defaults(func=cmd_pipeline)

    # Schedule command group
    parser_schedule = subparsers.add_parser(
        "schedule",
        parents=[parent_parser],
        help="Manage Azure ML pipeline schedules",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    schedule_subparsers = parser_schedule.add_subparsers(
        dest="schedule_action", required=True, help="Schedule management actions"
    )

    # Create schedule command
    parser_create_schedule = schedule_subparsers.add_parser(
        "create",
        parents=[parent_parser],
        help="Create a new pipeline schedule",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_schedule_create_args(parser_create_schedule)
    parser_create_schedule.set_defaults(func=cmd_schedule_create)

    # List schedules command
    parser_list_schedules = schedule_subparsers.add_parser(
        "list",
        parents=[parent_parser],
        help="List all pipeline schedules",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_list_schedules.set_defaults(func=cmd_schedule_list)

    # Schedule details command
    parser_schedule_details = schedule_subparsers.add_parser(
        "details",
        parents=[parent_parser],
        help="Get details of a specific schedule",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_schedule_details.add_argument(
        "schedule_name", type=str, help="Name of the schedule to get details for"
    )
    parser_schedule_details.set_defaults(func=cmd_schedule_details)

    # Enable schedule command
    parser_enable_schedule = schedule_subparsers.add_parser(
        "enable",
        parents=[parent_parser],
        help="Enable a pipeline schedule",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_enable_schedule.add_argument(
        "schedule_name", type=str, help="Name of the schedule to enable"
    )
    parser_enable_schedule.set_defaults(func=cmd_schedule_enable)

    # Disable schedule command
    parser_disable_schedule = schedule_subparsers.add_parser(
        "disable",
        parents=[parent_parser],
        help="Disable a pipeline schedule",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_disable_schedule.add_argument(
        "schedule_name", type=str, help="Name of the schedule to disable"
    )
    parser_disable_schedule.set_defaults(func=cmd_schedule_disable)

    # Delete schedule command
    parser_delete_schedule = schedule_subparsers.add_parser(
        "delete",
        parents=[parent_parser],
        help="Delete a pipeline schedule",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_delete_schedule.add_argument(
        "schedule_name", type=str, help="Name of the schedule to delete"
    )
    parser_delete_schedule.add_argument(
        "--confirm", action="store_true", help="Confirm deletion without prompting"
    )
    parser_delete_schedule.set_defaults(func=cmd_schedule_delete)

    # Setup default schedules command
    parser_setup_schedules = schedule_subparsers.add_parser(
        "setup-defaults",
        parents=[parent_parser],
        help="Setup common schedule patterns (daily, weekly, monthly)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_setup_schedules.add_argument(
        "--pipeline-name",
        type=str,
        default="deberta-full-pipeline",
        help="Name of the pipeline to schedule",
    )
    parser_setup_schedules.set_defaults(func=cmd_schedule_setup_defaults)

    # Status command
    parser_status = subparsers.add_parser(
        "status",
        parents=[parent_parser],
        help="Check Azure ML pipeline status",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_status.add_argument(
        "--job-id", type=str, help="Azure ML job ID to check status"
    )
    parser_status.set_defaults(func=cmd_status)

    # Endpoint command group
    parser_endpoint = subparsers.add_parser(
        "endpoint",
        parents=[parent_parser],
        help="Manage Azure ML Kubernetes endpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    endpoint_subparsers = parser_endpoint.add_subparsers(
        dest="endpoint_action", required=True, help="Endpoint management actions"
    )

    # Deploy endpoint command
    parser_deploy_endpoint = endpoint_subparsers.add_parser(
        "deploy",
        parents=[parent_parser],
        help="Deploy model to Azure ML Kubernetes endpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_endpoint_deploy_args(parser_deploy_endpoint)
    parser_deploy_endpoint.set_defaults(func=cmd_endpoint_deploy)

    # Status endpoint command
    parser_status_endpoint = endpoint_subparsers.add_parser(
        "status",
        parents=[parent_parser],
        help="Check status and traffic distribution of the endpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_endpoint_status_args(parser_status_endpoint)
    parser_status_endpoint.set_defaults(func=cmd_endpoint_status)

    # Switch traffic endpoint command
    parser_switch_traffic_endpoint = endpoint_subparsers.add_parser(
        "switch-traffic",
        parents=[parent_parser],
        help="Switch traffic between blue and green deployments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_endpoint_traffic_args(parser_switch_traffic_endpoint)
    parser_switch_traffic_endpoint.set_defaults(func=cmd_endpoint_switch_traffic)

    # Rollback endpoint command
    parser_rollback_endpoint = endpoint_subparsers.add_parser(
        "rollback",
        parents=[parent_parser],
        help="Rollback to the previous deployment by switching traffic",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_endpoint_rollback_args(parser_rollback_endpoint)
    parser_rollback_endpoint.set_defaults(func=cmd_endpoint_rollback)

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
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
