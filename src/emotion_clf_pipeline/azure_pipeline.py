"""
Azure ML Pipeline Integration for Emotion Classification Pipeline.

This module handles Azure ML pipeline submission and monitoring for:
- Data preprocessing pipelines
- Model training pipelines
- Job status monitoring

Integrates with existing Azure ML resources including compute instances,
environments, and data assets.
"""

import logging
import os
import shutil
import tempfile
from typing import Dict, Optional

# Azure ML Configuration
ENV_NAME, ENV_VERSION = "emotion-clf-pipeline-env", "23"
COMPUTE_NAME = "adsai-lambda-0"

# Data assets
# Leave NA for version for latest
RAW_TRAIN_DATA_ASSET_NAME = "emotion-raw-train"
RAW_TRAIN_DATA_ASSET_VERSION = "1"
RAW_TEST_DATA_ASSET_NAME = "emotion-raw-test"
RAW_TEST_DATA_ASSET_VERSION = "1"

logger = logging.getLogger(__name__)

try:
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import Job, Data
    from azure.ai.ml.constants import AssetTypes
    from azure.identity import DefaultAzureCredential
    AZURE_AVAILABLE = True
except ImportError:
    logger.warning("Azure ML SDK not available. Install with: pip install azure-ai-ml")
    AZURE_AVAILABLE = False


def get_ml_client() -> MLClient:
    """Get Azure ML client with authentication."""
    if not AZURE_AVAILABLE:
        raise ImportError("Azure ML SDK not available")    # Read Azure ML configuration
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP")
    workspace_name = os.getenv("AZURE_WORKSPACE_NAME")

    if not all([subscription_id, resource_group, workspace_name]):
        raise ValueError(
            "Missing Azure ML configuration. Please set environment variables: "
            "AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE_NAME"
        )

    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )

    return ml_client


def submit_preprocess_pipeline(args) -> Job:
    """Submit data preprocessing pipeline to Azure ML."""
    ml_client = get_ml_client()

    # Create temporary directory with required files
    temp_dir = create_temp_code_directory()

    try:
        # Define preprocessing command job
        from azure.ai.ml import command, Input, Output
        from azure.ai.ml.constants import AssetTypes
        job = command(
            code=temp_dir,
            command=(
                "python -c \"import nltk; "
                "nltk.download('punkt', quiet=True); "
                "nltk.download('punkt_tab', quiet=True); "
                "nltk.download('averaged_perceptron_tagger', quiet=True); "
                "nltk.download('vader_lexicon', quiet=True); "
                "nltk.download('stopwords', quiet=True)\" "
                "&& python -m src.emotion_clf_pipeline.data "
                f"--raw-train-path ${{inputs.train_data}} "
                f"--raw-test-path ${{inputs.test_data}} "
                f"--output-dir ${{outputs.processed_data}} "
                f"--encoders-dir ${{outputs.encoders}} "
                f"--model-name-tokenizer {args.model_name_tokenizer} "
                f"--max-length {args.max_length} "
                f"--output-tasks {args.output_tasks}"
            ),
            inputs={
                "train_data": Input(
                    type=AssetTypes.URI_FOLDER,
                    path=f"azureml:{RAW_TRAIN_DATA_ASSET_NAME}:{RAW_TRAIN_DATA_ASSET_VERSION}"  # noqa: E501
                ),
                "test_data": Input(
                    type=AssetTypes.URI_FOLDER,
                    path=f"azureml:{RAW_TEST_DATA_ASSET_NAME}:{RAW_TEST_DATA_ASSET_VERSION}"  # noqa: E501
                ),
            },
            outputs={
                "processed_data": Output(type=AssetTypes.URI_FOLDER),
                "encoders": Output(type=AssetTypes.URI_FOLDER),
            },
            environment=f"azureml:{ENV_NAME}:{ENV_VERSION}",
            compute=COMPUTE_NAME,
            display_name="emotion-clf-preprocess",
            description="Data preprocessing for emotion classification"
        )

        # Submit job
        job = ml_client.jobs.create_or_update(job)
        logger.info(f"Submitted preprocessing job: {job.name}")

        return job

    finally:
        # Clean up temporary directory after job submission
        cleanup_temp_directory(temp_dir)


def submit_training_pipeline(args) -> Job:
    """Submit model training pipeline to Azure ML."""
    ml_client = get_ml_client()

    # Create temporary directory with required files
    temp_dir = create_temp_code_directory()

    try:
        # Define training command job
        from azure.ai.ml import command, Input, Output

        job = command(
            code=temp_dir,            command=(
                "python -c \"import nltk; "
                "nltk.download('punkt', quiet=True); "
                "nltk.download('punkt_tab', quiet=True); "
                "nltk.download('averaged_perceptron_tagger', quiet=True); "
                "nltk.download('vader_lexicon', quiet=True); "
                "nltk.download('stopwords', quiet=True)\" "
                "&& python -m src.emotion_clf_pipeline.train "
                f"--model-name {args.model_name} "
                f"--batch-size {args.batch_size} "
                f"--learning-rate {args.learning_rate} "
                f"--epochs {args.epochs} "
                "--train-data ${{inputs.train_data}} "
                "--test-data ${{inputs.test_data}} "
                "--output-dir ${{outputs.model_output}}"
            ),
            environment=f"azureml:{ENV_NAME}:{ENV_VERSION}",
            compute=COMPUTE_NAME,
            inputs={
                "train_data": Input(
                    type="uri_file",
                    path="azureml:emotion-processed-train:1"
                ),
                "test_data": Input(
                    type="uri_file",
                    path="azureml:emotion-processed-test:1"
                )
            },
            outputs={
                "model_output": Output(type="uri_folder", mode="rw_mount")
            },
            display_name="emotion-clf-training",
            description="Model training for emotion classification"
        )

        # Submit job
        job = ml_client.jobs.create_or_update(job)
        logger.info(f"Submitted training job: {job.name}")

        return job

    finally:
        # Clean up temporary directory after job submission
        cleanup_temp_directory(temp_dir)


def get_pipeline_status(job_id: str) -> str:
    """Get the status of an Azure ML job."""
    ml_client = get_ml_client()

    try:
        job = ml_client.jobs.get(job_id)
        return job.status
    except Exception as e:
        logger.error(f"Failed to get job status: {str(e)}")
        return "Unknown"


def list_recent_jobs(limit: int = 10) -> list:
    """List recent Azure ML jobs."""
    ml_client = get_ml_client()

    try:
        jobs = list(ml_client.jobs.list(max_results=limit))
        return jobs
    except Exception as e:
        logger.error(f"Failed to list jobs: {str(e)}")
        return []


def download_job_outputs(job_id: str, output_path: str = "./outputs") -> bool:
    """Download outputs from an Azure ML job."""
    ml_client = get_ml_client()

    try:
        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        # Download job outputs
        ml_client.jobs.download(job_id, download_path=output_path)
        logger.info(f"Downloaded job outputs to: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to download job outputs: {str(e)}")
        return False


def create_data_asset(name: str, path: str, description: str = "") -> Optional[Data]:
    """Create or update a data asset in Azure ML."""
    ml_client = get_ml_client()

    try:
        data_asset = Data(
            name=name,
            path=path,
            type=AssetTypes.URI_FOLDER,
            description=description
        )

        created_data = ml_client.data.create_or_update(data_asset)
        logger.info(f"Created data asset: {created_data.name}")
        return created_data

    except Exception as e:
        logger.error(f"Failed to create data asset: {str(e)}")
        return None


def get_compute_status(compute_name: str) -> str:
    """Get the status of an Azure ML compute resource."""
    ml_client = get_ml_client()

    try:
        compute = ml_client.compute.get(compute_name)
        return compute.provisioning_state
    except Exception as e:
        logger.error(f"Failed to get compute status: {str(e)}")
        return "Unknown"


# Configuration helpers
def validate_azure_config() -> bool:
    """Validate Azure ML configuration."""
    required_vars = [
        "AZURE_SUBSCRIPTION_ID",
        "AZURE_RESOURCE_GROUP",
        "AZURE_WORKSPACE_NAME"
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return False

    return True


def get_azure_config() -> Dict[str, str]:
    """Get current Azure ML configuration."""
    return {
        "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID", ""),
        "resource_group": os.getenv("AZURE_RESOURCE_GROUP", ""),
        "workspace_name": os.getenv("AZURE_WORKSPACE_NAME", ""),
        "tenant_id": os.getenv("AZURE_TENANT_ID", ""),
    }


def create_temp_code_directory() -> str:
    """
    Create a temporary directory with only the required files for Azure ML.

    Returns:
        str: Path to the temporary directory
    """
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="azureml_code_")

    try:
        # Copy entire src directory to preserve structure
        shutil.copytree("./src", os.path.join(temp_dir, "src"))

        # Copy models/features directory (contains EmoLex lexicon)
        models_dest = os.path.join(temp_dir, "models")
        os.makedirs(models_dest, exist_ok=True)

        if os.path.exists("./models/features"):
            features_dest = os.path.join(models_dest, "features")
            shutil.copytree("./models/features", features_dest)

        # Copy models/encoders directory if it exists
        if os.path.exists("./models/encoders"):
            encoders_dest = os.path.join(models_dest, "encoders")
            shutil.copytree("./models/encoders", encoders_dest)

        logger.info(f"Created temporary code directory: {temp_dir}")
        return temp_dir

    except Exception as e:
        # Clean up on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValueError(f"Failed to create temporary directory: {str(e)}")


def cleanup_temp_directory(temp_dir: str) -> None:
    """Clean up temporary directory."""
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.warning(f"Failed to clean up temporary directory: {str(e)}")


def register_processed_data_assets(job: Job) -> bool:
    """
    Wait for preprocessing job to complete and register outputs as data assets.

    Args:
        job: The preprocessing job to monitor

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from azure.ai.ml.entities import Data
        from azure.ai.ml.constants import AssetTypes
        import os
        import tempfile
        import time

        ml_client = get_ml_client()

        # Wait for job completion
        logger.info(f"Waiting for job {job.name} to complete...")
        start_time = time.time()
        timeout = 3600  # 1 hour timeout

        while job.status not in ["Completed", "Failed", "Canceled"]:
            if time.time() - start_time > timeout:
                logger.error(f"Job {job.name} timed out after {timeout} seconds")
                return False

            time.sleep(30)  # Check every 30 seconds
            job = ml_client.jobs.get(job.name)
            logger.info(f"Job status: {job.status}")

        if job.status != "Completed":
            logger.error(f"Job {job.name} failed with status: {job.status}")
            return False

        logger.info(f"Job {job.name} completed successfully!")

        # Get job outputs
        processed_data_output = job.outputs.get("processed_data")
        if not processed_data_output:
            logger.error("No processed_data output found in job")
            return False

        # Download the processed data to a temporary location
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info("Downloading processed data from job outputs...")

            # Download the processed data folder
            ml_client.jobs.download(
                job.name,
                output_name="processed_data",
                download_path=temp_dir
            )

            # Check possible paths for downloaded data
            # Azure ML may download directly to temp_dir or create subdirectories
            possible_paths = [
                os.path.join(temp_dir, "processed_data"),  # Expected path
                temp_dir,  # Direct download to temp_dir
                os.path.join(temp_dir, "named-outputs", "processed_data"),
            ]

            processed_data_path = None
            for path in possible_paths:
                logger.info(f"Checking for processed data at: {path}")
                if os.path.exists(path):
                    # Check if this path contains the expected CSV files
                    train_csv = os.path.join(path, "train.csv")
                    test_csv = os.path.join(path, "test.csv")
                    if os.path.exists(train_csv) and os.path.exists(test_csv):
                        processed_data_path = path
                        logger.info(f"Found processed data at: {processed_data_path}")
                        break
                    else:
                        logger.info(
                            f"Path exists but doesn't contain train.csv and test.csv: "
                            f"{path}"
                        )

            if not processed_data_path:
                # List contents of temp_dir for debugging
                logger.error(f"Processed data not found. Contents of {temp_dir}:")
                for item in os.listdir(temp_dir):
                    item_path = os.path.join(temp_dir, item)
                    if os.path.isdir(item_path):
                        logger.error(f"  Directory: {item}/")
                        for subitem in os.listdir(item_path):
                            logger.error(f"    {subitem}")
                    else:
                        logger.error(f"  File: {item}")
                return False            # Check for train.csv and test.csv
            train_csv_path = os.path.join(processed_data_path, "train.csv")
            test_csv_path = os.path.join(processed_data_path, "test.csv")

            if not os.path.exists(train_csv_path):
                logger.error(f"train.csv not found at: {train_csv_path}")
                return False

            if not os.path.exists(test_csv_path):
                logger.error(f"test.csv not found at: {test_csv_path}")
                return False

            # Register train data asset (as UriFile pointing to CSV)
            logger.info("Registering emotion-processed-train data asset...")
            train_data_asset = Data(
                name="emotion-processed-train",
                description="Processed training data for emotion classification",
                path=train_csv_path,
                type=AssetTypes.URI_FILE
            )

            train_asset = ml_client.data.create_or_update(train_data_asset)
            logger.info(
                f"✅ Registered train data asset: {train_asset.name} "
                f"(version: {train_asset.version})"
            )

            # Register test data asset (as UriFile pointing to CSV)
            logger.info("Registering emotion-processed-test data asset...")
            test_data_asset = Data(
                name="emotion-processed-test",
                description="Processed test data for emotion classification",
                path=test_csv_path,
                type=AssetTypes.URI_FILE
            )

            test_asset = ml_client.data.create_or_update(test_data_asset)
            logger.info(
                f"✅ Registered test data asset: {test_asset.name} "
                f"(version: {test_asset.version})"
            )

        logger.info("Successfully registered processed data assets!")
        return True

    except Exception as e:
        logger.error(f"Failed to register processed data assets: {str(e)}")
        logger.error("Full error traceback:", exc_info=True)
        return False
