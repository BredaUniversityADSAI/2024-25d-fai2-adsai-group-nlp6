# Import the libraries
from typing import Union
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, Data
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml import command, Input, Output
from azure.ai.ml.dsl import pipeline
import os
import time
import logging

# Path setting
PIPELINE_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SRC_CODE_ROOT = os.path.join(PIPELINE_PROJECT_ROOT, "src")

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Initialization
COMPUTE_NAME = "adsai-lambda-0"


#############################
#    Azure ML Connection    #
#############################


class AzureMLConnector:
    """Class to handle Azure ML client connection and authentication."""

    def __init__(self):
        """Initialize the Azure ML client."""
        self.ml_client = self._get_azure_ml_client()

    def _get_azure_ml_client(self):
        """
        Get the Azure ML client with appropriate authentication.

        Returns:
            MLClient: An authenticated Azure ML client instance.
        """

        # Error handling
        try:

            # Project root for environment variables
            project_root_for_env = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )

            # Load environment variables
            dotenv_path = os.path.join(project_root_for_env, '.env')
            if os.path.exists(dotenv_path):
                load_dotenv(dotenv_path)

            # Option 1 - Authenticate through service principal
            credential = ClientSecretCredential(
                tenant_id=os.environ.get("AZURE_TENANT_ID"),
                client_id=os.environ.get("AZURE_CLIENT_ID"),
                client_secret=os.environ.get("AZURE_CLIENT_SECRET")
            )
            credential.get_token("https://management.azure.com/.default")

            # # Option 2 - Authenticate through default Azure credentials
            # credential = DefaultAzureCredential()
            # credential.get_token("https://management.azure.com/.default")

            # # Option 3 - Authenticate through interactive browser if needed
            # credential = InteractiveBrowserCredential()

            # Create and return the MLClient instance
            ml_client = MLClient(
                credential=credential,
                subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
                resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
                workspace_name=os.environ["AZURE_WORKSPACE_NAME"]
            )
            logger.info("✅ Successfully connected to Azure ML workspace.")

            return ml_client

        except Exception as e:
            logger.error("❌ Class 'AzureMLConnector' failed to create client.")
            logger.error(f"❌ Error: {e}")
            raise RuntimeError("❌ Class 'AzureMLConnector' failed to create client.")


######################
#    Data Handler    #
######################

class DataHandler:
    """Class to handle Azure ML data assets creation and management."""

    def __init__(self, ml_client: MLClient):
        """
        Initialize the DataHandler with an Azure ML client.

        Args:
            ml_client (MLClient): An authenticated Azure ML client instance.
        """
        self.ml_client = ml_client

    def create_data_asset_from_local_path(
        self,
        local_path: str,
        asset_name: str,
        asset_description: str,
        version: Union[str, None] = None
    ) -> Data:
        """
        Create or update a data asset from a local file or folder.
        The method automatically determines if the path is a file or a folder.

        Args:
            local_path (str): Path to the local file or folder to be uploaded.
            asset_name (str): Name of the data asset.
            asset_description (str): Description of the data asset.
            version (str | None): Version of the data asset. Defaults to None.

        Returns:
            Data: The created or updated data asset.

        Raises:
            ValueError: If the local_path is neither a file nor a directory.
        """

        # Error handling
        try:

            # Check if the local path is a file or directory
            if os.path.isfile(local_path):
                asset_type = AssetTypes.URI_FILE
            elif os.path.isdir(local_path):
                asset_type = AssetTypes.URI_FOLDER
            else:
                raise ValueError(f"'{local_path}' is not a valid file or directory.")

            # Create the data asset
            data_asset = Data(
                name=asset_name,
                description=asset_description,
                path=local_path,
                type=asset_type,
                version=version
            )

            # Create or update the data asset
            created_or_updated_asset = self.ml_client.data.create_or_update(data_asset)
            logger.info(f"✅ Data asset '{asset_name}' created/updated successfully.")

            return created_or_updated_asset

        except Exception as e:
            logger.error("❌ Class 'DataHandler' failed.")
            logger.error(f"❌ Error: {e}")
            raise RuntimeError("❌ Method 'create_data_asset_from_local_path' failed.")

    def get_data_asset(
        self,
        asset_name: str,
        version: Union[str, None] = None
    ) -> Data:
        """
        Retrieves a data asset from Azure ML.

        Args:
            asset_name (str): The name of the data asset.
            version (str | None): The specific version of the data asset.
                                  If None, the latest version is retrieved.

        Returns:
            Data: The retrieved data asset.
        """

        # Error handling
        try:

            # Retrieve data asset
            data_asset = self.ml_client.data.get(
                asset_name,
                version if version is not None else None
            )
            logger.info("✅ Data asset retrieved successfully.")
            return data_asset

        except Exception as e:
            logger.error("❌ Class 'DataHandler' method 'get_data_asset' failed.")
            logger.error(f"❌ Error: {e}")
            raise RuntimeError("❌ Class 'DataHandler' method 'get_data_asset' failed.")

    def download_data_asset(
        self,
        asset_name: str,
        version: Union[str, None] = None,
        download_path: str = "."
    ) -> str:
        """
        Downloads a data asset from Azure ML to a specified local path.

        Args:
            asset_name (str): The name of the data asset.
            version (str | None): The specific version of the data asset.
                                  If None, the latest version is downloaded.
            download_path (str): Local path where the data asset will be downloaded.

        Returns:
            str: The local path where the data asset was downloaded.
        """

        # Error handling
        try:

            # Download the data asset
            downloaded_asset = self.ml_client.data.download(
                name=asset_name,
                version=version if version is not None else None,
                download_path=download_path
            )
            logger.info(f"✅ Data asset '{asset_name}' downloaded successfully \
                to {downloaded_asset}.")
            return downloaded_asset

        except Exception as e:
            logger.error("❌ Class 'DataHandler' method 'download_data_asset' failed.")
            logger.error(f"❌ Error: {e}")
            raise RuntimeError("❌ Class 'DataHandler' method \
                'download_data_asset' failed.")


#############################
#    Environment Handler    #
#############################


class EnvironmentHandler:
    """Class to handle Azure ML environment creation and management."""

    def __init__(self, ml_client: MLClient):
        """
        Initialize the EnvironmentHandler with an Azure ML client.

        Args:
            ml_client (MLClient): An authenticated Azure ML client instance.
        """
        self.ml_client = ml_client

    def push_environment_from_yaml(
        self,
        conda_file_path: str,
        environment_name: str,
        description: str,
        base_image: str = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        version: Union[str, None] = None
    ):
        """
        Push an Azure ML environment from a conda YAML file.

        Args:
            conda_file_path (str): Path to the conda YAML file.
            environment_name (str): Name of the Azure ML environment.
            description (str): Description of the Azure ML environment.
            base_image (str): Base Docker image for the environment.
            version (str | None): Version of the environment. Defaults to None.

        Returns:
            Environment: The created or updated Azure ML environment.
        """

        # Error handling
        try:

            # Create the Azure ML environment
            azure_ml_environment = Environment({
                "name": environment_name,
                "description": description,
                "image": base_image,
                "conda_file": conda_file_path,
                "version": version
            })

            # Create or update the environment in Azure ML
            created_or_updated_env = self.ml_client.environments.create_or_update(
                azure_ml_environment
            )
            logger.info("✅ Environment created/updated successfully.")

            return created_or_updated_env

        except Exception as e:
            logger.error("❌ Class 'EnvironmentHandler' failed.")
            logger.error(f"❌ Error: {e}")
            raise RuntimeError("❌ Failed to create or update environment.")

    def get_environment(
        self,
        environment_name: str,
        version: Union[str, None] = None
    ) -> Environment:
        """
        Retrieves an Azure ML environment by name and version.

        Args:
            environment_name (str): The name of the Azure ML environment.
            version (str | None): The specific version of the environment.
                                  If None, the latest version is retrieved.

        Returns:
            Environment: The retrieved Azure ML environment.
        """

        # Error handling
        try:

            # Retrieve the environment
            azure_ml_environment = self.ml_client.environments.get(
                name=environment_name,
                version=version if version is not None else None
            )
            logger.info("✅ Environment retrieved successfully.")
            return azure_ml_environment

        except Exception as e:
            logger.error("❌ Method 'EnvironmentHandler.get_environment' failed.")
            logger.error(f"❌ Error: {e}")
            raise RuntimeError("❌ Failed to retrieve environment.")


##################
#    Pipeline    #
##################

class EmotionClassificationPipeline:
    """Class to handle the complete emotion classification pipeline in Azure ML."""

    def __init__(self, ml_client: MLClient, compute_name: str = "adsai-lambda-0"):
        """
        Initialize the pipeline with an Azure ML client.

        Args:
            ml_client (MLClient): An authenticated Azure ML client instance.
            compute_name (str): Name of the compute target to use.
        """
        self.ml_client = ml_client
        self.compute_name = compute_name

    def _create_data_pipeline_component(self, environment_name_version: str):
        """
        Creates a command component for preprocessing emotion data.

        Args:
            environment_name_version (str): The name and version of the Azure ML
                                            environment.

        Returns:
            command: An Azure ML command component for preprocessing emotion data.
        """
        return command(
            name="preprocess_emotion_data",
            display_name="Preprocess Emotion Data",
            description="Loads raw data, preprocesses text, generates features, \
                and saves processed data and encoders.",
            inputs={
                "raw_train_data": Input(
                    type=AssetTypes.URI_FILE,
                    description="Raw training CSV file from Azure ML Data Asset"
                ),
                "raw_test_data": Input(
                    type=AssetTypes.URI_FILE,
                    description="Raw test CSV file from Azure ML Data Asset"
                ),
                "output_tasks_str": Input(
                    type="string",
                    default="emotion,sub_emotion,intensity",
                    description="Comma-separated list of output tasks"
                ),
                "max_length": Input(
                    type="integer",
                    default=128,
                    description="Max sequence length for tokenizer"
                ),
                "model_name_tokenizer": Input(
                    type="string",
                    default="microsoft/deberta-v3-xsmall",
                    description="HF model name for tokenizer"
                )
            },
            outputs={
                "processed_train_data_dir": Output(
                    type=AssetTypes.URI_FOLDER,
                    mode=InputOutputModes.RW_MOUNT,
                    description="Folder for processed train.csv"
                ),
                "processed_test_data_dir": Output(
                    type=AssetTypes.URI_FOLDER,
                    mode=InputOutputModes.RW_MOUNT,
                    description="Folder for processed test.csv"
                ),                "encoders_dir": Output(
                    type=AssetTypes.URI_FOLDER,
                    mode=InputOutputModes.RW_MOUNT,
                    description="Folder for saved label encoders (.pkl files)"
                )
            },
            code=SRC_CODE_ROOT,            command=(
                "python -m emotion_clf_pipeline.cli preprocess "
                "--raw-train-path ${{inputs.raw_train_data}} "
                "--raw-test-path ${{inputs.raw_test_data}} "
                "--model-name-tokenizer ${{inputs.model_name_tokenizer}} "
                "--max-length ${{inputs.max_length}} "
                "--output-dir ${{outputs.processed_train_data_dir}} "
                "--encoders-dir ${{outputs.encoders_dir}} "
                "--output-tasks ${{inputs.output_tasks_str}}"
            ),
            environment=environment_name_version,
            compute=self.compute_name,
        )

    def _create_train_pipeline_component(self, environment_name_version: str):
        """
        Creates a command component for training an emotion classification model.

        Args:
            environment_name_version (str): The name and version of the Azure ML
                                            environment.

        Returns:
            command: An Azure ML command component for training an emotion
                     classification model.
        """
        return command(
            name="train_emotion_model",
            display_name="Train Emotion Model",
            description="Trains a DEBERTA-based model on preprocessed emotion data.",
            inputs={
                "processed_train_data_dir": Input(
                    type=AssetTypes.URI_FOLDER,
                    description="Directory with processed train.csv"),
                "processed_test_data_dir": Input(
                    type=AssetTypes.URI_FOLDER,
                    description="Directory with processed test.csv"),
                "encoders_dir": Input(
                    type=AssetTypes.URI_FOLDER,
                    description="Directory with label encoders"),
                "model_name_bert": Input(
                    type="string",
                    default="microsoft/deberta-v3-xsmall",
                    description="HF model name for BERT classifier"),
                "output_tasks_str": Input(
                    type="string",
                    default="emotion,sub_emotion,intensity"),
                "max_length": Input(
                    type="integer",
                    default=128),
                "batch_size": Input(
                    type="integer",
                    default=16),
                "epochs": Input(
                    type="integer",
                    default=1),
                "learning_rate": Input(
                    type="number",
                    default=2e-5),
            },
            outputs={
                "trained_model_dir": Output(
                    type=AssetTypes.URI_FOLDER,
                    mode=InputOutputModes.RW_MOUNT,
                    description="Directory for the trained model weights and config"
                ),                "training_metrics_file": Output(
                    type=AssetTypes.URI_FILE,
                    mode=InputOutputModes.RW_MOUNT,
                    description="JSON file with training metrics (e.g., best F1 scores)"
                )
            },
            code=SRC_CODE_ROOT,            command=(
                "python -m emotion_clf_pipeline.cli train "
                "--processed-train-dir ${{inputs.processed_train_data_dir}} "
                "--processed-test-dir ${{inputs.processed_test_data_dir}} "
                "--encoders-dir ${{inputs.encoders_dir}} "
                "--model-name ${{inputs.model_name_bert}} "
                "--output-dir ${{outputs.trained_model_dir}} "
                "--metrics-file ${{outputs.training_metrics_file}} "
                "--epochs ${{inputs.epochs}} "
                "--batch-size ${{inputs.batch_size}} "
                "--learning-rate ${{inputs.learning_rate}} "
                "--output-tasks ${{inputs.output_tasks_str}}"
            ),
            environment=environment_name_version,
            compute=self.compute_name,
        )

    def _create_evaluate_register_component(self, environment_name_version: str):
        """
        Creates a command component for evaluating and registering an emotion
        classification model.

        Args:
            environment_name_version (str): The name and version of the Azure ML
                                            environment.

        Returns:
            command: An Azure ML command component for evaluating and registering
                        an emotion classification model.
        """
        return command(
            name="evaluate_register_emotion_model",
            display_name="Evaluate and Register Emotion Model",
            description="Evaluates the trained model on test data and registers it \
                in Azure ML if performance criteria are met.",
            inputs={
                "trained_model_dir": Input(
                    type=AssetTypes.URI_FOLDER,
                    description="Directory with the trained model artifacts"
                ),
                "training_metrics_file": Input(
                    type=AssetTypes.URI_FILE,
                    description="JSON file with training metrics"
                ),
                "encoders_dir": Input(
                    type=AssetTypes.URI_FOLDER,
                    description="Directory with label encoders"
                ),
                "processed_test_data_dir": Input(
                    type=AssetTypes.URI_FOLDER,
                    description="Directory with processed test.csv for final evaluation"
                ),
                "registration_f1_threshold_emotion": Input(
                    type="number",
                    default=0.1,
                    description="Min F1 score for emotion task to register model"
                ),
                "train_path": Input(
                    type=AssetTypes.URI_FILE,
                    description="Training CSV file path for evaluation"
                ),
            },
            outputs={
                "registration_status_file": Output(
                    type=AssetTypes.URI_FILE,
                    mode=InputOutputModes.RW_MOUNT,
                    description="File indicating 'registered' or 'not_registered'"
                ),
                "final_evaluation_report_dir": Output(
                    type=AssetTypes.URI_FOLDER,
                    mode=InputOutputModes.RW_MOUNT,
                    description="Directory for final eval reports (e.g., evaluation.csv)"
                )
            },
            code=SRC_CODE_ROOT,            command=(
                "python -m emotion_clf_pipeline.cli evaluate_register "
                "--model-input-dir ${{inputs.trained_model_dir}} "
                "--processed-test-dir ${{inputs.processed_test_data_dir}} "
                "--train-path ${{inputs.train_path}} "
                "--encoders-dir ${{inputs.encoders_dir}} "
                "--final-eval-output-dir ${{outputs.final_evaluation_report_dir}} "
                "--registration-f1-threshold-emotion "
                "${{inputs.registration_f1_threshold_emotion}} "
                "--registration-status-output-file "
                "${{outputs.registration_status_file}}"
            ),
            environment=environment_name_version,
            compute=self.compute_name,
            identity={"type": "UserIdentity"},
        )

    def create_pipeline_definition(self, environment_name_version: str):
        """
        Create the pipeline definition with all components.

        Args:
            environment_name_version (str): The name and version of the Azure ML environment.

        Returns:
            function: The pipeline function decorated with @pipeline.
        """
        # Create component functions
        preprocess_data_component_func = self._create_data_pipeline_component(environment_name_version)
        train_model_component_func = self._create_train_pipeline_component(environment_name_version)
        evaluate_register_model_component_func = self._create_evaluate_register_component(environment_name_version)

        @pipeline(
            default_compute=self.compute_name,
            description="Pipeline: Preprocess, Train, Evaluate, Register",
        )
        def emotion_classification_pipeline(
            model_base_name_for_registration: str = "emotion_clf_from_pipeline",
            registration_f1_threshold: float = 0.10,
            epochs_param: int = 1
        ):
            """
            Pipeline for emotion classification: preprocess data, train model,
            evaluate and register the model in Azure ML.

            Args:
                model_base_name_for_registration (str): Base name for the registered model.
                registration_f1_threshold (float): Minimum F1 score for emotion task to
                                                   register the model.
                epochs_param (int): Number of training epochs.

            Returns:
                dict: Outputs of the pipeline including processed data directories,
                      trained model directory, registration status file, and evaluation
                      report directory.
            """

            # Initialize data pipeline components
            # TODO: Change path
            preprocess_step = preprocess_data_component_func(
                raw_train_data=Input(
                    type=AssetTypes.URI_FILE, 
                    path="azureml:emotion-train-data:1"
                ),
                raw_test_data=Input(
                    type=AssetTypes.URI_FILE, 
                    path="azureml:emotion-test-data:1"
                ),
            )

            # Initialize training pipeline components
            train_step = train_model_component_func(
                processed_train_data_dir=preprocess_step.outputs.processed_train_data_dir,
                processed_test_data_dir=preprocess_step.outputs.processed_test_data_dir,
                encoders_dir=preprocess_step.outputs.encoders_dir,
                epochs=epochs_param
            )

            # Initialize evaluation and registration pipeline components
            # TODO: Change path
            evaluate_register_step = evaluate_register_model_component_func(
                trained_model_dir=train_step.outputs.trained_model_dir,
                training_metrics_file=train_step.outputs.training_metrics_file,
                encoders_dir=preprocess_step.outputs.encoders_dir,
                processed_test_data_dir=preprocess_step.outputs.processed_test_data_dir,
                train_path=Input(type=AssetTypes.URI_FILE, path="azureml:emotion-train-data:1"),
                registration_f1_threshold_emotion=registration_f1_threshold
            )

            return {
                "train_preprocess_job": preprocess_step.outputs.processed_train_data_dir,
                "test_preprocess_job": preprocess_step.outputs.processed_test_data_dir,
                "train_model_job": train_step.outputs.trained_model_dir,
                "reg_status_job": evaluate_register_step.outputs.registration_status_file,
                "eval_report_job": evaluate_register_step.outputs.final_evaluation_report_dir
            }

        return emotion_classification_pipeline

    def create_and_submit_pipeline(
        self,
        environment_name_version: str,
        experiment_name: str,
        model_base_name_for_registration: str = "emotion_clf_from_pipeline",
        registration_f1_threshold: float = 0.10,
        epochs_param: int = 1
    ):
        """
        Create and submit the complete pipeline to Azure ML.

        Args:
            environment_name_version (str): The name and version of the Azure ML environment.
            experiment_name (str): Name of the experiment to submit the pipeline to.
            model_base_name_for_registration (str): Base name for the registered model.
            registration_f1_threshold (float): Minimum F1 score for emotion task to register the model.
            epochs_param (int): Number of training epochs.

        Returns:
            The submitted pipeline job.
        """
        try:
            # Create pipeline definition
            pipeline_func = self.create_pipeline_definition(environment_name_version)

            # Create pipeline job instance
            pipeline_job = pipeline_func(
                model_base_name_for_registration=model_base_name_for_registration,
                registration_f1_threshold=registration_f1_threshold,
                epochs_param=epochs_param
            )

            # Submit the job
            returned_job = self.ml_client.jobs.create_or_update(
                pipeline_job, 
                experiment_name=experiment_name
            )

            logger.info(f"✅ Pipeline submitted successfully. Job ID: {returned_job.id}")
            return returned_job

        except Exception as e:
            logger.error("❌ Failed to create and submit pipeline.")
            logger.error(f"❌ Error: {e}")
            raise RuntimeError("❌ Failed to create and submit pipeline.")


# Start the program
if __name__ == "__main__":

    # Initialization

    # Compute
    COMPUTE_NAME = "adsai-lambda-0"

    # Environment
    # Leave NA for version for latest
    ENV_NAME, ENV_VERSION = "emotion-clf-pipeline-env", "23"
    AML_ENVIRONMENT_NAME_VERSION = f"{ENV_NAME}:{ENV_VERSION}"
    conda_file_full_path = os.path.join(
        PIPELINE_PROJECT_ROOT, "environment/environment.yml"
    )

    # Data assets
    # Leave NA for version for latest
    RAW_TRAIN_DATA_ASSET_NAME = "emotion-raw-train"
    RAW_TRAIN_DATA_ASSET_VERSION = "1"
    RAW_TEST_DATA_ASSET_NAME = "emotion-raw-test"
    RAW_TEST_DATA_ASSET_VERSION = "1"

    # Experiment
    EXPERIMENT_NAME = "emotion_clf_pipeline_runs"

    # Retry logic
    MAX_RETRIES = 10
    RETRY_DELAY_SECONDS = 3

    # STEP 1 - Connect to azure
    for attempt in range(MAX_RETRIES):
        azure_connector = AzureMLConnector()
        ml_client = azure_connector.ml_client
        break
        time.sleep(RETRY_DELAY_SECONDS)

    # Step 2 - Get environment
    for attempt in range(MAX_RETRIES):
        env_handler = EnvironmentHandler(ml_client)
        azure_ml_env = env_handler.get_environment(
            environment_name=ENV_NAME,
            version=ENV_VERSION
        )
        break
        time.sleep(RETRY_DELAY_SECONDS)

    # Step 3 - Get data assets
    for attempt in range(MAX_RETRIES):
        data_handler = DataHandler(ml_client)
        train_data_asset = data_handler.get_data_asset(
            asset_name=RAW_TRAIN_DATA_ASSET_NAME,
            version=RAW_TRAIN_DATA_ASSET_VERSION
        )
        test_data_asset = data_handler.get_data_asset(
            asset_name=RAW_TEST_DATA_ASSET_NAME,
            version=RAW_TEST_DATA_ASSET_VERSION
        )
        break
        time.sleep(RETRY_DELAY_SECONDS)

    # Step 3.1 - Download data assets
    for attempt in range(MAX_RETRIES):
        train_data_path = data_handler.download_data_asset(
            asset_name=RAW_TRAIN_DATA_ASSET_NAME,
            version=RAW_TRAIN_DATA_ASSET_VERSION,
            download_path=os.path.join(PIPELINE_PROJECT_ROOT, "data", "raw")
        )
        test_data_path = data_handler.download_data_asset(
            asset_name=RAW_TEST_DATA_ASSET_NAME,
            version=RAW_TEST_DATA_ASSET_VERSION,
            download_path=os.path.join(PIPELINE_PROJECT_ROOT, "data", "raw")
        )
        break
        time.sleep(RETRY_DELAY_SECONDS)    # Step 4 - Pipeline
    # pipeline_handler = EmotionClassificationPipeline(ml_client, COMPUTE_NAME)
    # unique_model_suffix = str(uuid.uuid4())[:8]

    # returned_job = pipeline_handler.create_and_submit_pipeline(
    #     environment_name_version=AML_ENVIRONMENT_NAME_VERSION,
    #     experiment_name=EXPERIMENT_NAME,
    #     model_base_name_for_registration=f"emotion_clf_pipeline_{unique_model_suffix}",
    #     registration_f1_threshold=0.05,
    #     epochs_param=1
    # )