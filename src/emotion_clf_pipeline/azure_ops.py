# Import the libraries
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, BuildContext, Data
from azure.ai.ml.constants import AssetTypes, InputOutputModes
# Add these imports for Azure ML Pipelines
from azure.ai.ml import command, Input, Output
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import Model # For model registration type hint
import os
import logging
import uuid
import time # For optional job status polling

############################
#    Azure ML Connector    #
############################

class AzureMLConnector:
    """A class to manage connection to Azure Machine Learning workspace."""

    def __init__(self):
        """Initializes the AzureMLConnector and establishes the MLClient connection."""
        self.logger = logging.getLogger(__name__)
        self.ml_client = self._get_azure_ml_client()


    def _get_azure_ml_client(self) -> MLClient | None:
        """
        Connects to Azure ML workspace and returns an MLClient.
        Tries to load credentials from a .env file at the project root first.
        Then attempts to authenticate using DefaultAzureCredential, falling back
        to InteractiveBrowserCredential if needed.
        Returns:
            MLClient: An authenticated MLClient object or None on failure.
        """
        # Project root is three levels up from src/emotion_clf_pipeline/azure.py
        project_root_for_env = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dotenv_path = os.path.join(project_root_for_env, '.env')

        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)
            self.logger.info(f".env file loaded from {dotenv_path}")
        else:
            self.logger.info(".env file not found, relying on environment variables or other auth methods.")

        try:
            subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
            resource_group = os.environ["AZURE_RESOURCE_GROUP"]
            workspace_name = os.environ["AZURE_ML_WORKSPACE_NAME"]
            self.logger.info(f"Using Azure ML config: Subscription ID: {subscription_id}, RG: {resource_group}, Workspace: {workspace_name}")
        except KeyError as e:
            self.logger.error(f"Missing Azure ML config: {e}. Set in environment or .env.")
            return None

        credential = None
        try:
            self.logger.info("Attempting DefaultAzureCredential...")
            credential = DefaultAzureCredential()
            credential.get_token("https://management.azure.com/.default")  # Validate credential
            self.logger.info("DefaultAzureCredential successful.")
        except Exception as e_default:
            self.logger.warning(f"DefaultAzureCredential failed: {e_default}. Trying InteractiveBrowserCredential...")
            try:
                credential = InteractiveBrowserCredential()
                self.logger.info("InteractiveBrowserCredential successful. Please follow browser prompts if any.")
            except Exception as ex_interactive:
                self.logger.error(f"InteractiveBrowserCredential failed: {ex_interactive}. Azure authentication failed.")
                return None

        try:
            ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)
            self.logger.info(f"Successfully connected to Azure ML Workspace: {workspace_name}")
            return ml_client
        except Exception as e_ml_client:
            self.logger.error(f"Failed to create MLClient: {e_ml_client}")
            return None

    def test_connection(self) -> bool:
        """
        Tests the connection to the Azure ML workspace by listing compute targets.
        Returns:
            bool: True if the connection test is successful, False otherwise.
        """
        if not self.ml_client:
            self.logger.error("MLClient not initialized. Cannot test connection.")
            return False
        try:
            self.logger.info("\n--- Testing Azure ML Connection ---")
            computes = self.ml_client.compute.list()
            self.logger.info("Available compute targets:")
            for compute_target_info in computes: # Renamed variable to avoid conflict
                state = getattr(compute_target_info, 'state', 'N/A') # Use renamed variable
                self.logger.info(
                    f"- Name: {compute_target_info.name}, Type: {compute_target_info.type}, State: {state}"
                )
            self.logger.info("Azure ML connection test successful.")
            return True
        except Exception as e:
            self.logger.error(f"Error during Azure ML connection test: {e}")
            return False


#######################
#    Data Handling    #
#######################

class DataHandler:
    """Handles uploading data and creating versioned data assets in Azure ML."""

    def __init__(self, ml_client: MLClient):
        """
        Initializes the DataHandler with an MLClient.
        Args:
            ml_client (MLClient): The Azure ML client.
        """
        self.ml_client = ml_client
        self.logger = logging.getLogger(__name__)

    def create_data_asset_from_local_file(
        self,
        local_file_path: str,
        asset_name: str,
        asset_description: str,
        version: str | None = None,
    ) -> Data | None:
        """
        Creates or updates a versioned Azure ML Data Asset from a local file.
        The data will be uploaded to the default datastore.
        Args:
            local_file_path (str): Path to the local file.
            asset_name (str): The name for the Azure ML Data Asset.
            asset_description (str): A description for the Data Asset.
            version (str, optional): The version of the Data Asset. Auto-increments if None.
        Returns:
            Data | None: The created/updated Azure ML Data Asset or None on failure.
        """
        if not os.path.exists(local_file_path):
            self.logger.error(f"Local file not found: {local_file_path}")
            raise FileNotFoundError(f"Local file not found: {local_file_path}")

        self.logger.info(
            f"Attempting to create/update Data Asset: {asset_name} "
            f"from local file: {local_file_path}"
        )

        data_asset = Data(
            name=asset_name,
            description=asset_description,
            path=local_file_path,  # Local path for upload
            type=AssetTypes.URI_FILE,
            version=version,
        )

        try:
            created_asset = self.ml_client.data.create_or_update(data_asset)
            self.logger.info(
                f"Data Asset '{created_asset.name}' version '{created_asset.version}' "
                "created/updated successfully."
            )
            self.logger.info(f"Data uploaded to: {created_asset.path}")
            return created_asset
        except Exception as e:
            self.logger.error(
                f"Error creating/updating Data Asset '{asset_name}': {e}"
            )
            return None

    def create_data_asset_from_local_folder(
        self,
        local_folder_path: str,
        asset_name: str,
        asset_description: str,
        version: str | None = None,
    ) -> Data | None:
        """
        Creates or updates a versioned Azure ML Data Asset from a local folder.
        The data will be uploaded to the default datastore.
        Args:
            local_folder_path (str): Path to the local folder.
            asset_name (str): The name for the Azure ML Data Asset.
            asset_description (str): A description for the Data Asset.
            version (str, optional): The version of the Data Asset. Auto-increments if None.
        Returns:
            Data | None: The created/updated Azure ML Data Asset or None on failure.
        """
        if not os.path.isdir(local_folder_path):
            self.logger.error(f"Local folder not found or not a directory: {local_folder_path}")
            raise FileNotFoundError(f"Local folder not found: {local_folder_path}")

        self.logger.info(
            f"Attempting to create/update Data Asset: {asset_name} "
            f"from local folder: {local_folder_path}"
        )

        data_asset = Data(
            name=asset_name,
            description=asset_description,
            path=local_folder_path, # Local path for upload
            type=AssetTypes.URI_FOLDER,
            version=version,
        )

        try:
            created_asset = self.ml_client.data.create_or_update(data_asset)
            self.logger.info(
                f"Data Asset '{created_asset.name}' version '{created_asset.version}' "
                "created/updated successfully."
            )
            self.logger.info(f"Data uploaded to: {created_asset.path}")
            return created_asset
        except Exception as e:
            self.logger.error(
                f"Error creating/updating Data Asset '{asset_name}': {e}"
            )
            return None


#####################
#    Environment    #
##################### 


class EnvironmentHandler:
    """
    Handles creation and management of Azure ML Environments from either
    a conda environment.yml file or a Dockerfile.
    """

    def __init__(self, ml_client: MLClient):
        """
        Initializes the EnvironmentHandler with an MLClient.
        Args:
            ml_client (MLClient): The Azure ML client.
        """
        self.ml_client = ml_client
        self.logger = logging.getLogger(__name__)

    def _create_or_update_env(self, env_params: dict) -> Environment | None:
        """Helper function to create or update the environment."""
        azure_ml_environment = Environment(**env_params)
        try:
            created_env = self.ml_client.environments.create_or_update(
                azure_ml_environment
            )
            self.logger.info(
                f"Environment '{created_env.name}' version '{created_env.version}' "
                "created/updated successfully."
            )
            return created_env
        except Exception as e:  # Catching a broader exception for robustness
            self.logger.error(
                f"Error creating/updating environment '{env_params.get('name')}': {e}"
            )
            return None

    def push_environment_from_yaml(
        self,
        conda_file_path: str,
        environment_name: str,
        description: str,
        base_image: str = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04-py311", # Ensure py311 is intended
        version: str | None = None,
    ) -> Environment | None:
        """
        Creates or updates an Azure ML Environment from a local conda environment.yml file.
        Args:
            conda_file_path (str): Path to the local environment.yml file.
            environment_name (str): The name for the Azure ML Environment.
            description (str): A description for the environment.
            base_image (str, optional): The base Docker image for the environment.
            version (str, optional): The version of the environment. Auto-increments if None.
        Returns:
            Environment | None: The created/updated Azure ML Environment or None on failure.
        """
        if not os.path.exists(conda_file_path):
            self.logger.error(f"Conda file not found: {conda_file_path}")
            raise FileNotFoundError(f"Conda file not found: {conda_file_path}")

        self.logger.info(
            f"Attempting to create/update Conda environment: {environment_name} "
            f"from {conda_file_path}"
        )
        env_params = {
            "name": environment_name,
            "description": description,
            "image": base_image,
            "conda_file": conda_file_path,
            "version": version,
        }
        return self._create_or_update_env(env_params)

    def push_environment_from_dockerfile(
        self,
        dockerfile_path: str, # Relative to build_context_path
        build_context_path: str,
        environment_name: str,
        description: str,
        version: str | None = None,
    ) -> Environment | None:
        """
        Creates or updates an Azure ML Environment from a Dockerfile.
        Args:
            dockerfile_path (str): Relative path to the Dockerfile within the build context.
            build_context_path (str): Absolute path to the build context directory.
            environment_name (str): The name for the Azure ML Environment.
            description (str): A description for the environment.
            version (str, optional): The version of the environment. Auto-increments if None.
        Returns:
            Environment | None: The created/updated Azure ML Environment or None on failure.
        """
        if not os.path.isdir(build_context_path):
            self.logger.error(f"Build context path not found or not a directory: {build_context_path}")
            raise FileNotFoundError(f"Build context path not found: {build_context_path}")

        full_dockerfile_path = os.path.join(build_context_path, dockerfile_path)
        if not os.path.exists(full_dockerfile_path):
            self.logger.error(f"Dockerfile not found at: {full_dockerfile_path}")
            raise FileNotFoundError(f"Dockerfile not found at: {full_dockerfile_path}")

        self.logger.info(
            f"Attempting to create/update Docker environment: {environment_name} "
            f"from Dockerfile: {dockerfile_path} in context: {build_context_path}"
        )
        env_params = {
            "name": environment_name,
            "description": description,
            "build": BuildContext(
                path=build_context_path,
                dockerfile_path=dockerfile_path, # Must be relative to context path
            ),
            "version": version,
        }
        return self._create_or_update_env(env_params)


##################
#    Pipeline    #
################## 

# Define the compute target from your workspace
COMPUTE_NAME = "adsai-lambda-0" 

# Define base path for component scripts and code context
# This assumes azure.py is in src/emotion_clf_pipeline, so project_root is three levels up.
PIPELINE_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Global placeholders for component functions, to be initialized in main
# This allows MLClient and environment to be passed correctly during component creation.
preprocess_data_component_func = None
train_model_component_func = None
evaluate_register_model_component_func = None


# Component for Data Preprocessing
def get_preprocess_data_component(ml_client: MLClient, environment_name_version: str):
    """Factory function to create the data preprocessing component."""
    logger = logging.getLogger(__name__)
    logger.info(f"Defining preprocess component with env: {environment_name_version} and code: {PIPELINE_PROJECT_ROOT}")
    preprocess_command_instance = command(
        name="preprocess_emotion_data",
        display_name="Preprocess Emotion Data",
        description="Loads raw data, preprocesses text, generates features, and saves processed data and encoders.",
        inputs={
            "raw_train_data": Input(type=AssetTypes.URI_FILE, description="Raw training CSV file from Azure ML Data Asset"),
            "raw_test_data": Input(type=AssetTypes.URI_FILE, description="Raw test CSV file from Azure ML Data Asset"),
            "output_tasks_str": Input(type="string", default="emotion,sub_emotion,intensity", description="Comma-separated list of output tasks"),
            "max_length": Input(type="integer", default=128, description="Max sequence length for tokenizer"),
            "model_name_tokenizer": Input(type="string", default="microsoft/deberta-v3-xsmall", description="HF model name for tokenizer")
        },
        outputs={
            "processed_train_data_dir": Output(type=AssetTypes.URI_FOLDER, mode=InputOutputModes.RW_MOUNT, description="Folder for processed train.csv"),
            "processed_test_data_dir": Output(type=AssetTypes.URI_FOLDER, mode=InputOutputModes.RW_MOUNT, description="Folder for processed test.csv"),
            "encoders_dir": Output(type=AssetTypes.URI_FOLDER, mode=InputOutputModes.RW_MOUNT, description="Folder for saved label encoders (.pkl files)")
        },
        code=PIPELINE_PROJECT_ROOT, 
        command="""python src/emotion_clf_pipeline/train.py \
                    --action preprocess \
                    --raw_train_csv_path ${{inputs.raw_train_data}} \
                    --raw_test_csv_path ${{inputs.raw_test_data}} \
                    --output_tasks ${{inputs.output_tasks_str}} \
                    --max_length ${{inputs.max_length}} \
                    --model_name_tokenizer ${{inputs.model_name_tokenizer}} \
                    --processed_train_output_dir ${{outputs.processed_train_data_dir}} \
                    --processed_test_output_dir ${{outputs.processed_test_data_dir}} \
                    --encoders_output_dir ${{outputs.encoders_dir}}
                """,
        environment=environment_name_version,
        compute=COMPUTE_NAME,
    )
    return preprocess_command_instance

# Component for Model Training
def get_train_model_component(ml_client: MLClient, environment_name_version: str):
    """Factory function to create the model training component."""
    logger = logging.getLogger(__name__)
    logger.info(f"Defining train component with env: {environment_name_version} and code: {PIPELINE_PROJECT_ROOT}")
    train_command_instance = command(
        name="train_emotion_model",
        display_name="Train Emotion Model",
        description="Trains a DEBERTA-based model on preprocessed emotion data.",
        inputs={
            "processed_train_data_dir": Input(type=AssetTypes.URI_FOLDER, description="Directory with processed train.csv"),
            "processed_test_data_dir": Input(type=AssetTypes.URI_FOLDER, description="Directory with processed test.csv (for val split within train.py)"),
            "encoders_dir": Input(type=AssetTypes.URI_FOLDER, description="Directory with label encoders"),
            "model_name_bert": Input(type="string", default="microsoft/deberta-v3-xsmall", description="HF model name for BERT classifier"),
            "output_tasks_str": Input(type="string", default="emotion,sub_emotion,intensity"),
            "max_length": Input(type="integer", default=128),
            "batch_size": Input(type="integer", default=16),
            "epochs": Input(type="integer", default=1), # Adjust as needed, 1 is for quick testing
            "learning_rate": Input(type="number", default=2e-5),
        },
        outputs={
            "trained_model_dir": Output(type=AssetTypes.URI_FOLDER, mode=InputOutputModes.RW_MOUNT, description="Directory for the trained model weights and config"),
            "training_metrics_file": Output(type=AssetTypes.URI_FILE, mode=InputOutputModes.RW_MOUNT, description="JSON file with training metrics (e.g., best F1 scores)")
        },
        code=PIPELINE_PROJECT_ROOT,
        command="""python src/emotion_clf_pipeline/train.py \
                    --action train \
                    --processed_train_dir ${{inputs.processed_train_data_dir}} \
                    --processed_test_dir ${{inputs.processed_test_data_dir}} \
                    --encoders_input_dir ${{inputs.encoders_dir}} \
                    --model_name_bert ${{inputs.model_name_bert}} \
                    --output_tasks ${{inputs.output_tasks_str}} \
                    --max_length ${{inputs.max_length}} \
                    --batch_size ${{inputs.batch_size}} \
                    --epochs ${{inputs.epochs}} \
                    --learning_rate ${{inputs.learning_rate}} \
                    --trained_model_output_dir ${{outputs.trained_model_dir}} \
                    --metrics_output_file ${{outputs.training_metrics_file}}
                """,
        environment=environment_name_version,
        compute=COMPUTE_NAME,
    )
    return train_command_instance

# Component for Model Evaluation and Registration
def get_evaluate_register_model_component(ml_client: MLClient, environment_name_version: str):
    """Factory function to create the model evaluation and registration component."""
    logger = logging.getLogger(__name__)
    logger.info(f"Defining evaluate_register component with env: {environment_name_version} and code: {PIPELINE_PROJECT_ROOT}")
    eval_reg_command_instance = command(
        name="evaluate_register_emotion_model",
        display_name="Evaluate and Register Emotion Model",
        description="Evaluates the trained model on test data and registers it in Azure ML if performance criteria are met.",
        inputs={
            "trained_model_dir": Input(type=AssetTypes.URI_FOLDER, description="Directory with the trained model artifacts"),
            "training_metrics_file": Input(type=AssetTypes.URI_FILE, description="JSON file with training metrics"),
            "encoders_dir": Input(type=AssetTypes.URI_FOLDER, description="Directory with label encoders"),
            "processed_test_data_dir": Input(type=AssetTypes.URI_FOLDER, description="Directory with processed test.csv for final evaluation"),
            "model_name_for_registration": Input(type="string", default="emotion_clf_pipeline_model", description="Base name for the registered model"),
            "registration_f1_threshold_emotion": Input(type="number", default=0.1, description="Min F1 score for emotion task to register model"),
            "output_tasks_str": Input(type="string", default="emotion,sub_emotion,intensity"), # Needed for loading model/data
            "model_name_bert": Input(type="string", default="microsoft/deberta-v3-xsmall"), # Needed for loading model
            "max_length": Input(type="integer", default=128), # Needed for data prep for eval
        },
        outputs={
            "registration_status_file": Output(type=AssetTypes.URI_FILE, mode=InputOutputModes.RW_MOUNT, description="File indicating 'registered' or 'not_registered'"),
            "final_evaluation_report_dir": Output(type=AssetTypes.URI_FOLDER, mode=InputOutputModes.RW_MOUNT, description="Directory for final evaluation reports (e.g., evaluation.csv)")
        },
        code=PIPELINE_PROJECT_ROOT,
        command="""python src/emotion_clf_pipeline/train.py \
                    --action evaluate_register \
                    --model_input_dir ${{inputs.trained_model_dir}} \
                    --metrics_input_file ${{inputs.training_metrics_file}} \
                    --encoders_input_dir ${{inputs.encoders_dir}} \
                    --processed_test_dir ${{inputs.processed_test_data_dir}} \
                    --model_name_for_registration ${{inputs.model_name_for_registration}} \
                    --registration_f1_threshold_emotion ${{inputs.registration_f1_threshold_emotion}} \
                    --output_tasks ${{inputs.output_tasks_str}} \
                    --model_name_bert ${{inputs.model_name_bert}} \
                    --max_length ${{inputs.max_length}} \
                    --registration_status_output_file ${{outputs.registration_status_file}} \
                    --final_eval_output_dir ${{outputs.final_evaluation_report_dir}}
                """,
        environment=environment_name_version,
        compute=COMPUTE_NAME,
        identity={"type": "UserIdentity"}, # Required for model registration from component
    )
    return eval_reg_command_instance


# Define the pipeline
@pipeline(
    default_compute=COMPUTE_NAME,
    description="Emotion Classification Training Pipeline: Preprocess, Train, Evaluate, Register",
)
def emotion_classification_pipeline(
    # Pipeline inputs: raw data assets
    raw_train_data_asset_name: str,
    raw_train_data_asset_version: str,
    raw_test_data_asset_name: str,
    raw_test_data_asset_version: str,
    # Pipeline parameters
    model_base_name_for_registration: str = "emotion_clf_from_pipeline",
    registration_f1_threshold: float = 0.10, # Example threshold
    epochs_param: int = 1 # Allow overriding epochs at pipeline level
):
    """Azure ML pipeline for emotion classification."""
    # The actual component functions (preprocess_data_component_func, etc.)
    # are retrieved from global scope, where they're initialized in __main__.
    
    # Step 1: Preprocess Data
    # Note: component functions are already instantiated with ml_client and env
    preprocess_step = preprocess_data_component_func( 
        raw_train_data=Input(type=AssetTypes.URI_FILE, path=f"azureml:{raw_train_data_asset_name}:{raw_train_data_asset_version}"),
        raw_test_data=Input(type=AssetTypes.URI_FILE, path=f"azureml:{raw_test_data_asset_name}:{raw_test_data_asset_version}"),
        # Other preprocess params can be set here if needed, otherwise defaults are used
    )

    # Step 2: Train Model
    train_step = train_model_component_func(
        processed_train_data_dir=preprocess_step.outputs.processed_train_data_dir,
        processed_test_data_dir=preprocess_step.outputs.processed_test_data_dir, 
        encoders_dir=preprocess_step.outputs.encoders_dir,
        epochs=epochs_param # Pass pipeline parameter to component
        # Other train params can be set here
    )

    # Step 3: Evaluate and Register Model
    # Append a unique ID to the model name for registration to avoid conflicts
    # The component script itself should handle the full name construction if needed,
    # or this can be passed if the script expects the full unique name.
    # For now, passing the base name.
    evaluate_register_step = evaluate_register_model_component_func(
        trained_model_dir=train_step.outputs.trained_model_dir,
        training_metrics_file=train_step.outputs.training_metrics_file,
        encoders_dir=preprocess_step.outputs.encoders_dir, 
        processed_test_data_dir=preprocess_step.outputs.processed_test_data_dir, 
        model_name_for_registration=model_base_name_for_registration, # Base name
        registration_f1_threshold_emotion=registration_f1_threshold
    )

    # Define pipeline outputs (optional, but good practice)
    return {
        "pipeline_job_preprocess_output_train": preprocess_step.outputs.processed_train_data_dir,
        "pipeline_job_preprocess_output_test": preprocess_step.outputs.processed_test_data_dir,
        "pipeline_job_train_output_model": train_step.outputs.trained_model_dir,
        "pipeline_job_eval_reg_output_status": evaluate_register_step.outputs.registration_status_file,
        "pipeline_job_eval_reg_output_report": evaluate_register_step.outputs.final_evaluation_report_dir
    }


#####################
#    Run Program    #
#####################

if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")
    logger = logging.getLogger(__name__) # Logger for this main script part

    logger.info("Starting Azure ML Pipeline script execution.")

    azure_connector = AzureMLConnector()
    if not azure_connector.ml_client:
        logger.error("Failed to initialize MLClient. Exiting pipeline script.")
        exit(1)
    
    # Test connection (optional, as MLClient init implies some level of success)
    # if not azure_connector.test_connection():
    #     logger.warning("Azure ML connection test failed. Pipeline submission might encounter issues.")
    # else:
    #     logger.info("Azure ML connection established and tested successfully.")

    ml_client = azure_connector.ml_client
    
    # Define Environment Name and Version
    # It's good practice to version your environments.
    # If environment.yml changes, increment the version.
    ENV_NAME = "emotion-clf-pipeline-env" 
    ENV_VERSION = "3" # Increment if environment.yml or base image changes
    AML_ENVIRONMENT_NAME_VERSION = f"{ENV_NAME}:{ENV_VERSION}"

    try:
        env_handler = EnvironmentHandler(ml_client)
        conda_file_full_path = os.path.join(PIPELINE_PROJECT_ROOT, "environment.yml")
        if not os.path.exists(conda_file_full_path):
            logger.error(f"Conda environment file 'environment.yml' not found at: {conda_file_full_path}")
            raise FileNotFoundError(f"Conda environment file not found: {conda_file_full_path}")

        logger.info(f"Checking for or creating Azure ML Environment: {AML_ENVIRONMENT_NAME_VERSION}")
        try:
            # Try to get the environment first
            azure_ml_env = ml_client.environments.get(name=ENV_NAME, version=ENV_VERSION)
            logger.info(f"Environment {AML_ENVIRONMENT_NAME_VERSION} already exists and will be used.")
        except Exception:
            logger.info(f"Environment {AML_ENVIRONMENT_NAME_VERSION} not found. Attempting to create it...")
            azure_ml_env = env_handler.push_environment_from_yaml(
                conda_file_path=conda_file_full_path,
                environment_name=ENV_NAME,
                description="Environment for Emotion Classification Pipeline (based on train.py actions)",
                version=ENV_VERSION 
                # base_image can be specified here if different from default in push_environment_from_yaml
            )
            if azure_ml_env:
                logger.info(f"Environment '{azure_ml_env.name}:{azure_ml_env.version}' created/updated successfully.")
            else:
                logger.error(f"Failed to create Azure ML Environment {AML_ENVIRONMENT_NAME_VERSION}. Exiting.")
                exit(1)
    except Exception as e:
        logger.error(f"An error occurred during Azure ML Environment setup: {e}")
        exit(1)

    # --- Define Raw Data Assets ---
    # These should be the names and versions of your REGISTERED RAW data assets in Azure ML.
    # You might need to create these first using DataHandler or Azure ML Studio if they don't exist.
    # Example:
    # data_handler = DataHandler(ml_client)
    # train_raw_local_path = os.path.join(PIPELINE_PROJECT_ROOT, "data", "raw", "train", "your_combined_raw_train_file.csv") # You need to prepare this file
    # test_raw_local_path = os.path.join(PIPELINE_PROJECT_ROOT, "data", "raw", "test", "group 21_url1.csv")
    # if not (os.path.exists(train_raw_local_path) and os.path.exists(test_raw_local_path)):
    #    logger.error(f"Raw data files not found at expected local paths for asset creation. Please check paths.")
    #    exit(1)
    # train_asset = data_handler.create_data_asset_from_local_file(train_raw_local_path, "emotion-train-raw-data", "Raw train data for emotion pipeline", "1")
    # test_asset = data_handler.create_data_asset_from_local_file(test_raw_local_path, "emotion-test-raw-data", "Raw test data for emotion pipeline", "1")
    # if not (train_asset and test_asset):
    #    logger.error("Failed to create necessary raw data assets. Exiting.")
    #    exit(1)
    # RAW_TRAIN_DATA_ASSET_NAME = train_asset.name
    # RAW_TRAIN_DATA_ASSET_VERSION = train_asset.version
    # RAW_TEST_DATA_ASSET_NAME = test_asset.name
    # RAW_TEST_DATA_ASSET_VERSION = test_asset.version
    
    # Replace with your actual RAW data asset names and versions. Using placeholders for now.
    # IMPORTANT: Ensure these assets exist in your Azure ML workspace.
    RAW_TRAIN_DATA_ASSET_NAME = "emotion-train-raw-data"  # Example name
    RAW_TRAIN_DATA_ASSET_VERSION = "latest"  # Or a specific version like "1"
    RAW_TEST_DATA_ASSET_NAME = "emotion-test-raw-data"    # Example name
    RAW_TEST_DATA_ASSET_VERSION = "latest" # Or a specific version

    try:
        # Verify data assets exist
        ml_client.data.get(name=RAW_TRAIN_DATA_ASSET_NAME, version=RAW_TRAIN_DATA_ASSET_VERSION)
        ml_client.data.get(name=RAW_TEST_DATA_ASSET_NAME, version=RAW_TEST_DATA_ASSET_VERSION)
        logger.info(f"Successfully verified raw data assets: "
                    f"{RAW_TRAIN_DATA_ASSET_NAME}:{RAW_TRAIN_DATA_ASSET_VERSION} and "
                    f"{RAW_TEST_DATA_ASSET_NAME}:{RAW_TEST_DATA_ASSET_VERSION}.")
    except Exception as e:
        logger.error(f"Could not find required raw data assets. "
                     f"Please ensure '{RAW_TRAIN_DATA_ASSET_NAME}:{RAW_TRAIN_DATA_ASSET_VERSION}' and "
                     f"'{RAW_TEST_DATA_ASSET_NAME}:{RAW_TEST_DATA_ASSET_VERSION}' exist in your workspace. Error: {e}")
        logger.info("Tip: You can use the DataHandler class (commented out example above) to create these assets from local files if needed.")
        exit(1)

    # Initialize component function variables globally so pipeline definition can use them
    # These factory functions now correctly use the ml_client and AML_ENVIRONMENT_NAME_VERSION
    preprocess_data_component_func = get_preprocess_data_component(ml_client, AML_ENVIRONMENT_NAME_VERSION)
    train_model_component_func = get_train_model_component(ml_client, AML_ENVIRONMENT_NAME_VERSION)
    evaluate_register_model_component_func = get_evaluate_register_model_component(ml_client, AML_ENVIRONMENT_NAME_VERSION)

    logger.info("Constructing the Azure ML pipeline.")
    # Unique model name for registration for this run
    unique_model_suffix = str(uuid.uuid4())[:8]
    pipeline_job = emotion_classification_pipeline(
        raw_train_data_asset_name=RAW_TRAIN_DATA_ASSET_NAME,
        raw_train_data_asset_version=RAW_TRAIN_DATA_ASSET_VERSION,
        raw_test_data_asset_name=RAW_TEST_DATA_ASSET_NAME,
        raw_test_data_asset_version=RAW_TEST_DATA_ASSET_VERSION,
        model_base_name_for_registration=f"emotion_clf_pipeline_{unique_model_suffix}",
        registration_f1_threshold=0.05, # Example: Register if emotion F1 > 0.05
        epochs_param=1 # Example: run for 1 epoch, can be parameterized further
    )

    try:
        logger.info("Submitting the pipeline job to Azure ML...")
        # You can use a specific experiment name
        experiment_name = "emotion_classification_pipeline_runs"
        returned_job = ml_client.jobs.create_or_update(
            pipeline_job, experiment_name=experiment_name
        )
        logger.info(f"Pipeline job submitted successfully. Job Name: {returned_job.name}")
        logger.info(f"View in Azure ML Studio: {returned_job.studio_url}")

        # Optional: Stream job logs or wait for completion
        # logger.info("Streaming job logs... (Ctrl+C to stop streaming)")
        # ml_client.jobs.stream(returned_job.name)
        # logger.info(f"Job {returned_job.name} finished with status: {ml_client.jobs.get(returned_job.name).status}")

    except Exception as e:
        logger.error(f"Error submitting pipeline job: {e}")
        # More detailed error logging for validation issues
        if hasattr(e, 'message') and "ValidationException" in str(e.message if isinstance(e.message, str) else e):
             logger.error(f"Validation Error Details: {e}")
        elif hasattr(e, 'error') and hasattr(e.error, 'message'): # For AzureMLHTTPError
             logger.error(f"AzureMLHTTPError Details: {e.error.message}")
        # You might want to print the full stack trace for debugging
        # import traceback
        # logger.error(traceback.format_exc())


    logger.info("Azure ML Pipeline script execution finished.")

