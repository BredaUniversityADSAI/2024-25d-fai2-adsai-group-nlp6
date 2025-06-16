"""
Azure ML Model Synchronization Manager
Handles bidirectional sync between local weights and Azure ML Model Registry
"""

import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model as AzureModel
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class AzureMLSync:
    """
    Manages synchronization between local model weights and Azure ML Model Registry.

    Features:
    - Download models from Azure ML if local weights don't exist
    - Upload new models to Azure ML with proper tags
    - Sync baseline/dynamic model designations
    - Handle offline scenarios gracefully
    """

    def __init__(self, weights_dir: str = "models/weights"):
        """
        Initialize the Azure ML Model Manager.

        Args:
            weights_dir: Local directory for model weights
        """
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        # Load environment variables
        load_dotenv()

        # Azure ML configuration
        self.subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        self.resource_group = os.getenv("AZURE_RESOURCE_GROUP")
        self.workspace_name = os.getenv("AZURE_WORKSPACE_NAME")

        # Model names in Azure ML
        self.baseline_model_name = "emotion-clf-baseline"
        self.dynamic_model_name = "emotion-clf-dynamic"

        # Initialize Azure ML client
        self._ml_client = None
        self._azure_available = self._check_azure_availability()
        self._auth_method = "unknown"

        # API endpoint for model refresh
        self.refresh_endpoint = os.getenv(
            "API_REFRESH_ENDPOINT", "http://localhost:8000/refresh-model"
        )

    def _check_azure_availability(self) -> bool:
        """Check if Azure ML is available and configured with retry logic."""
        try:
            # Check required environment variables
            missing_vars = []
            if not self.subscription_id:
                missing_vars.append("AZURE_SUBSCRIPTION_ID")
            if not self.resource_group:
                missing_vars.append("AZURE_RESOURCE_GROUP")
            if not self.workspace_name:
                missing_vars.append("AZURE_WORKSPACE_NAME")

            if missing_vars:
                logger.warning(
                    f"Azure ML credentials incomplete. Missing: \
                        {', '.join(missing_vars)}"
                )
                logger.info("Operating in local-only mode. Set environment \
                    variables for Azure ML sync.")
                return False

            # Try multiple authentication methods in order of preference
            credential = None
            auth_method = "unknown"

            # Method 1: Service Principal (if client_id and client_secret are set)
            client_id = os.getenv("AZURE_CLIENT_ID")
            client_secret = os.getenv("AZURE_CLIENT_SECRET")
            tenant_id = os.getenv("AZURE_TENANT_ID")

            if client_id and client_secret and tenant_id:
                from azure.identity import ClientSecretCredential
                credential = ClientSecretCredential(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    client_secret=client_secret
                )
                auth_method = "service_principal"
            else:
                # Method 2: Try Azure CLI credentials first (most
                # reliable when az login was used)
                try:
                    from azure.identity import AzureCliCredential
                    credential = AzureCliCredential()
                    auth_method = "azure_cli"
                except Exception as cli_error:
                    logger.debug(f"Azure CLI credential failed: {cli_error}")

                    # Method 3: Fall back to default credential chain
                    credential = DefaultAzureCredential()
                    auth_method = "default_credential"
            self._ml_client = MLClient(
                credential=credential,
                subscription_id=self.subscription_id,
                resource_group_name=self.resource_group,
                workspace_name=self.workspace_name
            )

            # Store auth method and test connection with retry logic
            self._auth_method = auth_method
            return self._test_azure_connection_with_retry(auth_method)

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Azure ML not available: {error_msg}")

            # Provide specific guidance based on the error
            if "client_id should be the id of a Microsoft Entra application" \
                    in error_msg:
                logger.info("💡 Authentication failed. \
                    Try one of these options:")
                logger.info("   1. Run 'az login' in your terminal for \
                    interactive authentication")
                logger.info("   2. Set AZURE_CLIENT_ID and AZURE_CLIENT_SECRET \
                    for service principal auth")
                logger.info("   3. Use managed identity if running on Azure \
                    infrastructure")
            elif "AADSTS" in error_msg:
                logger.info("💡 Azure Active Directory authentication issue. \
                    Try 'az login' or check your credentials.")
            else:
                logger.info("💡 Check your Azure credentials and network connection.")
            return False

    def _test_azure_connection_with_retry(
        self, auth_method: str, max_retries: int = 2
    ) -> bool:
        """Test Azure ML connection with retry logic for network issues."""
        for attempt in range(max_retries + 1):
            try:
                # Test connection
                self._ml_client.workspaces.get(self.workspace_name)
                logger.info(f"Azure ML connection established successfully \
                    using {auth_method}")
                return True

            except (ConnectionResetError, HttpResponseError) as e:
                if "Connection aborted" in str(e) or "Connection broken" in \
                        str(e) or "reset" in str(e).lower():
                    if attempt < max_retries:
                        wait_time = (attempt + 1) * 2  # 2, 4 seconds
                        logger.warning(f"Connection test failed (attempt \
                            {attempt + 1}/{max_retries + 1}): {e}")
                        logger.info(f"Retrying connection test in {wait_time} \
                            seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.warning(f"Azure ML connection failed after \
                            {max_retries + 1} attempts due to network issues")
                        logger.info("💡 Network connectivity issues detected. \
                            Operations may fail.")
                        logger.info("   • Check your internet connection")
                        logger.info("   • Try again in a few moments")
                        return False
                else:
                    logger.warning(f"Azure ML connection failed: {e}")
                    return False

            except Exception as e:
                logger.warning(f"Azure ML connection failed: {e}")
                return False

        return False

    def _ensure_azure_connection(self) -> bool:
        """Ensure Azure ML connection is available, re-establishing if needed."""
        if not self._azure_available:
            return False

        # Test if connection is still valid
        try:
            self._ml_client.workspaces.get(self.workspace_name)
            return True
        except Exception:
            logger.info("Azure ML connection lost, attempting to reconnect...")
            self._azure_available = self._check_azure_availability()
            return self._azure_available

    def sync_on_startup(self) -> Tuple[bool, bool]:
        """
        Sync models on startup - download from Azure ML if local files don't exist.

        Returns:
            Tuple of (baseline_synced, dynamic_synced)
        """
        baseline_synced = False
        dynamic_synced = False

        if not self._azure_available:
            logger.info("Azure ML not available, using local weights only")
            return baseline_synced, dynamic_synced

        # Check and sync baseline model
        baseline_path = self.weights_dir / "baseline_weights.pt"
        if not baseline_path.exists():
            logger.info("Baseline weights not found locally, \
                downloading from Azure ML...")
            baseline_synced = self._download_model_from_azure(
                self.baseline_model_name, baseline_path
            )

        # Check and sync dynamic model
        dynamic_path = self.weights_dir / "dynamic_weights.pt"
        if not dynamic_path.exists():
            logger.info("Dynamic weights not found locally, downloading \
                from Azure ML...")
            dynamic_synced = self._download_model_from_azure(
                self.dynamic_model_name, dynamic_path
            )

        return baseline_synced, dynamic_synced

    def _download_model_from_azure(self, model_name: str, local_path: Path) -> bool:
        """Download a model from Azure ML to local path."""
        try:
            # Get latest version of the model
            model = self._ml_client.models.get(model_name, label="latest")

            # Download model to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                downloaded_path = self._ml_client.models.download(
                    name=model_name,
                    version=model.version,
                    download_path=temp_dir
                )

                # Handle case where download path might be None or the temp_dir itself
                search_path = Path(downloaded_path) if downloaded_path \
                    else Path(temp_dir)

                # Find the .pt file in downloaded content
                pt_files = list(search_path.rglob("*.pt"))
                if pt_files:
                    shutil.copy2(pt_files[0], local_path)
                    logger.info(f"Downloaded {model_name} v{model.version} \
                        to {local_path}")
                    return True
                else:
                    logger.error(f"No .pt file found in downloaded model {model_name}")
                    logger.info(f"Available files: {list(search_path.rglob('*'))}")
                    return False

        except Exception as e:
            logger.error(f"Failed to download {model_name} from Azure ML: {e}")
            return False

    def upload_dynamic_model(
        self, f1_score: float, metadata: Optional[Dict] = None
    ) -> bool:
        """
        Upload dynamic model to Azure ML with retry logic for reliability.

        Args:
            f1_score: F1 score of the model
            metadata: Additional metadata to store with the model

        Returns:
            True if upload successful
        """
        if not self._azure_available:
            logger.warning("Azure ML not available, skipping model upload")
            return False

        dynamic_path = self.weights_dir / "dynamic_weights.pt"
        if not dynamic_path.exists():
            logger.error("Dynamic weights file not found, cannot upload")
            return False

        # Check file size and log
        file_size_mb = dynamic_path.stat().st_size / (1024 * 1024)
        logger.info(f"Uploading dynamic model ({file_size_mb:.1f} MB) to Azure ML...")

        return self._upload_model_with_retry(
            model_path=dynamic_path,
            model_name=self.dynamic_model_name,
            model_type="dynamic",
            f1_score=f1_score,
            metadata=metadata,
            max_retries=3
        )

    def _upload_model_with_retry(
        self, model_path: Path, model_name: str, model_type: str,
        f1_score: float, metadata: Optional[Dict] = None, max_retries: int = 3
    ) -> bool:
        """
        Upload model with retry logic and exponential backoff.

        Args:
            model_path: Path to the model file
            model_name: Name for the model in Azure ML
            model_type: Type of model (baseline/dynamic)
            f1_score: F1 score of the model
            metadata: Additional metadata
            max_retries: Maximum number of retry attempts

        Returns:
            True if upload successful
        """
        for attempt in range(max_retries + 1):
            try:
                # Prepare model metadata
                model_metadata = {
                    "model_type": model_type,
                    "f1_score": str(f1_score),
                    "upload_time": datetime.now().isoformat(),
                    "framework": "pytorch",
                    "upload_attempt": str(attempt + 1),
                    "file_size_mb": str(
                        round(model_path.stat().st_size / (1024 * 1024), 2)
                    )
                }

                if metadata:
                    model_metadata.update(metadata)

                # Create temporary directory with model file
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_model_dir = Path(temp_dir) / "model"
                    temp_model_dir.mkdir()

                    # Copy model file
                    temp_model_file = temp_model_dir / f"{model_type}_weights.pt"
                    shutil.copy2(model_path, temp_model_file)

                    # Create model metadata file
                    with open(temp_model_dir / "metadata.json", "w") as f:
                        json.dump(model_metadata, f, indent=2)

                    # Create Azure ML model
                    azure_model = AzureModel(
                        path=str(temp_model_dir),
                        name=model_name,
                        description=f"{model_type.title()} emotion \
                            classification model (F1: {f1_score:.4f})",
                        type=AssetTypes.CUSTOM_MODEL,
                        tags=model_metadata
                    )

                    # Upload to Azure ML with timeout handling
                    logger.info(f"Attempting upload (attempt {attempt + 1}\
                        /{max_retries + 1})...")
                    registered_model = self._ml_client.models.create_or_update(
                        azure_model
                    )

                    logger.info(f"✅ Successfully uploaded {model_type} \
                        model v{registered_model.version} to Azure ML")
                    return True

            except (ConnectionResetError, HttpResponseError) as e:
                if "Connection aborted" in str(e) or "Connection broken" in \
                        str(e) or "reset" in str(e).lower():
                    if attempt < max_retries:
                        # Exponential backoff: 2, 5, 9 seconds
                        wait_time = (2 ** attempt) + 1
                        logger.warning(f"Upload failed due to connection \
                            issue (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Upload failed after {max_retries + 1} \
                            attempts due to connection issues: {e}")
                        logger.info("💡 Large file upload tips:")
                        logger.info("   • Check your internet connection stability")
                        logger.info("   • Try again during off-peak hours")
                        logger.info("   • Consider splitting large models if possible")
                        return False
                else:
                    logger.error(f"Upload failed with HTTP error: {e}")
                    return False

            except Exception as e:
                logger.error(f"Upload failed with unexpected error: {e}")
                if attempt < max_retries:
                    wait_time = (2 ** attempt) + 1
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Upload failed after {max_retries + 1} attempts")
                    return False

        return False

    def promote_dynamic_to_baseline(self) -> bool:
        """
        Promote the latest dynamic model to baseline.

        This involves:
        1. Finding the latest 'dynamic' model.
        2. Archiving the old 'baseline' model.
        3. Creating a new model version under the 'baseline' name that points to the dynamic model's assets.
        4. Triggering a refresh on the live API server.
        """
        if not self._ensure_azure_connection():
            logger.warning("Azure connection not available. Cannot promote model.")
            # Fallback to local-only promotion if Azure is offline
            return self._promote_local_only()

        try:
            # 1. Get the latest 'dynamic' model
            dynamic_models = list(self._ml_client.models.list(name=self.dynamic_model_name))
            if not dynamic_models:
                logger.warning(f"No models found with name '{self.dynamic_model_name}'. Nothing to promote.")
                return False

            # Sort by version (integer conversion) to find the latest
            latest_dynamic_model = max(dynamic_models, key=lambda m: int(m.version))
            logger.info(f"Found latest dynamic model to promote: {latest_dynamic_model.name} (Version: {latest_dynamic_model.version})")

            # 2. Find and archive the current 'baseline' model
            try:
                current_baseline_model = self._ml_client.models.get(self.baseline_model_name, label="latest")
                if current_baseline_model:
                    logger.info(f"Archiving current baseline model: {current_baseline_model.name} (Version: {current_baseline_model.version})")
                    self._ml_client.models.archive(name=self.baseline_model_name, version=current_baseline_model.version)
            except ResourceNotFoundError:
                logger.info("No existing baseline model found. Skipping archiving.")

            # 3. Promote the new model by creating it with the 'baseline' name
            promotion_tags = latest_dynamic_model.tags.copy() if latest_dynamic_model.tags else {}
            promotion_tags["promoted_from"] = f"{latest_dynamic_model.name}:{latest_dynamic_model.version}"

            promoted_model = AzureModel(
                name=self.baseline_model_name,
                path=latest_dynamic_model.path,
                version=latest_dynamic_model.version,
                tags=promotion_tags,
                description=f"Promoted from {latest_dynamic_model.name} v{latest_dynamic_model.version}",
                type=AssetTypes.CUSTOM_MODEL
            )

            logger.info(f"Promoting model to '{self.baseline_model_name}' (Version: {promoted_model.version})...")
            self._ml_client.models.create_or_update(promoted_model)
            logger.info("✅ --- Promotion successful in Azure ML Registry --- ✅")

            # 4. Trigger the /refresh-model endpoint on the API server
            self._trigger_api_refresh()

            # Local sync after promotion
            self._sync_promoted_model_locally(promoted_model.version)

            return True

        except Exception as e:
            logger.error(f"An unexpected error occurred during promotion: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _trigger_api_refresh(self):
        """Make a POST request to the API's /refresh-model endpoint."""
        if not self.refresh_endpoint:
            logger.warning("API_REFRESH_ENDPOINT is not set. \
                Skipping API refresh.")
            return

        logger.info(f"Triggering model refresh on API server at: \
            {self.refresh_endpoint}")
        try:
            response = requests.post(self.refresh_endpoint, timeout=30)
            response.raise_for_status()  # Raise an exception for bad status codes
            logger.info(f"✅ --- API server responded with success: \
                {response.json()} --- ✅")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ --- Failed to trigger API refresh: {e} --- ❌")
            logger.error("The model has been promoted in the registry, but the \
                live API may be serving a stale model.")
            logger.error("Please restart the API server manually or ensure it \
                can be reached at the configured endpoint.")

    def _sync_promoted_model_locally(self, promoted_version: str):
        """
        Sync the newly promoted baseline model to local file.

        Args:
            promoted_version: Version of the promoted baseline model
        """
        if not self._azure_available:
            return

        baseline_path = self.weights_dir / "baseline_weights.pt"

        logger.info(f"Syncing newly promoted baseline model \
            (Version: {promoted_version}) locally...")

        if self._download_specific_model_version(
            self.baseline_model_name, promoted_version, baseline_path
        ):
            logger.info("Local baseline model updated successfully.")
            # Optionally remove the old dynamic weights if they are now baseline
            dynamic_path = self.weights_dir / "dynamic_weights.pt"
            if dynamic_path.exists():
                logger.info("Removing old local dynamic weights.")
                dynamic_path.unlink()
        else:
            logger.warning("Failed to sync the new baseline model to local file.")

    def _promote_local_only(self) -> bool:
        """
        Fallback for promoting dynamic to baseline when Azure is not available.
        This simply renames the local weight files.
        """
        try:
            dynamic_path = self.weights_dir / "dynamic_weights.pt"
            baseline_path = self.weights_dir / "baseline_weights.pt"

            if dynamic_path.exists():
                shutil.copy2(dynamic_path, baseline_path)
                logger.info("Local promotion: dynamic → baseline completed")
                return True
            else:
                logger.error("Dynamic weights not found for local promotion")
                return False
        except Exception as e:
            logger.error(f"Local promotion failed: {e}")
            return False

    def get_model_info(self) -> Dict:
        """Get information about both local and Azure ML models."""
        info = {
            "local": {},
            "azure_ml": {},
            "azure_available": self._azure_available
        }

        # Local model info
        baseline_path = self.weights_dir / "baseline_weights.pt"
        dynamic_path = self.weights_dir / "dynamic_weights.pt"

        info["local"]["baseline_exists"] = baseline_path.exists()
        info["local"]["dynamic_exists"] = dynamic_path.exists()

        if baseline_path.exists():
            info["local"]["baseline_size"] = baseline_path.stat().st_size
            info["local"]["baseline_modified"] = datetime.fromtimestamp(
                baseline_path.stat().st_mtime
            ).isoformat()

        if dynamic_path.exists():
            info["local"]["dynamic_size"] = dynamic_path.stat().st_size
            info["local"]["dynamic_modified"] = datetime.fromtimestamp(
                dynamic_path.stat().st_mtime
            ).isoformat()

        # Azure ML model info
        if self._azure_available:
            try:
                for model_name in [self.baseline_model_name, self.dynamic_model_name]:
                    try:
                        model = self._ml_client.models.get(model_name, label="latest")
                        created_time = None
                        if model.creation_context:
                            created_time = model.creation_context.created_at.isoformat()
                        info["azure_ml"][model_name] = {
                            "version": model.version,
                            "created_time": created_time,
                            "tags": model.tags
                        }
                    except Exception:
                        info["azure_ml"][model_name] = {"status": "not_found"}
            except Exception as e:
                info["azure_ml"]["error"] = str(e)

        return info

    def get_configuration_status(self) -> Dict:
        """
        Get detailed Azure ML configuration status for troubleshooting.

        Returns:
            Dictionary with configuration details and status
        """
        # Check authentication methods available
        client_id = os.getenv("AZURE_CLIENT_ID")
        client_secret = os.getenv("AZURE_CLIENT_SECRET")
        tenant_id = os.getenv("AZURE_TENANT_ID")

        auth_methods = []
        if client_id and client_secret and tenant_id:
            auth_methods.append("Service Principal")
        if shutil.which("az"):  # Check if Azure CLI is installed
            auth_methods.append("Azure CLI")
        auth_methods.append("Default Credential Chain")

        # Checkings
        azure_subscription_id = "✓ Set" if self.subscription_id else "✗ Missing"
        azure_resource_group = "✓ Set" if self.resource_group else "✗ Missing"
        azure_workspace_name = "✓ Set" if self.workspace_name else "✗ Missing"
        azure_client_id = "✓ Set" if client_id else "✗ Not set (optional)"
        azure_client_secret = "✓ Set" if client_secret else "✗ Not set (optional)"
        azure_tenant_id = "✓ Set" if tenant_id else "✗ Not set (optional)"
        connection_status = "Connected" if self._azure_available else "Not connected"
        subscription_id_status = self.subscription_id[:8] \
            + "..." if self.subscription_id else "Not configured"

        return {
            "environment_variables": {
                "AZURE_SUBSCRIPTION_ID": azure_subscription_id,
                "AZURE_RESOURCE_GROUP": azure_resource_group,
                "AZURE_WORKSPACE_NAME": azure_workspace_name,
                "AZURE_CLIENT_ID": azure_client_id,
                "AZURE_CLIENT_SECRET": azure_client_secret,
                "AZURE_TENANT_ID": azure_tenant_id
            },
            "authentication": {
                "available_methods": auth_methods,
                "service_principal_configured": bool(
                    client_id and client_secret and tenant_id),
                "azure_cli_available": bool(shutil.which("az"))
            },
            "azure_available": self._azure_available,
            "connection_status": connection_status,
            "workspace_name": self.workspace_name or "Not configured",
            "resource_group": self.resource_group or "Not configured",
            "subscription_id": subscription_id_status
        }

    def auto_sync_on_startup(self, check_for_updates=True) -> Dict[str, bool]:
        """
        Comprehensive auto-sync on startup - downloads missing models and checks
        for updates.

        Args:
            check_for_updates: Whether to check for newer models in Azure ML

        Returns:
            Dict with sync results
        """
        results = {
            "baseline_downloaded": False,
            "dynamic_downloaded": False,
            "baseline_updated": False,
            "dynamic_updated": False
        }

        if not self._azure_available:
            logger.info("Azure ML not available, using local weights only")
            return results

        baseline_path = self.weights_dir / "baseline_weights.pt"
        dynamic_path = self.weights_dir / "dynamic_weights.pt"

        # Download missing models
        if not baseline_path.exists():
            logger.info("Baseline model missing, downloading from Azure ML...")
            results["baseline_downloaded"] = self._download_model_from_azure(
                self.baseline_model_name, baseline_path
            )

        if not dynamic_path.exists():
            logger.info("Dynamic model missing, downloading from Azure ML...")
            results["dynamic_downloaded"] = self._download_model_from_azure(
                self.dynamic_model_name, dynamic_path
            )

        # Check for updates if requested
        if check_for_updates:
            results["baseline_updated"] = self._check_and_update_model(
                self.baseline_model_name, baseline_path
            )
            results["dynamic_updated"] = self._check_and_update_model(
                self.dynamic_model_name, dynamic_path
            )

        return results

    def _check_and_update_model(self, model_name: str, local_path: Path) -> bool:
        """Check if Azure ML has a newer version and update if so."""
        try:
            if not local_path.exists():
                return False

            # Get latest Azure ML model info
            azure_model = self._ml_client.models.get(model_name, label="latest")

            # Get local file modification time
            local_mtime = datetime.fromtimestamp(local_path.stat().st_mtime)
            azure_created = None
            if azure_model.creation_context:
                azure_created = azure_model.creation_context.created_at

            if azure_created and azure_created > local_mtime:
                logger.info(f"Newer {model_name} found in Azure ML, updating...")
                return self._download_model_from_azure(model_name, local_path)

        except Exception as e:
            logger.debug(f"Could not check for updates for {model_name}: {e}")

        return False

    def auto_upload_after_training(
        self, f1_score: float, auto_promote_threshold: float = 0.85,
        metadata: Optional[Dict] = None
    ) -> Dict[str, bool]:
        """
        Automatically upload dynamic model after training and optionally promote
        to baseline.

        Args:
            f1_score: F1 score achieved by the model
            auto_promote_threshold: F1 threshold for automatic promotion
            metadata: Additional metadata

        Returns:
            Dict with upload and promotion results
        """
        results = {
            "uploaded": False,
            "promoted": False
        }

        # Always upload the new model
        results["uploaded"] = self.upload_dynamic_model(f1_score, metadata)

        if results["uploaded"] and f1_score >= auto_promote_threshold:
            logger.info(
                f"F1 score {f1_score:.4f} >= threshold \
                    {auto_promote_threshold:.4f}, auto-promoting to baseline..."
            )
            results["promoted"] = self.promote_dynamic_to_baseline()

        return results

    def get_auto_sync_config(self) -> Dict[str, any]:
        """Get configuration for automatic sync behavior."""
        return {
            "auto_download_on_startup": True,
            "auto_check_updates_on_startup": True,
            "auto_upload_after_training": True,
            "auto_promote_threshold": 0.85,
            "sync_on_model_load": True,
            "background_sync_enabled": False  # Future feature
        }

    def get_local_baseline_f1_score(self) -> Optional[float]:
        """
        Extract F1 score from local baseline model metadata.

        Returns:
            F1 score if found, None otherwise
        """
        try:
            # Check if sync status file exists with local model info
            sync_status_path = self.weights_dir / "sync_status.json"
            if sync_status_path.exists():
                with open(sync_status_path, 'r') as f:
                    sync_data = json.load(f)

                # Look for baseline model F1 score in sync status
                baseline_info = (
                    sync_data.get("models", {})
                    .get("azure_ml", {})
                    .get("emotion-clf-baseline", {})
                )
                f1_str = baseline_info.get("tags", {}).get("f1_score")

                if f1_str:
                    return float(f1_str)

            # Fallback: check if there's a model_config.json with F1 info
            config_path = self.weights_dir / "model_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    f1_str = config_data.get("f1_score")
                    if f1_str:
                        return float(f1_str)

            logger.debug("No local F1 score metadata found")
            return None

        except Exception as e:
            logger.warning(f"Error reading local baseline F1 score: {e}")
            return None

    def download_best_baseline_model(
        self, force_update: bool = False, min_f1_improvement: float = 0.01
    ) -> bool:
        """
        Download the best baseline model from Azure ML based on F1 comparison.

        Args:
            force_update: Download even if local F1 is equal or better
            min_f1_improvement: Minimum F1 improvement required to download

        Returns:
            True if model was downloaded and updated
        """
        if not self._azure_available:
            logger.info("Azure ML not available, keeping local baseline model")
            return False

        # Get the best model from Azure ML
        best_azure_model = self.get_best_baseline_model()
        if not best_azure_model:
            logger.info("No suitable baseline model found in Azure ML")
            return False

        azure_f1 = best_azure_model["f1_score"]
        baseline_path = self.weights_dir / "baseline_weights.pt"

        # Get local model F1 score for comparison
        local_f1 = self.get_local_baseline_f1_score()

        # Decide whether to download
        should_download = force_update

        if not should_download:
            if not baseline_path.exists():
                logger.info("Local baseline model missing, downloading from Azure")
                should_download = True
            elif local_f1 is None:
                logger.info(
                    "Local baseline F1 score unknown, "
                    "downloading latest from Azure ML"
                )
                should_download = True
            elif azure_f1 > (local_f1 + min_f1_improvement):
                logger.info(
                    f"Azure ML has better baseline model: "
                    f"F1 {azure_f1:.4f} vs local {local_f1:.4f}"
                )
                should_download = True
            else:
                logger.info(
                    f"Local baseline model is current: "
                    f"F1 {local_f1:.4f} vs Azure {azure_f1:.4f}"
                )

        if not should_download:
            return False

        # Download the best model
        model_info = best_azure_model["model"]
        logger.info(
            f"Downloading best baseline model {model_info.name}:"
            f"{model_info.version} with F1 score {azure_f1:.4f}"
        )

        success = self._download_specific_model_version(
            model_info.name,
            model_info.version,
            baseline_path
        )

        if success:
            # Update sync status with new model info
            self._update_sync_status_for_baseline(best_azure_model)
            logger.info(
                f"Successfully updated baseline model to version "
                f"{model_info.version} (F1: {azure_f1:.4f})"
            )

        return success

    def _download_specific_model_version(
        self, model_name: str, version: str, local_path: Path
    ) -> bool:
        """Download a specific model version from Azure ML."""
        try:
            # Download model to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                downloaded_path = self._ml_client.models.download(
                    name=model_name,
                    version=version,
                    download_path=temp_dir
                )

                # Handle case where download path might be None or temp_dir
                search_path = (
                    Path(downloaded_path) if downloaded_path
                    else Path(temp_dir)
                )

                # Find the .pt file in downloaded content
                pt_files = list(search_path.rglob("*.pt"))

                if pt_files:
                    # Copy the first .pt file found to the target location
                    shutil.copy2(pt_files[0], local_path)
                    file_size_mb = local_path.stat().st_size / (1024 * 1024)
                    logger.info(
                        f"Downloaded model to {local_path} "
                        f"({file_size_mb:.1f} MB)"
                    )
                    return True
                else:
                    logger.error(
                        f"No .pt file found in downloaded model "
                        f"{model_name}:{version}"
                    )
                    return False

        except Exception as e:
            logger.error(f"Error downloading model {model_name}:{version}: {e}")
            return False

    def _update_sync_status_for_baseline(self, model_info: Dict) -> None:
        """Update sync status file with baseline model information."""
        try:
            sync_status_path = self.weights_dir / "sync_status.json"

            # Load existing sync status or create new
            sync_data = {}
            if sync_status_path.exists():
                with open(sync_status_path, 'r') as f:
                    sync_data = json.load(f)

            # Ensure structure exists
            if "models" not in sync_data:
                sync_data["models"] = {}
            if "azure_ml" not in sync_data["models"]:
                sync_data["models"]["azure_ml"] = {}
            if "emotion-clf-baseline" not in sync_data["models"]["azure_ml"]:
                sync_data["models"]["azure_ml"]["emotion-clf-baseline"] = {}

            # Update baseline model info
            model = model_info["model"]
            sync_data["models"]["azure_ml"]["emotion-clf-baseline"]["tags"] = {
                "f1_score": str(model_info["f1_score"]),
                "model_type": "baseline",
                "download_time": datetime.now().isoformat(),
                "version": model.version,
                "framework": (
                    model.tags.get("framework", "pytorch")
                    if model.tags else "pytorch"
                )
            }

            # Save updated sync status
            with open(sync_status_path, 'w') as f:
                json.dump(sync_data, f, indent=2)

        except Exception as e:
            logger.warning(f"Error updating sync status: {e}")

    def get_best_baseline_model(self) -> Optional[Dict]:
        """
        Find and return the best baseline model from Azure ML based on F1 score.

        Returns:
            Dictionary containing model info and F1 score, or None if no model found
        """
        if not self._azure_available:
            logger.warning("Azure ML not available")
            return None

        try:
            # Get all baseline models from Azure ML
            baseline_models = list(
                self._ml_client.models.list(name=self.baseline_model_name)
            )

            if not baseline_models:
                logger.info("No baseline models found in Azure ML")
                return None

            best_model = None
            best_f1_score = -1.0

            # Find model with highest F1 score
            for model in baseline_models:
                try:
                    # Extract F1 score from model tags
                    f1_score = 0.0
                    if model.tags and "f1_score" in model.tags:
                        f1_score = float(model.tags["f1_score"])
                    elif model.tags and "test_f1" in model.tags:
                        f1_score = float(model.tags["test_f1"])

                    if f1_score > best_f1_score:
                        best_f1_score = f1_score
                        best_model = model

                except (ValueError, TypeError):
                    logger.debug(
                        f"Could not parse F1 score for model \
                            {model.name}:{model.version}"
                    )
                    continue

            if best_model:
                logger.info(
                    f"Found best baseline model: \
                        {best_model.name}:{best_model.version} "
                    f"with F1 score {best_f1_score:.4f}"
                )
                return {
                    "model": best_model,
                    "f1_score": best_f1_score
                }
            else:
                logger.warning("No baseline models with valid F1 scores found")
                return None

        except Exception as e:
            logger.error(f"Error finding best baseline model: {e}")
            return None

    def sync_best_baseline(
        self, force_update: bool = False, min_f1_improvement: float = 0.01
    ) -> bool:
        """
        Synchronize with the best available baseline model from Azure ML.

        This method finds the baseline model with the highest F1 score in Azure ML
        and downloads it if it's better than the local baseline model.

        Args:
            force_update: Download even if local F1 is equal or better
            min_f1_improvement: Minimum F1 improvement required to download

        Returns:
            True if a better model was downloaded and synchronized
        """
        logger.info("Synchronizing with best baseline model from Azure ML...")
        return self.download_best_baseline_model(force_update, min_f1_improvement)


# Convenience functions for integration with existing code
def sync_models_on_startup(weights_dir: str = "models/weights") -> bool:
    """Convenience function to sync models on startup."""
    manager = AzureMLSync(weights_dir)
    baseline_synced, dynamic_synced = manager.sync_on_startup()
    return baseline_synced or dynamic_synced


def upload_dynamic_model_to_azure(
    f1_score: float, weights_dir: str = "models/weights",
    metadata: Optional[Dict] = None
) -> bool:
    """Convenience function to upload dynamic model."""
    manager = AzureMLSync(weights_dir)
    return manager.upload_dynamic_model(f1_score, metadata)


def promote_to_baseline_with_azure(weights_dir: str = "models/weights") -> bool:
    """Promote dynamic model to baseline with Azure."""
    sync_manager = AzureMLSync(weights_dir=weights_dir)
    return sync_manager.promote_dynamic_to_baseline()


def get_azure_configuration_status(weights_dir: str = "models/weights") -> Dict:
    """Get Azure configuration status."""
    sync_manager = AzureMLSync(weights_dir=weights_dir)
    return sync_manager.get_configuration_status()


def sync_best_baseline(
    weights_dir: str = "models/weights",
    force_update: bool = False,
    min_f1_improvement: float = 0.01,
) -> bool:
    """
    Standalone function to find and sync the best baseline model from Azure ML.

    Args:
        weights_dir: Local directory for model weights.
        force_update: If True, force download even if a local file exists.
        min_f1_improvement: Minimum F1 score improvement to trigger a download.

    Returns:
        True if a new model was downloaded, False otherwise.
    """
    sync_manager = AzureMLSync(weights_dir=weights_dir)
    return sync_manager.sync_best_baseline(
        force_update=force_update, min_f1_improvement=min_f1_improvement
    )
