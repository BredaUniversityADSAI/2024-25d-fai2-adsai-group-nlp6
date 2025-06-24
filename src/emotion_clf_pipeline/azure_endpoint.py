"""
Azure ML Kubernetes Endpoint Deployment Manager
Handles blue-green deployment, traffic switching, and CLI integration.
"""

import time
import logging
from typing import List, Optional
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    KubernetesOnlineEndpoint,
    KubernetesOnlineDeployment,
    CodeConfiguration,
)
from azure.core.exceptions import ResourceNotFoundError

# Configure logging to provide detailed output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AzureEndpointManager:
    """
    Manages blue-green deployments for Azure ML Kubernetes Online Endpoints,
    including creation, deployment, traffic management, and status checks.
    """
    K8S_INSTANCE_TYPES: List[str] = [
        "defaultinstancetype",
    ]

    def __init__(self, subscription_id: str, resource_group: str, workspace_name: str, endpoint_name: str):
        """
        Initializes the AzureEndpointManager with connection details for Azure ML.

        Args:
            subscription_id: Your Azure subscription ID.
            resource_group: The name of the resource group.
            workspace_name: The name of the Azure ML workspace.
            endpoint_name: The name of the online endpoint to manage.
        """
        if not all([subscription_id, resource_group, workspace_name, endpoint_name]):
            raise ValueError("Subscription ID, resource group, workspace name, and endpoint name are required.")
            
        self.endpoint_name = endpoint_name
        try:
            self.ml_client = MLClient(
                DefaultAzureCredential(), subscription_id, resource_group, workspace_name
            )
        except Exception as e:
            logger.error(f"Failed to initialize MLClient: {e}", exc_info=True)
            raise

    def create_endpoint(self) -> None:
        """Create the Kubernetes endpoint if it doesn't exist."""
        try:
            self.ml_client.online_endpoints.get(self.endpoint_name)
            logger.info(f"Endpoint '{self.endpoint_name}' already exists.")
        except ResourceNotFoundError:
            logger.info(f"Endpoint '{self.endpoint_name}' not found, creating a new one.")
            endpoint = KubernetesOnlineEndpoint(
                name=self.endpoint_name, auth_mode="key", compute="adsai-lambda-0"
            )
            try:
                result = self.ml_client.online_endpoints.begin_create_or_update(endpoint)
                result.wait()  # Use wait() for synchronous completion
                logger.info(f"Successfully created endpoint '{self.endpoint_name}'.")
            except Exception as e:
                logger.error(f"Failed to create endpoint '{self.endpoint_name}': {e}", exc_info=True)
                raise

    def get_k8s_compatible_instance_type(self, preferred_instance_type: Optional[str] = None) -> Optional[str]:
        """
        Validates and returns a Kubernetes-compatible instance type.

        Args:
            preferred_instance_type: The user's preferred instance type.

        Returns:
            A valid Kubernetes-compatible instance type or None if auto-selection is intended.
        """
        if preferred_instance_type:
            if preferred_instance_type in self.K8S_INSTANCE_TYPES:
                logger.info(f"Using preferred instance type: '{preferred_instance_type}'")
                return preferred_instance_type
            else:
                logger.warning(
                    f"Instance type '{preferred_instance_type}' is not in the known compatible list. "
                    f"Attempting to use it, but it may fail."
                )
                return preferred_instance_type
        
        # If no preference, return the default, or None to let Azure decide
        default_type = self.K8S_INSTANCE_TYPES[0]
        logger.info(f"No preferred instance type specified, using default: '{default_type}'")
        return default_type

    def deploy(
        self,
        model_name: str,
        model_version: str,
        deployment_name: str,
        environment: str,
        code_path: str,
        scoring_script: str,
        instance_type: Optional[str] = None,
        retry: int = 3,
    ) -> bool:
        """
        Deploy a model to the specified deployment slot (blue/green).
        Includes robust fallback logic for Kubernetes clusters.
        """
        k8s_instance_type = self.get_k8s_compatible_instance_type(instance_type)

        for attempt in range(1, retry + 1):
            logger.info(f"Deployment attempt {attempt}/{retry} for '{deployment_name}'...")
            try:
                model = self.ml_client.models.get(model_name, version=model_version)

                deployment_kwargs = {
                    "name": deployment_name,
                    "endpoint_name": self.endpoint_name,
                    "model": model.id,
                    "environment": environment,
                    "code_configuration": CodeConfiguration(
                        code=code_path, scoring_script=scoring_script
                    ),
                    "instance_count": 1,
                    "instance_type": k8s_instance_type,
                }
                
                # Remove instance_type if it's None to allow auto-selection
                if not k8s_instance_type:
                    del deployment_kwargs["instance_type"]

                deployment = KubernetesOnlineDeployment(**deployment_kwargs)
                
                logger.info(f"Submitting deployment '{deployment_name}'...")
                self.ml_client.online_deployments.begin_create_or_update(
                    deployment
                ).wait()

                instance_info = f"instance type '{k8s_instance_type}'" if k8s_instance_type else "auto-selected instance"
                logger.info(
                    f"✅ Deployment '{deployment_name}' succeeded on attempt {attempt} with {instance_info}."
                )
                return True

            except Exception as e:
                error_msg = str(e).lower()
                logger.warning(f"Deployment attempt {attempt} failed: {error_msg}")

                if "instance type" in error_msg and "not found" in error_msg:
                    logger.warning("Instance type not found. Retrying with auto-selection.")
                    k8s_instance_type = None  # Fallback to auto-select
                    time.sleep(5 * attempt)
                    continue

                if attempt == retry:
                    logger.error(
                        f"All deployment attempts for '{deployment_name}' failed after {retry} tries.",
                        exc_info=True
                    )
                    logger.error(
                        "Suggested actions:\n"
                        "1. Check if the Kubernetes cluster 'adsai-lambda-0' is healthy.\n"
                        "2. Verify the model and environment are correctly registered in Azure ML.\n"
                        "3. Ensure the scoring script and its dependencies are correct.\n"
                        "4. Check the Azure ML workspace for detailed error logs."
                    )
                    return False
                
                time.sleep(5 * attempt)
                
        return False

    def update_traffic(self, blue_weight: int = 100, green_weight: int = 0) -> None:
        """Switch traffic between blue and green deployments."""
        if blue_weight + green_weight != 100:
            raise ValueError("Sum of blue and green traffic weights must be 100.")
            
        traffic = {"blue": blue_weight, "green": green_weight}
        logger.info(f"Attempting to update traffic to: {traffic}")

        try:
            endpoint = self.ml_client.online_endpoints.get(self.endpoint_name)
            endpoint.traffic = traffic
            self.ml_client.online_endpoints.begin_create_or_update(endpoint).wait()
            logger.info(f"✅ Successfully updated traffic: {traffic}")
        except Exception as e:
            logger.error(f"Failed to update traffic: {e}", exc_info=True)
            raise

    def get_active_deployment(self) -> Optional[str]:
        """
        Determines the active deployment (blue or green) based on traffic allocation.
        """
        try:
            endpoint = self.ml_client.online_endpoints.get(self.endpoint_name)
            traffic = endpoint.traffic or {}
            if traffic.get("blue", 0) == 100:
                return "blue"
            if traffic.get("green", 0) == 100:
                return "green"
            # If traffic is split or not set, consider none fully active
            return None
        except ResourceNotFoundError:
            logger.warning(f"Endpoint '{self.endpoint_name}' not found. Cannot determine active deployment.")
            return None

    def blue_green_deploy(
        self,
        model_name: str,
        model_version: str,
        environment: str,
        code_path: str,
        scoring_script: str,
        instance_type: str = "defaultinstancetype",
    ) -> None:
        """
        Perform blue-green deployment: deploy to inactive slot, test,
        then switch traffic. Uses Kubernetes-compatible instance types.
        """
        self.create_endpoint()
        
        active_deployment = self.get_active_deployment()
        target_deployment = "green" if active_deployment == "blue" else "blue"
        
        logger.info(f"Active deployment is '{active_deployment}'. Deploying to inactive slot: '{target_deployment}'.")

        deployment_success = self.deploy(
            model_name,
            model_version,
            target_deployment,
            environment,
            code_path,
            scoring_script,
            instance_type=instance_type,
        )

        if not deployment_success:
            logger.error(f"Deployment to '{target_deployment}' slot failed. Aborting traffic switch.")
            raise RuntimeError(f"Deployment to '{target_deployment}' failed")

        logger.info(f"✅ Deployment to '{target_deployment}' slot succeeded.")
        
        # It's critical to wait for the deployment to be fully ready before switching traffic.
        # The 'deploy' method now waits, so this explicit check is for added safety.
        if not self.wait_for_deployment_ready(target_deployment):
            raise RuntimeError(f"Deployment '{target_deployment}' did not become ready in time.")

        logger.info(f"Switching traffic to '{target_deployment}'...")
        if target_deployment == "blue":
            self.update_traffic(blue_weight=100, green_weight=0)
        else:
            self.update_traffic(blue_weight=0, green_weight=100)
            
        logger.info(f"✅ Blue-green deployment complete. Active deployment is now: '{target_deployment}'")

    def wait_for_deployment_ready(self, deployment_name: str, timeout: int = 900) -> bool:
        """
        Waits for a deployment to reach a 'Succeeded' state.

        Args:
            deployment_name: Name of the deployment to check.
            timeout: Maximum time to wait in seconds (default: 15 minutes).

        Returns:
            True if the deployment succeeded, False otherwise.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                deployment = self.ml_client.online_deployments.get(
                    endpoint_name=self.endpoint_name, name=deployment_name
                )
                state = getattr(deployment, "provisioning_state", "Unknown")
                logger.info(f"Deployment '{deployment_name}' current state: {state}")

                if state == "Succeeded":
                    logger.info(f"✅ Deployment '{deployment_name}' is ready.")
                    return True
                if state in ["Failed", "Canceled"]:
                    logger.error(f"Deployment '{deployment_name}' entered failed state: {state}")
                    # Attempt to get logs for diagnostics
                    try:
                        logs = self.ml_client.online_deployments.get_logs(
                            endpoint_name=self.endpoint_name,
                            name=deployment_name,
                            lines=100,
                        )
                        logger.error(f"Last 100 lines of logs for '{deployment_name}':\n{logs}")
                    except Exception as log_e:
                        logger.error(f"Could not retrieve logs for failed deployment: {log_e}")
                    return False

                time.sleep(30)
            except ResourceNotFoundError:
                logger.warning(f"Deployment '{deployment_name}' not found yet, retrying...")
                time.sleep(30)
            except Exception as e:
                logger.warning(f"An error occurred while checking deployment status: {e}", exc_info=True)
                time.sleep(30)

        logger.error(f"Timeout ({timeout}s) waiting for deployment '{deployment_name}' to become ready.")
        return False
