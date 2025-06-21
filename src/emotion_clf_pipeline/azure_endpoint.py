"""
Azure ML Kubernetes Endpoint Deployment Manager
Handles blue-green deployment, traffic switching, and CLI integration.
"""

import time
import logging
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    KubernetesOnlineEndpoint,
    KubernetesOnlineDeployment,
    CodeConfiguration,
)
from azure.core.exceptions import ResourceNotFoundError

logger = logging.getLogger(__name__)


class AzureEndpointManager:
    """
    Manages blue-green deployment for Azure ML Kubernetes Online Endpoints.
    """    # Kubernetes cluster-specific instance types (from adsai-lambda-0 cluster)
    K8S_INSTANCE_TYPES = [
        "defaultinstancetype",  # 8 CPU, 16Gi memory - primary instance type
        # Note: gpu and gpu-instance are available but require GPU resources
    ]

    def __init__(self, subscription_id, resource_group, workspace_name, endpoint_name):
        """Initialize the Azure ML client and endpoint configuration."""
        self.ml_client = MLClient(
            DefaultAzureCredential(), subscription_id, resource_group, workspace_name
        )
        self.endpoint_name = endpoint_name

    def create_endpoint(self):
        """Create the Kubernetes endpoint if it doesn't exist."""
        try:
            self.ml_client.online_endpoints.get(self.endpoint_name)
            logger.info(f"Endpoint '{self.endpoint_name}' already exists.")
        except ResourceNotFoundError:
            endpoint = KubernetesOnlineEndpoint(
                name=self.endpoint_name,
                auth_mode="key",
                compute="adsai-lambda-0"
            )
            result = self.ml_client.online_endpoints.begin_create_or_update(endpoint)
            result.result()
            logger.info(f"Created endpoint '{self.endpoint_name}'.")

    def get_k8s_compatible_instance_type(self, preferred_instance_type=None):
        """
        Get a Kubernetes-compatible instance type.

        Args:
            preferred_instance_type: User's preferred instance type

        Returns:            str: A Kubernetes-compatible instance type
        """
        if (
            preferred_instance_type
            and preferred_instance_type in self.K8S_INSTANCE_TYPES
        ):
            return preferred_instance_type

        # Log if user's preferred type is not K8s compatible
        if preferred_instance_type:
            logger.warning(
                f"Instance type '{preferred_instance_type}' may not be compatible "
                f"with Kubernetes. Using '{self.K8S_INSTANCE_TYPES[0]}' instead."
            )

        return self.K8S_INSTANCE_TYPES[0]

    def deploy(
        self,
        model_name,
        model_version,
        deployment_name,
        environment,
        code_path,
        scoring_script,
        instance_type=None,
        retry=3,
    ):
        """
        Deploy a model to the specified deployment slot (blue/green).
        Includes robust fallback logic for Kubernetes clusters.
        """
        use_auto_select = False
        k8s_instance_type = None

        # Try user-specified instance type first, then fallbacks, then auto-select
        if instance_type:
            k8s_instance_type = self.get_k8s_compatible_instance_type(instance_type)

        for attempt in range(1, retry + 1):
            try:
                model = self.ml_client.models.get(model_name, version=model_version)

                # Create deployment with or without instance type
                deployment_kwargs = {
                    "name": deployment_name,
                    "endpoint_name": self.endpoint_name,
                    "model": model.id,
                    "environment": environment,
                    "code_configuration": CodeConfiguration(
                        code=code_path, scoring_script=scoring_script
                    ),
                    "instance_count": 1,
                }

                # Only add instance_type if we're not using auto-select
                if not use_auto_select and k8s_instance_type:
                    deployment_kwargs["instance_type"] = k8s_instance_type

                deployment = KubernetesOnlineDeployment(**deployment_kwargs)
                self.ml_client.online_deployments.begin_create_or_update(
                    deployment
                ).result()

                instance_info = (
                    f"instance type '{k8s_instance_type}'"
                    if k8s_instance_type
                    else "auto-selected instance type"
                )
                logger.info(
                    f"Deployment '{deployment_name}' succeeded with "
                    f"{instance_info}."
                )
                return True

            except Exception as e:
                error_msg = str(e).lower()

                # Handle instance type errors with intelligent fallback
                if ("instance type" in error_msg and "not found" in error_msg) or (
                    "instance" in error_msg and "not available" in error_msg
                ):

                    if not use_auto_select:
                        # Try fallback instance types first
                        fallback_types = [
                            t for t in self.K8S_INSTANCE_TYPES if t != k8s_instance_type
                        ]

                        if fallback_types and len(fallback_types) > (attempt - 1):
                            k8s_instance_type = fallback_types[
                                min(attempt - 1, len(fallback_types) - 1)
                            ]
                            logger.warning(
                                f"Instance type failed, trying fallback: '{k8s_instance_type}'"
                            )
                            continue
                        else:
                            # All specific instance types failed, try auto-select
                            use_auto_select = True
                            k8s_instance_type = None
                            logger.warning(
                                "All specific instance types failed, trying auto-select"
                            )
                            continue

                # For other deployment errors, provide context
                if "validation" in error_msg or "invalid" in error_msg:
                    logger.error(f"Deployment validation failed: {e}")
                else:
                    logger.warning(f"Deployment attempt {attempt} failed: {e}")

                if attempt == retry:
                    logger.error(
                        f"All deployment attempts failed after {retry} tries. Last error: {e}"
                    )
                    logger.error(
                        "Suggested actions:\n"
                        "1. Check if the Kubernetes cluster 'adsai-lambda-0' is healthy\n"
                        "2. Verify model and environment are correctly registered\n"
                        "3. Check cluster node pool configuration\n"
                        f"4. Available fallback types tried: {', '.join(self.K8S_INSTANCE_TYPES[:3])}"
                    )
                    raise
                time.sleep(5 * attempt)
        return False

    def update_traffic(self, blue_weight=100, green_weight=0):
        """Switch traffic between blue and green deployments."""
        traffic = {"blue": blue_weight, "green": green_weight}

        # Get the existing endpoint
        endpoint = self.ml_client.online_endpoints.get(self.endpoint_name)

        # Update the traffic allocation
        endpoint.traffic = traffic

        # Apply the update
        self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        logger.info(f"Updated traffic: {traffic}")

    def get_active_deployment(self):
        """Return the active deployment (blue or green) based on traffic."""
        endpoint = self.ml_client.online_endpoints.get(self.endpoint_name)
        traffic = endpoint.traffic
        if traffic.get("blue", 0) >= 100:
            return "blue"
        elif traffic.get("green", 0) >= 100:
            return "green"
        return None

    def blue_green_deploy(
        self,
        model_name,
        model_version,
        environment,
        code_path,
        scoring_script,
        instance_type="defaultinstancetype",
    ):
        """
        Perform blue-green deployment: deploy to inactive slot, test,
        then switch traffic. Uses Kubernetes-compatible instance types.
        """
        self.create_endpoint()
        active = self.get_active_deployment()
        next_deploy = "green" if active == "blue" else "blue"
        logger.info(f"Deploying to '{next_deploy}' slot.")

        # Deploy to the inactive slot
        deployment_success = self.deploy(
            model_name,
            model_version,
            next_deploy,
            environment,
            code_path,
            scoring_script,
            instance_type=instance_type,
        )

        if not deployment_success:
            raise RuntimeError(f"Deployment to '{next_deploy}' slot failed")

        logger.info(f"Deployment to '{next_deploy}' slot initiated successfully")

        # Wait for deployment to be ready before switching traffic
        logger.info(f"Waiting for deployment '{next_deploy}' to be ready...")
        if not self.wait_for_deployment_ready(next_deploy):
            raise RuntimeError(f"Deployment '{next_deploy}' did not become ready")

        logger.info(f"Deployment '{next_deploy}' is ready")

        # (Optional) Add health check here

        # Switch traffic to the new deployment only if deployment succeeded
        if next_deploy == "blue":
            self.update_traffic(blue_weight=100, green_weight=0)
        else:
            self.update_traffic(blue_weight=0, green_weight=100)
        logger.info(f"Blue-green deployment complete. Active: {next_deploy}")

    def wait_for_deployment_ready(self, deployment_name, timeout=600):
        """
        Wait for a deployment to be in a ready state.

        Args:
            deployment_name: Name of the deployment to check
            timeout: Maximum time to wait in seconds (default: 10 minutes)

        Returns:
            bool: True if deployment is ready, False if timeout or failed
        """
        import time

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                deployment = self.ml_client.online_deployments.get(
                    endpoint_name=self.endpoint_name, name=deployment_name
                )

                if hasattr(deployment, "provisioning_state"):
                    state = deployment.provisioning_state
                    logger.info(f"Deployment '{deployment_name}' state: {state}")

                    if state == "Succeeded":
                        return True
                    elif state in ["Failed", "Canceled"]:
                        logger.error(
                            f"Deployment '{deployment_name}' failed with "
                            f"state: {state}"
                        )
                        return False

                # Wait before next check
                time.sleep(30)

            except Exception as e:
                logger.warning(f"Error checking deployment status: {e}")
                time.sleep(30)

        logger.error(f"Timeout waiting for deployment '{deployment_name}' to be ready")
        return False
