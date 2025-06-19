from azure.identity import ClientSecretCredential
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
from azure.mgmt.containerinstance.models import (
    ContainerGroup,
    Container,
    ResourceRequests,
    ResourceRequirements,
    ContainerGroupNetworkProtocol,
    OperatingSystemTypes,
    IpAddress,
    Port,
)
import os
from dotenv import load_dotenv

load_dotenv()


def deploy_to_aci():
    """
    Deploys a container to Azure Container Instances.
    """
    # Replace with your own values or load from environment
    SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
    RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")
    CONTAINER_NAME = os.getenv("ACI_CONTAINER_NAME", "emotion-clf-api")
    IMAGE = os.getenv("DOCKER_IMAGE")  # e.g., 'yourdockerhub/emotion-clf-api:latest'
    CPU_CORE_COUNT = 1.0
    MEMORY_GB = 1.5
    TENANT_ID = os.getenv("AZURE_TENANT_ID")
    CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
    CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")
    LOCATION = os.getenv("AZURE_LOCATION", "westeurope")

    if not all(
        [
            SUBSCRIPTION_ID,
            RESOURCE_GROUP,
            IMAGE,
            TENANT_ID,
            CLIENT_ID,
            CLIENT_SECRET,
        ]
    ):
        raise ValueError(
            "Please set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, DOCKER_IMAGE, "
            "AZURE_TENANT_ID, AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET environment variables."
        )

    print("Authenticating with Azure...")
    # Get credentials
    credentials = ClientSecretCredential(
        client_id=CLIENT_ID, client_secret=CLIENT_SECRET, tenant_id=TENANT_ID
    )

    print("Creating Container Instance Management client...")
    # Create a Container Instance Management client
    container_client = ContainerInstanceManagementClient(credentials, SUBSCRIPTION_ID)

    print(f"Defining container '{CONTAINER_NAME}'...")
    # Define the container
    container_resource_requests = ResourceRequests(
        memory_in_gb=MEMORY_GB, cpu=CPU_CORE_COUNT
    )
    container_resource_requirements = ResourceRequirements(
        requests=container_resource_requests
    )
    container = Container(
        name=CONTAINER_NAME,
        image=IMAGE,
        resources=container_resource_requirements,
        ports=[Port(port=3120)],
    )

    print("Defining container group...")
    # Define the group of containers
    container_group = ContainerGroup(
        location=LOCATION,
        containers=[container],
        os_type=OperatingSystemTypes.linux,
        ip_address=IpAddress(
            ports=[Port(protocol=ContainerGroupNetworkProtocol.tcp, port=3120)],
            type="Public",
        ),
    )

    print(f"Creating container group '{CONTAINER_NAME}' in resource group '{RESOURCE_GROUP}'...")
    # Create the container group
    result = container_client.container_groups.begin_create_or_update(
        RESOURCE_GROUP, CONTAINER_NAME, container_group
    )

    print("Waiting for container group creation to complete...")
    result.wait()
    print("Container group deployment completed.")

    # Get the IP address of the container group
    container_group_info = container_client.container_groups.get(RESOURCE_GROUP, CONTAINER_NAME)
    if container_group_info.ip_address:
        print(f"Container group '{CONTAINER_NAME}' is running at IP address: {container_group_info.ip_address.ip}")
    else:
        print("Could not retrieve the IP address. Please check the Azure portal.")


if __name__ == "__main__":
    deploy_to_aci() 