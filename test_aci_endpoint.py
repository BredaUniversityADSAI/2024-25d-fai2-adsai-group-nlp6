import requests
import argparse
import os


def test_endpoint(ip_address: str):
    """
    Tests the /docs endpoint of the deployed API.

    Args:
        ip_address (str): The public IP address of the container instance.
    """
    if not ip_address:
        print("IP address not provided. Trying to get from environment variable ACI_IP_ADDRESS.")
        ip_address = os.getenv("ACI_IP_ADDRESS")
        if not ip_address:
            print("Error: ACI_IP_ADDRESS environment variable not set.")
            print("Please provide the IP address using --ip-address or set the ACI_IP_ADDRESS environment variable.")
            return

    api_url = f"http://{ip_address}:3120/"
    docs_url = f"{api_url}docs"
    print(f"Testing endpoint at: {docs_url}")

    try:
        response = requests.get(docs_url, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes

        if response.status_code == 200:
            print("API is up and running!")
            print("You can access the documentation at:", docs_url)
        else:
            print(f"API returned status code: {response.status_code}")
            print("Response content:", response.text)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while trying to reach the API: {e}")
        print("Please check the following:")
        print("1. The container instance is running correctly in the Azure portal.")
        print("2. The IP address is correct.")
        print("3. There are no network issues blocking access to port 3120.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the FastAPI endpoint on Azure Container Instances."
    )
    parser.add_argument(
        "--ip-address",
        type=str,
        help="The public IP address of the ACI.",
        default=os.getenv("ACI_IP_ADDRESS"),
    )
    args = parser.parse_args()
    test_endpoint(args.ip_address) 