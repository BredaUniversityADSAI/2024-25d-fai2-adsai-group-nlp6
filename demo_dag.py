from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from datetime import datetime
import os
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
from azure.storage.blob import BlobServiceClient
from azure.ai.ml.entities import Data

# === Fill these in from Azure ML Studio UI ===
# Datastore name (Default workspace Blob store name): e.g. "workspaceblobstore"
DATASTORE_NAME = "workspaceblobstore"
# Blob path inside the container: e.g. "LocalUpload/"
BLOB_PATH = "LocalUpload/33458cd81c28a1c114908e3b31881430/test.csv"
# Local filename under /tmp/
LOCAL_FILE_NAME = "<test.csv>"

LOCAL_PATH = f"/tmp/{LOCAL_FILE_NAME}"


def download_blob():
    # Get Azure ML connection
    conn = BaseHook.get_connection("azure_ml_conn")
    extras = conn.extra_dejson

    # Extract SP credentials and workspace info
    tenant_id = extras["tenant_id"]
    client_id = extras["client_id"]
    client_secret = extras["client_secret"]
    subscription_id = extras["subscription_id"]
    resource_group = extras["resource_group"]
    workspace_name = extras["workspace_name"]

    # Authenticate and instantiate MLClient
    credential = ClientSecretCredential(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret
    )
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )

    # Lookup blob datastore details
    datastore = ml_client.datastores.get(DATASTORE_NAME)
    account_name = datastore.account_name
    container_name = datastore.container_name

    # Create BlobServiceClient
    blob_service_client = BlobServiceClient(
        account_url=f"https://{account_name}.blob.core.windows.net",
        credential=credential
    )
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(BLOB_PATH)

    # Download blob to /tmp
    os.makedirs(os.path.dirname(LOCAL_PATH), exist_ok=True)
    with open(LOCAL_PATH, "wb") as f:
        download_stream = blob_client.download_blob()
        f.write(download_stream.readall())

    print(f"Downloaded blob '{BLOB_PATH}' to '{LOCAL_PATH}'")


def preprocess_blob():
    # Insert your Python preprocessing logic here
    print(f"Starting preprocessing on {LOCAL_PATH}")
    # e.g., open(LOCAL_PATH) and transform data


def create_data_asset():
    # Get connection and parse extras
    conn = BaseHook.get_connection("azure_ml_conn")
    extras = conn.extra_dejson

    tenant_id = extras.get("tenant_id")
    client_id = extras.get("client_id")
    client_secret = extras.get("client_secret")
    subscription_id = extras.get("subscription_id")
    resource_group = extras.get("resource_group")
    workspace_name = extras.get("workspace_name")

    # Authenticate using Service Principal
    credential = ClientSecretCredential(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret
    )

    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )

    # # Locate CSV file
    # csv_path = os.path.abspath(os.path.join(
    #     os.path.dirname(__file__), '..', 'data', 'test.csv')
    # )

    data_asset = Data(
        path=LOCAL_PATH,
        type="uri_file",
        description="CSV registered via Airflow using AzureML connection",
        name="airflow_csv_conn_asset",
        version="1"
    )

    ml_client.data.create_or_update(data_asset)
    print("Data asset created using Airflow connection.")


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
}

with DAG(
    dag_id='azureml_demo',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['azureml', 'datastore', 'blob', 'download'],
) as dag:

    download_task = PythonOperator(
        task_id='download_blob',
        python_callable=download_blob,
    )

    preprocess_task = PythonOperator(
        task_id='preprocess_blob',
        python_callable=preprocess_blob,
    )

    create_asset_task = PythonOperator(
        task_id='create_data_asset',
        python_callable=create_data_asset,
    )

    download_task >> preprocess_task >> create_asset_task
