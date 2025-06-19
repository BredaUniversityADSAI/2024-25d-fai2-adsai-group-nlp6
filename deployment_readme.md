# Deployment Strategy: Blue-Green

This document outlines the Blue-Green deployment strategy used for the emotion classification model in this project. This strategy allows for safe, zero-downtime updates to the production model.

## 1. Core Concepts

Our deployment revolves around two key model designations within the Azure ML Model Registry:

*   **Blue Environment (The Baseline Model):** This is the stable, live production model. The API server is configured to load and use the model tagged as `emotion-clf-baseline`. It serves 100% of user traffic.

*   **Green Environment (The Dynamic Model):** This is a new, candidate model that has been trained on new data or with a new architecture. It is tagged as `emotion-clf-dynamic`. This model is in a staging environment and receives no live traffic.

The process of updating the production model involves training a new "dynamic" model, validating it, and then "flipping the switch" to make it the new "baseline".

## 2. The Deployment Workflow

The deployment process is managed by two key components: `azure_pipeline.py` for automated training and `azure_sync.py` for model promotion.

### Step 1: Training the "Green" Model

*   A new model is trained using the Azure ML pipeline, triggered either manually or on a schedule (defined in `azure_pipeline.py`).
*   Upon successful training and evaluation, the pipeline registers this new model in the Azure ML Model Registry with the `emotion-clf-dynamic` tag.
*   At this stage, the new model is in the "Green" environment. It is completely isolated from the production API and has no impact on users.

### Step 2: Validation (Offline)

*   The performance of the "Green" (dynamic) model is compared against the "Blue" (baseline) model. This can be done by reviewing the metrics (e.g., F1-score, accuracy) from the training pipeline's output and comparing them to the metrics of the current baseline model.
*   This is a critical decision point. Only models that show a significant performance improvement should be promoted.

### Step 3: Flipping the Switch (Promotion)

*   Once the "Green" model is validated and approved for deployment, the promotion is handled by the `azure_sync.py` script.
*   Executing the `promote_dynamic_to_baseline()` function within this script performs the "flip":
    1.  It finds the latest version of the `emotion-clf-dynamic` model.
    2.  It removes the `emotion-clf-baseline` tag from the old production model.
    3.  It applies the `emotion-clf-baseline` tag to the new, validated dynamic model.
*   This switch is nearly instantaneous within the Azure ML Model Registry.

### Step 4: Activating the New Model

*   The production API server is designed to load the model tagged as `emotion-clf-baseline` on startup or on a regular refresh cycle.
*   After the switch, the next time the API server loads its model, it will automatically fetch the new version that now carries the "baseline" tag.
*   The old model is now effectively offline but remains in the model registry, available for a quick rollback if needed.

## 3. Rollback Procedure

The Blue-Green strategy provides a straightforward rollback mechanism. If the newly deployed model shows unexpected issues in production:

1.  Identify the version number of the previous stable model from the Azure ML Model Registry.
2.  Manually re-apply the `emotion-clf-baseline` tag to that specific version.
3.  Restart the API server. It will now load the old, stable model, effectively rolling back the deployment.

This process ensures that you can quickly and safely recover from a problematic deployment with minimal impact on users.

## 4. Implemented Blue-Green Workflow

The core logic for promoting a model in the Azure ML Registry has been fully integrated with the API server to enable a seamless, zero-downtime Blue-Green deployment workflow.

- [x] **Integrate Model Sync on API Startup:**
    -   `api.py` now includes a startup event handler. When the FastAPI application launches, it automatically calls the `sync_best_baseline` function from `azure_sync.py`.
    -   This ensures that any new or restarted API instance always downloads the latest version of the model tagged as `emotion-clf-baseline` from the Azure ML Registry and loads it into memory.

- [x] **Implement a Model Refresh Mechanism:**
    -   A new endpoint, `POST /refresh-model`, has been created in `api.py`.
    -   When this endpoint is called, it triggers the API to re-run the model synchronization process, downloading and loading the latest `emotion-clf-baseline` model without requiring a full server restart.
    -   **Note:** This endpoint is currently unsecured. Access should be restricted in a production environment (e.g., via firewall rules or an API gateway).

- [x] **Update the Promotion Script:**
    -   The `promote_dynamic_to_baseline()` function in `azure_sync.py` has been updated.
    -   After successfully promoting a model in the registry, it now automatically makes a `POST` request to the `/refresh-model` endpoint.
    -   The endpoint URL is configurable via the `API_REFRESH_ENDPOINT` environment variable (defaults to `http://localhost:8000/refresh-model`). This call completes the Blue-Green flip by forcing all running API instances to adopt the new model.

## 5. TODO for Azure-Native Deployment Strategies

The project has now been upgraded to support deployment via Azure ML Managed Endpoints, which is the recommended approach for scalable, production-grade serving on Azure.

- [x] **Deploy with Azure ML Managed Endpoints:**
    -   **Goal:** The model is deployed using a fully managed and scalable endpoint service designed for ML.
    -   **Implementation Details:**
        -   A scoring script (`src/emotion_clf_pipeline/score.py`) has been created to load the model and process inference requests.
        -   An Azure ML Environment is defined in `environment/environment.yml` to ensure all dependencies are correctly installed.
        -   A deployment script (`src/emotion_clf_pipeline/deploy_endpoint.py`) automates the creation of the endpoint and the deployment of the `emotion-clf-baseline` model.
    -   **How to Deploy:**
        -   Ensure you are logged into Azure (`az login`).
        -   Run the deployment script from the root of the project:
          ```bash
          python -m src.emotion_clf_pipeline.deploy_endpoint --endpoint-name <your-unique-endpoint-name>
          ```
    -   **Next Step:** Integrate this deployment script into a CI/CD pipeline (e.g., GitHub Actions) to be triggered after a model is successfully promoted.

- [ ] **Explore Azure Container Instances (ACI):**
    -   **Goal:** Deploy the existing Docker container as a simple, serverless container instance.
    -   **Tasks:**
        - [x] Build and push the Docker image from the `Dockerfile` to a container registry (like Azure Container Registry or Docker Hub).
        - [x] Write a script using the `azure-mgmt-containerinstance` SDK to deploy the container image to ACI.
        - [x] Configure necessary networking and environment variables for the ACI instance.
    -   **How to Deploy:**
        1.  **Build and Push the Docker Image:**
            First, ensure your Docker image is available in a public or private container registry. Docker Hub is a straightforward option.
            - **Log in to Docker Hub:**
              ```bash
              docker login
              ```
            - **Build the image:** This command builds the backend image using the main `Dockerfile`.
              ```bash
              docker build -t soheilmp/emotion-clf-backend:latest .
              ```
              Replace `soheilmp` with your Docker Hub username.
            - **Push the image:**
              ```bash
              docker push soheilmp/emotion-clf-backend:latest
              ```

        2.  **Set Up Environment Variables:**
            Create a `.env` file in the root of the project and add your Azure service principal credentials and deployment configuration. The deployment script will load these automatically.
            ```env
            # .env file
            AZURE_SUBSCRIPTION_ID="<your-subscription-id>"
            AZURE_RESOURCE_GROUP="<your-resource-group>"
            AZURE_TENANT_ID="<your-tenant-id>"
            AZURE_CLIENT_ID="<your-client-id>"
            AZURE_CLIENT_SECRET="<your-client-secret>"
            AZURE_LOCATION="westeurope" # Or your preferred Azure region

            # Docker Image from Step 1
            DOCKER_IMAGE="soheilmp/emotion-clf-backend:latest"

            # Optional: Name for the container instance
            ACI_CONTAINER_NAME="emotion-clf-backend"
            ```

        3.  **Run the Deployment Script:**
            Execute the deployment script from the root of the project. It will create the Azure Container Instance.
            ```bash
            python -m src.emotion_clf_pipeline.deploy_aci
            ```
            The script will print the public IP address of your container once it's deployed.

        4.  **Test the Deployed Endpoint:**
            After deployment, use the `test_aci_endpoint.py` script to verify that the API is running.
            - You can pass the IP address directly:
              ```bash
              python test_aci_endpoint.py --ip-address <your-container-ip-address>
              ```
            - Or, set it as an environment variable in your `.env` file and run the script:
              ```
              # .env file
              ACI_IP_ADDRESS="<your-container-ip-address>"
              ```
              ```bash
              python test_aci_endpoint.py
              ```

- [ ] **Explore Azure Container Apps (ACA):**
    -   **Goal:** Deploy the application for more complex, scalable production workloads with built-in HTTPS, and advanced features.
    -   **Tasks:**
        -   Build and push the Docker image to a container registry.
        -   Use the Azure CLI (`az containerapp up` or `az containerapp create`) or ARM/Bicep templates to define and deploy the container app.
        -   Configure ingress, scaling rules, and secrets for the container app.
        -   Integrate the Azure CLI deployment commands into the CI/CD pipeline.

## 6. TODO for Monitoring

The following is a phased plan to implement a comprehensive monitoring strategy, covering everything from basic system health to advanced model performance tracking.

### Phase 1: Implement Foundational Batch Inference
The cornerstone of most monitoring activities is the ability to process large datasets offline.

- [ ] **Enhance the Command-Line Interface (CLI):**
    - [ ] Modify `src/emotion_clf_pipeline/cli.py` to support batch predictions from a file (`--input-file`).
    - [ ] Make the existing `url` argument optional, ensuring a user provides either a single `url` or an `--input-file`.
    - [ ] Implement the logic in the `run_predict` function to process a file of URLs and aggregate the results.

### Phase 2: Implement API Health & System Monitoring
This phase focuses on the operational health of the deployed API.

- [ ] **Create a Health Check Script (`src/emotion_clf_pipeline/monitoring/health_check.py`):**
    - [ ] **Scenario Testing:** Implement a function to call the `/predict` endpoint with valid, invalid, and edge-case inputs to verify API robustness.
    - [ ] **Latency Monitoring:** Add timing logic to the `/predict` endpoint in `api.py` to log the processing time for each request.
    - [ ] **Basic Load Testing:** Create a function in the health check script to send multiple requests to the API and calculate average latency and throughput.

### Phase 3: Implement Model Quality Monitoring (Drift Detection)
This answers the question: "Is my model still behaving like it did during training?"

- [ ] **Create a Drift Detection Script (`src/emotion_clf_pipeline/monitoring/drift_detector.py`):**
    - [ ] **Create a Reference Profile:** Add a feature to analyze the training data and save a statistical profile of its predictions (e.g., emotion distribution).
    - [ ] **Compare Current Data:** Add a feature to take a new batch of data, create a new profile, and statistically compare it to the reference profile.
    - [ ] **Generate a Drift Report:** Output a report indicating if significant Data Drift or Concept Drift has been detected.

### Phase 4: Implement Performance Monitoring (with Ground Truth)
This phase calculates the "true" performance of the model using the feedback loop.

- [ ] **Create a Performance Reporting Script (`src/emotion_clf_pipeline/monitoring/performance_report.py`):**
    - [ ] **Utilize Feedback Data:** Process the CSV files generated by the `/save-feedback` endpoint.
    - [ ] **Re-run Predictions:** Run batch predictions on the text from the feedback files.
    - [ ] **Calculate True Metrics:** Compare the model's predictions against the human-corrected labels to calculate F1-score, accuracy, etc.
    - [ ] **Generate a Performance Report:** Save a timestamped classification report to track true model performance over time. 