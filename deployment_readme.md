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

Currently, the project is containerized using Docker, which provides a portable foundation for deployment. However, it does not yet implement specific Azure-native deployment services. The following are potential strategies to explore for a more robust, scalable, and managed deployment on Azure.

- [ ] **Explore Azure ML Managed Endpoints:**
    -   **Goal:** Deploy the model using a fully managed and scalable endpoint service designed for ML.
    -   **Tasks:**
        -   Create a scoring script (`score.py`) that loads the model and processes inference requests.
        -   Define an Azure ML Environment that includes all necessary dependencies.
        -   Write a script to programmatically create a `ManagedOnlineEndpoint` and deploy the model to it using a `ManagedOnlineDeployment` configuration.
        -   Integrate this deployment step into the CI/CD pipeline.

- [ ] **Explore Azure Container Instances (ACI):**
    -   **Goal:** Deploy the existing Docker container as a simple, serverless container instance.
    -   **Tasks:**
        -   Build and push the Docker image from the `Dockerfile` to a container registry (like Azure Container Registry or Docker Hub).
        -   Write a script using the `azure-mgmt-containerinstance` SDK to deploy the container image to ACI.
        -   Configure necessary networking and environment variables for the ACI instance.
        -   Consider this for development, testing, or low-traffic applications, as it lacks auto-scaling.

- [ ] **Explore Azure Container Apps (ACA):**
    -   **Goal:** Deploy the application for more complex, scalable production workloads with built-in HTTPS, and advanced features.
    -   **Tasks:**
        -   Build and push the Docker image to a container registry.
        -   Use the Azure CLI (`az containerapp up` or `az containerapp create`) or ARM/Bicep templates to define and deploy the container app.
        -   Configure ingress, scaling rules, and secrets for the container app.
        -   Integrate the Azure CLI deployment commands into the CI/CD pipeline. 