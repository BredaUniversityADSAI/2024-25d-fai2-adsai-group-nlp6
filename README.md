<div align="center">

<!-- PROJECT LOGO -->
<!-- <br /> -->
<!-- <img src="./assets/logo.png" alt="Logo" width="120" height="120" style="border-radius: 10px;"> -->
<h1>Emotion Classification Pipeline</h1>

<!-- BADGES -->
[![Python 3.9](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Poetry](https://img.shields.io/badge/packaging-poetry-cyan.svg)](https://python-poetry.org/)
[![Lint Workflow](https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp6/actions/workflows/lint.yaml/badge.svg)](https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp6/actions/workflows/lint.yaml)
[![Test Suite Workflow](https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp6/actions/workflows/test.yaml/badge.svg)](https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp6/actions/workflows/test.yaml)
[![Docker Build, Scan & Push](https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp6/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp6/actions/workflows/docker-publish.yml)
[![Build and Deploy Sphinx Documentation](https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp6/actions/workflows/sphinx-docs.yml/badge.svg)](https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp6/actions/workflows/sphinx-docs.yml)

<p align="center">
  <b>An advanced NLP tool for delivering actionable emotional insights from video and audio content.</b>
  <br />
  <i>Transforming unstructured media into meaningful emotional analytics</i>
</p>

<p>Documentation page: <a href="https://bredauniversityadsai.github.io/2024-25d-fai2-adsai-group-nlp6/">https://bredauniversityadsai.github.io/2024-25d-fai2-adsai-group-nlp6/</a></p>


<!-- PROJECT DEMO -->
<p align="center">
  <a href="#overview">
    <img src="./assets/dashboard_screenshot.png" alt="Dashboard Screenshot" width="800" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
  </a>
</p>


<p align="center">
  <a href="#installation"><strong>Installation & Usage</strong></a> •
  <a href="#contributing"><strong>Contributing</strong></a> •
  <a href="#license"><strong>License</strong></a>
</p>

</div>

<br>

## 📋 Table of Contents

<ol>
  <li><a href="#overview">🌟 Overview</a></li>
  <li><a href="#project-structure">📁 Project Structure</a></li>  <li><a href="#installation">🚀 Installation and Usage</a>
    <ul>
      <li><a href="#step-1---prerequisites">Step 1 - Prerequisites</a></li>
      <li><a href="#step-1.5---setting-up-git-lfs-for-large-model-files">Step 1.5 - Setting up Git LFS (for Large Model Files)</a></li>
      <li><a href="#step-2---cloning-the-repository">Step 2 - Cloning the Repository</a></li>
      <li><a href="#step-3---creating-the-env-file">Step 3 - Creating the .env File</a></li>
      <li><a href="#step-4---setup--run">Step 4 - Setup & Run Options</a></li>
      <li><a href="#step-5---usage">Step 5 - Usage</a></li>
    </ul>
  </li>
  <li><a href="#pipeline-plans">🚀 Pipeline Plans</a>
    <ul>
      <li><a href="#local-pipeline-commands">Local Pipeline Commands</a></li>
      <li><a href="#azure-ml-pipeline-commands">Azure ML Pipeline Commands</a></li>
      <li><a href="#advanced-training--evaluation-pipeline">Advanced Training & Evaluation Pipeline</a></li>
    </ul>
  </li>
  <li><a href="#azure-ml-pipeline-scheduling">⏰ Azure ML Pipeline Scheduling</a>
    <ul>
      <li><a href="#quick-start-examples">Quick Start Examples</a></li>
      <li><a href="#schedule-management">Schedule Management</a></li>
      <li><a href="#azure-ml-studio-integration">Azure ML Studio Integration</a></li>
    </ul>
  </li>
  <li><a href="#contributing">👥 Contributing</a>
    <ul>
      <li><a href="#branch-naming-convention">Branch Naming Convention</a></li>
      <li><a href="#pull-request-process">Pull Request Process</a></li>
    </ul>
  </li>
  <li><a href="#architecture-diagrams">🗺️ Architecture Diagrams</a>
    <ul>
      <li><a href="#system-architecture-high-level">System Architecture (High-Level)</a></li>
      <li><a href="#data-flow-for-predict-endpoint">Data Flow for <code>/predict</code> Endpoint</a></li>
      <li><a href="#internal-component-diagram-srcemotion_clf_pipeline">Internal Component Diagram (<code>src/emotion_clf_pipeline</code>)</a></li>
    </ul>
  </li>
  <li><a href="#testing-procedures">🧪 Testing Procedures</a>
    <ul>
      <li><a href="#running-pytest">Running <code>pytest</code></a></li>
      <li><a href="#running-unittest">Running <code>unittest</code></a></li>
      <li><a href="#test-coverage">Test Coverage</a></li>
    </ul>
  </li>
  <li><a href="#license">📄 License</a></li>
</ol>

<br>

<a id="overview"></a>
## 🌟 Overview

Emotion Classification Pipeline is a sophisticated natural language processing tool designed to extract and analyze emotional content from video and audio data. Built with modern ML/AI techniques, our system delivers actionable emotional insights that can be used for content analysis, customer sentiment tracking, and more.

<br>

<a id="project-structure"></a>
## 📁 Project Structure

```
./
├── .github/                  # GitHub Actions workflows
├── assets/                   # Static assets (images, logos)
├── data/                     # Datasets (raw, processed)
├── dist/                     # Distribution files (build artifacts)
├── docs/                     # Project documentation
├── frontend/                 # React frontend application
├── logs/                     # Log files
├── models/                   # Trained machine learning models
├── notebooks/                # Jupyter notebooks for exploration
├── src/                      # Source code
│   └── emotion_clf_pipeline/ # Main Python package
│       ├── __init__.py
│       ├── api.py            # FastAPI application
│       ├── cli.py            # Command-line interface
│       ├── data.py           # Data loading and preprocessing
│       ├── model.py          # Model architecture
│       ├── predict.py        # Prediction logic
│       └── train.py          # Training scripts
├── tests/                    # Unit and integration tests
├── .flake8                   # Flake8 configuration
├── .gitignore                # Git ignore rules
├── .pre-commit-config.yaml   # Pre-commit hook configurations
├── Dockerfile                # Backend Docker configuration
├── docker-compose.yml        # Docker Compose for full-stack
├── LICENSE                   # Project license
├── poetry.lock               # Poetry lock file
├── pyproject.toml            # Python project configuration (Poetry)
└── README.md                 # This file
```

<br>

<a id="installation"></a>
## 🚀 Installation and Usage

This project offers several ways to get started, depending on your needs. Choose the method that best suits your workflow.

<br>

### Step 1 - Prerequisites

Before you begin, ensure you have the following installed:

- **Python**: Version 3.11 or higher.
- **Poetry**: For managing Python dependencies. ([Installation Guide](https://python-poetry.org/docs/#installation))
- **Docker**: For containerized deployment (optional but recommended for full-stack). ([Installation Guide](https://docs.docker.com/get-docker/))
- **Git**: For cloning the repository. ([Installation Guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git))

<br>


### Step 2 - Cloning the Repository

First, clone the project to your local machine:

```bash
git clone https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp6.git
cd 2024-25d-fai2-adsai-group-nlp6
```

<br>

### Step 3 - Creating the .env File

Create a file named `.env` in the project root directory.
```bash
# From the project root directory
touch .env
```

Include all the required API keys inside the `.env`.
```
ASSEMBLYAI_API_KEY="your_actual_assemblyai_api_key"
```

This file is used by `docker-compose.yml` and the `Dockerfile` to provide secrets to the application.

<br>

### Step 4 - Setup & Run

#### Option 1. Docker Compose (Run Frontend and Backend Containers) <mark> Recommented </mark>

This is the recommended method for running the complete application, including the React frontend and the Python backend.

```bash
# Ensure you are in the project root directory
docker-compose up --build
```

This will start both the frontend (accessible at `http://localhost:3121`) and the backend API (accessible at `http://localhost:3120`).


#### Option 2. Run Docker Container (Backend API only)

This method containerizes the backend API, making it easy to deploy and run in isolation.

```bash
# Build image from Dockerfile
docker build -t emotion-clf-api .

# Run a container for "emotion-clf-api" image
docker run -p 3120:80 emotion-clf-api
```

#### Option 3. Run Backend and Frontend Separately (using Poetry & npm)

This method allows you to run the backend API and frontend independently, which is ideal for development, debugging, or when you want to customize either service.

```mermaid
graph LR
    A[YouTube URL] --> C[Audio Download]
    C --> D[Speech-to-Text]
    D --> E[Emotion Classifier]
    E --> G[API/CLI Output]
```

**Setup:**
```bash
# Install project dependencies using Poetry
poetry install

# Activate the virtual environment managed by Poetry
poetry shell
```

**Option 3a - Start Backend API (Port 3120):**
```bash
# Navigate to project root and start backend API
uvicorn src.emotion_clf_pipeline.api:app --host 0.0.0.0 --port 3120 --reload
```

**Option 3b - Start Frontend (Port 3121):**
```bash
# Navigate to frontend directory and start React app
cd "frontend"
set PORT=3121 && npm start
```

**Option 3c - CLI Usage:**
```bash
# Process a YouTube video
poetry run python -m emotion_clf_pipeline.cli "https://www.youtube.com/watch?v=jNQXAC9IVRw"

# With custom options
poetry run python -m emotion_clf_pipeline.cli "YOUR_YOUTUBE_URL" --filename my_video_output --transcription whisper
```

> **Note**: When running separately, ensure both backend (3120) and frontend (3121) are running simultaneously for full functionality. The `--host 0.0.0.0` setting allows the backend to be accessible from other machines on the network.

REST API Endpoints:

| Endpoint  | Method | Request Body                | Response Body Example                                                                                                | Description                                                                 |
|-----------|--------|-----------------------------|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| `/`       | GET    | N/A                         | `{"message": "Welcome..."}`                                                                                           | Root endpoint providing a welcome message.                                  |
| `/predict`| POST   | `{"url": "youtube_url"}`    | `{"videoId": "...", "title": "...", "transcript": [{"sentence": ..., "emotion": ...}, ...]}` | Analyzes YouTube URL for emotion.                                           |
| `/docs`   | GET    | N/A                         | HTML Page                                                                                                            | Interactive API documentation (Swagger UI).                                 |
| `/redoc`  | GET    | N/A                         | HTML Page                                                                                                            | Alternative API documentation (ReDoc).                                      |

Error Codes:

| Code | Description                  | Potential Resolution / Source                                      |
|------|------------------------------|--------------------------------------------------------------------|
| 422  | Unprocessable Entity         | Request validation error (e.g., incorrect `url` format in `/predict`). Review request payload. |
| 500  | Internal Server Error        | An unexpected error occurred on the server. Check server logs for details.     |
| -    | Service-Specific Errors      | Errors from external services (e.g., YouTube, AssemblyAI) or internal pipeline steps (e.g., audio download, transcription, model prediction). Check server logs for detailed error messages originating from `predict.py` or other modules. Ensure API keys (e.g., `ASSEMBLYAI_API_KEY`) are correctly configured if external services fail. |

<br>

### Step 5 - Usage

#### Option 1 - Send API Requests

Send requests to the API:

```bash
curl -X POST "http://127.0.0.1:3120/predict" -H "Content-Type: application/json" -d '{"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```
Windows:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:3120/predict" -Method Post -ContentType "application/json" -Body '{"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

#### Option 2 - Use the Frontend Interface

When running with docker-compose, access the frontend interface at:

- **Frontend UI**: http://localhost:3121

The UI allows you to input YouTube URLs and view emotional analysis visualizations.

### 🛠️ Quick Development Setup

For rapid development with both backend and frontend running simultaneously:

**Terminal 1 (Backend API):**
```bash
uvicorn src.emotion_clf_pipeline.api:app --host 0.0.0.0 --port 3120 --reload
```

**Terminal 2 (Frontend):**
```bash
cd "frontend"
set PORT=3121 && npm start
```

**Access Points:**
- **Backend API**: http://localhost:3120
- **Frontend UI**: http://localhost:3121
- **API Documentation**: http://localhost:3120/docs

<br>

## � Pipeline Plans

The emotion classification pipeline supports both local and Azure ML execution modes, with comprehensive automation options for training, evaluation, and scheduling.

### 📍 Local Pipeline Commands

For local development and testing, use these streamlined commands:

#### Data Preprocessing (Local)
```bash
# Basic preprocessing with verbose output
python -m src.emotion_clf_pipeline.cli preprocess --verbose --raw-train-path "data/raw/train" --raw-test-path "data/raw/test/test_data-0001.csv"
```

#### Training and Evaluation (Local)
```bash
# Quick training with minimal epochs for testing
python -m src.emotion_clf_pipeline.cli train --verbose --epochs 1 --batch-size 8
```

### ☁️ Azure ML Pipeline Commands

For production-ready pipelines with automatic asset management:

#### Data Pipeline (Azure ML)
```bash
# Data preprocessing with automatic registration of data assets
poetry run python -m emotion_clf_pipeline.cli preprocess --azure --register-data-assets --verbose
```

#### Training Pipeline (Azure ML)
```bash
# Training and evaluation with automatic model registration
poetry run python -m emotion_clf_pipeline.cli train --azure --verbose
```

#### Full Pipeline (Azure ML)
```bash
# Complete end-to-end pipeline execution
poetry run python -m emotion_clf_pipeline.cli train-pipeline --azure --verbose
```

### 👟 Advanced Training & Evaluation Pipeline

For advanced users who want granular control over the pipeline, the following detailed CLI commands are available:

##### Step 1 - Preprocess Data (Detailed)
```bash
python -m emotion_clf_pipeline.cli preprocess --raw_train_csv_path data\raw\train\ --raw_test_csv_path data\raw\test\test_data-0001.csv --model_name_tokenizer "microsoft/deberta-v3-base" --max_length 128 --processed_train_output_dir data\processed\ --processed_test_output_dir data\processed\ --encoders_output_dir models\encoders\ --output_tasks "emotion,sub_emotion,intensity"
```

##### Step 2 - Train Model (Detailed)
```bash
python -m emotion_clf_pipeline.cli train --processed_train_dir "data/processed" --processed_test_dir "data/processed" --encoders_input_dir "models/encoders" --model_name_bert "microsoft/deberta-v3-xsmall" --trained_model_output_dir "models/weights" --metrics_output_file "models/evaluation/metrics.json" --epochs 1 --batch_size 8 --learning_rate 5e-5 --output_tasks "emotion,sub_emotion,intensity"
```

##### Step 3 - Evaluate Model (Detailed)
```bash
python -m emotion_clf_pipeline.cli evaluate_register --model_input_dir models/weights --processed_test_dir data/processed --train_path data/processed/train.csv --encoders_input_dir models/encoders --final_eval_output_dir results/evaluation --registration_f1_threshold_emotion 0.5 --registration_status_output_file results/evaluation/registration_status.json
```

##### Step 4 - Predict Emotions from YouTube URL
```bash
poetry run python -m emotion_clf_pipeline.cli predict "https://www.youtube.com/watch?v=7yVFZn87TkY&ab_channel=IBuildStuff"
```

## ⏰ Azure ML Pipeline Scheduling

The emotion classification pipeline supports automated scheduling through Azure ML, allowing you to run training pipelines on a regular basis (e.g., daily at midnight) for continuous model improvement.

### 🚀 Quick Start with Scheduling

#### Enhanced Schedule Manager Commands

Create comprehensive scheduled pipelines with the enhanced schedule manager:

```bash
# Create a daily schedule at midnight UTC (enabled by default)
python -m src.emotion_clf_pipeline.cli schedule create --schedule-name 'daily-retraining' --daily --hour 0 --minute 0 --enabled --mode azure

# Create a weekly schedule on Sundays at 2 AM
python -m src.emotion_clf_pipeline.cli schedule create \
  --schedule-name 'weekly-retraining' \
  --weekly 0 --hour 2 --minute 0 \
  --enabled --mode azure

# Create a monthly schedule on the 1st at 3 AM
python -m src.emotion_clf_pipeline.cli schedule create \
  --schedule-name 'monthly-retraining' \
  --monthly 1 --hour 3 --minute 0 \
  --enabled --mode azure

# List all schedules
python -m src.emotion_clf_pipeline.cli schedule list --mode azure

# Setup default schedule patterns
python -m src.emotion_clf_pipeline.cli schedule setup-defaults --mode azure
```

#### Legacy Schedule Commands (Still Supported)

For backward compatibility, the following commands are also supported:

```bash
# Create a Daily Schedule (Midnight UTC)
poetry run python -m emotion_clf_pipeline.cli schedule create --daily --schedule-name "daily-training" --hour 0 --minute 0 --timezone "UTC"

# Create a Weekly Schedule (Sunday at 2 AM)
poetry run python -m emotion_clf_pipeline.cli schedule create --weekly 0 --schedule-name "weekly-training" --hour 2 --minute 0 --timezone "UTC"

# Create a Custom Cron Schedule (Every 6 Hours)
poetry run python -m emotion_clf_pipeline.cli schedule create --cron "0 */6 * * *" --schedule-name "frequent-training" --timezone "UTC"
```

### 📋 Schedule Management Commands

| Command | Description | Example |
|---------|-------------|---------|
| `schedule list` | List all pipeline schedules | `poetry run python -m emotion_clf_pipeline.cli schedule list` |
| `schedule details <name>` | Get detailed schedule information | `poetry run python -m emotion_clf_pipeline.cli schedule details daily-training` |
| `schedule enable <name>` | Enable a schedule | `poetry run python -m emotion_clf_pipeline.cli schedule enable daily-training` |
| `schedule disable <name>` | Disable a schedule | `poetry run python -m emotion_clf_pipeline.cli schedule disable daily-training` |
| `schedule delete <name>` | Delete a schedule | `poetry run python -m emotion_clf_pipeline.cli schedule delete daily-training --confirm` |
| `schedule setup-defaults` | Create common schedule templates | `poetry run python -m emotion_clf_pipeline.cli schedule setup-defaults` |

### 🕐 Common Cron Expressions

| Pattern | Cron Expression | Description |
|---------|----------------|-------------|
| Daily at midnight | `0 0 * * *` | Every day at 00:00 UTC |
| Every 6 hours | `0 */6 * * *` | At 00:00, 06:00, 12:00, 18:00 UTC |
| Weekly (Sunday) | `0 0 * * 0` | Every Sunday at midnight UTC |
| Monthly (1st day) | `0 0 1 * *` | First day of every month at midnight |
| Weekdays only | `0 0 * * 1-5` | Monday to Friday at midnight |
| Twice daily | `0 0,12 * * *` | Daily at midnight and noon |

### 📍 Azure ML Studio Integration

All schedules created through this system are fully integrated with Azure ML and will appear in:

- **Azure ML Studio** → **Schedules** section
- **Pipelines** → **Schedule** tab
- **Experiments** with the prefix `scheduled-`

### 🔧 Advanced Scheduling Configuration

For more complex scheduling needs, you can customize pipeline parameters:

```bash
# Schedule with custom pipeline configuration
poetry run python -m emotion_clf_pipeline.cli schedule create \
  --daily \
  --schedule-name "custom-daily-training" \
  --hour 2 \
  --minute 30 \
  --timezone "America/New_York" \
  --pipeline-name "emotion_clf_custom" \
  --model-name "microsoft/deberta-v3-base" \
  --batch-size 32 \
  --epochs 5 \
  --learning-rate 3e-5
```

### ⚠️ Important Notes

- **Schedules are created in disabled state** for safety - enable them when ready
- **Timezone support**: Use standard timezone names (e.g., "UTC", "America/New_York", "Europe/London")
- **Compute resources**: Schedules use the same compute targets as regular pipeline runs
- **Cost management**: Monitor your Azure costs as scheduled runs consume compute resources
- **Data freshness**: Ensure your data assets are updated before scheduled runs

### 🧹 Demo and Testing

Try the scheduling functionality with the demo script:

```bash
# Run scheduling demo
python demo_scheduling.py

# Clean up demo schedules
python demo_scheduling.py cleanup
```

<br>

<a id="contributing"></a>
## 👥 Contributing

Contributions are welcome! Please follow our branch naming convention and code style guidelines.

```bash
# Run pre-commit hooks to ensure code quality
poetry run pre-commit run --all-files
```

> **Note for Windows Users:** If `pre-commit` hooks (like `trailing-whitespace` or `end-of-file-fixer`) repeatedly modify files due to line ending differences, you may need to configure Git to better handle line endings for this project. 
> In your local repository, run the following commands:
> ```bash
> git config core.autocrlf false
> git add --renormalize .
> ```
> This prevents Git from automatically converting line endings in a way that conflicts with `pre-commit`.

### Branch Naming Convention

To ensure consistent collaboration and traceability, all branches should follow the naming convention:

```
<type>/<sprint>-<scope>-<action>
```

Example: `feature/s2-data-add-youtube-transcript`

Type Prefixes:

| Prefix     | Description                     |
| ---------- | ------------------------------- |
| `feature`  | New functionality               |
| `fix`      | Bug fixes                       |
| `test`     | Unit/integration testing        |
| `docs`     | Documentation updates           |
| `config`   | Environment or dependency setup |
| `chore`    | Maintenance and cleanup         |
| `refactor` | Code restructuring              |


### Pull Request Process

1. Create a feature branch
2. Make your changes
3. Submit a pull request
4. Wait for code review and approval

<br>

### Setting up Git LFS (for Large Model Files)

If you plan to contribute or manage large model files (e.g., in the `models/` directory), you should use Git LFS (Large File Storage). Git LFS replaces large files such as audio samples, videos, datasets, and graphics with text pointers inside Git, while storing the file contents on a remote server.

**1. Install Git LFS:**

   Download and install Git LFS from [git-lfs.com](https://git-lfs.com). Installation instructions vary by operating system.

**2. Initialize Git LFS for your repository:**

   After cloning the repository (see Step 2), navigate to the project's root directory and run the following command ONCE per local repository:
   ```bash
   git lfs install
   ```
   This command installs Git LFS hooks in your local repository.

**3. Track file types:**

   To tell Git LFS which files to manage, use the `git lfs track` command. For this project, we want to track files in the `models/` directory.
   ```bash
   # From the project root directory
   git lfs track "models/*"
   ```
   This command will create or update a `.gitattributes` file in your repository. This file tells Git which files should be handled by LFS.

**4. Stage and commit changes:**

   Add the `.gitattributes` file and any large model files to your Git staging area and commit them:
   ```bash
   git add .gitattributes
   git add models/  # Or add specific model files if you prefer
   git commit -m "feat: Integrate Git LFS for model files"
   ```

**5. Push to the remote repository:**

   When you push your changes, the LFS-tracked files will be uploaded to the LFS storage:
   ```bash
   git push origin <your-branch-name>
   ```

   Now, Git LFS is configured to handle large files in the `models/` directory.

<br>


## 🗺️ Architecture Diagrams

This section provides an overview of the system's architecture and data flow.

### System Architecture (High-Level)

This diagram illustrates the main components of the Emotion Classification Pipeline and how they interact, including user interfaces, backend services, external dependencies, and data storage.

```mermaid
graph TD
    subgraph User Interaction
        UI[Browser - React Frontend]
        CLI[Command Line Interface]
        CURL[cURL/Postman]
    end

    subgraph Backend Services [Emotion Classification Pipeline API - FastAPI]
        API[api.py - Endpoints /predict, /health]
        PRED[predict.py - Orchestration Logic]
        DATA[data.py - Data Handling]
        MODEL[model.py - Emotion Model]
    end

    subgraph External Services
        YT[YouTube API/Service]
        ASSEMBLY[AssemblyAI API]
        WHISPER[Whisper Model - Local/HuggingFace]
    end

    subgraph Data Storage
        DS_AUDIO[Local File System: /data/youtube_audio]
        DS_TRANS[Local File System: /data/transcripts]
        DS_RESULTS[Local File System: /data/results]
    end

    UI --> API
    CLI --> PRED
    CURL --> API

    API --> PRED

    PRED --> DATA
    PRED --> MODEL
    PRED --> ASSEMBLY
    PRED --> WHISPER

    DATA --> YT
    DATA --> DS_AUDIO
    ASSEMBLY --> DS_TRANS
    WHISPER --> DS_TRANS
    MODEL --> DS_RESULTS


    classDef userStyle fill:#C9DAF8,stroke:#000,stroke-width:2px,color:#000
    class UI,CLI,CURL userStyle

    classDef backendStyle fill:#D9EAD3,stroke:#000,stroke-width:2px,color:#000
    class API,PRED,DATA,MODEL backendStyle

    classDef externalStyle fill:#FCE5CD,stroke:#000,stroke-width:2px,color:#000
    class YT,ASSEMBLY,WHISPER externalStyle

    classDef storageStyle fill:#FFF2CC,stroke:#000,stroke-width:2px,color:#000
    class DS_AUDIO,DS_TRANS,DS_RESULTS storageStyle
```

### Data Flow for `/predict` Endpoint

This sequence diagram details the process from a user submitting a YouTube URL to receiving the emotion analysis results. It highlights the interactions between the frontend, backend API, prediction service, data handling, transcription, and the emotion model.

```mermaid
sequenceDiagram
    actor User
    participant Frontend_UI as Frontend UI (React)
    participant Backend_API as FastAPI Backend (api.py)
    participant PredictionService as Prediction Service (predict.py)
    participant DataHandler as Data Handler (data.py)
    participant TranscriptionService as Transcription (AssemblyAI/Whisper)
    participant EmotionModel as Emotion Model (model.py)
    participant FileSystem as Local File System (data/*)

    User->>Frontend_UI: Inputs YouTube URL
    Frontend_UI->>Backend_API: POST /predict (URL)
    activate Backend_API

    Backend_API->>PredictionService: process_youtube_url_and_predict(URL)
    activate PredictionService

    PredictionService->>DataHandler: save_youtube_audio(URL)
    activate DataHandler
    DataHandler-->>FileSystem: Saves audio.mp3
    DataHandler-->>PredictionService: Returns audio_file_path
    deactivate DataHandler

    PredictionService->>TranscriptionService: Transcribe(audio_file_path)
    activate TranscriptionService
    TranscriptionService-->>FileSystem: Saves transcript.xlsx/json
    TranscriptionService-->>PredictionService: Returns transcript_data (text, timestamps)
    deactivate TranscriptionService

    PredictionService->>EmotionModel: predict_emotion(transcript_sentences)
    activate EmotionModel
    EmotionModel-->>PredictionService: Returns emotion_predictions (emotion, sub_emotion, intensity)
    deactivate EmotionModel

    PredictionService-->>FileSystem: Saves results.xlsx (optional)
    PredictionService-->>Backend_API: Formatted JSON with predictions
    deactivate PredictionService

    Backend_API-->>Frontend_UI: JSON Response
    deactivate Backend_API
    Frontend_UI->>User: Displays emotional analysis
```

### Internal Component Diagram (`src/emotion_clf_pipeline`)

This diagram shows the primary Python modules within the `src/emotion_clf_pipeline` package and their main dependencies, focusing on the prediction pathway.

```mermaid
graph LR
    subgraph src/emotion_clf_pipeline
        A[api.py]
        B[cli.py]
        C[predict.py]
        D[model.py]
        E[data.py]
        F[train.py] -- Not directly in /predict flow --> D
    end

    A --> C
    B --> C
    C --> D
    C --> E

    D --> E


    classDef moduleStyle fill:#E6E6FA,stroke:#333,stroke-width:2px,color:#000
    class A,B,C,D,E,F moduleStyle
```

<br>

## 🧪 Testing Procedures

This project uses `pytest` as the primary test runner, which can also discover and execute `unittest` tests. All commands should be run using `poetry`.

### Running `pytest`

```bash
# Run all tests with verbose output
poetry run pytest -v

# Unit test
poetry run pytest -v tests/unit

# Integration Tests
poetry run pytest -v tests/integration
```

### Running `unittest`

```bash
# Discover all tests in the 'tests' directory
poetry run python -m unittest discover -v tests

# Discover for specific types
poetry run python -m unittest discover -v tests/unit
poetry run python -m unittest discover -v tests/integration
```

### Test Coverage

To generate a test coverage report (requires `pytest-cov` or `coverage` package):
```bash
poetry run coverage run -m pytest
poetry run coverage report
poetry run coverage html  # Generates HTML report in htmlcov/
```

<br>

<a id="license"></a>
## 📄 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
