# Emotion Classification Pipeline

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Poetry](https://img.shields.io/badge/packaging-poetry-cyan.svg)](https://python-poetry.org/)
[![Lint Workflow](https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp6/actions/workflows/lint.yaml/badge.svg)](https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp6/actions/workflows/lint.yaml)
[![Test Suite Workflow](https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp6/actions/workflows/test.yaml/badge.svg)](https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp6/actions/workflows/test.yaml)

This project delivers an end-to-end NLP pipeline that processes video or audio content, transcribes spoken language, and classifies the emotional content. Built with modern ML/AI techniques and deployed on Azure using MLOps principles, the system enables:

- **Automated transcription** of video/audio content
- **Accurate emotion classification** from text
- **Scalable cloud deployment** with monitoring
- **CI/CD integration** for streamlined updates

## Table of Contents

- [Overview](#emotion-classification-pipeline)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Running with Docker](#-running-with-docker)
- [Usage](#-usage)
  - [Using the Docker Container](#-using-the-docker-container)
  - [API (Directly)](#️-api-directly)
  - [CLI](#-cli)
- [Contributing Guide](#-contributing-guide)
- [License](#-license)
- [Docker Image CI/CD](#docker-image-ci-cd)

## 📁 Project Structure

```
2024-25d-fai2-adsai-group-nlp6/
├── src/                   # Source code for the project
│   └── emotion_clf_pipeline/ # Main package directory
│       ├── __init__.py    # Makes emotion_clf_pipeline a Python package
│       ├── api.py         # FastAPI application
│       ├── cli.py         # Command-line interface script
│       ├── data.py        # Functions for loading and preprocessing data
│       ├── model.py       # Model architecture definition
│       ├── train.py       # Training and evaluation logic
│       └── predict.py     # Functions for loading model and making predictions
│
├── data/                  # Datasets used for the project
│   ├── raw/               # Original, unprocessed data
│   └── processed/         # Cleaned and preprocessed data ready for use
│
├── models/                # Trained ML models, artifacts, and checkpoints for retrieval and deployment
│
├── notebooks/             # Jupyter notebooks for exploration and prototyping
│
├── docs/                  # Project documentation and references
│
├── tests/                 # Unit tests for the codebase
│
├── .gitignore             # Specifies files and folders to ignore in Git
├── Dockerfile             # Docker configuration for the API
├── LICENSE                # Project license
├── README.md              # Project overview and instructions (THIS ONE MOVES TO ROOT)
└── pyproject.toml         # Project metadata and dependencies (Python packaging)
```

## ✅ Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop/) installed on your system.

### Training

```
poetry run python -m emotion_clf_pipeline.train
```


## 🚀 Running with Docker

This is the recommended way to run the API.

1.  **Build the Docker Image:**
    Open a terminal in the project root directory (`2024-25d-fai2-adsai-group-nlp6`) and run:
    ```bash
    docker build -t emotion-clf-api .
    ```

2.  **Configure API Keys (AssemblyAI):**
    This application uses AssemblyAI for audio transcription, which requires an API key.
    *   Create a file named `.env` in the project root directory (`2024-25d-fai2-adsai-group-nlp6`).
    *   Add your AssemblyAI API key to this file:
        ```
        ASSEMBLYAI_API_KEY="your_actual_assemblyai_api_key"
        ```
    *   **Important:** Ensure `.env` is listed in your `.gitignore` file to prevent committing your secret key (it should be there by default in this project's .gitignore). The `.env` file will be copied into the Docker image when you build it.

3.  **Run the Docker Container:**
    Once the image is built (which now includes your `.env` file), run a container with the following command:

    ```bash
    docker run -p 8000:80 emotion-clf-api
    ```
    This command maps port 80 inside the container (where the app runs) to port 8000 on your host machine. The API will be accessible at `http://localhost:8000`.

    **Note:** If you update the `.env` file, you will need to rebuild the Docker image for the changes to take effect within the container:
    ```bash
    docker build -t emotion-clf-api .
    ```

## 🛠️ Usage

### 🐳 Using the Docker Container

With the container running (see [Running with Docker](#-running-with-docker)), you can send requests to the API.

**Example using `curl` (PowerShell/Windows):**
```powershell
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"url\": \"https://www.youtube.com/watch?v=dQw4w9WgXcQ\"}"
```

**Example using `curl` (Bash/Linux/macOS):**
```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

### 🗣️ API (Directly)

If you prefer not to use Docker, you can run the API directly using Uvicorn (requires installing dependencies via Poetry first).

1.  **Install Dependencies (if not done):**
    ```bash
    # Use python 3.9
    poetry env use python3.9

    # Assumes Poetry is installed and you are in the project root
    poetry install --only main
    ```
2.  **Run Uvicorn:**
    ```bash
    uvicorn src.emotion_clf_pipeline.api:app --reload --host 127.0.0.1 --port 8000
    ```
3.  **Send Request:**
    Use the same `curl` commands as shown in the [Using the Docker Container](#-using-the-docker-container) section.

### 🧑‍💻 CLI

The project also includes a command-line interface for quick predictions.

1.  **Install Dependencies & Activate Environment (if not done):**
    Follow steps 1 & 2 from the [API (Directly)](#️-api-directly) section.

2.  **Run the CLI Script:**
    Execute the script from the project root, providing the text as an argument:
    ```bash
    python -m src.emotion_clf_pipeline.cli "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    ```
    You can also specify a base filename for outputs and the transcription method:
    ```bash
    python src/emotion_clf_pipeline/cli.py "YOUR_YOUTUBE_URL" --filename my_video_output --transcription whisper
    ```
    The predicted emotion details for transcribed sentences will be printed to the console in JSON format.

## 👥 Contributing Guide

### Code Formatting and Linting

To maintain code quality and consistency, this project uses `black` for code formatting, `isort` for import sorting, and `flake8` for linting. These are enforced by pre-commit hooks.

**Before committing your changes, and especially before pushing to the repository, please ensure your code is properly formatted and passes all linting checks.**

You can (and should) run these checks and automatic formatting locally using Poetry:

```bash
poetry run pre-commit run --all-files
```

This command will automatically format your files with `black` and `isort`, and then `flake8` will report any remaining linting issues that need manual attention.

**Using VS Code Extensions (Recommended):**

For a smoother development experience, it's highly recommended to use VS Code extensions for these tools:

*   **Python (Microsoft):** Essential for Python development, provides linting and formatting capabilities.
*   **Black Formatter (Microsoft):** Automatically formats your Python code with `black` on save.
*   **isort (Microsoft):** Automatically sorts your imports with `isort` on save.

Configure your VS Code settings (`settings.json`) to enable format on save and to use `black` as the default formatter and `isort` for organizing imports. This helps catch and fix issues early.

By following these steps, you help ensure that all code merged into the repository is clean, consistent, and adheres to our coding standards.

The `main` branch is protected and cannot be pushed to directly. All changes must be made through pull requests.

### Pull Request Process

1. **Create a new branch**: Always create a new branch for your work
   ```bash
   git checkout -b your-feature-name
   ```

2. **Make your changes**: Implement your feature or bug fix

3. **Submit a pull request**: Push your branch and create a PR on GitHub
   ```bash
   git push -u origin your-feature-name
   ```

4. **Code review**: At least one team member must review and approve your PR

5. **Merge**: After approval, your PR will be merged into the master branch


## Docker Image CI/CD

This project uses GitHub Actions to automatically build and push Docker images to Docker Hub.

### Workflow Setup

The workflow is defined in `.github/workflows/docker-publish.yml`.

To enable the workflow to push images to Docker Hub, you need to configure the following secrets in your GitHub repository settings (under "Settings" > "Secrets and variables" > "Actions"):

1.  **`DOCKERHUB_USERNAME`**: Your Docker Hub username.
2.  **`DOCKERHUB_TOKEN`**: A Docker Hub access token. You can generate an access token from your Docker Hub account settings (Account Settings > Security > New Access Token). **Do not use your actual Docker Hub password.**

Once these secrets are in place, the workflow will trigger on every push to the `main` branch and will also be available for manual dispatch via the "Actions" tab in GitHub.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
