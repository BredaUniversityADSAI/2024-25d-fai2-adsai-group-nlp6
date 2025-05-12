# Emotion Classification Pipeline

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Poetry](https://img.shields.io/badge/packaging-poetry-cyan.svg)](https://python-poetry.org/)

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
├── README.md              # Project overview and instructions
└── pyproject.toml         # Project metadata and dependencies (Python packaging)
```

## ✅ Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop/) installed on your system.

## 🚀 Running with Docker

This is the recommended way to run the API.

1.  **Build the Docker Image:**
    Open a terminal in the project root directory (`2024-25d-fai2-adsai-group-nlp6`) and run:
    ```bash
    docker build -t emotion-clf-api .
    ```

2.  **Run the Docker Container:**
    Once the image is built, run a container:
    ```bash
    docker run -p 8000:80 emotion-clf-api
    ```
    This command maps port 80 inside the container to port 8000 on your host machine. The API will be accessible at `http://localhost:8000`.

## 🛠️ Usage

### 🐳 Using the Docker Container

With the container running (see [Running with Docker](#-running-with-docker)), you can send requests to the API.

**Example using `curl` (PowerShell/Windows):**
```powershell
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{""text"": ""This is a sample text to test the emotion prediction.""}"
```

**Example using `curl` (Bash/Linux/macOS):**
```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"text": "This is a sample text to test the emotion prediction."}'
```

### 🗣️ API (Directly)

If you prefer not to use Docker, you can run the API directly using Uvicorn (requires installing dependencies via Poetry first).

1.  **Install Dependencies (if not done):**
    ```bash
    # Assumes Poetry is installed and you are in the project root
    poetry install --only main
    ```
2.  **Activate Environment:**
    ```bash
    poetry shell
    ```
3.  **Run Uvicorn:**
    ```bash
    uvicorn src.emotion_clf_pipeline.api:app --reload --host 127.0.0.1 --port 8000
    ```
4.  **Send Request:**
    Use the same `curl` commands as shown in the [Using the Docker Container](#-using-the-docker-container) section.

### 🧑‍💻 CLI

The project also includes a command-line interface for quick predictions.

1.  **Install Dependencies & Activate Environment (if not done):**
    Follow steps 1 & 2 from the [API (Directly)](#️-api-directly) section.

2.  **Run the CLI Script:**
    Execute the script from the project root, providing the text as an argument:
    ```bash
    python src/emotion_clf_pipeline/cli.py "Feeling really happy today!"
    ```
    The predicted emotion details will be printed to the console in JSON format.

## 👥 Contributing Guide

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

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
