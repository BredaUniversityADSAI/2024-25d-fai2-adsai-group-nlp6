# Emotion Classification Pipeline

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Poetry](https://img.shields.io/badge/packaging-poetry-cyan.svg)](https://python-poetry.org/)

This project delivers an end-to-end NLP pipeline that processes video or audio content, transcribes spoken language, and classifies the emotional content. Built with modern ML/AI techniques and deployed on Azure using MLOps principles, the system enables:

- **Automated transcription** of video/audio content
- **Accurate emotion classification** from text
- **Scalable cloud deployment** with monitoring
- **CI/CD integration** for streamlined updates

<br>

## Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Contributing Guide](#-contributing-guide)
- [License](#-license)

<br>

## 📁 Project Structure

```
2024-25d-fai2-adsai-group-nlp6/
├── src/                   # Source code for the project
│   ├── __init__.py        # Makes src a Python package
│   ├── data.py            # Functions for loading and preprocessing data
│   ├── model.py           # Model architecture definition
│   ├── train.py           # Training and evaluation logic
│   └── predict.py         # Functions for loading model and making predictions
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
├── LICENSE                # Project license
├── README.md              # Project overview and instructions
└── pyproject.toml         # Project metadata and dependencies (Python packaging)
```

<br>

## 🚀 Installation

This project uses Poetry for dependency management. To install and set up the project:

```bash
# Specify Python 3.12 for project
poetry env use python3.12

# Install dependencies
poetry install
```

To activate the Poetry virtual environment:

```bash
poetry shell
```

<br>

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

<br>

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
