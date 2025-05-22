# Emotion Classification Full-Stack Application

This project provides an end-to-end system for analyzing emotions in YouTube videos. It consists of a Python backend for transcription and emotion classification, and a React frontend for user interaction and visualization.

## Project Components

- **Backend (`./src/emotion_clf_pipeline`):** A Python-based NLP pipeline using FastAPI. It downloads audio from YouTube, transcribes it, and performs emotion classification on the text.
  - For detailed backend documentation, see `src/emotion_clf_pipeline/README.md`.
- **Frontend (`./frontend`):** A React application that allows users to input a YouTube URL, view the video, see the transcribed text with emotion highlighting, and explore emotion visualizations.
  - For detailed frontend documentation, see `frontend/README.md`.

## Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop/) and [Docker Compose](https://docs.docker.com/compose/install/) installed on your system.
- An AssemblyAI API key (if using AssemblyAI for transcription). See step 2 below.

## Running the Full Application (Backend + Frontend)

This is the recommended way to run the entire application.

1.  **Clone the Repository (if you haven't already):**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Configure Backend API Keys (AssemblyAI):**
    The backend uses AssemblyAI for audio transcription by default, which requires an API key.
    *   Create a file named `.env` in the project root directory (alongside `docker-compose.yml`).
    *   Add your AssemblyAI API key to this file:
        ```env
        ASSEMBLYAI_API_KEY="your_actual_assemblyai_api_key"
        ```
    *   You can create a `.env.example` file to show the format (see below).
    *   **Important:** Ensure `.env` is listed in your `.gitignore` file to prevent committing your secret key.

3.  **Build and Run with Docker Compose:**
    Open a terminal in the project root directory (where `docker-compose.yml` is located) and run:
    ```bash
    docker-compose up --build
    ```
    *   `docker-compose up` starts both the backend and frontend services.
    *   `--build` ensures that Docker images for both services are built (or rebuilt if they've changed).

4.  **Accessing the Application:**
    Once the services are up and running:
    *   The **Frontend UI** will be accessible at: `http://localhost:3000`
    *   The **Backend API** will be accessible at: `http://localhost:8000`
        *   You can test the backend directly, e.g., `POST http://localhost:8000/predict` with JSON `{"url": "<youtube_url>"}`.

5.  **Stopping the Application:**
    To stop the services, press `Ctrl+C` in the terminal where Docker Compose is running. To remove the containers and network created by Compose, run:
    ```bash
    docker-compose down
    ```

## Development Notes

- **Backend Hot Reloading:** The `docker-compose.yml` for the backend mounts the `./src` directory. If you're running Uvicorn with `--reload` (as configured in the backend `Dockerfile` by default for development), changes to the backend Python code should trigger an automatic reload of the backend service within Docker.
- **Frontend Development:** For more intensive frontend development, you might prefer to run the React development server (`npm start`) directly on your host machine in the `frontend` directory, while running the backend via Docker Compose. Ensure your frontend's `API_BASE_URL` in `frontend/src/api.js` still points to `http://localhost:8000`.

## Project Structure Overview

```
./
├── Dockerfile             # For the backend service
├── docker-compose.yml     # Defines how to run both services
├── frontend/
│   ├── Dockerfile         # For the frontend service
│   ├── README.md          # Frontend specific documentation
│   └── ...                # React app source code
├── src/
│   └── emotion_clf_pipeline/
│       ├── README.md      # Backend specific documentation
│       └── ...            # Python backend source code
├── .env                   # (To be created by user) API keys, etc.
├── .env.example           # Example for .env file structure
├── README.md              # This file (main project overview)
└── ...                    # Other project files (pyproject.toml, etc.)
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
