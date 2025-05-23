# 1. Use an official Python runtime as a parent image
# Using slim variant for smaller image size
FROM python:3.11-slim

# 2. Set environment variables
# Prevents Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Ensures Python output is sent straight to terminal
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory
WORKDIR /app

# 4. Install Poetry
# Install globally in the image
RUN pip install poetry==1.8.3 # Pinning version

# 5. Configure Poetry
# Disable virtualenv creation (managing environment in container)
RUN poetry config virtualenvs.create false

# 6. Copy dependency definition files
# Copy first to leverage Docker cache for dependency installation
COPY pyproject.toml ./

# 7. Install project dependencies
# --no-root: Don't install the project package
# --only main: Install only main dependencies (excludes dev/optional)
RUN poetry install --no-interaction --no-ansi --no-root --only main

# 7a. Download NLTK resources
ENV NLTK_DATA=/app/nltk_data
RUN python -m nltk.downloader -d /app/nltk_data vader_lexicon punkt averaged_perceptron_tagger punkt_tab

# 8. Copy application source code and model files
COPY ./src /app/src
COPY ./models /app/models
COPY ./.env /app/.env

# 9. Expose port
# Matches port in ENTRYPOINT
EXPOSE 80

# 10. Set startup command
# Runs Uvicorn, pointing to FastAPI app in api.py
# --host 0.0.0.0: accessible from outside container
ENTRYPOINT ["uvicorn", "src.emotion_clf_pipeline.api:app", "--host", "0.0.0.0", "--port", "80"]
