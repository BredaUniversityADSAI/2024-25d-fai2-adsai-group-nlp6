# 1. Use an official Python runtime as a parent image
# Using the slim variant for a smaller image size
FROM python:3.9-slim

# 2. Set environment variables
# Prevents Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE=1
# Ensures Python output is sent straight to terminal without being buffered
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory in the container
WORKDIR /app
 
# 4. Install Poetry
# We install it globally in the image
RUN pip install poetry==1.8.3 # Pinning version for consistency, adjust if needed

# 5. Configure Poetry
# Disable virtualenv creation as we are managing the environment within the container
RUN poetry config virtualenvs.create false

# 6. Copy dependency definition files
# Copy only these files first to leverage Docker cache for dependency installation layer
COPY pyproject.toml ./

# 7. Install project dependencies
# --no-root: Don't install the project package itself
# --only main: Install only dependencies specified under [tool.poetry.dependencies]
#             Excludes dev dependencies and optional groups like [cpu], [cuda]
RUN poetry install --no-interaction --no-ansi --no-root --only main

# 8. Copy the application source code into the container
COPY ./src /app/src

# 9. Expose the port the app runs on
# Matches the port used in the ENTRYPOINT command
EXPOSE 80

# 10. Set the startup command to run the API
# Runs Uvicorn, pointing to the FastAPI app instance inside api.py
# --host 0.0.0.0 makes it accessible from outside the container
ENTRYPOINT ["uvicorn", "src.emotion_clf_pipeline.api:app", "--host", "0.0.0.0", "--port", "80"] 