# 🚀 Development Commands Reference

This document contains all the common commands used for developing, running, and deploying the Emotion Classification Pipeline.

## 📋 Table of Contents

- [🏃‍♂️ Quick Start](#quick-start)
- [🖥️ Local Development](#local-development)
- [☁️ Azure ML Operations](#azure-ml-operations)
- [⏰ Schedule Management](#schedule-management)
- [🎯 Hyperparameter Tuning](#hyperparameter-tuning)
- [🚀 Deployment](#deployment)
- [🔧 Code Quality](#code-quality)
- [🐳 Docker Commands](#docker-commands)
- [📊 Monitoring & Logs](#monitoring--logs)

---

## 🏃‍♂️ Quick Start

### Start Full Application Stack

```bash
# Start backend API (Port 3120)
uvicorn src.emotion_clf_pipeline.api:app --host 0.0.0.0 --port 3120 --reload

# Start frontend (Port 3121) - In a new terminal
cd frontend
set PORT=3121 && npm start
```

### Alternative: Docker Compose
```bash
# Start both backend and frontend with Docker
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## 🖥️ Local Development

### Data Pipeline

```bash
# Process raw data locally
python -m src.emotion_clf_pipeline.cli preprocess \
  --verbose \
  --raw-train-path "data/raw/train" \
  --raw-test-path "data/raw/test/test_data-0001.csv"
```

### Training & Evaluation

```bash
# Train model locally (quick test)
python -m src.emotion_clf_pipeline.cli train \
  --verbose \
  --epochs 1 \
  --batch-size 8

# Full training run
python -m src.emotion_clf_pipeline.cli train \
  --verbose \
  --epochs 50 \
  --batch-size 16
```

### Development Server

```bash
# Start backend with auto-reload
uvicorn src.emotion_clf_pipeline.api:app \
  --host 0.0.0.0 \
  --port 3120 \
  --reload \
  --log-level debug

# Alternative with gunicorn (production-like)
gunicorn src.emotion_clf_pipeline.api:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:3120
```

### Frontend Development

```bash
# Navigate to frontend and install dependencies
cd frontend
npm install

# Start development server
set PORT=3121 && npm start

# Build for production
npm run build

# Run tests
npm test

# Run linting
npm run lint
```

---

## ☁️ Azure ML Operations

### Data Pipeline in Azure

```bash
# Process data in Azure ML with automatic registration
poetry run python -m emotion_clf_pipeline.cli preprocess \
  --azure \
  --register-data-assets \
  --verbose
```

### Training in Azure

```bash
# Training and evaluation pipeline in Azure ML
poetry run python -m emotion_clf_pipeline.cli train \
  --azure \
  --verbose

# Full pipeline (data + training)
poetry run python -m emotion_clf_pipeline.cli train-pipeline \
  --azure \
  --verbose
```

### Azure Configuration

```bash
# Set up Azure ML workspace (if not already configured)
az login
az account set --subscription "your-subscription-id"

# Configure Azure ML CLI
az configure --defaults group=your-resource-group workspace=your-workspace
```

---

## ⏰ Schedule Management

### Create Schedules

```bash
# Daily schedule at midnight UTC
python -m src.emotion_clf_pipeline.cli schedule create \
  --schedule-name 'scheduled-deberta-full-pipeline' \
  --daily \
  --hour 0 \
  --minute 0 \
  --enabled \
  --mode azure

# Weekly schedule on Sundays at 2 AM
python -m src.emotion_clf_pipeline.cli schedule create \
  --schedule-name 'weekly-sunday-training' \
  --weekly 0 \
  --hour 2 \
  --minute 0 \
  --enabled \
  --mode azure

# Monthly schedule on the 1st at 3 AM
python -m src.emotion_clf_pipeline.cli schedule create \
  --schedule-name 'monthly-first-training' \
  --monthly 1 \
  --hour 3 \
  --minute 0 \
  --enabled \
  --mode azure
```

### Manage Schedules

```bash
# List all schedules
python -m src.emotion_clf_pipeline.cli schedule list --mode azure

# Setup default schedule patterns
python -m src.emotion_clf_pipeline.cli schedule setup-defaults --mode azure

# Disable a schedule
python -m src.emotion_clf_pipeline.cli schedule disable \
  --schedule-name 'scheduled-deberta-full-pipeline' \
  --mode azure

# Delete a schedule
python -m src.emotion_clf_pipeline.cli schedule delete \
  --schedule-name 'old-schedule-name' \
  --mode azure
```

---

## 🎯 Hyperparameter Tuning

### Run Hyperparameter Optimization

```bash
# Run hyperparameter tuning in Azure (recommended)
python src/emotion_clf_pipeline/hyperparameter_tuning.py

# Local hyperparameter tuning (limited resources)
python src/emotion_clf_pipeline/hyperparameter_tuning.py --local

# Tune specific parameters
python src/emotion_clf_pipeline/hyperparameter_tuning.py \
  --learning-rate-range 1e-5 1e-3 \
  --batch-size-options 8 16 32 \
  --epochs 20
```

### View Tuning Results

```bash
# Check hyperparameter tuning database
sqlite3 models/hpt/emotion-clf-hpt.db

# View MLflow experiments
mlflow ui --host 0.0.0.0 --port 5000
```

---

## 🚀 Deployment

### Azure Endpoint Deployment

```bash
# Deploy model to Azure endpoint
python -m src.emotion_clf_pipeline.cli endpoint deploy \
  --model-name emotion-clf-baseline \
  --model-version 2 \
  --endpoint-name deberta-endpoint \
  --environment emotion-clf-pipeline-env:30 \
  --instance-type Standard_D2ads_v6

# Deploy with custom configuration
python -m src.emotion_clf_pipeline.cli endpoint deploy \
  --model-name emotion-clf-baseline \
  --model-version latest \
  --endpoint-name production-endpoint \
  --environment emotion-clf-pipeline-env:latest \
  --instance-type Standard_D4ads_v5 \
  --min-instances 2 \
  --max-instances 10
```

### Test Deployed Endpoint

```bash
# Test local endpoint
python test_endpoint.py

# Test Azure endpoint
python test_endpoint.py --endpoint-url "https://your-endpoint.region.inference.ml.azure.com"

# Load test endpoint
python test_endpoint.py --load-test --concurrent-requests 10 --duration 60
```

### Docker Deployment

```bash
# Build Docker image
docker build -t emotion-clf-pipeline:latest .

# Run container locally
docker run -p 3120:3120 emotion-clf-pipeline:latest

# Deploy to Azure Container Instances
az container create \
  --resource-group your-rg \
  --name emotion-clf-aci \
  --image your-registry/emotion-clf-pipeline:latest \
  --ports 3120 \
  --environment-variables KEY=value
```

---

## 🔧 Code Quality

### Formatting

```bash
# Format Python code with Black
python -m black --line-length 88 src/emotion_clf_pipeline/
python -m black --line-length 88 tests/
python -m black --line-length 88 "src\emotion_clf_pipeline\score.py"

# Format specific file
python -m black --line-length 88 --check src/emotion_clf_pipeline/api.py
```

### Linting

```bash
# Check with flake8
python -m flake8 --max-line-length=88 --extend-ignore=E203,W503 src/
python -m flake8 --max-line-length=88 --extend-ignore=E203,W503 tests/
python -m flake8 --max-line-length=88 --extend-ignore=E203,W503 "src\emotion_clf_pipeline\score.py"

# Run with specific configuration
python -m flake8 --config=.flake8 src/emotion_clf_pipeline/
```

### Type Checking

```bash
# Type checking with mypy
python -m mypy src/emotion_clf_pipeline/
python -m mypy --strict src/emotion_clf_pipeline/api.py
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
pre-commit run flake8 --all-files
```

---

## 🐳 Docker Commands

### Local Development

```bash
# Build development image
docker build -t emotion-clf-dev -f Dockerfile.dev .

# Run with volume mounting for development
docker run -it \
  -v ${PWD}:/app \
  -p 3120:3120 \
  emotion-clf-dev

# Interactive shell in container
docker run -it emotion-clf-dev /bin/bash
```

### Production

```bash
# Build production image
docker build -t emotion-clf-pipeline:latest .

# Run production container
docker run -d \
  --name emotion-clf-app \
  -p 3120:3120 \
  --restart unless-stopped \
  emotion-clf-pipeline:latest

# View container logs
docker logs -f emotion-clf-app

# Execute commands in running container
docker exec -it emotion-clf-app python -m emotion_clf_pipeline.cli --help
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d backend
docker-compose up -d frontend

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Scale services
docker-compose up -d --scale backend=3

# Stop and remove
docker-compose down
docker-compose down --volumes  # Also remove volumes
```

---

## 📊 Monitoring & Logs

### Application Logs

```bash
# View real-time logs
tail -f logs/app.log
tail -f logs/training.log
tail -f logs/api.log

# Search logs
grep "ERROR" logs/app.log
grep -i "exception" logs/*.log

# Log rotation and cleanup
find logs/ -name "*.log" -mtime +7 -delete
```

### MLflow Tracking

```bash
# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# MLflow with specific backend
mlflow ui \
  --backend-store-uri sqlite:///mlruns.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```

### System Monitoring

```bash
# Monitor system resources
htop
nvidia-smi  # For GPU monitoring

# Monitor Docker containers
docker stats
docker system df  # Check disk usage
docker system prune  # Clean up unused resources
```

### Performance Testing

```bash
# API performance testing with curl
curl -X POST "http://localhost:3120/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling great today!"}'

# Load testing with Apache Bench
ab -n 1000 -c 10 http://localhost:3120/health

# Performance profiling
python -m cProfile -o profile.stats src/emotion_clf_pipeline/train.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats()"
```

---

## 📝 Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Azure Configuration
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=your-resource-group
AZURE_WORKSPACE_NAME=your-workspace
AZURE_TENANT_ID=your-tenant-id

# API Configuration
API_HOST=0.0.0.0
API_PORT=3120
DEBUG=true

# Model Configuration
MODEL_NAME=emotion-clf-baseline
MODEL_VERSION=latest

# Database Configuration
DATABASE_URL=sqlite:///models/hpt/emotion-clf-hpt.db

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

---

## 🆘 Troubleshooting

### Common Issues

1. **Port already in use**:
   ```bash
   # Kill process using port 3120
   netstat -ano | findstr :3120
   taskkill /PID <PID> /F
   ```

2. **Poetry dependency issues**:
   ```bash
   # Clear poetry cache
   poetry cache clear --all pypi
   
   # Reinstall dependencies
   poetry install --no-cache
   ```

3. **Azure authentication**:
   ```bash
   # Re-authenticate with Azure
   az logout
   az login
   az account set --subscription "your-subscription-id"
   ```

4. **Docker build failures**:
   ```bash
   # Clean Docker cache
   docker system prune -a
   
   # Build without cache
   docker build --no-cache -t emotion-clf-pipeline:latest .
   ```

### Performance Optimization

```bash
# Enable GPU acceleration (if available)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Monitor memory usage during training
python -m memory_profiler src/emotion_clf_pipeline/train.py

# Optimize batch size for available memory
python -m src.emotion_clf_pipeline.cli train --auto-batch-size
```

---

## 📚 Additional Resources

- [Project Documentation](https://bredauniversityadsai.github.io/2024-25d-fai2-adsai-group-nlp6/)
- [Azure ML Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [Docker Documentation](https://docs.docker.com/)

---

*This document is automatically generated and maintained. Last updated: June 2025*
