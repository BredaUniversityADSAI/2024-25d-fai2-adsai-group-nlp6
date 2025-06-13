# System Architecture - Emotion Classification Pipeline

## 🏗️ Overview

The Emotion Classification Pipeline is a sophisticated, cloud-native system that analyzes emotional content from YouTube videos through advanced natural language processing. The system follows a microservices architecture with AI/ML capabilities, real-time processing, and comprehensive visualization features.

---

## 🏛️ High-Level System Architecture

### System Components Overview

```mermaid
graph TB
    subgraph "User Interface Layer"
        WEB[Web Frontend<br/>React + Material-UI]
        CLI[Command Line<br/>Interface]
        API_CLIENT[API Clients<br/>cURL/Postman]
    end
    
    subgraph "API Gateway Layer"
        FASTAPI[FastAPI Backend<br/>Port 3120]
        CORS[CORS Middleware]
        AUTH[Authentication<br/>& Validation]
    end
    
    subgraph "Core Processing Services"
        ORCHESTRATOR[Prediction Orchestrator<br/>predict.py]
        DOWNLOADER[Video Downloader<br/>YouTube/PyTube]
        STT_SERVICE[Speech-to-Text<br/>AssemblyAI + Whisper]
        ML_SERVICE[Emotion Classifier<br/>DeBERTa Multi-task]
        FEATURE_ENG[Feature Engineering<br/>NLP + Linguistic]
    end
    
    subgraph "ML Training Pipeline"
        TRAINER[Model Trainer<br/>train.py]
        EVALUATOR[Model Evaluator<br/>Metrics & Validation]
        MODEL_REG[Model Registry<br/>MLflow + Azure ML]
    end
    
    subgraph "Data Layer"
        FILE_SYS[(Local File System<br/>Audio, Transcripts, Results)]
        MODEL_STORE[(Model Storage<br/>Weights & Encoders)]
        CACHE[(Redis Cache<br/>Optional)]
    end
    
    subgraph "External Services"
        YOUTUBE[YouTube API]
        ASSEMBLY[AssemblyAI API]
        AZURE_ML[Azure ML<br/>Pipeline Execution]
        AZURE_STORAGE[Azure Blob<br/>Storage]
    end
    
    subgraph "Infrastructure"
        DOCKER[Docker Containers]
        COMPOSE[Docker Compose<br/>Orchestration]
        NGINX[Nginx<br/>Frontend Serving]
    end

    %% User Interface Connections
    WEB --> FASTAPI
    CLI --> ORCHESTRATOR
    API_CLIENT --> FASTAPI
    
    %% API Gateway Connections
    FASTAPI --> CORS
    FASTAPI --> AUTH
    FASTAPI --> ORCHESTRATOR
    
    %% Core Processing Connections
    ORCHESTRATOR --> DOWNLOADER
    ORCHESTRATOR --> STT_SERVICE
    ORCHESTRATOR --> ML_SERVICE
    ML_SERVICE --> FEATURE_ENG
    
    %% Training Pipeline Connections
    TRAINER --> EVALUATOR
    TRAINER --> MODEL_REG
    EVALUATOR --> MODEL_REG
    
    %% Data Layer Connections
    ORCHESTRATOR --> FILE_SYS
    ML_SERVICE --> MODEL_STORE
    TRAINER --> MODEL_STORE
    
    %% External Service Connections
    DOWNLOADER --> YOUTUBE
    STT_SERVICE --> ASSEMBLY
    TRAINER --> AZURE_ML
    MODEL_REG --> AZURE_STORAGE
    
    %% Infrastructure Connections
    WEB --> NGINX
    FASTAPI --> DOCKER
    NGINX --> DOCKER
    DOCKER --> COMPOSE

    %% Styling
    classDef uiLayer fill:#E1F5FE,stroke:#0277BD,stroke-width:2px
    classDef apiLayer fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px
    classDef coreService fill:#E8F5E8,stroke:#2E7D32,stroke-width:2px
    classDef mlLayer fill:#FFF3E0,stroke:#F57C00,stroke-width:2px
    classDef dataLayer fill:#FFEBEE,stroke:#C62828,stroke-width:2px
    classDef external fill:#F9FBE7,stroke:#827717,stroke-width:2px
    classDef infra fill:#EFEBE9,stroke:#5D4037,stroke-width:2px
    
    class WEB,CLI,API_CLIENT uiLayer
    class FASTAPI,CORS,AUTH apiLayer
    class ORCHESTRATOR,DOWNLOADER,STT_SERVICE,ML_SERVICE,FEATURE_ENG coreService
    class TRAINER,EVALUATOR,MODEL_REG mlLayer
    class FILE_SYS,MODEL_STORE,CACHE dataLayer
    class YOUTUBE,ASSEMBLY,AZURE_ML,AZURE_STORAGE external
    class DOCKER,COMPOSE,NGINX infra
```

---

## 📊 Data Flow Architecture

### End-to-End Processing Pipeline

```mermaid
sequenceDiagram
    participant User as 👤 User
    participant Frontend as 🌐 React Frontend
    participant API as 🚀 FastAPI Backend
    participant Orchestrator as 🎭 Prediction Service
    participant Downloader as 📥 Video Downloader
    participant STT as 🎤 Speech-to-Text
    participant AI as 🧠 Emotion Classifier
    participant Storage as 💾 File System
    
    User->>Frontend: Input YouTube URL
    Frontend->>API: POST /predict {url}
    
    activate API
    API->>Orchestrator: process_youtube_url_and_predict()
    
    activate Orchestrator
    Orchestrator->>Downloader: download_audio(youtube_url)
    
    activate Downloader
    Downloader->>Storage: Save audio.mp3
    Downloader-->>Orchestrator: audio_file_path
    deactivate Downloader
    
    Orchestrator->>STT: transcribe_audio(audio_path)
    
    activate STT
    STT->>Storage: Save transcript.json
    STT-->>Orchestrator: transcript_segments[]
    deactivate STT
    
    loop For each transcript segment
        Orchestrator->>AI: predict_emotion(text_segment)
        
        activate AI
        AI->>AI: Feature extraction
        AI->>AI: DeBERTa inference
        AI->>AI: Multi-task prediction
        AI-->>Orchestrator: {emotion, sub_emotion, intensity}
        deactivate AI
    end
    
    Orchestrator->>Storage: Save results.json
    Orchestrator-->>API: Structured predictions
    deactivate Orchestrator
    
    API-->>Frontend: JSON response with timeline
    deactivate API
    
    Frontend->>Frontend: Render visualizations
    Frontend->>User: Display emotion analysis
```

---

## 🧩 Microservices Architecture

### Service Decomposition

```mermaid
graph TD
    subgraph "Frontend Service"
        A[React Application<br/>Port: 3121]
        A1[Components Layer]
        A2[State Management<br/>VideoContext]
        A3[API Communication]
        A --> A1
        A --> A2
        A --> A3
    end
    
    subgraph "Backend Service"
        B[FastAPI Application<br/>Port: 3120]
        B1[API Endpoints<br/>/predict, /feedback]
        B2[Request Validation<br/>Pydantic Models]
        B3[Error Handling<br/>& Logging]
        B --> B1
        B --> B2
        B --> B3
    end
    
    subgraph "Video Processing Service"
        C[Video Orchestrator]
        C1[YouTube Integration<br/>PyTubefix]
        C2[Audio Extraction<br/>FFmpeg]
        C3[File Management]
        C --> C1
        C --> C2
        C --> C3
    end
    
    subgraph "Transcription Service"
        D[Speech-to-Text Engine]
        D1[AssemblyAI Client<br/>Primary STT]
        D2[Whisper Model<br/>Fallback STT]
        D3[Transcript Processing]
        D --> D1
        D --> D2
        D --> D3
    end
    
    subgraph "ML Inference Service"
        E[Emotion Classifier]
        E1[DeBERTa Model<br/>Transformer]
        E2[Feature Engineering<br/>NLP Pipeline]
        E3[Multi-task Prediction<br/>Emotion/Sub/Intensity]
        E --> E1
        E --> E2
        E --> E3
    end
    
    subgraph "Training Service"
        F[ML Training Pipeline]
        F1[Data Preprocessing]
        F2[Model Training<br/>Multi-task Learning]
        F3[Model Evaluation<br/>& Registration]
        F --> F1
        F --> F2
        F --> F3
    end
    
    subgraph "Data Services"
        G[Data Management]
        G1[Dataset Loading<br/>CSV Processing]
        G2[Feature Extraction<br/>Linguistic Analysis]
        G3[Data Validation<br/>& Cleaning]
        G --> G1
        G --> G2
        G --> G3
    end
    
    %% Service Interactions
    A3 -.->|HTTP/REST| B1
    B1 -->|Orchestrate| C
    C -->|Audio Data| D
    D -->|Transcript| E
    E -->|Predictions| C
    F -->|Models| E
    G -->|Training Data| F
    
    %% External Dependencies
    C1 -.->|API Calls| YT[YouTube]
    D1 -.->|API Calls| ASM[AssemblyAI]
    F3 -.->|Model Registry| AZ[Azure ML]
    
    %% Styling
    classDef frontend fill:#E3F2FD,stroke:#1976D2,stroke-width:2px
    classDef backend fill:#E8F5E8,stroke:#388E3C,stroke-width:2px
    classDef processing fill:#FFF3E0,stroke:#F57C00,stroke-width:2px
    classDef ml fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px
    classDef data fill:#FFEBEE,stroke:#D32F2F,stroke-width:2px
    classDef external fill:#F1F8E9,stroke:#689F38,stroke-width:2px
    
    class A,A1,A2,A3 frontend
    class B,B1,B2,B3 backend
    class C,C1,C2,C3,D,D1,D2,D3 processing
    class E,E1,E2,E3,F,F1,F2,F3 ml
    class G,G1,G2,G3 data
    class YT,ASM,AZ external
```

---

## 🛠️ Component Architecture

### Backend Module Structure

```mermaid
graph LR
    subgraph "src/emotion_clf_pipeline"
        API[api.py<br/>🚀 FastAPI Endpoints]
        CLI[cli.py<br/>💻 Command Interface]
        PREDICT[predict.py<br/>🎭 Main Orchestrator]
        MODEL[model.py<br/>🧠 AI/ML Models]
        DATA[data.py<br/>📊 Data Processing]
        TRAIN[train.py<br/>🏋️ Model Training]
        STT[stt.py<br/>🎤 Speech-to-Text]
        FEATURES[features.py<br/>🔤 Feature Engineering]
        AZURE[azure_pipeline.py<br/>☁️ Cloud Integration]
        TRANSCRIPT[transcript.py<br/>📝 Text Processing]
    end
    
    %% Main Entry Points
    API --> PREDICT
    CLI --> PREDICT
    CLI --> TRAIN
    CLI --> DATA
    
    %% Core Processing Flow
    PREDICT --> MODEL
    PREDICT --> STT
    PREDICT --> DATA
    
    %% Training Pipeline
    TRAIN --> MODEL
    TRAIN --> DATA
    TRAIN --> FEATURES
    
    %% Model Dependencies
    MODEL --> FEATURES
    MODEL --> DATA
    
    %% Azure Integration
    TRAIN --> AZURE
    DATA --> AZURE
    
    %% Speech Processing
    STT --> TRANSCRIPT
    
    %% Styling
    classDef entryPoint fill:#E1F5FE,stroke:#0277BD,stroke-width:3px
    classDef coreLogic fill:#E8F5E8,stroke:#2E7D32,stroke-width:2px
    classDef mlComponent fill:#FFF3E0,stroke:#F57C00,stroke-width:2px
    classDef dataComponent fill:#FFEBEE,stroke:#C62828,stroke-width:2px
    classDef cloudComponent fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px
    
    class API,CLI entryPoint
    class PREDICT,STT coreLogic
    class MODEL,TRAIN,FEATURES mlComponent
    class DATA,TRANSCRIPT dataComponent
    class AZURE cloudComponent
```

### Frontend Component Hierarchy

```mermaid
graph TD
    subgraph "React Application Structure"
        APP[App.js<br/>🎯 Main Application]
        CONTEXT[VideoContext.js<br/>🔄 State Management]
        
        subgraph "Core Components"
            URLGPT[UrlInput.js<br/>📝 URL Input]
            PLAYER[VideoPlayer.js<br/>📺 Video Playback]
            TRANSCRIPT[Transcript.js<br/>📜 Text Display]
            HISTORY[VideoHistory.js<br/>📋 Analysis History]
        end
        
        subgraph "Visualization Components"
            EMO_CUR[EmotionCurrent.js<br/>💫 Live Emotion Display]
            EMO_BAR[EmotionBarChart.js<br/>📊 Emotion Distribution]
            EMO_TIME[EmotionTimeline.js<br/>📈 Timeline Visualization]
            FEEDBACK[FeedbackModal.js<br/>💬 User Feedback]
        end
        
        subgraph "Utility Components"
            SEARCH[SearchBar.js<br/>🔍 Search Interface]
            MEMORY[VideoMemoryHeader.js<br/>🧠 Memory Header]
            EMPTY[EmptyState.js<br/>🌟 Empty State UI]
        end
        
        subgraph "Services & Utils"
            API_SVC[api.js<br/>🌐 API Communication]
            UTILS[utils.js<br/>🛠️ Utility Functions]
        end
    end
    
    %% Component Relationships
    APP --> CONTEXT
    APP --> URLGPT
    APP --> PLAYER
    APP --> TRANSCRIPT
    APP --> HISTORY
    APP --> EMO_CUR
    APP --> EMO_BAR
    APP --> EMO_TIME
    APP --> FEEDBACK
    
    CONTEXT --> API_SVC
    TRANSCRIPT --> SEARCH
    HISTORY --> MEMORY
    APP --> EMPTY
    
    EMO_CUR --> UTILS
    EMO_BAR --> UTILS
    EMO_TIME --> UTILS
    
    %% Styling
    classDef mainApp fill:#E1F5FE,stroke:#0277BD,stroke-width:3px
    classDef coreComp fill:#E8F5E8,stroke:#2E7D32,stroke-width:2px
    classDef vizComp fill:#FFF3E0,stroke:#F57C00,stroke-width:2px
    classDef utilComp fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px
    classDef service fill:#FFEBEE,stroke:#C62828,stroke-width:2px
    
    class APP,CONTEXT mainApp
    class URLGPT,PLAYER,TRANSCRIPT,HISTORY coreComp
    class EMO_CUR,EMO_BAR,EMO_TIME,FEEDBACK vizComp
    class SEARCH,MEMORY,EMPTY utilComp
    class API_SVC,UTILS service
```

---

## 🤖 AI/ML Architecture

### Machine Learning Pipeline

```mermaid
graph TB
    subgraph "Data Ingestion Layer"
        RAW_DATA[Raw Emotion Dataset<br/>CSV Files]
        YT_VIDEO[YouTube Video Input<br/>URL Processing]
        AUDIO_EXT[Audio Extraction<br/>FFmpeg Processing]
    end
    
    subgraph "Data Processing Layer"
        CLEAN[Data Cleaning<br/>Text Normalization]
        SPLIT[Train/Validation Split<br/>Stratified Sampling]
        TOKENIZE[Tokenization<br/>DeBERTa Tokenizer]
        FEATURE[Feature Engineering<br/>NLP + Linguistic]
    end
    
    subgraph "Feature Engineering Pipeline"
        POS[POS Tagging<br/>NLTK]
        SENTIMENT[Sentiment Analysis<br/>VADER + TextBlob]
        LEXICON[Emotion Lexicon<br/>EmoLex Features]
        TFIDF[TF-IDF Vectors<br/>Term Frequency]
        CONCAT[Feature Concatenation<br/>Multi-modal Fusion]
    end
    
    subgraph "Model Architecture"
        DEBERTA[DeBERTa Transformer<br/>Base Model]
        PROJECTION[Projection Layers<br/>Feature Integration]
        
        subgraph "Multi-task Heads"
            EMOTION_HEAD[Emotion Classification<br/>7 Classes]
            SUB_HEAD[Sub-emotion Classification<br/>28 Classes]
            INTENSITY_HEAD[Intensity Prediction<br/>3 Levels]
        end
        
        LOSS[Multi-task Loss<br/>Weighted Combination]
    end
    
    subgraph "Training Infrastructure"
        OPTIM[AdamW Optimizer<br/>Learning Rate Scheduling]
        VALID[Validation Loop<br/>Early Stopping]
        CHECKPT[Model Checkpointing<br/>Best Model Selection]
        METRICS[Evaluation Metrics<br/>F1, Precision, Recall]
    end
    
    subgraph "Model Deployment"
        PREDICTOR[Emotion Predictor<br/>Inference Engine]
        REGISTRY[Model Registry<br/>MLflow + Azure ML]
        VERSIONING[Model Versioning<br/>A/B Testing Ready]
    end
    
    %% Data Flow
    RAW_DATA --> CLEAN
    YT_VIDEO --> AUDIO_EXT
    AUDIO_EXT --> CLEAN
    
    CLEAN --> SPLIT
    SPLIT --> TOKENIZE
    TOKENIZE --> FEATURE
    
    %% Feature Engineering Flow
    FEATURE --> POS
    FEATURE --> SENTIMENT
    FEATURE --> LEXICON
    FEATURE --> TFIDF
    POS --> CONCAT
    SENTIMENT --> CONCAT
    LEXICON --> CONCAT
    TFIDF --> CONCAT
    
    %% Model Training Flow
    CONCAT --> DEBERTA
    TOKENIZE --> DEBERTA
    DEBERTA --> PROJECTION
    PROJECTION --> EMOTION_HEAD
    PROJECTION --> SUB_HEAD
    PROJECTION --> INTENSITY_HEAD
    
    EMOTION_HEAD --> LOSS
    SUB_HEAD --> LOSS
    INTENSITY_HEAD --> LOSS
    
    %% Training Infrastructure
    LOSS --> OPTIM
    OPTIM --> VALID
    VALID --> CHECKPT
    CHECKPT --> METRICS
    
    %% Deployment
    CHECKPT --> PREDICTOR
    PREDICTOR --> REGISTRY
    REGISTRY --> VERSIONING
    
    %% Styling
    classDef dataLayer fill:#E3F2FD,stroke:#1565C0,stroke-width:2px
    classDef processLayer fill:#E8F5E8,stroke:#2E7D32,stroke-width:2px
    classDef featureLayer fill:#FFF3E0,stroke:#EF6C00,stroke-width:2px
    classDef modelLayer fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px
    classDef trainLayer fill:#FFEBEE,stroke:#C62828,stroke-width:2px
    classDef deployLayer fill:#E0F2F1,stroke:#00695C,stroke-width:2px
    
    class RAW_DATA,YT_VIDEO,AUDIO_EXT dataLayer
    class CLEAN,SPLIT,TOKENIZE,FEATURE processLayer
    class POS,SENTIMENT,LEXICON,TFIDF,CONCAT featureLayer
    class DEBERTA,PROJECTION,EMOTION_HEAD,SUB_HEAD,INTENSITY_HEAD,LOSS modelLayer
    class OPTIM,VALID,CHECKPT,METRICS trainLayer
    class PREDICTOR,REGISTRY,VERSIONING deployLayer
```

### Model Architecture Details

```mermaid
graph LR
    subgraph "Input Processing"
        TEXT[Text Input<br/>Sentence/Segment]
        TOKENS[DeBERTa Tokenizer<br/>Subword Tokenization]
    end
    
    subgraph "Feature Extraction Parallel Paths"
        subgraph "Transformer Path"
            DEBERTA[DeBERTa Model<br/>microsoft/deberta-v3-base]
            CLS_TOKEN[CLS Token<br/>Sentence Representation]
        end
        
        subgraph "Engineered Features Path"
            POS_FEAT[POS Features<br/>Part-of-Speech Tags]
            SENT_FEAT[Sentiment Features<br/>VADER + TextBlob]
            LEX_FEAT[Lexicon Features<br/>EmoLex Mappings]
        end
    end
    
    subgraph "Feature Fusion"
        PROJECTION[Projection Layer<br/>Linear + Dropout]
        CONCAT[Feature Concatenation<br/>Transformer + Engineered]
        HIDDEN[Hidden Layer<br/>ReLU Activation]
    end
    
    subgraph "Multi-task Output Heads"
        EMO_LINEAR[Emotion Head<br/>Linear(hidden_dim → 7)]
        SUB_LINEAR[Sub-emotion Head<br/>Linear(hidden_dim → 28)]
        INT_LINEAR[Intensity Head<br/>Linear(hidden_dim → 3)]
        
        EMO_SOFTMAX[Emotion Softmax<br/>7 Emotions]
        SUB_SOFTMAX[Sub-emotion Softmax<br/>28 Sub-emotions]
        INT_SOFTMAX[Intensity Softmax<br/>3 Levels]
    end
    
    subgraph "Loss Computation"
        EMO_LOSS[Emotion CE Loss<br/>CrossEntropyLoss]
        SUB_LOSS[Sub-emotion CE Loss<br/>CrossEntropyLoss]
        INT_LOSS[Intensity CE Loss<br/>CrossEntropyLoss]
        WEIGHTED_LOSS[Weighted Multi-task Loss<br/>α*L_emo + β*L_sub + γ*L_int]
    end
    
    %% Input Flow
    TEXT --> TOKENS
    
    %% Parallel Feature Extraction
    TOKENS --> DEBERTA
    TOKENS --> POS_FEAT
    TOKENS --> SENT_FEAT
    TOKENS --> LEX_FEAT
    
    %% Feature Processing
    DEBERTA --> CLS_TOKEN
    
    %% Feature Fusion
    CLS_TOKEN --> PROJECTION
    POS_FEAT --> CONCAT
    SENT_FEAT --> CONCAT
    LEX_FEAT --> CONCAT
    PROJECTION --> CONCAT
    CONCAT --> HIDDEN
    
    %% Multi-task Outputs
    HIDDEN --> EMO_LINEAR
    HIDDEN --> SUB_LINEAR
    HIDDEN --> INT_LINEAR
    
    EMO_LINEAR --> EMO_SOFTMAX
    SUB_LINEAR --> SUB_SOFTMAX
    INT_LINEAR --> INT_SOFTMAX
    
    %% Loss Calculation
    EMO_SOFTMAX --> EMO_LOSS
    SUB_SOFTMAX --> SUB_LOSS
    INT_SOFTMAX --> INT_LOSS
    
    EMO_LOSS --> WEIGHTED_LOSS
    SUB_LOSS --> WEIGHTED_LOSS
    INT_LOSS --> WEIGHTED_LOSS
    
    %% Styling
    classDef inputLayer fill:#E3F2FD,stroke:#1565C0,stroke-width:2px
    classDef transformerPath fill:#E8F5E8,stroke:#2E7D32,stroke-width:2px
    classDef featurePath fill:#FFF3E0,stroke:#EF6C00,stroke-width:2px
    classDef fusionLayer fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px
    classDef outputLayer fill:#FFEBEE,stroke:#C62828,stroke-width:2px
    classDef lossLayer fill:#E0F2F1,stroke:#00695C,stroke-width:2px
    
    class TEXT,TOKENS inputLayer
    class DEBERTA,CLS_TOKEN transformerPath
    class POS_FEAT,SENT_FEAT,LEX_FEAT featurePath
    class PROJECTION,CONCAT,HIDDEN fusionLayer
    class EMO_LINEAR,SUB_LINEAR,INT_LINEAR,EMO_SOFTMAX,SUB_SOFTMAX,INT_SOFTMAX outputLayer
    class EMO_LOSS,SUB_LOSS,INT_LOSS,WEIGHTED_LOSS lossLayer
```

---

## 🌐 Deployment Architecture

### Container Infrastructure

```mermaid
graph TB
    subgraph "Docker Compose Orchestration"
        subgraph "Frontend Container"
            REACT[React Application<br/>Node.js Build]
            NGINX[Nginx Web Server<br/>Port: 80 → 3121]
            STATIC[Static Assets<br/>JS/CSS/Images]
        end
        
        subgraph "Backend Container"
            FASTAPI_CONT[FastAPI Application<br/>uvicorn Server]
            PYTHON_ENV[Python 3.11 Environment<br/>Poetry Dependencies]
            ML_MODELS[Pre-trained Models<br/>DeBERTa Weights]
            VOLUMES[Mounted Volumes<br/>/app/src, /models]
        end
        
        subgraph "Shared Resources"
            NETWORK[emotion_network<br/>Bridge Network]
            ENV_FILE[Environment Variables<br/>.env File]
            MODELS_VOL[Models Volume<br/>Persistent Storage]
        end
    end
    
    subgraph "External Dependencies"
        YOUTUBE_API[YouTube Service<br/>Video Download]
        ASSEMBLY_API[AssemblyAI API<br/>Speech-to-Text]
        AZURE_CLOUD[Azure ML Cloud<br/>Training Pipelines]
        DOCKER_HUB[Docker Hub<br/>Image Registry]
    end
    
    subgraph "GPU Support (Optional)"
        NVIDIA_RUNTIME[NVIDIA Container Runtime]
        CUDA_LIBS[CUDA Libraries<br/>GPU Acceleration]
        GPU_MEMORY[GPU Memory<br/>Model Inference]
    end
    
    %% Container Relationships
    REACT --> NGINX
    NGINX --> STATIC
    FASTAPI_CONT --> PYTHON_ENV
    PYTHON_ENV --> ML_MODELS
    FASTAPI_CONT --> VOLUMES
    
    %% Networking
    NGINX -.->|HTTP Port 3121| NETWORK
    FASTAPI_CONT -.->|HTTP Port 3120| NETWORK
    NETWORK -.->|Internal Communication| NGINX
    NETWORK -.->|Internal Communication| FASTAPI_CONT
    
    %% Environment Configuration
    ENV_FILE -.-> FASTAPI_CONT
    MODELS_VOL -.-> ML_MODELS
    
    %% External Connections
    FASTAPI_CONT -.->|API Calls| YOUTUBE_API
    FASTAPI_CONT -.->|API Calls| ASSEMBLY_API
    FASTAPI_CONT -.->|Pipeline Jobs| AZURE_CLOUD
    DOCKER_HUB -.->|Image Pull| REACT
    DOCKER_HUB -.->|Image Pull| FASTAPI_CONT
    
    %% GPU Support
    FASTAPI_CONT -.->|GPU Access| NVIDIA_RUNTIME
    NVIDIA_RUNTIME --> CUDA_LIBS
    CUDA_LIBS --> GPU_MEMORY
    ML_MODELS -.->|Accelerated Inference| GPU_MEMORY
    
    %% Styling
    classDef frontend fill:#E3F2FD,stroke:#1565C0,stroke-width:2px
    classDef backend fill:#E8F5E8,stroke:#2E7D32,stroke-width:2px
    classDef shared fill:#FFF3E0,stroke:#EF6C00,stroke-width:2px
    classDef external fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px
    classDef gpu fill:#FFEBEE,stroke:#C62828,stroke-width:2px
    
    class REACT,NGINX,STATIC frontend
    class FASTAPI_CONT,PYTHON_ENV,ML_MODELS,VOLUMES backend
    class NETWORK,ENV_FILE,MODELS_VOL shared
    class YOUTUBE_API,ASSEMBLY_API,AZURE_CLOUD,DOCKER_HUB external
    class NVIDIA_RUNTIME,CUDA_LIBS,GPU_MEMORY gpu
```

### Cloud Architecture (Azure ML Integration)

```mermaid
graph TB
    subgraph "Local Development Environment"
        DEV_CLI[Developer CLI<br/>emotion_clf_pipeline.cli]
        LOCAL_DATA[Local Data<br/>CSV Files]
        LOCAL_MODELS[Local Models<br/>Development Testing]
    end
    
    subgraph "Azure ML Workspace"
        WORKSPACE[Azure ML Workspace<br/>Resource Group]
        
        subgraph "Compute Resources"
            COMPUTE[Compute Instance<br/>adsai-lambda-0]
            GPU_CLUSTER[GPU Cluster<br/>Training Workloads]
            CPU_CLUSTER[CPU Cluster<br/>Data Processing]
        end
        
        subgraph "Data Assets"
            RAW_TRAIN[Raw Training Data<br/>emotion-raw-train]
            RAW_TEST[Raw Test Data<br/>emotion-raw-test]
            PROCESSED[Processed Data<br/>Auto-registered]
        end
        
        subgraph "Model Assets"
            MODEL_REGISTRY[Model Registry<br/>Versioned Models]
            EXPERIMENTS[MLflow Experiments<br/>Tracking & Metrics]
            ENDPOINTS[Model Endpoints<br/>Real-time Inference]
        end
        
        subgraph "Pipeline Orchestration"
            DATA_PIPELINE[Data Processing Pipeline<br/>Preprocessing Jobs]
            TRAIN_PIPELINE[Training Pipeline<br/>Multi-task Learning]
            EVAL_PIPELINE[Evaluation Pipeline<br/>Model Validation]
            SCHEDULE[Pipeline Scheduling<br/>Automated Retraining]
        end
    end
    
    subgraph "Azure Storage Services"
        BLOB_STORAGE[Azure Blob Storage<br/>Model Artifacts]
        DATA_LAKE[Azure Data Lake<br/>Large Datasets]
        KEY_VAULT[Azure Key Vault<br/>API Keys & Secrets]
    end
    
    subgraph "Monitoring & Logging"
        APP_INSIGHTS[Application Insights<br/>Performance Monitoring]
        LOG_ANALYTICS[Log Analytics<br/>Centralized Logging]
        ALERTS[Azure Alerts<br/>Failure Notifications]
    end
    
    %% Development to Cloud Flow
    DEV_CLI -.->|Submit Jobs| WORKSPACE
    LOCAL_DATA -.->|Upload| RAW_TRAIN
    LOCAL_DATA -.->|Upload| RAW_TEST
    
    %% Azure ML Internal Flow
    WORKSPACE --> COMPUTE
    WORKSPACE --> GPU_CLUSTER
    WORKSPACE --> CPU_CLUSTER
    
    RAW_TRAIN --> DATA_PIPELINE
    RAW_TEST --> DATA_PIPELINE
    DATA_PIPELINE --> PROCESSED
    
    PROCESSED --> TRAIN_PIPELINE
    TRAIN_PIPELINE --> MODEL_REGISTRY
    MODEL_REGISTRY --> EVAL_PIPELINE
    EVAL_PIPELINE --> ENDPOINTS
    
    TRAIN_PIPELINE --> EXPERIMENTS
    SCHEDULE -.->|Trigger| TRAIN_PIPELINE
    
    %% Storage Integration
    MODEL_REGISTRY -.-> BLOB_STORAGE
    PROCESSED -.-> DATA_LAKE
    DEV_CLI -.->|Secrets| KEY_VAULT
    
    %% Monitoring Integration
    TRAIN_PIPELINE -.-> APP_INSIGHTS
    ENDPOINTS -.-> APP_INSIGHTS
    WORKSPACE -.-> LOG_ANALYTICS
    LOG_ANALYTICS -.-> ALERTS
    
    %% Environment Sync
    LOCAL_MODELS -.->|Model Sync| MODEL_REGISTRY
    MODEL_REGISTRY -.->|Download| LOCAL_MODELS
    
    %% Styling
    classDef local fill:#E3F2FD,stroke:#1565C0,stroke-width:2px
    classDef compute fill:#E8F5E8,stroke:#2E7D32,stroke-width:2px
    classDef data fill:#FFF3E0,stroke:#EF6C00,stroke-width:2px
    classDef model fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px
    classDef pipeline fill:#FFEBEE,stroke:#C62828,stroke-width:2px
    classDef storage fill:#E0F2F1,stroke:#00695C,stroke-width:2px
    classDef monitoring fill:#FFF8E1,stroke:#F57F17,stroke-width:2px
    
    class DEV_CLI,LOCAL_DATA,LOCAL_MODELS local
    class WORKSPACE,COMPUTE,GPU_CLUSTER,CPU_CLUSTER compute
    class RAW_TRAIN,RAW_TEST,PROCESSED data
    class MODEL_REGISTRY,EXPERIMENTS,ENDPOINTS model
    class DATA_PIPELINE,TRAIN_PIPELINE,EVAL_PIPELINE,SCHEDULE pipeline
    class BLOB_STORAGE,DATA_LAKE,KEY_VAULT storage
    class APP_INSIGHTS,LOG_ANALYTICS,ALERTS monitoring
```

---

## 🔄 Real-time Processing Flow

### Live Emotion Analysis Pipeline

```mermaid
stateDiagram-v2
    [*] --> UserInput
    
    UserInput : User Submits YouTube URL
    UserInput --> URLValidation
    
    URLValidation : Validate URL Format
    URLValidation --> AudioDownload : Valid URL
    URLValidation --> ErrorState : Invalid URL
    
    AudioDownload : Download Audio from YouTube
    AudioDownload --> Transcription : Audio Downloaded
    AudioDownload --> ErrorState : Download Failed
    
    Transcription : Speech-to-Text Processing
    Transcription --> AssemblyAI : Primary STT
    Transcription --> WhisperFallback : AssemblyAI Failed
    
    AssemblyAI : AssemblyAI API Transcription
    AssemblyAI --> SegmentProcessing : Transcript Ready
    AssemblyAI --> WhisperFallback : API Failed
    
    WhisperFallback : Local Whisper Model
    WhisperFallback --> SegmentProcessing : Transcript Ready
    WhisperFallback --> ErrorState : Both STT Failed
    
    SegmentProcessing : Process Text Segments
    SegmentProcessing --> FeatureExtraction : For Each Segment
    
    FeatureExtraction : Extract NLP Features
    FeatureExtraction --> ModelInference : Features Ready
    
    ModelInference : DeBERTa Multi-task Prediction
    ModelInference --> EmotionOutput : Predictions Generated
    
    EmotionOutput : Emotion Classification Results
    EmotionOutput --> TimelineUpdate : Add to Timeline
    
    TimelineUpdate : Update Emotion Timeline
    TimelineUpdate --> SegmentProcessing : Next Segment
    TimelineUpdate --> ResponseGeneration : All Segments Done
    
    ResponseGeneration : Format JSON Response
    ResponseGeneration --> LiveVisualization : Send to Frontend
    
    LiveVisualization : Real-time UI Updates
    LiveVisualization --> UserInteraction : Display Results
    
    UserInteraction : User Views/Interacts
    UserInteraction --> FeedbackCollection : User Provides Feedback
    UserInteraction --> NewAnalysis : New URL Input
    
    FeedbackCollection : Collect User Corrections
    FeedbackCollection --> TrainingData : Store for Retraining
    
    TrainingData : Update Training Dataset
    TrainingData --> UserInteraction : Feedback Saved
    
    NewAnalysis : Start New Analysis
    NewAnalysis --> UserInput : Process New Video
    
    ErrorState : Error Handling & Recovery
    ErrorState --> UserInput : Retry/New Input
    
    %% Parallel States for Real-time Features
    LiveVisualization --> EmotionPulse
    LiveVisualization --> TimelineTracking
    LiveVisualization --> StatisticsUpdate
    
    EmotionPulse : Live Emotion Display
    TimelineTracking : Video Timeline Sync
    StatisticsUpdate : Real-time Statistics
    
    EmotionPulse --> UserInteraction
    TimelineTracking --> UserInteraction
    StatisticsUpdate --> UserInteraction
```

---

## 📱 User Interface Architecture

### Frontend Component Interaction Flow

```mermaid
graph TD
    subgraph "User Interface Layers"
        subgraph "Presentation Layer"
            MATERIAL_UI[Material-UI Components<br/>Design System]
            FRAMER[Framer Motion<br/>Animations]
            CHART_JS[Chart.js<br/>Data Visualization]
        end
        
        subgraph "Component Layer"
            URL_INPUT[URL Input Component<br/>YouTube URL Entry]
            VIDEO_PLAYER[Video Player Component<br/>React Player + Controls]
            LIVE_EMOTION[Live Emotion Display<br/>Real-time Emotion Orb]
            TIMELINE_VIZ[Timeline Visualization<br/>Emotion Over Time]
            TRANSCRIPT_VIEW[Transcript Component<br/>Interactive Text Display]
            ANALYTICS[Analytics Dashboard<br/>Statistics & Charts]
            FEEDBACK_UI[Feedback Interface<br/>User Corrections]
        end
        
        subgraph "State Management Layer"
            VIDEO_CONTEXT[Video Context<br/>Global State Provider]
            USE_VIDEO[useVideo Hook<br/>State Access Hook]
            LOCAL_STORAGE[Local Storage<br/>Persistence Layer]
        end
        
        subgraph "Service Layer"
            API_CLIENT[API Client<br/>Axios HTTP Client]
            ERROR_HANDLER[Error Handling<br/>Try/Catch Wrapper]
            LOADING_STATE[Loading States<br/>UI Feedback]
        end
    end
    
    subgraph "User Interaction Flow"
        USER_ACTION[User Action<br/>URL Submit/Video Seek]
        STATE_UPDATE[State Update<br/>Context Modification]
        API_CALL[API Request<br/>Backend Communication]
        RESPONSE_HANDLE[Response Handling<br/>Data Processing]
        UI_UPDATE[UI Update<br/>Component Re-render]
        USER_FEEDBACK[User Feedback<br/>Visual Response]
    end
    
    %% Component Dependencies
    URL_INPUT --> MATERIAL_UI
    VIDEO_PLAYER --> MATERIAL_UI
    LIVE_EMOTION --> FRAMER
    TIMELINE_VIZ --> CHART_JS
    TRANSCRIPT_VIEW --> MATERIAL_UI
    ANALYTICS --> CHART_JS
    FEEDBACK_UI --> MATERIAL_UI
    
    %% State Management Flow
    URL_INPUT --> VIDEO_CONTEXT
    VIDEO_PLAYER --> VIDEO_CONTEXT
    LIVE_EMOTION --> USE_VIDEO
    TIMELINE_VIZ --> USE_VIDEO
    TRANSCRIPT_VIEW --> USE_VIDEO
    ANALYTICS --> USE_VIDEO
    
    VIDEO_CONTEXT --> LOCAL_STORAGE
    USE_VIDEO --> VIDEO_CONTEXT
    
    %% Service Layer Connections
    VIDEO_CONTEXT --> API_CLIENT
    API_CLIENT --> ERROR_HANDLER
    API_CLIENT --> LOADING_STATE
    
    %% User Interaction Flow
    USER_ACTION --> STATE_UPDATE
    STATE_UPDATE --> API_CALL
    API_CALL --> RESPONSE_HANDLE
    RESPONSE_HANDLE --> UI_UPDATE
    UI_UPDATE --> USER_FEEDBACK
    
    %% Cross-layer Connections
    STATE_UPDATE --> VIDEO_CONTEXT
    API_CALL --> API_CLIENT
    UI_UPDATE --> LIVE_EMOTION
    UI_UPDATE --> TIMELINE_VIZ
    UI_UPDATE --> TRANSCRIPT_VIEW
    
    %% Styling
    classDef presentation fill:#E3F2FD,stroke:#1565C0,stroke-width:2px
    classDef component fill:#E8F5E8,stroke:#2E7D32,stroke-width:2px
    classDef state fill:#FFF3E0,stroke:#EF6C00,stroke-width:2px
    classDef service fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px
    classDef interaction fill:#FFEBEE,stroke:#C62828,stroke-width:2px
    
    class MATERIAL_UI,FRAMER,CHART_JS presentation
    class URL_INPUT,VIDEO_PLAYER,LIVE_EMOTION,TIMELINE_VIZ,TRANSCRIPT_VIEW,ANALYTICS,FEEDBACK_UI component
    class VIDEO_CONTEXT,USE_VIDEO,LOCAL_STORAGE state
    class API_CLIENT,ERROR_HANDLER,LOADING_STATE service
    class USER_ACTION,STATE_UPDATE,API_CALL,RESPONSE_HANDLE,UI_UPDATE,USER_FEEDBACK interaction
```

---

## 🔒 Security & Authentication Architecture

### Security Implementation Layers

```mermaid
graph TB
    subgraph "Frontend Security"
        CSP[Content Security Policy<br/>XSS Prevention]
        CORS_CLIENT[CORS Handling<br/>Cross-Origin Requests]
        INPUT_VALID[Input Validation<br/>URL Sanitization]
        HTTPS_ONLY[HTTPS Only<br/>Secure Transport]
    end
    
    subgraph "API Security"
        CORS_SERVER[CORS Middleware<br/>Origin Validation]
        RATE_LIMIT[Rate Limiting<br/>DoS Prevention]
        REQUEST_VALID[Request Validation<br/>Pydantic Models]
        ERROR_SANITIZE[Error Sanitization<br/>Information Leakage Prevention]
    end
    
    subgraph "Infrastructure Security"
        NETWORK_ISOLATION[Network Isolation<br/>Docker Bridge Networks]
        ENV_SECRETS[Environment Secrets<br/>.env File Protection]
        CONTAINER_SECURITY[Container Security<br/>Non-root User]
        VOLUME_MOUNT[Secure Volume Mounts<br/>Read-only Where Possible]
    end
    
    subgraph "External API Security"
        API_KEYS[API Key Management<br/>Azure Key Vault]
        KEY_ROTATION[Key Rotation<br/>Automated Updates]
        TLS_VERIFY[TLS Verification<br/>Certificate Validation]
        FALLBACK_SECURITY[Fallback Security<br/>Local Processing Option]
    end
    
    subgraph "Data Security"
        DATA_ENCRYPTION[Data Encryption<br/>At Rest & In Transit]
        PII_HANDLING[PII Data Handling<br/>Minimal Collection]
        TEMP_CLEANUP[Temporary File Cleanup<br/>Auto-deletion]
        AUDIT_LOGS[Security Audit Logs<br/>Access Tracking]
    end
    
    subgraph "Azure Cloud Security"
        IDENTITY[Azure Identity<br/>DefaultAzureCredential]
        RBAC[Role-Based Access<br/>Least Privilege]
        VNET[Virtual Network<br/>Network Isolation]
        PRIVATE_ENDPOINTS[Private Endpoints<br/>Internal Communication]
    end
    
    %% Security Flow Connections
    CSP --> CORS_CLIENT
    INPUT_VALID --> REQUEST_VALID
    HTTPS_ONLY --> TLS_VERIFY
    
    CORS_CLIENT -.->|Secure Requests| CORS_SERVER
    REQUEST_VALID --> RATE_LIMIT
    RATE_LIMIT --> ERROR_SANITIZE
    
    ENV_SECRETS --> API_KEYS
    CONTAINER_SECURITY --> NETWORK_ISOLATION
    VOLUME_MOUNT --> DATA_ENCRYPTION
    
    API_KEYS --> KEY_ROTATION
    TLS_VERIFY --> FALLBACK_SECURITY
    
    DATA_ENCRYPTION --> PII_HANDLING
    TEMP_CLEANUP --> AUDIT_LOGS
    
    IDENTITY --> RBAC
    RBAC --> VNET
    VNET --> PRIVATE_ENDPOINTS
    
    %% Cross-layer Security
    API_KEYS -.->|Secure Storage| IDENTITY
    AUDIT_LOGS -.->|Cloud Logging| VNET
    ERROR_SANITIZE -.->|Safe Responses| CSP
    
    %% Styling
    classDef frontend fill:#E3F2FD,stroke:#1565C0,stroke-width:2px
    classDef api fill:#E8F5E8,stroke:#2E7D32,stroke-width:2px
    classDef infra fill:#FFF3E0,stroke:#EF6C00,stroke-width:2px
    classDef external fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px
    classDef data fill:#FFEBEE,stroke:#C62828,stroke-width:2px
    classDef cloud fill:#E0F2F1,stroke:#00695C,stroke-width:2px
    
    class CSP,CORS_CLIENT,INPUT_VALID,HTTPS_ONLY frontend
    class CORS_SERVER,RATE_LIMIT,REQUEST_VALID,ERROR_SANITIZE api
    class NETWORK_ISOLATION,ENV_SECRETS,CONTAINER_SECURITY,VOLUME_MOUNT infra
    class API_KEYS,KEY_ROTATION,TLS_VERIFY,FALLBACK_SECURITY external
    class DATA_ENCRYPTION,PII_HANDLING,TEMP_CLEANUP,AUDIT_LOGS data
    class IDENTITY,RBAC,VNET,PRIVATE_ENDPOINTS cloud
```

---

## 📊 Performance & Scalability Architecture

### Performance Optimization Strategy

```mermaid
graph TB
    subgraph "Frontend Performance"
        CODE_SPLIT[Code Splitting<br/>Dynamic Imports]
        LAZY_LOAD[Lazy Loading<br/>Component-based]
        MEMOIZATION[React Memoization<br/>useMemo + useCallback]
        VIRTUAL_SCROLL[Virtual Scrolling<br/>Large Dataset Handling]
    end
    
    subgraph "Backend Performance"
        ASYNC_PROC[Async Processing<br/>FastAPI + asyncio]
        MODEL_CACHE[Model Caching<br/>Memory-resident Models]
        BATCH_PREDICT[Batch Prediction<br/>Efficient Inference]
        WORKER_POOL[Worker Pool<br/>Parallel Processing]
    end
    
    subgraph "Caching Strategy"
        BROWSER_CACHE[Browser Caching<br/>Static Assets]
        API_CACHE[API Response Caching<br/>Redis/Memory]
        MODEL_CACHE_DISK[Model Disk Cache<br/>Persistent Storage]
        TRANSCRIPT_CACHE[Transcript Caching<br/>Avoid Re-transcription]
    end
    
    subgraph "Resource Optimization"
        GPU_UTIL[GPU Utilization<br/>CUDA Acceleration]
        MEMORY_OPT[Memory Optimization<br/>Gradient Checkpointing]
        CPU_AFFINITY[CPU Affinity<br/>Core Binding]
        IO_OPTIMIZATION[I/O Optimization<br/>Async File Operations]
    end
    
    subgraph "Scalability Patterns"
        HORIZONTAL[Horizontal Scaling<br/>Multiple Container Instances]
        LOAD_BALANCE[Load Balancing<br/>nginx/HAProxy]
        QUEUE_SYSTEM[Queue System<br/>Celery/RQ for Long Tasks]
        MICROSERVICE[Microservice Split<br/>Dedicated STT Service]
    end
    
    subgraph "Monitoring & Metrics"
        PERF_MONITOR[Performance Monitoring<br/>Response Time Tracking]
        RESOURCE_MONITOR[Resource Monitoring<br/>CPU/Memory/GPU Usage]
        BOTTLENECK[Bottleneck Detection<br/>Profiling & Analysis]
        AUTO_SCALE[Auto-scaling<br/>Dynamic Resource Allocation]
    end
    
    %% Performance Flow
    CODE_SPLIT --> LAZY_LOAD
    LAZY_LOAD --> MEMOIZATION
    MEMOIZATION --> VIRTUAL_SCROLL
    
    ASYNC_PROC --> MODEL_CACHE
    MODEL_CACHE --> BATCH_PREDICT
    BATCH_PREDICT --> WORKER_POOL
    
    %% Caching Integration
    BROWSER_CACHE -.-> CODE_SPLIT
    API_CACHE -.-> ASYNC_PROC
    MODEL_CACHE_DISK -.-> MODEL_CACHE
    TRANSCRIPT_CACHE -.-> WORKER_POOL
    
    %% Resource Optimization
    GPU_UTIL --> MEMORY_OPT
    MEMORY_OPT --> CPU_AFFINITY
    CPU_AFFINITY --> IO_OPTIMIZATION
    
    %% Scalability Implementation
    HORIZONTAL --> LOAD_BALANCE
    LOAD_BALANCE --> QUEUE_SYSTEM
    QUEUE_SYSTEM --> MICROSERVICE
    
    %% Monitoring Integration
    PERF_MONITOR --> RESOURCE_MONITOR
    RESOURCE_MONITOR --> BOTTLENECK
    BOTTLENECK --> AUTO_SCALE
    
    %% Cross-layer Optimization
    MODEL_CACHE -.-> GPU_UTIL
    WORKER_POOL -.-> HORIZONTAL
    BATCH_PREDICT -.-> QUEUE_SYSTEM
    VIRTUAL_SCROLL -.-> PERF_MONITOR
    
    %% Styling
    classDef frontend fill:#E3F2FD,stroke:#1565C0,stroke-width:2px
    classDef backend fill:#E8F5E8,stroke:#2E7D32,stroke-width:2px
    classDef caching fill:#FFF3E0,stroke:#EF6C00,stroke-width:2px
    classDef resource fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px
    classDef scaling fill:#FFEBEE,stroke:#C62828,stroke-width:2px
    classDef monitoring fill:#E0F2F1,stroke:#00695C,stroke-width:2px
    
    class CODE_SPLIT,LAZY_LOAD,MEMOIZATION,VIRTUAL_SCROLL frontend
    class ASYNC_PROC,MODEL_CACHE,BATCH_PREDICT,WORKER_POOL backend
    class BROWSER_CACHE,API_CACHE,MODEL_CACHE_DISK,TRANSCRIPT_CACHE caching
    class GPU_UTIL,MEMORY_OPT,CPU_AFFINITY,IO_OPTIMIZATION resource
    class HORIZONTAL,LOAD_BALANCE,QUEUE_SYSTEM,MICROSERVICE scaling
    class PERF_MONITOR,RESOURCE_MONITOR,BOTTLENECK,AUTO_SCALE monitoring
```

---

## 🔄 DevOps & CI/CD Architecture

### Development Workflow & Automation

```mermaid
graph TB
    subgraph "Development Environment"
        DEV_LOCAL[Local Development<br/>Poetry + React Dev Server]
        GIT_HOOKS[Pre-commit Hooks<br/>Code Quality Checks]
        TESTING_LOCAL[Local Testing<br/>Unit + Integration Tests]
        DOCKER_LOCAL[Local Docker<br/>docker-compose up]
    end
    
    subgraph "Version Control & CI/CD"
        GITHUB[GitHub Repository<br/>Source Code Management]
        
        subgraph "GitHub Actions"
            LINT_ACTION[Lint Workflow<br/>Code Quality Checks]
            TEST_ACTION[Test Workflow<br/>Automated Testing]
            BUILD_ACTION[Build Workflow<br/>Docker Image Creation]
            DEPLOY_ACTION[Deploy Workflow<br/>Container Registry Push]
        end
    end
    
    subgraph "Container Registry"
        DOCKER_HUB[Docker Hub<br/>Public Image Registry]
        AZURE_ACR[Azure Container Registry<br/>Private Enterprise Registry]
        VERSION_TAGS[Version Tagging<br/>Semantic Versioning]
    end
    
    subgraph "Deployment Environments"
        DEV_ENV[Development Environment<br/>Feature Testing]
        STAGING_ENV[Staging Environment<br/>Pre-production Testing]
        PROD_ENV[Production Environment<br/>Live System]
        AZURE_ENV[Azure ML Environment<br/>Cloud Training Pipeline]
    end
    
    subgraph "Monitoring & Observability"
        HEALTH_CHECK[Health Checks<br/>Service Availability]
        LOG_AGGREGATION[Log Aggregation<br/>Centralized Logging]
        METRICS_COLLECTION[Metrics Collection<br/>Performance Data]
        ALERT_SYSTEM[Alert System<br/>Failure Notifications]
    end
    
    subgraph "Quality Assurance"
        CODE_COVERAGE[Code Coverage<br/>Testing Metrics]
        SECURITY_SCAN[Security Scanning<br/>Vulnerability Detection]
        PERF_TESTING[Performance Testing<br/>Load & Stress Tests]
        MODEL_VALIDATION[Model Validation<br/>ML Model Quality]
    end
    
    %% Development Flow
    DEV_LOCAL --> GIT_HOOKS
    GIT_HOOKS --> TESTING_LOCAL
    TESTING_LOCAL --> DOCKER_LOCAL
    DOCKER_LOCAL --> GITHUB
    
    %% CI/CD Pipeline
    GITHUB --> LINT_ACTION
    GITHUB --> TEST_ACTION
    LINT_ACTION --> BUILD_ACTION
    TEST_ACTION --> BUILD_ACTION
    BUILD_ACTION --> DEPLOY_ACTION
    
    %% Registry Management
    DEPLOY_ACTION --> DOCKER_HUB
    DEPLOY_ACTION --> AZURE_ACR
    DOCKER_HUB --> VERSION_TAGS
    AZURE_ACR --> VERSION_TAGS
    
    %% Environment Deployment
    VERSION_TAGS --> DEV_ENV
    DEV_ENV --> STAGING_ENV
    STAGING_ENV --> PROD_ENV
    BUILD_ACTION --> AZURE_ENV
    
    %% Monitoring Integration
    DEV_ENV --> HEALTH_CHECK
    STAGING_ENV --> HEALTH_CHECK
    PROD_ENV --> HEALTH_CHECK
    HEALTH_CHECK --> LOG_AGGREGATION
    LOG_AGGREGATION --> METRICS_COLLECTION
    METRICS_COLLECTION --> ALERT_SYSTEM
    
    %% Quality Assurance
    TEST_ACTION --> CODE_COVERAGE
    BUILD_ACTION --> SECURITY_SCAN
    STAGING_ENV --> PERF_TESTING
    AZURE_ENV --> MODEL_VALIDATION
    
    %% Feedback Loops
    ALERT_SYSTEM -.->|Issues| GITHUB
    MODEL_VALIDATION -.->|Results| DEV_LOCAL
    PERF_TESTING -.->|Optimization| DEV_LOCAL
    CODE_COVERAGE -.->|Coverage Reports| GITHUB
    
    %% Styling
    classDef development fill:#E3F2FD,stroke:#1565C0,stroke-width:2px
    classDef cicd fill:#E8F5E8,stroke:#2E7D32,stroke-width:2px
    classDef registry fill:#FFF3E0,stroke:#EF6C00,stroke-width:2px
    classDef deployment fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px
    classDef monitoring fill:#FFEBEE,stroke:#C62828,stroke-width:2px
    classDef quality fill:#E0F2F1,stroke:#00695C,stroke-width:2px
    
    class DEV_LOCAL,GIT_HOOKS,TESTING_LOCAL,DOCKER_LOCAL development
    class GITHUB,LINT_ACTION,TEST_ACTION,BUILD_ACTION,DEPLOY_ACTION cicd
    class DOCKER_HUB,AZURE_ACR,VERSION_TAGS registry
    class DEV_ENV,STAGING_ENV,PROD_ENV,AZURE_ENV deployment
    class HEALTH_CHECK,LOG_AGGREGATION,METRICS_COLLECTION,ALERT_SYSTEM monitoring
    class CODE_COVERAGE,SECURITY_SCAN,PERF_TESTING,MODEL_VALIDATION quality
```

---

## 📋 Technology Stack Summary

### Complete Technology Matrix

```mermaid
mindmap
  root((Emotion Classification Pipeline))
    Frontend
      React 18
        Material-UI
        Framer Motion
        Chart.js
        React Player
      State Management
        Context API
        Local Storage
      Build Tools
        Node.js
        npm/yarn
        Webpack
    Backend
      Python 3.11
        FastAPI
        uvicorn
        Pydantic
      ML/AI Stack
        PyTorch
        Transformers (HuggingFace)
        DeBERTa
        scikit-learn
        NLTK
        TextBlob
      External APIs
        AssemblyAI
        OpenAI Whisper
        YouTube (PyTubefix)
    Infrastructure
      Containerization
        Docker
        Docker Compose
        NVIDIA Container Runtime
      Cloud Platform
        Azure ML
        Azure Blob Storage
        Azure Key Vault
      Development
        Poetry (Python)
        Git + GitHub
        GitHub Actions
        Pre-commit Hooks
    Data & Storage
      File System
        Local Storage
        JSON/CSV
        Model Weights
      Databases
        Optional Redis
        Azure Storage
      Model Registry
        MLflow
        Azure ML Registry
    Monitoring
      Logging
        Python logging
        Azure App Insights
      Performance
        Response Time Tracking
        Resource Monitoring
      Quality
        Code Coverage
        Security Scanning
```

---

## 🎯 Key Architectural Decisions

### Design Principles & Rationale

| **Decision** | **Rationale** | **Trade-offs** |
|--------------|---------------|----------------|
| **Microservices Architecture** | Separation of concerns, independent scaling, technology diversity | Increased complexity, network overhead |
| **FastAPI Backend** | High performance, automatic documentation, type safety | Learning curve, Python-specific |
| **React Frontend** | Component reusability, large ecosystem, real-time updates | Bundle size, client-side complexity |
| **DeBERTa Multi-task Model** | State-of-the-art NLP, efficient single model for multiple outputs | Large model size, inference latency |
| **Docker Containerization** | Environment consistency, easy deployment, isolation | Resource overhead, complexity |
| **Azure ML Integration** | Scalable training, managed infrastructure, MLOps features | Cloud vendor lock-in, cost considerations |
| **AssemblyAI + Whisper Fallback** | High accuracy transcription with local backup | API dependency, cost per request |
| **Real-time Visualization** | Better user experience, immediate feedback | Increased frontend complexity |

---

## 🚀 Future Architecture Enhancements

### Planned Improvements & Scalability

```mermaid
graph TB
    subgraph "Short-term Enhancements (0-6 months)"
        REDIS[Redis Caching<br/>Response Optimization]
        API_RATE[API Rate Limiting<br/>Resource Protection]
        WEBSOCKET[WebSocket Support<br/>Real-time Updates]
        BATCH_API[Batch Processing API<br/>Multiple Video Analysis]
    end
    
    subgraph "Medium-term Improvements (6-18 months)"
        KUBERNETES[Kubernetes Deployment<br/>Container Orchestration]
        SERVICE_MESH[Service Mesh<br/>Istio/Linkerd]
        STREAMING[Streaming Processing<br/>Apache Kafka]
        ADVANCED_ML[Advanced ML Models<br/>Multimodal Analysis]
    end
    
    subgraph "Long-term Vision (18+ months)"
        EDGE_DEPLOY[Edge Deployment<br/>CDN + Edge Computing]
        FEDERATED[Federated Learning<br/>Privacy-preserving Training]
        REALTIME_VIDEO[Real-time Video Analysis<br/>Live Stream Processing]
        AI_ORCHESTRATOR[AI Orchestrator<br/>Multi-model Ensemble]
    end
    
    subgraph "Scalability Targets"
        CONCURRENT[1000+ Concurrent Users<br/>High Throughput]
        GLOBAL[Global Deployment<br/>Multi-region Setup]
        ENTERPRISE[Enterprise Features<br/>SSO, RBAC, Audit]
        MARKETPLACE[Model Marketplace<br/>Plugin Architecture]
    end
    
    %% Enhancement Flow
    REDIS --> KUBERNETES
    API_RATE --> SERVICE_MESH
    WEBSOCKET --> STREAMING
    BATCH_API --> ADVANCED_ML
    
    KUBERNETES --> EDGE_DEPLOY
    SERVICE_MESH --> FEDERATED
    STREAMING --> REALTIME_VIDEO
    ADVANCED_ML --> AI_ORCHESTRATOR
    
    %% Scalability Achievement
    EDGE_DEPLOY --> CONCURRENT
    FEDERATED --> GLOBAL
    REALTIME_VIDEO --> ENTERPRISE
    AI_ORCHESTRATOR --> MARKETPLACE
    
    %% Styling
    classDef shortTerm fill:#E8F5E8,stroke:#2E7D32,stroke-width:2px
    classDef mediumTerm fill:#FFF3E0,stroke:#EF6C00,stroke-width:2px
    classDef longTerm fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px
    classDef scalability fill:#E3F2FD,stroke:#1565C0,stroke-width:2px
    
    class REDIS,API_RATE,WEBSOCKET,BATCH_API shortTerm
    class KUBERNETES,SERVICE_MESH,STREAMING,ADVANCED_ML mediumTerm
    class EDGE_DEPLOY,FEDERATED,REALTIME_VIDEO,AI_ORCHESTRATOR longTerm
    class CONCURRENT,GLOBAL,ENTERPRISE,MARKETPLACE scalability
```

---

## 📖 Conclusion

The Emotion Classification Pipeline represents a comprehensive, production-ready system that combines cutting-edge AI/ML capabilities with modern software architecture principles. The system is designed for:

- **Scalability**: Microservices architecture with cloud-native deployment
- **Reliability**: Robust error handling, fallback mechanisms, and monitoring
- **Performance**: Optimized inference pipeline with caching and GPU acceleration
- **Maintainability**: Clean code structure, comprehensive testing, and automated CI/CD
- **Extensibility**: Modular design allowing for easy feature additions and model updates

The architecture successfully balances complexity with functionality, providing a solid foundation for emotional intelligence applications while maintaining the flexibility to evolve with changing requirements and technological advances.
