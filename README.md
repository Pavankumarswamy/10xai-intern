# AI Intern - Unified Agent (3 Tasks)

## 📌 Submission Information
- **Source Code Repository**: [https://github.com/Pavankumarswamy/10xai-intern.git](https://github.com/Pavankumarswamy/10xai-intern.git)
- **Description**: A unified Dockerized AI application performing Text Chat, Speech-to-Speech Assistant (LUCA), and RAG-based School Information retrieval.

---

## 🛠️ Setup and Run Instructions

### Prerequisites
1. **Docker Desktop** installed and running.
2. **Git** installed.

### Step 1: Clone the Repository
```bash
git clone https://github.com/Pavankumarswamy/10xai-intern.git
cd 10xai-intern
```

### Step 2: Build the Docker Image
```bash
# This build step downloads the LLM model (2GB) and bakes it into the image.
# It may take a few minutes, but ensures subsequent runs are instant.
docker build -t ai-intern-agent .
```

### Step 3: Run the Container
```bash
docker run -p 7860:7860 ai-intern-agent
```
*The application starts instantly as the model is pre-loaded.*

### Step 4: Access the Application
Open your browser and navigate to:
👉 **[http://localhost:7860](http://localhost:7860)**

---

## 🏗️ Architecture & Workflow

The application is built as a single-container microservice exposing a Gradio web interface.

### Core Components
- **Frontend**: Gradio (Web UI with Tabs).
- **LLM Backend**: Ollama (Running `gemma3:1b` locally).
- **Speech Stack**: 
  - **STT**: OpenAI Whisper (`tiny.en`) running on CPU.
  - **TTS**: Edge-TTS (Microsoft Edge Cloud).
- **RAG Stack**:
  - **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`.
  - **Vector Store**: FAISS (Facebook AI Similarity Search) running on CPU.

### Workflow by Task

#### 1. Task 1: Open Source Models
- **Text Sub-Task**: Chat interacts directly with Gemma 3 (streaming enabled).
- **Speech Sub-Task**: 
  - **Input**: User Audio → Whisper Transcribe.
  - **Logic**: Transcript → Gemma 3.
  - **Output**: Response → Edge TTS → Audio Playback.

#### 2. Task 2: Speech-to-Speech (LUCA)
- **Identity Enforcement**: The system strictly overrides answers to "Who are you?" with: *"I am LUCA from 10X Technologies."*
- **Workflow**:
  - Supports both **Voice** and **Text** input.
  - Uses Whisper for transcription.
  - Uses Gemma 3 for general conversation.
  - Returns both Audio (TTS) and Text response.

#### 3. Task 3: School Info (RAG)
- **Domain Constraint**: Answers **strictly** based on `school_information.pdf`.
- **Logic**:
  1. PDF is ingested & chunked on startup.
  2. Queries are matched via FAISS vector search.
  3. If relevant chunks are found: LLM answers using *only* that context.
  4. If no relevance: Returns *"I don't know"*.

### Design Decisions
1. **Model Baking**: The `gemma3:1b` model is pulled during the Docker `build` phase. This increases build time but makes the container startup nearly instantaneous and robust offline.
2. **Unified Docker Container**: All dependencies (Ollama, Python, ffmpeg) are packaged together for portability.
3. **Streaming Responses**: Task 1 and Task 3 utilize text streaming to provide better user feedback during inference.

---

## ⚠️ Known Limitations & Incomplete Areas

1. **Build Time**: The initial `docker build` takes time due to model downloads (~2-3GB total for LLM + Base Image).
2. **Audio Processing**: Whisper `tiny.en` is lightweight but may have lower accuracy on non-native accents compared to larger models.
3. **Persistence**: Chat history is session-based and clears on page refresh.
4. **Single PDF**: The RAG system is designed for a specific `school_information.pdf` included in the image.

---
**Author**: Pavan Kumar Swamy
**Date**: January 2026
