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
docker build -t ai-intern-agent .
```
*Note: The build process sets up Python 3.10 and installs necessary dependencies (ffmpeg, transformers, ollama, etc).*

### Step 3: Run the Container
```bash
docker run -p 7860:7860 ai-intern-agent
```
*Note: On the first run, the container will automatically pull the `gemma3:1b` model. This may take a few minutes depending on your internet connection.*

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

#### 1. Task 1: Text Chat
- **Input**: User types text.
- **Process**: Text sent to Ollama (`gemma3:1b`).
- **Output**: Streaming text response.

#### 2. Task 2: Speech-to-Speech (LUCA)
- **Input**: User speaks into microphone.
- **Process**: 
  1. **Whisper** converts Audio → Text.
  2. Identity Check: If asking "Who are you?", responds as "LUCA from 10X Technologies".
  3. **Ollama** generates response text.
  4. **Edge-TTS** converts Response Text → Audio.
- **Output**: Audio playback + Text transcript.

#### 3. Task 3: School Info (RAG)
- **Input**: User asks a question about the school.
- **Process**:
  1. PDF (`school_information.pdf`) is chunked and embedded (on startup).
  2. User query is embedded and matched against chunks using FAISS.
  3. Relevant chunks + Query are sent to Ollama.
  4. Model answers **strictly** based on context.
- **Output**: Accurate answer or "I don't know" if outside scope.

### Design Decisions
1. **Unified Docker Container**: To simplify deployment, all services (Ollama, Python App, Vector Store) run inside one container.
2. **Gemma 3 (1B)**: Chosen as the LLM to balance performance and memory usage for a local dockerized environment.
3. **FAISS CPU**: Used for vector search to avoid heavy dependencies like ChromaDB/Pinecone for this scale.

---

## ⚠️ Known Limitations & Incomplete Areas

1. **First-Run Latency**: The application downloads the LLM (~2GB) and Embedding models on the very first run, causing a startup delay.
2. **Audio Processing**: Whisper `tiny.en` is fast but may struggle with accents compared to larger models.
3. **Persistence**: Chat history is ephemeral and clears when the browser refresh or container restarts.
4. **PDF Handling**: Currently supports a single hardcoded PDF (`school_information.pdf`). Dynamic upload is not implemented in this version.

---
**Author**: Pavan Kumar Swamy
**Date**: January 2026
