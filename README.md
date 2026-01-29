# 🤖 Unified AI Assistant - 10X Technologies Intern Assignment

> **Assignment Objective**: Evaluate practical understanding of modern AI systems, including open-source language models, speech-based AI, and Retrieval-Augmented Generation (RAG). This project demonstrates clarity of thinking, execution ability, and system design.

## 📌 Assignment Overview

This project implements three distinct AI tasks in a unified application:

### **Task 1: Deploy Open-Source Models (Text & Speech)**
- ✅ Deploy one text-to-text language model
- ✅ Deploy one speech-to-speech model  
- ✅ Demonstrate basic input/output functionality
- ✅ Explain model choice and deployment method

### **Task 2: LUCA Voice Assistant**
- ✅ Build voice-enabled assistant using open-source LLM
- ✅ Identity enforcement: "LUCA from 10X Technologies" (hardcoded logic, not just system prompt)
- ✅ Integrate ASR (Automatic Speech Recognition) for voice input
- ✅ Integrate TTS (Text-to-Speech) for voice output
- ✅ Demonstrate voice conversation and direct text interaction

### **Task 3: RAG-Based School AI Assistant**
- ✅ Ingest school PDF into vector store
- ✅ Answer domain-specific questions only
- ✅ Reject out-of-domain queries with "I don't know"
- ✅ Handle structured and unstructured data correctly

---

## 🚀 Quick Start

### **Prerequisites**
Before running this project, ensure you have:
- **Docker Desktop** installed and running ([Download here](https://www.docker.com/products/docker-desktop/))
- **Git** (to clone the repository)
- At least **4GB RAM** available
- **10GB free disk space** (for Docker images and models)

### **Step 1: Clone the Repository**
```bash
git clone <your-repo-url>
cd 10xai
```

### **Step 2: Run with Docker (Recommended)**
```bash
# Build the Docker image (first time only, ~5-10 minutes)
docker build -t ai-intern-final .

# Run the container
docker run -p 7860:7860 ai-intern-final
```

**That's it!** Open your browser to **http://localhost:7860**

### **Alternative: Quick Development Run**
If you're making code changes and want to test quickly:
```powershell
# Windows PowerShell
.\run-fast.ps1
```
```bash
# Linux/Mac
chmod +x run-fast.sh
./run-fast.sh
```

This mounts your local code into the container without rebuilding.

### **What You'll See**
Once the app starts, you'll see three tabs:
1. **Task 1** - Text chat with Gemma 3 OR speech-to-speech pipeline
2. **Task 2** - LUCA Assistant with text chat AND voice interaction (separate sections)
3. **Task 3** - School RAG assistant (try the suggested questions)

---

## 📋 Table of Contents

- [Quick Start](#-quick-start) ⚡ *Start here to run the project*
- [Assignment Overview](#-assignment-overview)
- [Architecture Overview](#architecture-overview)
- [Task Breakdown](#task-breakdown)
  - [Task 1: Open Source Models](#task-1-open-source-models)
  - [Task 2: LUCA Voice Assistant](#task-2-luca-voice-assistant)
  - [Task 3: School RAG System](#task-3-school-rag-system)
- [Execution Flow](#execution-flow)
- [Running Locally](#running-locally)
- [Technical Stack](#technical-stack)
- [Deployment Method Explanation](#-deployment-method-explanation)
- [Design Decisions](#-design-decisions)
- [Troubleshooting](#-troubleshooting)
- [Known Limitations & Future Improvements](#️-known-limitations--future-improvements)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Gradio Web Interface                      │
│                  (Tabbed Multi-Task UI)                      │
└────────────┬────────────┬────────────┬─────────────────────┘
             │            │            │
    ┌────────▼────┐  ┌───▼────┐  ┌───▼──────────────────┐
    │   Task 1    │  │ Task 2 │  │      Task 3          │
    │ Text/Speech │  │  LUCA  │  │   School RAG         │
    └─────┬───────┘  └───┬────┘  └──────┬───────────────┘
          │              │               │
    ┌─────▼──────┐  ┌───▼─────┐   ┌────▼─────────────────┐
    │  Gemma 3   │  │ Whisper │   │  FAISS Vector Store  │
    │  (Ollama)  │  │ EdgeTTS │   │  + Keyword Search    │
    └────────────┘  └─────────┘   └──────────────────────┘
```

---

## 📝 Task Breakdown

### **Task 1: Open Source Models**

#### **Objective**
Deploy and demonstrate one text-to-text LLM and one speech-to-speech pipeline.

#### **Sub-Task A: Text-to-Text Model**

**Model**: `gemma3:1b` (via Ollama)

**Flow**:
```
User Input (Text) 
    ↓
System Prompt Injection (Tutor Style)
    ↓
Ollama Chat API (Streaming)
    ↓
Gradio ChatInterface (Real-time Display)
```

**Code Logic**:
```python
def task1_text_response(message, history):
    messages = normalize_history(history)
    messages.insert(0, {"role": "system", "content": TASK1_SYSTEM_PROMPT})
    messages.append({"role": "user", "content": _force_string(message)})
    
    stream = ollama.chat(model=LLM_MODEL, messages=messages, stream=True)
    partial_response = ""
    for chunk in stream:
        partial_response += chunk['message']['content']
        yield partial_response  # Streaming to UI
```

**Key Features**:
- Professional tutor-style responses
- Streaming output for real-time feedback
- Exam-oriented answer formatting

**Model Choice Justification**:
- **Gemma 3:1b**: Selected for its balance of performance and resource efficiency
  - Small footprint (~1GB) enables local deployment
  - Fast inference suitable for real-time streaming
  - Instruction-tuned for conversational tasks
  - Open-source (Apache 2.0 license)
  - Runs efficiently on CPU-only systems

---

#### **Sub-Task B: Speech-to-Speech Model**

**Pipeline**: Whisper (STT) → Gemma 3 (LLM) → EdgeTTS (TTS)

**Flow**:
```
Audio Input (Microphone/Upload)
    ↓
Whisper Transcription (STT)
    ↓
Gemma 3 Processing (LLM)
    ↓
EdgeTTS Synthesis (TTS)
    ↓
Audio Output + Transcript Display
```

**Code Logic**:
```python
def task1_speech_response(audio_path):
    # 1. Speech-to-Text
    transcribed = whisper_model.transcribe(audio_path)["text"].strip()
    
    # 2. LLM Processing
    messages = [
        {"role": "system", "content": TASK1_SYSTEM_PROMPT},
        {"role": "user", "content": transcribed}
    ]
    llm_response = call_ollama_non_stream(messages)
    
    # 3. Text-to-Speech
    async def get_tts(text):
        fd, path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        await edge_tts.Communicate(text, TTS_VOICE).save(path)
        return path
    
    audio_out = asyncio.run(get_tts(llm_response))
    return audio_out, f"**User:** {transcribed}\n\n**AI:** {llm_response}"
```

**Key Features**:
- End-to-end voice interaction
- Async TTS generation
- Visual transcript feedback

**Model Choice Justification**:
- **Whisper (tiny.en)**: OpenAI's robust ASR model
  - High accuracy for English speech recognition
  - Tiny variant (~39MB) for fast startup
  - Handles various accents and audio quality
  - Open-source and well-documented
- **EdgeTTS (en-US-AriaNeural)**: Microsoft's cloud TTS
  - Natural-sounding voice synthesis
  - No API key required
  - Low latency for real-time responses
  - Free tier sufficient for demo purposes

---

### **Task 2: LUCA Voice Assistant**

#### **Objective**
Create a voice-enabled assistant that always identifies itself as "LUCA from 10X Technologies". Supports both text chat and voice interaction.

#### **Architecture**:
```
┌─────────────────────────────────────────┐
│         Task 2: LUCA Assistant          │
├─────────────────────────────────────────┤
│                                         │
│  ┌─────────────────────────────────┐   │
│  │   Text Chat (Accordion 1)       │   │
│  │   - Streaming chat interface    │   │
│  │   - Identity check hardcoded    │   │
│  │   - Real-time responses         │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │   Voice Interaction (Accordion 2)│  │
│  │   - Audio input (mic/upload)    │   │
│  │   - Whisper STT                 │   │
│  │   - Identity check hardcoded    │   │
│  │   - LLM processing              │   │
│  │   - EdgeTTS synthesis           │   │
│  │   - Audio output + transcript   │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

#### **Code Logic**:

**Text Chat Handler**:
```python
def task2_text_chat(message, history):
    """Text-based chat with LUCA (streaming)"""
    user_input = _force_string(message)
    
    # Identity Check (Hardcoded override)
    if is_identity_question(user_input):
        yield LUCA_IDENTITY_TEXT  # "I am LUCA from 10X Technologies."
        return
    
    # Standard chat with System Prompt
    msgs = normalize_history(history)
    msgs.insert(0, {"role": "system", "content": TASK2_SYSTEM_PROMPT})
    msgs.append({"role": "user", "content": user_input})
    
    # Stream response
    stream = ollama.chat(model=LLM_MODEL, messages=msgs, stream=True)
    partial_response = ""
    for chunk in stream:
        partial_response += chunk['message']['content']
        yield partial_response
```

**Voice Handler**:
```python
def task2_voice_handler(audio_path):
    """Voice-based interaction with LUCA"""
    # 1. Speech-to-Text
    transcribed = whisper_model.transcribe(audio_path)["text"].strip()
    
    # 2. Identity Check (Hardcoded override)
    if is_identity_question(transcribed):
        response_text = LUCA_IDENTITY_TEXT
    else:
        messages = [
            {"role": "system", "content": TASK2_SYSTEM_PROMPT},
            {"role": "user", "content": transcribed}
        ]
        response_text = call_ollama_non_stream(messages)
    
    # 3. Text-to-Speech
    audio_out = asyncio.run(get_tts(response_text))
    return audio_out, f"**User:** {transcribed}\n\n**LUCA:** {response_text}"
```


**Key Features**:
- **Text Chat**: Streaming responses with conversation history
- **Voice Interaction**: Complete STT → LLM → TTS pipeline
- **Hardcoded Identity**: Pattern matching bypasses LLM for identity questions
- **Dual Mode**: Separate accordions for text and voice (like Task 1 structure)

---

### **Task 3: School RAG System**

#### **Objective**
Build a RAG (Retrieval-Augmented Generation) system that answers questions strictly from a school information PDF.

#### **Architecture**:
```
PDF Document (school_information.pdf)
    ↓
Text Extraction (PyPDF2)
    ↓
Chunking (RecursiveCharacterTextSplitter)
    ↓
Embedding (SentenceTransformer: all-MiniLM-L6-v2)
    ↓
Vector Store (FAISS IndexFlatIP)
    ↓
┌─────────────────────────────────────┐
│   Query Processing (Hybrid Search)  │
│  ┌──────────────┬─────────────────┐ │
│  │ Vector Search│ Keyword Search  │ │
│  │ (Semantic)   │ (Exact Match)   │ │
│  └──────┬───────┴────────┬────────┘ │
│         └────────┬────────┘          │
│           Retrieved Chunks           │
└──────────────────┬──────────────────┘
                   ↓
         LLM Context Injection
                   ↓
         Gemma 3 Response (Streaming)
```

#### **Code Logic**:

**1. Vector Store Creation**:
```python
def create_vector_store(pdf_path):
    # Extract text from PDF
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = "".join([p.extract_text() or "" for p in reader.pages])
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, 
        chunk_overlap=120
    )
    chunks = splitter.split_text(text)
    
    # Generate embeddings
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    
    # Create FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    return index, chunks
```

**2. Hybrid Retrieval (Vector + Keyword)**:
```python
def retrieve_context(query, index, chunks):
    # 1. Vector Search
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k_vector=10)
    
    retrieved_indices = set()
    
    # Collect vector matches (threshold: 0.1)
    for i, score in zip(I[0], D[0]):
        if i < len(chunks) and score >= SIMILARITY_THRESHOLD:
            retrieved_indices.add(i)
    
    # 2. Keyword Fallback
    query_words = [w.lower() for w in query.split() if len(w) > 3]
    for idx, chunk in enumerate(chunks):
        matches = sum(1 for w in query_words if w in chunk.lower())
        if matches >= 1:
            retrieved_indices.add(idx)
    
    # Return sorted chunks
    final_indices = sorted(list(retrieved_indices))
    return [chunks[i] for i in final_indices]
```

**3. RAG Response Generation**:
```python
def task3_rag_response(message, history, rag_state):
    # Retrieve relevant chunks
    docs = retrieve_context(message, index, chunks)
    
    if not docs:
        yield REJECTION_RESPONSE
        return
    
    # Build context
    context_str = "\n\n".join([f"Excerpt: {d}" for d in docs])
    
    # Create prompt
    prompt = (
        "You are a helpful school assistant. Use the following excerpts "
        "to answer the user's question.\n"
        "If the exact answer isn't stated, try to infer it reasonably.\n\n"
        f"Excerpts:\n{context_str}\n\n"
        f"Question: {message}"
    )
    
    # Stream LLM response
    stream = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], stream=True)
    partial_response = ""
    for chunk in stream:
        partial_response += chunk['message']['content']
        yield partial_response
```

**Key Features**:
- Hybrid search (semantic + keyword) for robust retrieval
- Dynamic PDF upload support (max 2MB)
- Auto-generated suggested questions
- Strict domain adherence with graceful rejection

---

## 🔄 Execution Flow

### **Startup Sequence**:
```
1. Docker Container Start
   ↓
2. Ollama Server Launch (Background)
   ↓
3. Model Loading
   - Whisper (tiny.en)
   - SentenceTransformer (all-MiniLM-L6-v2)
   - Gemma 3:1b (Pre-pulled in Docker)
   ↓
4. PDF Processing (Default: school_information.pdf)
   ↓
5. Gradio Interface Launch (Port 7860)
```

### **Runtime Interaction Flow**:

**Task 1 (Text)**:
```
User Types → History Normalization → System Prompt Injection → 
Ollama Streaming → Real-time UI Update
```

**Task 1 (Speech)**:
```
User Records Audio → Whisper STT → LLM Processing → 
EdgeTTS Synthesis → Audio Playback + Transcript
```

**Task 2**:
```
User Input (Text/Voice) → Identity Check → 
[Hardcoded Response OR LLM Call] → TTS → Voice Output
```

**Task 3**:
```
User Question → Hybrid Retrieval (Vector + Keyword) → 
Context Assembly → LLM Streaming → Answer Display
```

---

## 🚀 Running Locally

### **Option 1: Fast Run (Recommended for Development)**
```powershell
# From the 10xai folder
.\run-fast.ps1
```
This mounts your local `app.py` into the container, skipping rebuild.

### **Option 2: Full Docker Build**
```bash
# Build the image
docker build -t ai-intern-final .

# Run the container
docker run -p 7860:7860 ai-intern-final
```

### **Option 3: Direct Python (Requires Dependencies)**
```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama server separately
ollama serve &
ollama pull gemma3:1b

# Run the app
python app.py
```

**Access**: Open browser to `http://localhost:7860`

---

## 🛠️ Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Gemma 3:1b (Ollama) | Text generation for all tasks |
| **STT** | OpenAI Whisper (tiny.en) | Speech-to-text transcription |
| **TTS** | EdgeTTS (en-US-AriaNeural) | Text-to-speech synthesis |
| **Embeddings** | SentenceTransformer (all-MiniLM-L6-v2) | Semantic search for RAG |
| **Vector DB** | FAISS (IndexFlatIP) | Efficient similarity search |
| **PDF Processing** | PyPDF2 | Text extraction from documents |
| **Text Splitting** | LangChain RecursiveCharacterTextSplitter | Intelligent chunking |
| **UI Framework** | Gradio | Interactive web interface |
| **Container** | Docker | Unified deployment |

---

## 📊 Key Parameters

```python
# Task 1 & 2
LLM_MODEL = "gemma3:1b"
WHISPER_MODEL_SIZE = "tiny.en"
TTS_VOICE = "en-US-AriaNeural"

# Task 3 RAG
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
SIMILARITY_THRESHOLD = 0.1  # Lowered for better recall
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

---

## 🚢 Deployment Method Explanation

### **Why Docker?**

This project uses **Docker containerization** for deployment. Here's the rationale:

**Advantages**:
1. **Reproducibility**: Ensures consistent environment across different systems
2. **Dependency Management**: All dependencies (Python packages, Ollama, models) bundled together
3. **Isolation**: No conflicts with system Python or other applications
4. **Easy Distribution**: Single `docker run` command to start everything
5. **Cross-Platform**: Works on Windows, macOS, and Linux without modification

**Deployment Architecture**:
```
Docker Container
├── Ubuntu Base Image
├── Ollama Server (Background Process)
│   └── Gemma 3:1b Model (Pre-pulled)
├── Python 3.11 Environment
│   ├── Whisper Model (Auto-downloaded)
│   ├── SentenceTransformer Model (Auto-downloaded)
│   └── Application Dependencies
└── Gradio Web Server (Port 7860)
```

**Alternative Deployment Options** (not implemented, but possible):
- **Cloud Deployment**: Deploy to Hugging Face Spaces, AWS EC2, or Google Cloud Run
- **Local Python**: Direct `python app.py` (requires manual Ollama setup)
- **Kubernetes**: For production multi-replica deployment
- **Serverless**: Split into microservices (STT, LLM, TTS as separate functions)

**Why Local Over Cloud?**:
- No API costs (completely free to run)
- Data privacy (school PDF stays local)
- No rate limits
- Works offline (except EdgeTTS)
- Educational value (understanding full stack)

---

## 🎯 Design Decisions

1. **Hybrid Search (Task 3)**: Combines vector similarity with keyword matching to handle both semantic and exact-match queries.

2. **Hardcoded Identity (Task 2)**: Bypasses LLM for identity questions to prevent hallucination and ensure consistent branding.

3. **Streaming Responses**: All LLM interactions use streaming for better UX and perceived performance.

4. **Async TTS**: EdgeTTS runs asynchronously to prevent blocking the main thread.

5. **Low Threshold RAG**: SIMILARITY_THRESHOLD set to 0.1 to maximize recall, relying on LLM to filter irrelevant context.

---

## 📁 Project Structure

```
10xai/
├── app.py                    # Main application
├── Dockerfile                # Container definition
├── requirements.txt          # Python dependencies
├── school_information.pdf    # Default RAG document
├── run-fast.ps1             # Quick run script
└── README.md                # This file
```

---

---

## 🐛 Troubleshooting

### **Common Issues & Solutions**

#### **1. Docker Build Fails**
**Error**: `ERROR: failed to solve: process "/bin/sh -c apt-get update..."`

**Solution**:
```bash
# Clear Docker cache and rebuild
docker system prune -a
docker build --no-cache -t ai-intern-final .
```

#### **2. Port 7860 Already in Use**
**Error**: `Bind for 0.0.0.0:7860 failed: port is already allocated`

**Solution**:
```bash
# Option 1: Stop the conflicting container
docker ps
docker stop <container-id>

# Option 2: Use a different port
docker run -p 8080:7860 ai-intern-final
# Then access at http://localhost:8080
```

#### **3. Ollama Model Not Loading**
**Error**: `Error communicating with Ollama`

**Solution**:
- Wait 30-60 seconds after container start for Ollama to initialize
- Check logs: `docker logs <container-id>`
- Verify model is pulled: Inside container, run `ollama list`

#### **4. Out of Memory**
**Error**: Container crashes or freezes

**Solution**:
- Increase Docker memory limit to at least 4GB:
  - Docker Desktop → Settings → Resources → Memory
- Close other applications to free RAM

#### **5. Whisper Model Download Fails**
**Error**: `Error loading Whisper`

**Solution**:
- Check internet connection
- Manually download model:
  ```python
  import whisper
  whisper.load_model("tiny.en")
  ```

### **Debug Mode**

Enable verbose logging by checking the terminal output:
- **Task 3 RAG**: Shows retrieved chunks, similarity scores, and keyword matches
  ```
  DEBUG: Query 'What is the fee structure?'
    - Vector Match: Chunk 1 (Score: 0.6001)
    - Keyword Match: Chunk 2 (Matches: 2)
  ```
- **Task 2 Identity**: Displays pattern matching logic
  ```
  DEBUG Identity Check: 'who are you' -> is_identity: True
  ```
- **Ollama**: Shows LLM request/response timing
  ```
  [GIN] 2026/01/29 - 13:32:18 | 200 | 15.76s | POST "/api/chat"
  ```

### **Performance Tips**

1. **First Run**: Initial startup takes 5-10 minutes (model downloads)
2. **Subsequent Runs**: ~30 seconds (models cached)
3. **Fast Development**: Use `run-fast.ps1` to skip rebuild
4. **Production**: Build once, run multiple times with same image

---

## ⚠️ Known Limitations & Future Improvements

### **Task 1: Text & Speech Models**

**Current Limitations**:
1. **Model Size**: Gemma 3:1b is a small model, may produce less sophisticated responses compared to larger models (7B+)
2. **CPU-Only**: No GPU acceleration implemented, slower inference on complex queries
3. **Whisper Accuracy**: Tiny variant may struggle with heavy accents or noisy environments
4. **TTS Dependency**: EdgeTTS requires internet connection (not fully offline)

**Potential Improvements**:
- Add GPU support for faster inference
- Implement model quantization (GGUF) for better performance
- Add fallback TTS (e.g., piper-tts) for offline mode
- Support multiple languages

---

### **Task 2: LUCA Voice Assistant**

**Current Limitations**:
1. **Identity Pattern Matching**: Hardcoded patterns may miss variations like "what's ur name" or "introduce yourself"
2. **Voice Quality**: EdgeTTS voice may sound robotic compared to premium TTS services
3. **Latency**: Sequential pipeline (STT → LLM → TTS) adds 5-10 seconds delay
4. **No Conversation Memory**: Each voice interaction is stateless (history not preserved across audio inputs)

**Potential Improvements**:
- Expand identity pattern list with fuzzy matching
- Implement streaming TTS for lower latency
- Add conversation memory for multi-turn voice dialogs
- Support wake word detection ("Hey LUCA")

---

### **Task 3: RAG-Based School Assistant**

**Current Limitations**:
1. **Small LLM**: Gemma 3:1b sometimes mixes information from different sections despite explicit prompts
2. **Chunk Size**: Fixed 600-char chunks may split tables awkwardly
3. **Similarity Threshold**: Set to 0.1 (very low) to maximize recall, may retrieve irrelevant chunks
4. **No Multi-Document**: Only supports single PDF at a time
5. **Table Parsing**: PyPDF2 may not preserve table structure perfectly
6. **Keyword Search**: Simple string matching, no stemming or synonyms

**Potential Improvements**:
- Use larger LLM (7B+) for better reasoning
- Implement table-aware chunking (preserve row integrity)
- Add re-ranking step after retrieval (cross-encoder)
- Support multiple PDFs with metadata filtering
- Use advanced PDF parsers (e.g., unstructured.io, pdfplumber)
- Implement semantic keyword expansion

---

### **General System Limitations**

1. **Resource Requirements**: Requires 4GB+ RAM, may not run on low-end systems
2. **Docker Size**: Image is ~5GB due to model weights
3. **No Authentication**: No user management or access control
4. **No Persistence**: Uploaded PDFs and conversations not saved between restarts
5. **Single User**: Not designed for concurrent multi-user access
6. **Error Handling**: Limited graceful degradation (e.g., if Ollama fails, entire app breaks)

**Potential Improvements**:
- Add user authentication and session management
- Implement database for conversation history
- Add Redis for caching and session storage
- Implement proper error boundaries and fallbacks
- Add monitoring and logging (e.g., Prometheus, Grafana)

---

### **Assignment-Specific Notes**

As per assignment guidelines:
- ✅ **Code Quality**: Modular design with clear separation of concerns
- ✅ **Documentation**: Comprehensive README with architecture and workflows
- ✅ **Reasoning**: Model choices justified based on resource constraints and performance
- ✅ **Limitations**: Clearly documented above
- ⚠️ **UI Polish**: Minimal (as stated: "architecture matters more than UI polish")

---

## 📝 License

This project was created as part of the 10X Technologies AI Intern assignment.

---

**Built with ❤️ for 10X Technologies**
